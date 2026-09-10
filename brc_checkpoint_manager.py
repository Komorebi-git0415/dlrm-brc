"""Application-level BRC checkpoint manager.

This module connects:

    ext4 BRC lifecycle
        +
    dirty-block planning
        +
    block materialization

It intentionally does not implement training-state sidecars, manifests,
restore policy, or interruption orchestration. Those belong to later
Stage-C steps.

Kernel ABI baseline:
    Linux 6.12.103-brc3
    commit 1983e6e246c9
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

import brc_uapi
from brc_block_materializer import reconstruct_blocks
from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    CheckpointLayout,
    DirtyBlockPlan,
    plan_dirty_blocks,
)
from brc_materialization import (
    MaterializationStats,
    materialize_dirty_blocks,
)
from brc_materialization_policy import (
    MaterializationPolicy,
)


@dataclass(frozen=True)
class CheckpointResult:
    """Result of one successfully sealed BRC checkpoint."""

    generation: int
    path: str
    dirty_blocks: int
    dirty_rows: int
    materialization: MaterializationStats | None


def _pwrite_all(
    fd: int,
    data: bytes,
    offset: int,
) -> None:
    written = 0

    while written < len(data):
        count = os.pwrite(
            fd,
            data[written:],
            offset + written,
        )

        if count <= 0:
            raise RuntimeError(
                "pwrite made no progress"
            )

        written += count


def _root_plans(
    layout: CheckpointLayout,
) -> tuple[DirtyBlockPlan, ...]:
    """Treat every real embedding row as dirty for generation zero."""

    plans = []

    for block_id in range(
        layout.num_blocks
    ):
        rows = layout.row_refs_for_block(
            block_id
        )

        plans.append(
            DirtyBlockPlan(
                block_id=block_id,
                dirty_rows=rows,
            )
        )

    return tuple(plans)


class BRCCkptManager:
    """Manage one linear BRC embedding checkpoint lineage."""

    def __init__(
        self,
        checkpoint_dir,
        layout: CheckpointLayout,
        policy: MaterializationPolicy,
    ):
        if not isinstance(
            layout,
            CheckpointLayout,
        ):
            raise TypeError(
                "layout must be a CheckpointLayout"
            )

        if not isinstance(
            policy,
            MaterializationPolicy,
        ):
            raise TypeError(
                "policy must be a MaterializationPolicy"
            )

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.layout = layout
        self.policy = policy

        self._ledger_fd = None
        self._session_fd = None
        self._parent_fd = None
        self._parent_path = None
        self._generation = None

        self._started = False
        self._broken = False

    @property
    def generation(self):
        return self._generation

    @property
    def parent_path(self):
        if self._parent_path is None:
            return None

        return str(
            self._parent_path
        )

    @property
    def ledger_path(self):
        return str(
            self.checkpoint_dir
            / "lineage.ledger"
        )

    def _ensure_usable(self):
        if not self._started:
            raise RuntimeError(
                "BRC checkpoint manager has not been started"
            )

        if self._broken:
            raise RuntimeError(
                "BRC checkpoint manager is unusable after "
                "an incomplete child checkpoint"
            )

    def start_new(self) -> None:
        """Create a new persistent lineage.

        The checkpoint directory must already exist and must not contain
        an existing lineage ledger.
        """

        if self._started:
            raise RuntimeError(
                "BRC checkpoint manager is already started"
            )

        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(
                "checkpoint directory does not exist: "
                f"{self.checkpoint_dir}"
            )

        ledger_path = (
            self.checkpoint_dir
            / "lineage.ledger"
        )

        ledger_fd = os.open(
            ledger_path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL,
            0o600,
        )

        try:
            session_fd = (
                brc_uapi.lineage_begin(
                    ledger_fd
                )
            )
        except Exception:
            os.close(
                ledger_fd
            )
            raise

        self._ledger_fd = ledger_fd
        self._session_fd = session_fd
        self._started = True
        self._generation = None

    def resume(
        self,
        generation: int,
        parent_path,
    ) -> None:
        """Resume an existing published BRC lineage.

        ``generation`` and ``parent_path`` are application metadata.
        Later stages obtain them from the published checkpoint manifest.
        The kernel ledger restores the BRC session itself.
        """

        if self._started:
            raise RuntimeError(
                "BRC checkpoint manager is already started"
            )

        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 0
        ):
            raise ValueError(
                "generation must be a non-negative integer"
            )

        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(
                "checkpoint directory does not exist: "
                f"{self.checkpoint_dir}"
            )

        ledger_path = (
            self.checkpoint_dir
            / "lineage.ledger"
        )

        parent_path = Path(parent_path)

        if not parent_path.is_absolute():
            parent_path = (
                self.checkpoint_dir
                / parent_path
            )

        ledger_fd = os.open(
            ledger_path,
            os.O_RDWR,
        )

        parent_fd = None

        try:
            parent_fd = os.open(
                parent_path,
                os.O_RDONLY,
            )

            ledger_stat = os.fstat(
                ledger_fd
            )

            parent_stat = os.fstat(
                parent_fd
            )

            if not stat.S_ISREG(
                parent_stat.st_mode
            ):
                raise ValueError(
                    "resume parent must be a regular file"
                )

            if (
                parent_stat.st_size
                != self.layout.image_size
            ):
                raise ValueError(
                    "resume parent size does not match "
                    "checkpoint layout"
                )

            if (
                parent_stat.st_dev
                != ledger_stat.st_dev
            ):
                raise ValueError(
                    "resume parent and lineage ledger must "
                    "be on the same filesystem"
                )

            session_fd = (
                brc_uapi.lineage_resume(
                    ledger_fd
                )
            )

        except Exception:
            if parent_fd is not None:
                os.close(
                    parent_fd
                )

            os.close(
                ledger_fd
            )

            raise

        self._ledger_fd = ledger_fd
        self._session_fd = session_fd
        self._parent_fd = parent_fd
        self._parent_path = parent_path
        self._generation = generation
        self._started = True
        self._broken = False

    def create_root(
        self,
        embedding_tables,
    ) -> CheckpointResult:
        """Write and seal generation zero from current live weights."""

        self._ensure_usable()

        if self._generation is not None:
            raise RuntimeError(
                "root checkpoint already exists"
            )

        path = (
            self.checkpoint_dir
            / "C0.img"
        )

        fd = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL,
            0o600,
        )

        try:
            os.ftruncate(
                fd,
                self.layout.image_size,
            )

            plans = _root_plans(
                self.layout
            )

            blocks, stats = (
                reconstruct_blocks(
                    self.layout,
                    plans,
                    embedding_tables,
                )
            )

            for block in blocks:
                _pwrite_all(
                    fd,
                    block.data,
                    block.block_id
                    * BLOCK_SIZE,
                )

            brc_uapi.seal(
                fd,
                self._session_fd,
            )

        except Exception:
            os.close(fd)
            raise

        self._parent_fd = fd
        self._parent_path = path
        self._generation = 0

        return CheckpointResult(
            generation=0,
            path=str(path),
            dirty_blocks=len(plans),
            dirty_rows=self.layout.total_rows,
            materialization=None,
        )

    def create_child(
        self,
        dirty_rows_by_table,
        embedding_tables,
    ) -> CheckpointResult:
        """Create and seal the next BRC generation."""

        self._ensure_usable()

        if self._generation is None:
            raise RuntimeError(
                "root checkpoint must be created first"
            )

        next_generation = (
            self._generation + 1
        )

        path = (
            self.checkpoint_dir
            / f"C{next_generation}.img"
        )

        child_fd = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL,
            0o600,
        )

        create_succeeded = False

        try:
            if os.fstat(
                child_fd
            ).st_size != 0:
                raise AssertionError(
                    "new child must be empty before BRC_CREATE"
                )

            brc_uapi.create(
                child_fd=child_fd,
                parent_fd=self._parent_fd,
                session_fd=self._session_fd,
            )

            create_succeeded = True

            os.ftruncate(
                child_fd,
                self.layout.image_size,
            )

            plans = plan_dirty_blocks(
                self.layout,
                dirty_rows_by_table,
            )

            blocks, stats = (
                materialize_dirty_blocks(
                    self.layout,
                    plans,
                    embedding_tables,
                    self.policy,
                    parent_fd=self._parent_fd,
                )
            )

            for block in blocks:
                _pwrite_all(
                    child_fd,
                    block.data,
                    block.block_id
                    * BLOCK_SIZE,
                )

            brc_uapi.seal(
                child_fd,
                self._session_fd,
            )

        except Exception:
            if create_succeeded:
                # brc3 deliberately has no Abort UAPI.
                # Once CREATE succeeded, failure before a successful
                # SEAL leaves an unpublished child outside the
                # supported continuation path for this manager.
                self._broken = True

            os.close(
                child_fd
            )

            raise

        old_parent_fd = (
            self._parent_fd
        )

        self._parent_fd = child_fd
        self._parent_path = path
        self._generation = (
            next_generation
        )

        if old_parent_fd is not None:
            os.close(
                old_parent_fd
            )

        dirty_rows = sum(
            plan.dirty_count
            for plan in plans
        )

        return CheckpointResult(
            generation=next_generation,
            path=str(path),
            dirty_blocks=len(plans),
            dirty_rows=dirty_rows,
            materialization=stats,
        )

    def close(self) -> None:
        """Close application-owned descriptors."""

        if self._parent_fd is not None:
            os.close(
                self._parent_fd
            )

            self._parent_fd = None

        if self._session_fd is not None:
            brc_uapi.close_session(
                self._session_fd
            )

            self._session_fd = None

        if self._ledger_fd is not None:
            os.close(
                self._ledger_fd
            )

            self._ledger_fd = None

        self._started = False

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()
