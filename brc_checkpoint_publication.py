"""Durable application-level publication for BRC checkpoints.

A BRC checkpoint becomes application-visible only after:

    1. Cn.img has already been successfully BRC_SEALed.
    2. Cn.state.pt is durably written.
    3. manifest.json is atomically updated and durably published.

The manifest replacement is the application-level publication point.

This module does not implement restore. Restore belongs to C7.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    EMBEDDING_DIM,
    ROW_BYTES,
    ROWS_PER_BLOCK,
    CheckpointLayout,
    checkpoint_layout_fingerprint,
)
from brc_checkpoint_manager import CheckpointResult


SIDECAR_FORMAT = "brc-training-state"
SIDECAR_VERSION = 1

MANIFEST_FORMAT = "brc-checkpoint-manifest"
MANIFEST_VERSION = 2
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class PublishedCheckpoint:
    generation: int
    checkpoint_path: str
    sidecar_path: str
    manifest_path: str


def _fsync_directory(directory: Path) -> None:
    """Durably persist directory-entry changes."""

    fd = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY,
    )

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_torch_save_no_overwrite(
    payload,
    final_path: Path,
) -> None:
    """Create one generation-specific sidecar durably."""

    if final_path.exists():
        raise FileExistsError(
            f"sidecar already exists: {final_path}"
        )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.tmp-",
        dir=final_path.parent,
    )

    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(
                payload,
                stream,
            )

            stream.flush()
            os.fsync(
                stream.fileno()
            )

        # The current design assumes one checkpoint publisher.
        if final_path.exists():
            raise FileExistsError(
                f"sidecar already exists: {final_path}"
            )

        os.replace(
            tmp_path,
            final_path,
        )

        _fsync_directory(
            final_path.parent
        )

    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

        raise


def _atomic_write_json(
    payload,
    final_path: Path,
) -> None:
    """Atomically and durably replace manifest.json."""

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.tmp-",
        dir=final_path.parent,
    )

    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                indent=2,
            )

            stream.write("\n")
            stream.flush()
            os.fsync(
                stream.fileno()
            )

        os.replace(
            tmp_path,
            final_path,
        )

        _fsync_directory(
            final_path.parent
        )

    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

        raise


def _read_manifest(
    manifest_path: Path,
):
    if not manifest_path.exists():
        return None

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        manifest = json.load(
            stream
        )

    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            "existing manifest has unexpected format"
        )

    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            "existing manifest has unsupported version"
        )

    generation = manifest.get(
        "generation"
    )

    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
    ):
        raise ValueError(
            "existing manifest has invalid generation"
        )

    return manifest


class CheckpointPublisher:
    """Publish sidecar state and the latest application manifest."""

    def __init__(
        self,
        checkpoint_dir,
        layout: CheckpointLayout,
    ):
        if not isinstance(
            layout,
            CheckpointLayout,
        ):
            raise TypeError(
                "layout must be a CheckpointLayout"
            )

        checkpoint_dir = Path(
            checkpoint_dir
        )

        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                "checkpoint directory does not exist: "
                f"{checkpoint_dir}"
            )

        self.checkpoint_dir = (
            checkpoint_dir.resolve()
        )
        self.layout = layout

    @property
    def manifest_path(self) -> str:
        return str(
            self.checkpoint_dir
            / MANIFEST_FILENAME
        )

    def publish(
        self,
        checkpoint: CheckpointResult,
        training_state: dict,
    ) -> PublishedCheckpoint:
        """Publish one already-sealed checkpoint generation."""

        if not isinstance(
            checkpoint,
            CheckpointResult,
        ):
            raise TypeError(
                "checkpoint must be a CheckpointResult"
            )

        if not isinstance(
            training_state,
            dict,
        ):
            raise TypeError(
                "training_state must be a dict"
            )

        generation = checkpoint.generation

        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 0
        ):
            raise ValueError(
                "checkpoint generation must be "
                "a non-negative integer"
            )

        manifest_path = (
            self.checkpoint_dir
            / MANIFEST_FILENAME
        )

        current_manifest = _read_manifest(
            manifest_path
        )

        if current_manifest is None:
            if generation != 0:
                raise ValueError(
                    "first application publication "
                    "must be generation 0"
                )
        else:
            expected_generation = (
                current_manifest["generation"]
                + 1
            )

            if generation != expected_generation:
                raise ValueError(
                    "checkpoint generations must be "
                    "published consecutively: expected "
                    f"{expected_generation}, got {generation}"
                )

        checkpoint_path = Path(
            checkpoint.path
        )

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                self.checkpoint_dir
                / checkpoint_path
            )

        checkpoint_path = checkpoint_path.resolve(
            strict=True
        )

        if checkpoint_path.parent != self.checkpoint_dir:
            raise ValueError(
                "checkpoint file must be directly inside "
                "the checkpoint directory"
            )

        expected_name = (
            f"C{generation}.img"
        )

        if checkpoint_path.name != expected_name:
            raise ValueError(
                "checkpoint filename does not match "
                f"generation {generation}: "
                f"expected {expected_name}"
            )

        checkpoint_stat = checkpoint_path.stat()

        if not stat.S_ISREG(
            checkpoint_stat.st_mode
        ):
            raise ValueError(
                "checkpoint must be a regular file"
            )

        if checkpoint_stat.st_size != self.layout.image_size:
            raise ValueError(
                "checkpoint size does not match "
                "checkpoint layout"
            )

        sidecar_path = (
            self.checkpoint_dir
            / f"C{generation}.state.pt"
        )

        sidecar_payload = {
            "format": SIDECAR_FORMAT,
            "version": SIDECAR_VERSION,
            "generation": generation,
            "state": training_state,
        }

        # Sidecar becomes durable BEFORE manifest publication.
        _atomic_torch_save_no_overwrite(
            sidecar_payload,
            sidecar_path,
        )

        sidecar_size = (
            sidecar_path.stat().st_size
        )

        manifest = {
            "format": MANIFEST_FORMAT,
            "version": MANIFEST_VERSION,
            "generation": generation,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_size": checkpoint_stat.st_size,
            "sidecar_file": sidecar_path.name,
            "sidecar_size": sidecar_size,
            "layout": {
                "mode": self.layout.mode,
                "num_tables": self.layout.num_tables,
                "total_rows": self.layout.total_rows,
                "image_size": self.layout.image_size,
                "block_size": BLOCK_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "row_bytes": ROW_BYTES,
                "rows_per_block": ROWS_PER_BLOCK,
                "table_sizes": [
                    int(value)
                    for value
                    in self.layout.table_sizes
                ],
                "fingerprint": (
                    checkpoint_layout_fingerprint(
                        self.layout
                    )
                ),
            },
        }

        # This atomic replacement is the application publication point.
        _atomic_write_json(
            manifest,
            manifest_path,
        )

        return PublishedCheckpoint(
            generation=generation,
            checkpoint_path=str(
                checkpoint_path
            ),
            sidecar_path=str(
                sidecar_path
            ),
            manifest_path=str(
                manifest_path
            ),
        )
