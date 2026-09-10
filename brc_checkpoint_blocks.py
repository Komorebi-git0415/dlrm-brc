"""Checkpoint storage layout and dirty-block planning for BRC.

This module implements the common planning layer shared by future
checkpoint block materializers:

    dirty runtime rows
        -> checkpoint storage slots
        -> unique dirty 4 KiB blocks

The runtime row IDs are the post-stage-1 ``new_local_id`` values used by
the materialized Criteo dataset.

Two checkpoint storage modes are supported:

``local-only``
    Rows remain table-contiguous after the stage-1 local permutation.

``global-mixed``
    Rows use the stage-2 global storage layout produced by
    ``brc_global_layout.build_global_storage_layout``.

This module does not read parent checkpoints and does not access live
embedding weights. Those responsibilities belong to later C4
materialization stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from brc_global_layout import split_global_slots_by_table


BLOCK_SIZE = 4096
EMBEDDING_DIM = 16
EMBEDDING_DTYPE = np.dtype(np.float32)

ROW_BYTES = EMBEDDING_DIM * EMBEDDING_DTYPE.itemsize

if BLOCK_SIZE % ROW_BYTES != 0:
    raise RuntimeError(
        "embedding row size must divide the BRC block size"
    )

ROWS_PER_BLOCK = BLOCK_SIZE // ROW_BYTES

LAYOUT_LOCAL_ONLY = "local-only"
LAYOUT_GLOBAL_MIXED = "global-mixed"

_VALID_LAYOUT_MODES = {
    LAYOUT_LOCAL_ONLY,
    LAYOUT_GLOBAL_MIXED,
}


@dataclass(frozen=True)
class StorageRowRef:
    """One runtime embedding row in checkpoint storage order."""

    table_id: int
    row_id: int
    storage_slot: int
    row_in_block: int


@dataclass(frozen=True)
class DirtyBlockPlan:
    """Dirty-row information for one checkpoint storage block."""

    block_id: int
    dirty_rows: tuple[StorageRowRef, ...]

    @property
    def dirty_count(self) -> int:
        return len(self.dirty_rows)

    @property
    def dirty_slots(self) -> tuple[int, ...]:
        return tuple(
            row.storage_slot
            for row in self.dirty_rows
        )


def _validate_table_sizes(
    table_sizes: Sequence[int],
) -> np.ndarray:
    sizes = np.asarray(
        table_sizes,
        dtype=np.int64,
    )

    if sizes.ndim != 1:
        raise ValueError(
            "table_sizes must be one-dimensional"
        )

    if len(sizes) == 0:
        raise ValueError(
            "at least one embedding table is required"
        )

    if np.any(sizes <= 0):
        raise ValueError(
            "every embedding table must contain at least one row"
        )

    return sizes


class CheckpointLayout:
    """Map runtime embedding rows to checkpoint storage slots."""

    def __init__(
        self,
        table_sizes: Sequence[int],
        mode: str,
        global_layout=None,
    ):
        self._table_sizes = _validate_table_sizes(
            table_sizes
        )

        if mode not in _VALID_LAYOUT_MODES:
            raise ValueError(
                "unsupported checkpoint layout mode: "
                f"{mode!r}"
            )

        self._mode = mode
        self._num_tables = len(
            self._table_sizes
        )

        self._table_offsets = np.zeros(
            self._num_tables + 1,
            dtype=np.int64,
        )

        self._table_offsets[1:] = np.cumsum(
            self._table_sizes,
            dtype=np.int64,
        )

        self._total_rows = int(
            self._table_offsets[-1]
        )

        self._global_layout = None
        self._global_slots_by_table = None
        self._flat_local_by_global_slot = None

        if mode == LAYOUT_LOCAL_ONLY:
            if global_layout is not None:
                raise ValueError(
                    "global_layout must not be provided "
                    "for local-only mode"
                )

        else:
            if global_layout is None:
                raise ValueError(
                    "global_layout is required for "
                    "global-mixed mode"
                )

            self._initialize_global_layout(
                global_layout
            )

    def _initialize_global_layout(
        self,
        global_layout,
    ) -> None:
        try:
            num_tables = int(
                global_layout["num_tables"]
            )
            total_rows = int(
                global_layout["total_rows"]
            )
            layout_offsets = np.asarray(
                global_layout["table_offsets"],
                dtype=np.int64,
            )
            inverse = np.asarray(
                global_layout[
                    "flat_local_by_global_slot"
                ],
                dtype=np.int64,
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "invalid global_layout"
            ) from None

        if num_tables != self._num_tables:
            raise ValueError(
                "global_layout table count does not "
                "match table_sizes"
            )

        if total_rows != self._total_rows:
            raise ValueError(
                "global_layout total_rows does not "
                "match table_sizes"
            )

        if not np.array_equal(
            layout_offsets,
            self._table_offsets,
        ):
            raise ValueError(
                "global_layout table_offsets do not "
                "match table_sizes"
            )

        if inverse.shape != (
            self._total_rows,
        ):
            raise ValueError(
                "invalid flat_local_by_global_slot"
            )

        expected = np.arange(
            self._total_rows,
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(inverse),
            expected,
        ):
            raise ValueError(
                "flat_local_by_global_slot must be "
                "a permutation"
            )

        slots_by_table = (
            split_global_slots_by_table(
                global_layout
            )
        )

        if len(slots_by_table) != self._num_tables:
            raise ValueError(
                "invalid global slot table count"
            )

        for table_id, slots in enumerate(
            slots_by_table
        ):
            slots = np.asarray(
                slots,
                dtype=np.int64,
            )

            if slots.shape != (
                int(
                    self._table_sizes[
                        table_id
                    ]
                ),
            ):
                raise ValueError(
                    "global slot vector length mismatch "
                    f"for table {table_id}"
                )

            if (
                np.any(slots < 0)
                or np.any(
                    slots >= self._total_rows
                )
            ):
                raise ValueError(
                    "global storage slot out of range"
                )

            slots_by_table[
                table_id
            ] = slots

        self._global_layout = global_layout
        self._global_slots_by_table = (
            slots_by_table
        )
        self._flat_local_by_global_slot = (
            inverse
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def num_tables(self) -> int:
        return self._num_tables

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def rows_per_block(self) -> int:
        return ROWS_PER_BLOCK

    @property
    def num_blocks(self) -> int:
        return (
            self._total_rows
            + ROWS_PER_BLOCK
            - 1
        ) // ROWS_PER_BLOCK

    @property
    def image_size(self) -> int:
        return self.num_blocks * BLOCK_SIZE

    @property
    def table_sizes(self) -> np.ndarray:
        return self._table_sizes.copy()

    def _validate_runtime_row(
        self,
        table_id: int,
        row_id: int,
    ) -> tuple[int, int]:
        if not isinstance(
            table_id,
            (int, np.integer),
        ):
            raise TypeError(
                "table_id must be an integer"
            )

        if not isinstance(
            row_id,
            (int, np.integer),
        ):
            raise TypeError(
                "row_id must be an integer"
            )

        table_id = int(table_id)
        row_id = int(row_id)

        if (
            table_id < 0
            or table_id >= self._num_tables
        ):
            raise IndexError(
                f"table_id out of range: {table_id}"
            )

        table_size = int(
            self._table_sizes[
                table_id
            ]
        )

        if (
            row_id < 0
            or row_id >= table_size
        ):
            raise IndexError(
                "row_id out of range for table "
                f"{table_id}: {row_id}"
            )

        return table_id, row_id

    def storage_slot(
        self,
        table_id: int,
        row_id: int,
    ) -> int:
        """Return checkpoint storage slot for one runtime row."""

        table_id, row_id = (
            self._validate_runtime_row(
                table_id,
                row_id,
            )
        )

        if self._mode == LAYOUT_LOCAL_ONLY:
            return int(
                self._table_offsets[
                    table_id
                ]
                + row_id
            )

        return int(
            self._global_slots_by_table[
                table_id
            ][row_id]
        )

    def _flat_local_to_runtime_row(
        self,
        flat_local: int,
    ) -> tuple[int, int]:
        table_id = int(
            np.searchsorted(
                self._table_offsets,
                flat_local,
                side="right",
            )
            - 1
        )

        row_id = int(
            flat_local
            - self._table_offsets[
                table_id
            ]
        )

        return table_id, row_id

    def row_ref_for_slot(
        self,
        storage_slot: int,
    ) -> StorageRowRef:
        """Reverse-map one valid checkpoint storage slot."""

        if not isinstance(
            storage_slot,
            (int, np.integer),
        ):
            raise TypeError(
                "storage_slot must be an integer"
            )

        storage_slot = int(
            storage_slot
        )

        if (
            storage_slot < 0
            or storage_slot >= self._total_rows
        ):
            raise IndexError(
                "storage_slot out of range: "
                f"{storage_slot}"
            )

        if self._mode == LAYOUT_LOCAL_ONLY:
            flat_local = storage_slot
        else:
            flat_local = int(
                self._flat_local_by_global_slot[
                    storage_slot
                ]
            )

        table_id, row_id = (
            self._flat_local_to_runtime_row(
                flat_local
            )
        )

        return StorageRowRef(
            table_id=table_id,
            row_id=row_id,
            storage_slot=storage_slot,
            row_in_block=(
                storage_slot
                % ROWS_PER_BLOCK
            ),
        )

    def row_refs_for_block(
        self,
        block_id: int,
    ) -> tuple[StorageRowRef, ...]:
        """Return all real embedding rows stored in one block.

        The final block can contain fewer than ROWS_PER_BLOCK rows.
        Padding bytes are not represented by StorageRowRef objects.
        """

        if not isinstance(
            block_id,
            (int, np.integer),
        ):
            raise TypeError(
                "block_id must be an integer"
            )

        block_id = int(
            block_id
        )

        if (
            block_id < 0
            or block_id >= self.num_blocks
        ):
            raise IndexError(
                f"block_id out of range: {block_id}"
            )

        start_slot = (
            block_id
            * ROWS_PER_BLOCK
        )

        end_slot = min(
            start_slot + ROWS_PER_BLOCK,
            self._total_rows,
        )

        return tuple(
            self.row_ref_for_slot(slot)
            for slot in range(
                start_slot,
                end_slot,
            )
        )


def _normalize_dirty_rows(
    layout: CheckpointLayout,
    dirty_rows_by_table,
) -> list[np.ndarray]:
    if len(dirty_rows_by_table) != layout.num_tables:
        raise ValueError(
            "dirty row table count does not "
            "match checkpoint layout"
        )

    normalized = []

    table_sizes = layout.table_sizes

    for table_id, rows in enumerate(
        dirty_rows_by_table
    ):
        array = np.asarray(
            rows,
        )

        if array.ndim != 1:
            raise ValueError(
                "dirty rows for table "
                f"{table_id} must be one-dimensional"
            )

        if not np.issubdtype(
            array.dtype,
            np.integer,
        ):
            raise TypeError(
                "dirty rows for table "
                f"{table_id} must be integers"
            )

        array = array.astype(
            np.int64,
            copy=False,
        )

        if len(array) != 0:
            if (
                np.any(array < 0)
                or np.any(
                    array
                    >= table_sizes[
                        table_id
                    ]
                )
            ):
                raise IndexError(
                    "dirty row out of range for table "
                    f"{table_id}"
                )

            # A row can appear multiple times during one checkpoint
            # interval. It is dirty only once for planning purposes.
            array = np.unique(
                array
            )

        normalized.append(
            array
        )

    return normalized


def plan_dirty_blocks(
    layout: CheckpointLayout,
    dirty_rows_by_table,
) -> tuple[DirtyBlockPlan, ...]:
    """Build deterministic dirty-block plans.

    ``dirty_rows_by_table[t]`` contains post-stage-1 runtime row IDs for
    table ``t``.

    Duplicate dirty rows are removed. Returned blocks are ordered by
    increasing block ID, and dirty rows within each block are ordered by
    increasing checkpoint storage slot.
    """

    dirty_rows_by_table = (
        _normalize_dirty_rows(
            layout,
            dirty_rows_by_table,
        )
    )

    rows_by_block = {}

    for table_id, dirty_rows in enumerate(
        dirty_rows_by_table
    ):
        for row_id_value in dirty_rows:
            row_id = int(
                row_id_value
            )

            storage_slot = (
                layout.storage_slot(
                    table_id,
                    row_id,
                )
            )

            block_id = (
                storage_slot
                // ROWS_PER_BLOCK
            )

            ref = StorageRowRef(
                table_id=table_id,
                row_id=row_id,
                storage_slot=storage_slot,
                row_in_block=(
                    storage_slot
                    % ROWS_PER_BLOCK
                ),
            )

            rows_by_block.setdefault(
                block_id,
                [],
            ).append(
                ref
            )

    plans = []

    for block_id in sorted(
        rows_by_block
    ):
        dirty_rows = tuple(
            sorted(
                rows_by_block[
                    block_id
                ],
                key=lambda ref: (
                    ref.storage_slot
                ),
            )
        )

        plans.append(
            DirtyBlockPlan(
                block_id=block_id,
                dirty_rows=dirty_rows,
            )
        )

    return tuple(
        plans
    )
