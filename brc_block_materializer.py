"""BRC checkpoint block materialization.

The materialization layer converts DirtyBlockPlan objects into complete
4 KiB checkpoint blocks.

This module is device agnostic. Embedding weights may reside on CPU or
CUDA devices. The same reconstruction interface is used in both cases.

Current stage:
    full reconstruction from current live embedding weights.

Later stages add:
    parent block RMW
    cache-residency-aware hybrid selection
"""

from __future__ import annotations

from collections import defaultdict
import os
from dataclasses import dataclass
from typing import Sequence

import torch

from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    EMBEDDING_DIM,
    ROW_BYTES,
    CheckpointLayout,
    DirtyBlockPlan,
    StorageRowRef,
)


@dataclass(frozen=True)
class MaterializedBlock:
    """One complete checkpoint block ready for pwrite."""

    block_id: int
    data: bytes

    def __post_init__(self) -> None:
        if len(self.data) != BLOCK_SIZE:
            raise ValueError(
                "materialized block must contain exactly "
                f"{BLOCK_SIZE} bytes"
            )


@dataclass(frozen=True)
class ReconstructionStats:
    """Basic accounting for reconstruction materialization."""

    num_blocks: int
    gathered_rows: int
    host_bytes: int


def _embedding_weight(
    embedding_tables,
    table_id: int,
) -> torch.Tensor:
    """Return the live weight tensor for one embedding table.

    The target BRC experiment uses standard nn.EmbeddingBag modules.
    A raw Tensor is also accepted to keep the materialization layer
    independently testable.
    """

    table = embedding_tables[table_id]

    if isinstance(table, torch.Tensor):
        weight = table
    else:
        try:
            weight = table.weight
        except AttributeError:
            raise TypeError(
                "embedding table {} does not expose a "
                ".weight tensor".format(table_id)
            ) from None

    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            "embedding table {} weight is not a torch Tensor".format(
                table_id
            )
        )

    if weight.ndim != 2:
        raise ValueError(
            "embedding table {} weight must be two-dimensional".format(
                table_id
            )
        )

    if weight.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            "embedding table {} has embedding dimension {}; "
            "expected {}".format(
                table_id,
                weight.shape[1],
                EMBEDDING_DIM,
            )
        )

    if weight.dtype != torch.float32:
        raise TypeError(
            "embedding table {} must use float32 weights; "
            "got {}".format(
                table_id,
                weight.dtype,
            )
        )

    return weight


def _validate_embedding_tables(
    layout: CheckpointLayout,
    embedding_tables,
) -> list[torch.Tensor]:
    if len(embedding_tables) != layout.num_tables:
        raise ValueError(
            "embedding table count does not match checkpoint layout"
        )

    table_sizes = layout.table_sizes
    weights = []

    for table_id in range(layout.num_tables):
        weight = _embedding_weight(
            embedding_tables,
            table_id,
        )

        if weight.shape[0] != int(table_sizes[table_id]):
            raise ValueError(
                "embedding table {} row count {} does not match "
                "checkpoint layout row count {}".format(
                    table_id,
                    weight.shape[0],
                    int(table_sizes[table_id]),
                )
            )

        weights.append(weight)

    return weights


def _validate_plans(
    layout: CheckpointLayout,
    plans: Sequence[DirtyBlockPlan],
) -> tuple[DirtyBlockPlan, ...]:
    plans = tuple(plans)

    previous = -1
    seen = set()

    for plan in plans:
        if not isinstance(plan, DirtyBlockPlan):
            raise TypeError(
                "plans must contain DirtyBlockPlan objects"
            )

        block_id = plan.block_id

        if block_id < 0 or block_id >= layout.num_blocks:
            raise IndexError(
                f"dirty block out of range: {block_id}"
            )

        if block_id in seen:
            raise ValueError(
                f"duplicate dirty block plan: {block_id}"
            )

        if block_id <= previous:
            raise ValueError(
                "dirty block plans must be ordered by "
                "increasing block_id"
            )

        seen.add(block_id)
        previous = block_id

    return plans


def reconstruct_blocks(
    layout: CheckpointLayout,
    plans: Sequence[DirtyBlockPlan],
    embedding_tables,
) -> tuple[
    tuple[MaterializedBlock, ...],
    ReconstructionStats,
]:
    """Reconstruct complete dirty blocks from current live weights.

    The function is intentionally batch-oriented.

    All rows required by all reconstruction blocks are first grouped by
    embedding table. Each table is then gathered with one index_select
    operation. This avoids one GPU operation per checkpoint block when
    embedding weights reside on CUDA.

    The returned block payloads always contain exactly 4096 bytes.
    Unused rows in the final checkpoint block are zero padded.
    """

    plans = _validate_plans(
        layout,
        plans,
    )

    weights = _validate_embedding_tables(
        layout,
        embedding_tables,
    )

    if not plans:
        return (
            (),
            ReconstructionStats(
                num_blocks=0,
                gathered_rows=0,
                host_bytes=0,
            ),
        )

    # One host-side output buffer per dirty block.
    block_buffers = {
        plan.block_id: bytearray(BLOCK_SIZE)
        for plan in plans
    }

    # Group all rows needed by all reconstruction blocks by table.
    #
    # Each item is:
    #   (StorageRowRef, destination_block_id)
    #
    # row_ref_for_block() already returns rows in checkpoint storage
    # order. Grouping changes gather order but not final placement,
    # because every result is scattered back by row_in_block.
    refs_by_table = defaultdict(list)

    gathered_rows = 0

    for plan in plans:
        refs = layout.row_refs_for_block(
            plan.block_id
        )

        for ref in refs:
            refs_by_table[
                ref.table_id
            ].append(
                (
                    ref,
                    plan.block_id,
                )
            )

            gathered_rows += 1

    # Gather once per embedding table, regardless of the number of
    # reconstruction blocks that reference that table.
    for table_id in sorted(refs_by_table):
        items = refs_by_table[
            table_id
        ]

        weight = weights[
            table_id
        ]

        row_ids = torch.tensor(
            [
                ref.row_id
                for ref, _block_id in items
            ],
            dtype=torch.long,
            device=weight.device,
        )

        with torch.no_grad():
            gathered = torch.index_select(
                weight.detach(),
                0,
                row_ids,
            )

        # This is the device boundary.
        #
        # CPU weights:
        #     this is effectively a host-side contiguous conversion.
        #
        # CUDA weights:
        #     the same code performs a batched D2H transfer for the
        #     rows gathered from this table.
        #
        # Future GPU optimization may use pinned host buffers and
        # asynchronous copies without changing this public interface.
        gathered_host = (
            gathered
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )

        if gathered_host.shape != (
            len(items),
            EMBEDDING_DIM,
        ):
            raise RuntimeError(
                "unexpected gathered embedding shape"
            )

        gathered_bytes = (
            gathered_host
            .numpy()
            .tobytes(order="C")
        )

        expected_bytes = (
            len(items)
            * ROW_BYTES
        )

        if len(gathered_bytes) != expected_bytes:
            raise RuntimeError(
                "unexpected gathered embedding byte count"
            )

        for index, (
            ref,
            block_id,
        ) in enumerate(items):
            source_start = (
                index
                * ROW_BYTES
            )
            source_end = (
                source_start
                + ROW_BYTES
            )

            destination_start = (
                ref.row_in_block
                * ROW_BYTES
            )
            destination_end = (
                destination_start
                + ROW_BYTES
            )

            block_buffers[
                block_id
            ][
                destination_start:
                destination_end
            ] = gathered_bytes[
                source_start:
                source_end
            ]

    materialized = tuple(
        MaterializedBlock(
            block_id=plan.block_id,
            data=bytes(
                block_buffers[
                    plan.block_id
                ]
            ),
        )
        for plan in plans
    )

    stats = ReconstructionStats(
        num_blocks=len(materialized),
        gathered_rows=gathered_rows,
        host_bytes=(
            gathered_rows
            * ROW_BYTES
        ),
    )

    return materialized, stats


@dataclass(frozen=True)
class ParentRMWStats:
    """Accounting for parent read-modify-write materialization."""

    num_blocks: int
    gathered_rows: int
    host_bytes: int
    parent_read_bytes: int


def parent_rmw_blocks(
    layout: CheckpointLayout,
    plans: Sequence[DirtyBlockPlan],
    embedding_tables,
    parent_fd: int,
) -> tuple[
    tuple[MaterializedBlock, ...],
    ParentRMWStats,
]:
    """Materialize dirty blocks by parent read-modify-write.

    For every dirty block:

        1. read the complete 4 KiB parent block;
        2. gather only the currently dirty embedding rows;
        3. patch those rows into the parent block;
        4. return one complete 4 KiB child block.

    The function is batch-oriented with respect to embedding access.
    Dirty rows from all RMW blocks are grouped by embedding table, so
    each table is gathered with one index_select operation.

    Embedding weights may reside on CPU or CUDA devices. The public
    interface and checkpoint semantics are identical on both.
    """

    plans = _validate_plans(
        layout,
        plans,
    )

    weights = _validate_embedding_tables(
        layout,
        embedding_tables,
    )

    if not isinstance(parent_fd, int):
        raise TypeError(
            "parent_fd must be an integer file descriptor"
        )

    if not plans:
        return (
            (),
            ParentRMWStats(
                num_blocks=0,
                gathered_rows=0,
                host_bytes=0,
                parent_read_bytes=0,
            ),
        )

    # Read every complete parent block before patching.
    #
    # This intentionally uses normal buffered pread(). Later hybrid
    # selection will decide whether using this path is worthwhile based
    # on parent page-cache residency and dirty-row density.
    block_buffers = {}

    for plan in plans:
        offset = (
            plan.block_id
            * BLOCK_SIZE
        )

        parent_data = os.pread(
            parent_fd,
            BLOCK_SIZE,
            offset,
        )

        if len(parent_data) != BLOCK_SIZE:
            raise RuntimeError(
                "short parent checkpoint read for block {}: "
                "{} != {}".format(
                    plan.block_id,
                    len(parent_data),
                    BLOCK_SIZE,
                )
            )

        block_buffers[
            plan.block_id
        ] = bytearray(
            parent_data
        )

    # Gather only dirty rows, grouped across all RMW blocks by table.
    #
    # Each item:
    #   (StorageRowRef, destination_block_id)
    refs_by_table = defaultdict(list)

    gathered_rows = 0

    for plan in plans:
        for ref in plan.dirty_rows:
            refs_by_table[
                ref.table_id
            ].append(
                (
                    ref,
                    plan.block_id,
                )
            )

            gathered_rows += 1

    for table_id in sorted(
        refs_by_table
    ):
        items = refs_by_table[
            table_id
        ]

        weight = weights[
            table_id
        ]

        row_ids = torch.tensor(
            [
                ref.row_id
                for ref, _block_id in items
            ],
            dtype=torch.long,
            device=weight.device,
        )

        with torch.no_grad():
            gathered = torch.index_select(
                weight.detach(),
                0,
                row_ids,
            )

        # Same device boundary as reconstruction:
        #
        # CPU weights -> host contiguous tensor
        # CUDA weights -> batched D2H transfer
        gathered_host = (
            gathered
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )

        if gathered_host.shape != (
            len(items),
            EMBEDDING_DIM,
        ):
            raise RuntimeError(
                "unexpected gathered embedding shape"
            )

        gathered_bytes = (
            gathered_host
            .numpy()
            .tobytes(order="C")
        )

        expected_bytes = (
            len(items)
            * ROW_BYTES
        )

        if len(gathered_bytes) != expected_bytes:
            raise RuntimeError(
                "unexpected gathered embedding byte count"
            )

        for index, (
            ref,
            block_id,
        ) in enumerate(items):
            source_start = (
                index
                * ROW_BYTES
            )
            source_end = (
                source_start
                + ROW_BYTES
            )

            destination_start = (
                ref.row_in_block
                * ROW_BYTES
            )
            destination_end = (
                destination_start
                + ROW_BYTES
            )

            block_buffers[
                block_id
            ][
                destination_start:
                destination_end
            ] = gathered_bytes[
                source_start:
                source_end
            ]

    materialized = tuple(
        MaterializedBlock(
            block_id=plan.block_id,
            data=bytes(
                block_buffers[
                    plan.block_id
                ]
            ),
        )
        for plan in plans
    )

    stats = ParentRMWStats(
        num_blocks=len(materialized),
        gathered_rows=gathered_rows,
        host_bytes=(
            gathered_rows
            * ROW_BYTES
        ),
        parent_read_bytes=(
            len(materialized)
            * BLOCK_SIZE
        ),
    )

    return materialized, stats
