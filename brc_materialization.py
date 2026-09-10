"""Unified BRC dirty-block materialization.

This module combines:

    DirtyBlockPlan
        +
    MaterializationPolicy
        +
    Reconstruction materializer
        +
    Parent-RMW materializer

Supported experiment modes:

    only_reconstruct
    only_parent_rmw
    hybrid(threshold)

The implementation is device agnostic. Embedding weights can reside on
CPU in the current development VM or on CUDA devices on the final
experimental platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from brc_block_materializer import (
    MaterializedBlock,
    parent_rmw_blocks,
    reconstruct_blocks,
)
from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    CheckpointLayout,
    DirtyBlockPlan,
)
from brc_materialization_policy import (
    METHOD_PARENT_RMW,
    METHOD_RECONSTRUCT,
    MaterializationPolicy,
)


@dataclass(frozen=True)
class MaterializationStats:
    """Unified statistics for one checkpoint materialization."""

    num_blocks: int

    reconstruct_blocks: int
    parent_rmw_blocks: int

    reconstruct_gathered_rows: int
    parent_rmw_gathered_rows: int

    reconstruct_host_bytes: int
    parent_rmw_host_bytes: int

    parent_read_bytes: int
    output_bytes: int

    @property
    def total_gathered_rows(self) -> int:
        return (
            self.reconstruct_gathered_rows
            + self.parent_rmw_gathered_rows
        )

    @property
    def total_host_bytes(self) -> int:
        return (
            self.reconstruct_host_bytes
            + self.parent_rmw_host_bytes
        )


def _validate_plans(
    layout: CheckpointLayout,
    plans: Sequence[DirtyBlockPlan],
) -> tuple[DirtyBlockPlan, ...]:
    plans = tuple(plans)

    previous_block_id = -1
    seen = set()

    for plan in plans:
        if not isinstance(
            plan,
            DirtyBlockPlan,
        ):
            raise TypeError(
                "plans must contain DirtyBlockPlan objects"
            )

        block_id = plan.block_id

        if (
            block_id < 0
            or block_id >= layout.num_blocks
        ):
            raise IndexError(
                f"dirty block out of range: {block_id}"
            )

        if block_id in seen:
            raise ValueError(
                f"duplicate dirty block plan: {block_id}"
            )

        if block_id <= previous_block_id:
            raise ValueError(
                "dirty block plans must be ordered by "
                "increasing block_id"
            )

        seen.add(block_id)
        previous_block_id = block_id

    return plans


def materialize_dirty_blocks(
    layout: CheckpointLayout,
    plans: Sequence[DirtyBlockPlan],
    embedding_tables,
    policy: MaterializationPolicy,
    parent_fd: int | None = None,
) -> tuple[
    tuple[MaterializedBlock, ...],
    MaterializationStats,
]:
    """Materialize all dirty checkpoint blocks under one policy.

    Plans selected for reconstruction are processed together.
    Plans selected for parent-RMW are processed together.

    This preserves batch-oriented embedding access instead of invoking
    one materializer independently for every block.

    ``parent_fd`` is required only when at least one block is selected
    for parent-RMW.
    """

    if not isinstance(
        policy,
        MaterializationPolicy,
    ):
        raise TypeError(
            "policy must be a MaterializationPolicy"
        )

    plans = _validate_plans(
        layout,
        plans,
    )

    if not plans:
        return (
            (),
            MaterializationStats(
                num_blocks=0,
                reconstruct_blocks=0,
                parent_rmw_blocks=0,
                reconstruct_gathered_rows=0,
                parent_rmw_gathered_rows=0,
                reconstruct_host_bytes=0,
                parent_rmw_host_bytes=0,
                parent_read_bytes=0,
                output_bytes=0,
            ),
        )

    reconstruct_plans = []
    parent_rmw_plans = []

    for plan in plans:
        method = policy.select(
            plan
        )

        if method == METHOD_RECONSTRUCT:
            reconstruct_plans.append(
                plan
            )

        elif method == METHOD_PARENT_RMW:
            parent_rmw_plans.append(
                plan
            )

        else:
            raise RuntimeError(
                "materialization policy returned "
                f"unknown method: {method!r}"
            )

    if (
        parent_rmw_plans
        and parent_fd is None
    ):
        raise ValueError(
            "parent_fd is required when at least one "
            "block uses parent-RMW"
        )

    if reconstruct_plans:
        (
            reconstruct_result,
            reconstruct_stats,
        ) = reconstruct_blocks(
            layout,
            reconstruct_plans,
            embedding_tables,
        )

        reconstruct_gathered_rows = (
            reconstruct_stats.gathered_rows
        )
        reconstruct_host_bytes = (
            reconstruct_stats.host_bytes
        )

    else:
        reconstruct_result = ()
        reconstruct_gathered_rows = 0
        reconstruct_host_bytes = 0

    if parent_rmw_plans:
        (
            parent_rmw_result,
            parent_rmw_stats,
        ) = parent_rmw_blocks(
            layout,
            parent_rmw_plans,
            embedding_tables,
            parent_fd,
        )

        parent_rmw_gathered_rows = (
            parent_rmw_stats.gathered_rows
        )
        parent_rmw_host_bytes = (
            parent_rmw_stats.host_bytes
        )
        parent_read_bytes = (
            parent_rmw_stats.parent_read_bytes
        )

    else:
        parent_rmw_result = ()
        parent_rmw_gathered_rows = 0
        parent_rmw_host_bytes = 0
        parent_read_bytes = 0

    # Merge both execution paths back into the original deterministic
    # checkpoint block order.
    by_block_id = {}

    for block in (
        tuple(reconstruct_result)
        + tuple(parent_rmw_result)
    ):
        if block.block_id in by_block_id:
            raise RuntimeError(
                "materializer returned duplicate block "
                f"{block.block_id}"
            )

        by_block_id[
            block.block_id
        ] = block

    expected_ids = [
        plan.block_id
        for plan in plans
    ]

    actual_ids = sorted(
        by_block_id
    )

    if actual_ids != expected_ids:
        raise RuntimeError(
            "materialized block set does not match "
            "dirty block plans"
        )

    materialized = tuple(
        by_block_id[
            block_id
        ]
        for block_id in expected_ids
    )

    stats = MaterializationStats(
        num_blocks=len(
            materialized
        ),
        reconstruct_blocks=len(
            reconstruct_plans
        ),
        parent_rmw_blocks=len(
            parent_rmw_plans
        ),
        reconstruct_gathered_rows=(
            reconstruct_gathered_rows
        ),
        parent_rmw_gathered_rows=(
            parent_rmw_gathered_rows
        ),
        reconstruct_host_bytes=(
            reconstruct_host_bytes
        ),
        parent_rmw_host_bytes=(
            parent_rmw_host_bytes
        ),
        parent_read_bytes=(
            parent_read_bytes
        ),
        output_bytes=(
            len(materialized)
            * BLOCK_SIZE
        ),
    )

    return materialized, stats
