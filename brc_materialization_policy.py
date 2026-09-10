"""Materialization policy for BRC dirty checkpoint blocks.

The policy layer decides how each DirtyBlockPlan should be materialized.

Supported modes:

    only_reconstruct
        Always reconstruct the complete 4 KiB block from current
        live embedding weights.

    only_parent_rmw
        Always read the parent 4 KiB block and patch its dirty rows.

    hybrid
        Use parent-RMW when:

            dirty_count <= threshold

        otherwise use reconstruction.

The policy intentionally does not inspect parent page-cache residency.
That keeps the decision deterministic and makes threshold selection an
explicit experimental variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from brc_checkpoint_blocks import (
    ROWS_PER_BLOCK,
    DirtyBlockPlan,
)


MODE_ONLY_RECONSTRUCT = "only_reconstruct"
MODE_ONLY_PARENT_RMW = "only_parent_rmw"
MODE_HYBRID = "hybrid"

METHOD_RECONSTRUCT = "reconstruct"
METHOD_PARENT_RMW = "parent_rmw"

_VALID_MODES = {
    MODE_ONLY_RECONSTRUCT,
    MODE_ONLY_PARENT_RMW,
    MODE_HYBRID,
}


@dataclass(frozen=True)
class MaterializationPolicy:
    """Select reconstruction or parent-RMW for each dirty block."""

    mode: str
    threshold: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                "unsupported materialization mode: "
                f"{self.mode!r}"
            )

        if self.mode == MODE_HYBRID:
            if not isinstance(self.threshold, Integral):
                raise TypeError(
                    "hybrid threshold must be an integer"
                )

            threshold = int(self.threshold)

            if threshold < 0 or threshold > ROWS_PER_BLOCK:
                raise ValueError(
                    "hybrid threshold must be in range "
                    f"[0, {ROWS_PER_BLOCK}]"
                )

            object.__setattr__(
                self,
                "threshold",
                threshold,
            )

        elif self.threshold is not None:
            raise ValueError(
                "threshold is only valid for hybrid mode"
            )

    def select(
        self,
        plan: DirtyBlockPlan,
    ) -> str:
        """Return materialization method for one dirty block."""

        if not isinstance(plan, DirtyBlockPlan):
            raise TypeError(
                "plan must be a DirtyBlockPlan"
            )

        dirty_count = plan.dirty_count

        if (
            dirty_count <= 0
            or dirty_count > ROWS_PER_BLOCK
        ):
            raise ValueError(
                "dirty block must contain between 1 and "
                f"{ROWS_PER_BLOCK} dirty rows"
            )

        if self.mode == MODE_ONLY_RECONSTRUCT:
            return METHOD_RECONSTRUCT

        if self.mode == MODE_ONLY_PARENT_RMW:
            return METHOD_PARENT_RMW

        if dirty_count <= self.threshold:
            return METHOD_PARENT_RMW

        return METHOD_RECONSTRUCT
