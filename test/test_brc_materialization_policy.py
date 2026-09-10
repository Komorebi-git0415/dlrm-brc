import unittest

from brc_checkpoint_blocks import (
    ROWS_PER_BLOCK,
    DirtyBlockPlan,
    StorageRowRef,
)
from brc_materialization_policy import (
    METHOD_PARENT_RMW,
    METHOD_RECONSTRUCT,
    MODE_HYBRID,
    MODE_ONLY_PARENT_RMW,
    MODE_ONLY_RECONSTRUCT,
    MaterializationPolicy,
)


def make_plan(dirty_count):
    rows = tuple(
        StorageRowRef(
            table_id=0,
            row_id=i,
            storage_slot=i,
            row_in_block=i,
        )
        for i in range(dirty_count)
    )

    return DirtyBlockPlan(
        block_id=0,
        dirty_rows=rows,
    )


class TestOnlyReconstructPolicy(unittest.TestCase):
    def test_always_reconstructs(self):
        policy = MaterializationPolicy(
            MODE_ONLY_RECONSTRUCT
        )

        for dirty_count in (
            1,
            2,
            8,
            32,
            64,
        ):
            self.assertEqual(
                policy.select(
                    make_plan(dirty_count)
                ),
                METHOD_RECONSTRUCT,
            )


class TestOnlyParentRMWPolicy(unittest.TestCase):
    def test_always_uses_parent_rmw(self):
        policy = MaterializationPolicy(
            MODE_ONLY_PARENT_RMW
        )

        for dirty_count in (
            1,
            2,
            8,
            32,
            64,
        ):
            self.assertEqual(
                policy.select(
                    make_plan(dirty_count)
                ),
                METHOD_PARENT_RMW,
            )


class TestHybridPolicy(unittest.TestCase):
    def test_threshold_boundary(self):
        policy = MaterializationPolicy(
            MODE_HYBRID,
            threshold=8,
        )

        self.assertEqual(
            policy.select(make_plan(1)),
            METHOD_PARENT_RMW,
        )

        self.assertEqual(
            policy.select(make_plan(8)),
            METHOD_PARENT_RMW,
        )

        self.assertEqual(
            policy.select(make_plan(9)),
            METHOD_RECONSTRUCT,
        )

        self.assertEqual(
            policy.select(make_plan(64)),
            METHOD_RECONSTRUCT,
        )

    def test_zero_threshold_is_all_reconstruct(self):
        policy = MaterializationPolicy(
            MODE_HYBRID,
            threshold=0,
        )

        for dirty_count in (
            1,
            8,
            64,
        ):
            self.assertEqual(
                policy.select(
                    make_plan(dirty_count)
                ),
                METHOD_RECONSTRUCT,
            )

    def test_max_threshold_is_all_parent_rmw(self):
        policy = MaterializationPolicy(
            MODE_HYBRID,
            threshold=ROWS_PER_BLOCK,
        )

        for dirty_count in (
            1,
            8,
            64,
        ):
            self.assertEqual(
                policy.select(
                    make_plan(dirty_count)
                ),
                METHOD_PARENT_RMW,
            )


class TestPolicyValidation(unittest.TestCase):
    def test_hybrid_requires_threshold(self):
        with self.assertRaises(TypeError):
            MaterializationPolicy(
                MODE_HYBRID
            )

    def test_threshold_range_is_checked(self):
        with self.assertRaises(ValueError):
            MaterializationPolicy(
                MODE_HYBRID,
                threshold=-1,
            )

        with self.assertRaises(ValueError):
            MaterializationPolicy(
                MODE_HYBRID,
                threshold=65,
            )

    def test_threshold_rejected_for_fixed_mode(self):
        with self.assertRaises(ValueError):
            MaterializationPolicy(
                MODE_ONLY_RECONSTRUCT,
                threshold=8,
            )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MaterializationPolicy(
                "unknown"
            )

    def test_invalid_plan_is_rejected(self):
        policy = MaterializationPolicy(
            MODE_HYBRID,
            threshold=8,
        )

        with self.assertRaises(TypeError):
            policy.select(None)

    def test_empty_dirty_block_is_rejected(self):
        policy = MaterializationPolicy(
            MODE_HYBRID,
            threshold=8,
        )

        empty = DirtyBlockPlan(
            block_id=0,
            dirty_rows=(),
        )

        with self.assertRaises(ValueError):
            policy.select(empty)


if __name__ == "__main__":
    unittest.main()
