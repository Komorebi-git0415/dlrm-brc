import unittest

import numpy as np

from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    ROW_BYTES,
    ROWS_PER_BLOCK,
    LAYOUT_GLOBAL_MIXED,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
    plan_dirty_blocks,
)
from brc_global_layout import (
    build_global_storage_layout,
)


class TestCheckpointLayoutConstants(
    unittest.TestCase
):
    def test_fixed_embedding_geometry(self):
        self.assertEqual(
            ROW_BYTES,
            64,
        )
        self.assertEqual(
            ROWS_PER_BLOCK,
            64,
        )
        self.assertEqual(
            BLOCK_SIZE,
            4096,
        )


class TestLocalOnlyCheckpointLayout(
    unittest.TestCase
):
    def test_table_contiguous_mapping(self):
        layout = CheckpointLayout(
            table_sizes=[3, 2],
            mode=LAYOUT_LOCAL_ONLY,
        )

        self.assertEqual(
            layout.storage_slot(0, 0),
            0,
        )
        self.assertEqual(
            layout.storage_slot(0, 2),
            2,
        )
        self.assertEqual(
            layout.storage_slot(1, 0),
            3,
        )
        self.assertEqual(
            layout.storage_slot(1, 1),
            4,
        )

    def test_inverse_mapping(self):
        layout = CheckpointLayout(
            table_sizes=[3, 2],
            mode=LAYOUT_LOCAL_ONLY,
        )

        ref = layout.row_ref_for_slot(
            3
        )

        self.assertEqual(
            ref.table_id,
            1,
        )
        self.assertEqual(
            ref.row_id,
            0,
        )
        self.assertEqual(
            ref.storage_slot,
            3,
        )
        self.assertEqual(
            ref.row_in_block,
            3,
        )

    def test_image_is_block_aligned(self):
        layout = CheckpointLayout(
            table_sizes=[70, 30],
            mode=LAYOUT_LOCAL_ONLY,
        )

        self.assertEqual(
            layout.total_rows,
            100,
        )
        self.assertEqual(
            layout.num_blocks,
            2,
        )
        self.assertEqual(
            layout.image_size,
            8192,
        )

        final_rows = (
            layout.row_refs_for_block(
                1
            )
        )

        self.assertEqual(
            len(final_rows),
            36,
        )

        self.assertEqual(
            final_rows[0].storage_slot,
            64,
        )

        self.assertEqual(
            final_rows[-1].storage_slot,
            99,
        )


class TestGlobalMixedCheckpointLayout(
    unittest.TestCase
):
    @staticmethod
    def _build():
        frequencies = [
            np.array(
                [5, 1, 5],
                dtype=np.int64,
            ),
            np.array(
                [5, 2],
                dtype=np.int64,
            ),
        ]

        new_to_old = [
            np.array(
                [0, 2, 1],
                dtype=np.int64,
            ),
            np.array(
                [0, 1],
                dtype=np.int64,
            ),
        ]

        global_layout = (
            build_global_storage_layout(
                frequencies,
                new_to_old,
            )
        )

        layout = CheckpointLayout(
            table_sizes=[3, 2],
            mode=LAYOUT_GLOBAL_MIXED,
            global_layout=global_layout,
        )

        return layout

    def test_reuses_stage_b_global_mapping(self):
        layout = self._build()

        # Stage-B expected order:
        #
        # slot 0 -> T0:L0
        # slot 1 -> T0:L1
        # slot 2 -> T1:L0
        # slot 3 -> T1:L1
        # slot 4 -> T0:L2
        self.assertEqual(
            layout.storage_slot(0, 0),
            0,
        )
        self.assertEqual(
            layout.storage_slot(0, 1),
            1,
        )
        self.assertEqual(
            layout.storage_slot(1, 0),
            2,
        )
        self.assertEqual(
            layout.storage_slot(1, 1),
            3,
        )
        self.assertEqual(
            layout.storage_slot(0, 2),
            4,
        )

    def test_global_inverse_mapping(self):
        layout = self._build()

        expected = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
        ]

        actual = []

        for slot in range(
            layout.total_rows
        ):
            ref = (
                layout.row_ref_for_slot(
                    slot
                )
            )

            actual.append(
                (
                    ref.table_id,
                    ref.row_id,
                )
            )

        self.assertEqual(
            actual,
            expected,
        )


class TestDirtyBlockPlanning(
    unittest.TestCase
):
    def test_local_only_deduplicates_and_groups(self):
        layout = CheckpointLayout(
            table_sizes=[70, 70],
            mode=LAYOUT_LOCAL_ONLY,
        )

        # Storage:
        #
        # T0 rows 0..69 -> slots 0..69
        # T1 rows 0..69 -> slots 70..139
        #
        # Dirty slots:
        # T0:L1  -> 1   -> block 0
        # T0:L63 -> 63  -> block 0
        # T0:L64 -> 64  -> block 1
        # T1:L0  -> 70  -> block 1
        # T1:L58 -> 128 -> block 2
        plans = plan_dirty_blocks(
            layout,
            [
                np.array(
                    [1, 63, 64, 64],
                    dtype=np.int64,
                ),
                np.array(
                    [0, 58],
                    dtype=np.int64,
                ),
            ],
        )

        self.assertEqual(
            [p.block_id for p in plans],
            [0, 1, 2],
        )

        self.assertEqual(
            plans[0].dirty_slots,
            (1, 63),
        )

        self.assertEqual(
            plans[0].dirty_count,
            2,
        )

        self.assertEqual(
            plans[1].dirty_slots,
            (64, 70),
        )

        self.assertEqual(
            plans[2].dirty_slots,
            (128,),
        )

    def test_global_mixed_dirty_rows_use_global_slots(
        self,
    ):
        frequencies = [
            np.array(
                [5, 1, 5],
                dtype=np.int64,
            ),
            np.array(
                [5, 2],
                dtype=np.int64,
            ),
        ]

        new_to_old = [
            np.array(
                [0, 2, 1],
                dtype=np.int64,
            ),
            np.array(
                [0, 1],
                dtype=np.int64,
            ),
        ]

        global_layout = (
            build_global_storage_layout(
                frequencies,
                new_to_old,
            )
        )

        layout = CheckpointLayout(
            table_sizes=[3, 2],
            mode=LAYOUT_GLOBAL_MIXED,
            global_layout=global_layout,
        )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array(
                    [2],
                    dtype=np.int64,
                ),
                np.array(
                    [0],
                    dtype=np.int64,
                ),
            ],
        )

        self.assertEqual(
            len(plans),
            1,
        )

        # T1:L0 -> global slot 2
        # T0:L2 -> global slot 4
        self.assertEqual(
            plans[0].dirty_slots,
            (2, 4),
        )

        self.assertEqual(
            [
                (
                    ref.table_id,
                    ref.row_id,
                )
                for ref in plans[
                    0
                ].dirty_rows
            ],
            [
                (1, 0),
                (0, 2),
            ],
        )

    def test_no_dirty_rows_returns_no_blocks(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[3, 2],
            mode=LAYOUT_LOCAL_ONLY,
        )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array(
                    [],
                    dtype=np.int64,
                ),
                np.array(
                    [],
                    dtype=np.int64,
                ),
            ],
        )

        self.assertEqual(
            plans,
            (),
        )


class TestCheckpointLayoutValidation(
    unittest.TestCase
):
    def test_global_layout_is_required(self):
        with self.assertRaises(
            ValueError
        ):
            CheckpointLayout(
                table_sizes=[3, 2],
                mode=LAYOUT_GLOBAL_MIXED,
            )

    def test_global_layout_rejected_for_local_only(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            CheckpointLayout(
                table_sizes=[3, 2],
                mode=LAYOUT_LOCAL_ONLY,
                global_layout={},
            )

    def test_dirty_row_out_of_range_is_rejected(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[3],
            mode=LAYOUT_LOCAL_ONLY,
        )

        with self.assertRaises(
            IndexError
        ):
            plan_dirty_blocks(
                layout,
                [
                    np.array(
                        [3],
                        dtype=np.int64,
                    )
                ],
            )

    def test_dirty_rows_must_be_integer_ids(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[3],
            mode=LAYOUT_LOCAL_ONLY,
        )

        with self.assertRaises(
            TypeError
        ):
            plan_dirty_blocks(
                layout,
                [
                    np.array(
                        [1.5],
                        dtype=np.float64,
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
