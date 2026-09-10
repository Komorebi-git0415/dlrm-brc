import unittest

import numpy as np
import torch

from brc_block_materializer import (
    reconstruct_blocks,
)
from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    ROW_BYTES,
    LAYOUT_GLOBAL_MIXED,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
    plan_dirty_blocks,
)
from brc_global_layout import (
    build_global_storage_layout,
)


def make_table(num_rows, base):
    values = torch.empty(
        (num_rows, 16),
        dtype=torch.float32,
    )

    for row in range(num_rows):
        values[row].fill_(
            float(base + row)
        )

    return values


def decode_block_rows(block):
    array = np.frombuffer(
        block.data,
        dtype=np.float32,
    ).reshape(
        BLOCK_SIZE // ROW_BYTES,
        16,
    )

    return array


class TestLocalOnlyReconstruction(
    unittest.TestCase
):
    def test_reconstructs_complete_block_from_live_weights(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[70, 10],
            mode=LAYOUT_LOCAL_ONLY,
        )

        tables = [
            make_table(70, 1000),
            make_table(10, 2000),
        ]

        # Only one dirty row is needed to mark block 0 dirty.
        # Reconstruction must still rebuild all 64 rows.
        plans = plan_dirty_blocks(
            layout,
            [
                np.array([5], dtype=np.int64),
                np.array([], dtype=np.int64),
            ],
        )

        blocks, stats = reconstruct_blocks(
            layout,
            plans,
            tables,
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_id, 0)
        self.assertEqual(len(blocks[0].data), BLOCK_SIZE)

        rows = decode_block_rows(blocks[0])

        for row in range(64):
            np.testing.assert_array_equal(
                rows[row],
                np.full(
                    16,
                    float(1000 + row),
                    dtype=np.float32,
                ),
            )

        self.assertEqual(stats.num_blocks, 1)
        self.assertEqual(stats.gathered_rows, 64)
        self.assertEqual(
            stats.host_bytes,
            64 * 64,
        )

    def test_final_partial_block_is_zero_padded(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[70],
            mode=LAYOUT_LOCAL_ONLY,
        )

        tables = [
            make_table(70, 100),
        ]

        plans = plan_dirty_blocks(
            layout,
            [
                np.array([69], dtype=np.int64),
            ],
        )

        blocks, stats = reconstruct_blocks(
            layout,
            plans,
            tables,
        )

        self.assertEqual(
            blocks[0].block_id,
            1,
        )

        rows = decode_block_rows(
            blocks[0]
        )

        # Real rows 64..69 occupy block rows 0..5.
        for block_row, source_row in enumerate(
            range(64, 70)
        ):
            np.testing.assert_array_equal(
                rows[block_row],
                np.full(
                    16,
                    float(100 + source_row),
                    dtype=np.float32,
                ),
            )

        # Remaining block rows must be deterministic zero padding.
        np.testing.assert_array_equal(
            rows[6:],
            np.zeros(
                (58, 16),
                dtype=np.float32,
            ),
        )

        self.assertEqual(
            stats.gathered_rows,
            6,
        )

    def test_multiple_dirty_blocks_are_materialized_together(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[140],
            mode=LAYOUT_LOCAL_ONLY,
        )

        tables = [
            make_table(140, 0),
        ]

        plans = plan_dirty_blocks(
            layout,
            [
                np.array(
                    [1, 70, 130],
                    dtype=np.int64,
                ),
            ],
        )

        blocks, stats = reconstruct_blocks(
            layout,
            plans,
            tables,
        )

        self.assertEqual(
            [block.block_id for block in blocks],
            [0, 1, 2],
        )

        self.assertEqual(
            stats.num_blocks,
            3,
        )

        # Blocks 0 and 1 have 64 rows, block 2 has 12.
        self.assertEqual(
            stats.gathered_rows,
            140,
        )


class TestGlobalMixedReconstruction(
    unittest.TestCase
):
    def test_reconstruction_uses_global_storage_order(
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

        table0 = make_table(
            3,
            100,
        )
        table1 = make_table(
            2,
            200,
        )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array([2], dtype=np.int64),
                np.array([], dtype=np.int64),
            ],
        )

        blocks, stats = reconstruct_blocks(
            layout,
            plans,
            [table0, table1],
        )

        rows = decode_block_rows(
            blocks[0]
        )

        # Stage-B global order:
        #
        # slot 0 -> T0:L0 -> 100
        # slot 1 -> T0:L1 -> 101
        # slot 2 -> T1:L0 -> 200
        # slot 3 -> T1:L1 -> 201
        # slot 4 -> T0:L2 -> 102
        expected = [
            100.0,
            101.0,
            200.0,
            201.0,
            102.0,
        ]

        for slot, value in enumerate(
            expected
        ):
            np.testing.assert_array_equal(
                rows[slot],
                np.full(
                    16,
                    value,
                    dtype=np.float32,
                ),
            )

        np.testing.assert_array_equal(
            rows[5:],
            np.zeros(
                (59, 16),
                dtype=np.float32,
            ),
        )

        self.assertEqual(
            stats.gathered_rows,
            5,
        )


class TestEmbeddingBagCompatibility(
    unittest.TestCase
):
    def test_standard_embedding_bag_weight_is_supported(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[4],
            mode=LAYOUT_LOCAL_ONLY,
        )

        embedding = torch.nn.EmbeddingBag(
            4,
            16,
            mode="sum",
            sparse=True,
        )

        with torch.no_grad():
            for row in range(4):
                embedding.weight[row].fill_(
                    float(10 + row)
                )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array([0], dtype=np.int64),
            ],
        )

        blocks, stats = reconstruct_blocks(
            layout,
            plans,
            [embedding],
        )

        rows = decode_block_rows(
            blocks[0]
        )

        for row in range(4):
            np.testing.assert_array_equal(
                rows[row],
                np.full(
                    16,
                    float(10 + row),
                    dtype=np.float32,
                ),
            )

        self.assertEqual(
            stats.gathered_rows,
            4,
        )


class TestReconstructionValidation(
    unittest.TestCase
):
    def test_wrong_dtype_is_rejected(self):
        layout = CheckpointLayout(
            table_sizes=[3],
            mode=LAYOUT_LOCAL_ONLY,
        )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array([0], dtype=np.int64),
            ],
        )

        table = torch.zeros(
            (3, 16),
            dtype=torch.float64,
        )

        with self.assertRaises(TypeError):
            reconstruct_blocks(
                layout,
                plans,
                [table],
            )

    def test_wrong_embedding_dimension_is_rejected(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[3],
            mode=LAYOUT_LOCAL_ONLY,
        )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array([0], dtype=np.int64),
            ],
        )

        table = torch.zeros(
            (3, 8),
            dtype=torch.float32,
        )

        with self.assertRaises(ValueError):
            reconstruct_blocks(
                layout,
                plans,
                [table],
            )

    def test_empty_plan_returns_empty_result(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[3],
            mode=LAYOUT_LOCAL_ONLY,
        )

        table = torch.zeros(
            (3, 16),
            dtype=torch.float32,
        )

        blocks, stats = reconstruct_blocks(
            layout,
            (),
            [table],
        )

        self.assertEqual(blocks, ())
        self.assertEqual(stats.num_blocks, 0)
        self.assertEqual(stats.gathered_rows, 0)
        self.assertEqual(stats.host_bytes, 0)


if __name__ == "__main__":
    unittest.main()
