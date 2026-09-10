import tempfile
import unittest

import numpy as np
import torch

from brc_block_materializer import (
    reconstruct_blocks,
)
from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    ROW_BYTES,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
    plan_dirty_blocks,
)
from brc_materialization import (
    materialize_dirty_blocks,
)
from brc_materialization_policy import (
    MODE_HYBRID,
    MODE_ONLY_PARENT_RMW,
    MODE_ONLY_RECONSTRUCT,
    MaterializationPolicy,
)


def make_table(
    num_rows,
    base,
):
    table = torch.empty(
        (num_rows, 16),
        dtype=torch.float32,
    )

    for row in range(
        num_rows
    ):
        table[row].fill_(
            float(
                base + row
            )
        )

    return table


def build_checkpoint_image(
    layout,
    tables,
):
    image = bytearray(
        layout.image_size
    )

    for slot in range(
        layout.total_rows
    ):
        ref = layout.row_ref_for_slot(
            slot
        )

        payload = (
            tables[
                ref.table_id
            ][
                ref.row_id
            ]
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
            .tobytes(
                order="C"
            )
        )

        if len(payload) != ROW_BYTES:
            raise AssertionError(
                "unexpected embedding row size"
            )

        start = (
            slot
            * ROW_BYTES
        )

        image[
            start:
            start + ROW_BYTES
        ] = payload

    return bytes(image)


class ParentFile:
    def __init__(
        self,
        payload,
    ):
        self._file = (
            tempfile.TemporaryFile()
        )

        self._file.write(
            payload
        )
        self._file.flush()

    @property
    def fd(self):
        return self._file.fileno()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self._file.close()


class TestUnifiedFixedPolicies(
    unittest.TestCase
):
    def setUp(self):
        self.layout = CheckpointLayout(
            table_sizes=[128],
            mode=LAYOUT_LOCAL_ONLY,
        )

        self.parent_table = (
            make_table(
                128,
                1000,
            )
        )

        self.current_table = (
            self.parent_table.clone()
        )

        self.current_table[
            1
        ].fill_(9001.0)

        self.current_table[
            70
        ].fill_(9070.0)

        self.dirty_rows = [
            np.array(
                [1, 70],
                dtype=np.int64,
            )
        ]

        self.plans = (
            plan_dirty_blocks(
                self.layout,
                self.dirty_rows,
            )
        )

        self.parent_image = (
            build_checkpoint_image(
                self.layout,
                [self.parent_table],
            )
        )

    def test_only_reconstruct_does_not_require_parent(
        self,
    ):
        policy = MaterializationPolicy(
            MODE_ONLY_RECONSTRUCT
        )

        blocks, stats = (
            materialize_dirty_blocks(
                self.layout,
                self.plans,
                [self.current_table],
                policy,
                parent_fd=None,
            )
        )

        self.assertEqual(
            [b.block_id for b in blocks],
            [0, 1],
        )

        self.assertEqual(
            stats.num_blocks,
            2,
        )

        self.assertEqual(
            stats.reconstruct_blocks,
            2,
        )

        self.assertEqual(
            stats.parent_rmw_blocks,
            0,
        )

        self.assertEqual(
            stats.parent_read_bytes,
            0,
        )

        self.assertEqual(
            stats.output_bytes,
            2 * BLOCK_SIZE,
        )

    def test_only_parent_rmw_uses_parent_for_all_blocks(
        self,
    ):
        policy = MaterializationPolicy(
            MODE_ONLY_PARENT_RMW
        )

        with ParentFile(
            self.parent_image
        ) as parent:
            blocks, stats = (
                materialize_dirty_blocks(
                    self.layout,
                    self.plans,
                    [self.current_table],
                    policy,
                    parent_fd=parent.fd,
                )
            )

        self.assertEqual(
            [b.block_id for b in blocks],
            [0, 1],
        )

        self.assertEqual(
            stats.reconstruct_blocks,
            0,
        )

        self.assertEqual(
            stats.parent_rmw_blocks,
            2,
        )

        self.assertEqual(
            stats.parent_rmw_gathered_rows,
            2,
        )

        self.assertEqual(
            stats.parent_rmw_host_bytes,
            2 * ROW_BYTES,
        )

        self.assertEqual(
            stats.parent_read_bytes,
            2 * BLOCK_SIZE,
        )

    def test_parent_is_required_only_if_rmw_selected(
        self,
    ):
        policy = MaterializationPolicy(
            MODE_ONLY_PARENT_RMW
        )

        with self.assertRaises(
            ValueError
        ):
            materialize_dirty_blocks(
                self.layout,
                self.plans,
                [self.current_table],
                policy,
                parent_fd=None,
            )


class TestUnifiedHybridPolicy(
    unittest.TestCase
):
    def test_hybrid_splits_blocks_by_dirty_count(
        self,
    ):
        # Three complete blocks.
        layout = CheckpointLayout(
            table_sizes=[192],
            mode=LAYOUT_LOCAL_ONLY,
        )

        parent_table = make_table(
            192,
            1000,
        )

        current_table = (
            parent_table.clone()
        )

        # block 0: 1 dirty row
        #
        # block 1: 4 dirty rows
        #
        # block 2: 10 dirty rows
        #
        # With threshold=4:
        #
        # blocks 0,1 -> parent-RMW
        # block 2   -> reconstruction
        dirty = [
            1,
            64,
            65,
            66,
            67,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
        ]

        for row in dirty:
            current_table[
                row
            ].fill_(
                float(
                    9000 + row
                )
            )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array(
                    dirty,
                    dtype=np.int64,
                )
            ],
        )

        parent_image = (
            build_checkpoint_image(
                layout,
                [parent_table],
            )
        )

        policy = MaterializationPolicy(
            MODE_HYBRID,
            threshold=4,
        )

        with ParentFile(
            parent_image
        ) as parent:
            hybrid_blocks, stats = (
                materialize_dirty_blocks(
                    layout,
                    plans,
                    [current_table],
                    policy,
                    parent_fd=parent.fd,
                )
            )

        # Full reconstruction is the correctness oracle:
        # both strategies must produce the same current
        # checkpoint contents when dirty tracking is complete.
        oracle_blocks, _ = (
            reconstruct_blocks(
                layout,
                plans,
                [current_table],
            )
        )

        self.assertEqual(
            [
                b.block_id
                for b in hybrid_blocks
            ],
            [0, 1, 2],
        )

        self.assertEqual(
            len(hybrid_blocks),
            len(oracle_blocks),
        )

        for hybrid, oracle in zip(
            hybrid_blocks,
            oracle_blocks,
        ):
            self.assertEqual(
                hybrid.block_id,
                oracle.block_id,
            )

            self.assertEqual(
                hybrid.data,
                oracle.data,
            )

        self.assertEqual(
            stats.num_blocks,
            3,
        )

        self.assertEqual(
            stats.parent_rmw_blocks,
            2,
        )

        self.assertEqual(
            stats.reconstruct_blocks,
            1,
        )

        # RMW:
        # block 0 -> 1 dirty row
        # block 1 -> 4 dirty rows
        self.assertEqual(
            stats.parent_rmw_gathered_rows,
            5,
        )

        self.assertEqual(
            stats.parent_rmw_host_bytes,
            5 * ROW_BYTES,
        )

        # Reconstruction:
        # block 2 is complete -> all 64 rows.
        self.assertEqual(
            stats.reconstruct_gathered_rows,
            64,
        )

        self.assertEqual(
            stats.reconstruct_host_bytes,
            64 * ROW_BYTES,
        )

        self.assertEqual(
            stats.parent_read_bytes,
            2 * BLOCK_SIZE,
        )

        self.assertEqual(
            stats.output_bytes,
            3 * BLOCK_SIZE,
        )

        self.assertEqual(
            stats.total_gathered_rows,
            69,
        )

        self.assertEqual(
            stats.total_host_bytes,
            69 * ROW_BYTES,
        )

    def test_zero_threshold_needs_no_parent(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        table = make_table(
            64,
            0,
        )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array(
                    [1],
                    dtype=np.int64,
                )
            ],
        )

        policy = MaterializationPolicy(
            MODE_HYBRID,
            threshold=0,
        )

        blocks, stats = (
            materialize_dirty_blocks(
                layout,
                plans,
                [table],
                policy,
                parent_fd=None,
            )
        )

        self.assertEqual(
            len(blocks),
            1,
        )

        self.assertEqual(
            stats.reconstruct_blocks,
            1,
        )

        self.assertEqual(
            stats.parent_rmw_blocks,
            0,
        )


class TestUnifiedMaterializationValidation(
    unittest.TestCase
):
    def test_empty_plan_returns_zero_stats(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[3],
            mode=LAYOUT_LOCAL_ONLY,
        )

        table = make_table(
            3,
            0,
        )

        policy = MaterializationPolicy(
            MODE_ONLY_PARENT_RMW
        )

        blocks, stats = (
            materialize_dirty_blocks(
                layout,
                (),
                [table],
                policy,
                parent_fd=None,
            )
        )

        self.assertEqual(
            blocks,
            (),
        )

        self.assertEqual(
            stats.num_blocks,
            0,
        )

        self.assertEqual(
            stats.output_bytes,
            0,
        )

    def test_policy_type_is_checked(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[3],
            mode=LAYOUT_LOCAL_ONLY,
        )

        with self.assertRaises(
            TypeError
        ):
            materialize_dirty_blocks(
                layout,
                (),
                [make_table(3, 0)],
                policy=None,
            )


if __name__ == "__main__":
    unittest.main()
