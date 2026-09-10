import os
import tempfile
import unittest

import numpy as np
import torch

from brc_block_materializer import (
    parent_rmw_blocks,
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


def build_checkpoint_image(
    layout,
    tables,
):
    """Build a complete padded checkpoint image for test setup."""

    image = bytearray(
        layout.image_size
    )

    for slot in range(
        layout.total_rows
    ):
        ref = layout.row_ref_for_slot(
            slot
        )

        row = (
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

        if len(row) != ROW_BYTES:
            raise AssertionError(
                "unexpected test row size"
            )

        start = slot * ROW_BYTES
        end = start + ROW_BYTES

        image[start:end] = row

    return bytes(image)


class ParentFile:
    def __init__(self, payload):
        self._file = tempfile.TemporaryFile()
        self._file.write(payload)
        self._file.flush()

    @property
    def fd(self):
        return self._file.fileno()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()


class TestParentRMWLocalOnly(
    unittest.TestCase
):
    def test_single_dirty_row_reads_4k_and_gathers_64b(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        parent_table = make_table(
            64,
            100,
        )

        current_table = parent_table.clone()

        # Only row 7 changes.
        current_table[7].fill_(
            9999.0
        )

        plans = plan_dirty_blocks(
            layout,
            [
                np.array(
                    [7],
                    dtype=np.int64,
                )
            ],
        )

        parent_image = build_checkpoint_image(
            layout,
            [parent_table],
        )

        with ParentFile(
            parent_image
        ) as parent:
            blocks, stats = parent_rmw_blocks(
                layout,
                plans,
                [current_table],
                parent.fd,
            )

        self.assertEqual(
            len(blocks),
            1,
        )

        self.assertEqual(
            len(blocks[0].data),
            BLOCK_SIZE,
        )

        self.assertEqual(
            stats.num_blocks,
            1,
        )

        self.assertEqual(
            stats.gathered_rows,
            1,
        )

        self.assertEqual(
            stats.host_bytes,
            64,
        )

        self.assertEqual(
            stats.parent_read_bytes,
            4096,
        )

        rows = np.frombuffer(
            blocks[0].data,
            dtype=np.float32,
        ).reshape(
            64,
            16,
        )

        for row in range(64):
            expected = (
                9999.0
                if row == 7
                else float(
                    100 + row
                )
            )

            np.testing.assert_array_equal(
                rows[row],
                np.full(
                    16,
                    expected,
                    dtype=np.float32,
                ),
            )

    def test_multiple_dirty_blocks_patch_only_dirty_rows(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[140],
            mode=LAYOUT_LOCAL_ONLY,
        )

        parent_table = make_table(
            140,
            1000,
        )

        current_table = parent_table.clone()

        dirty = [
            1,
            70,
            130,
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

        parent_image = build_checkpoint_image(
            layout,
            [parent_table],
        )

        with ParentFile(
            parent_image
        ) as parent:
            blocks, stats = parent_rmw_blocks(
                layout,
                plans,
                [current_table],
                parent.fd,
            )

        self.assertEqual(
            [b.block_id for b in blocks],
            [0, 1, 2],
        )

        self.assertEqual(
            stats.gathered_rows,
            3,
        )

        self.assertEqual(
            stats.host_bytes,
            3 * ROW_BYTES,
        )

        self.assertEqual(
            stats.parent_read_bytes,
            3 * BLOCK_SIZE,
        )


class TestParentRMWMatchesReconstruction(
    unittest.TestCase
):
    def test_local_only_matches_full_reconstruction(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[70, 70],
            mode=LAYOUT_LOCAL_ONLY,
        )

        parent_tables = [
            make_table(
                70,
                1000,
            ),
            make_table(
                70,
                2000,
            ),
        ]

        current_tables = [
            parent_tables[
                0
            ].clone(),
            parent_tables[
                1
            ].clone(),
        ]

        dirty_rows = [
            np.array(
                [1, 63, 64],
                dtype=np.int64,
            ),
            np.array(
                [0, 58],
                dtype=np.int64,
            ),
        ]

        for table_id, rows in enumerate(
            dirty_rows
        ):
            for row in rows:
                current_tables[
                    table_id
                ][
                    int(row)
                ].fill_(
                    float(
                        10000
                        + table_id * 1000
                        + int(row)
                    )
                )

        plans = plan_dirty_blocks(
            layout,
            dirty_rows,
        )

        parent_image = build_checkpoint_image(
            layout,
            parent_tables,
        )

        with ParentFile(
            parent_image
        ) as parent:
            rmw_blocks, _ = parent_rmw_blocks(
                layout,
                plans,
                current_tables,
                parent.fd,
            )

        reconstructed, _ = reconstruct_blocks(
            layout,
            plans,
            current_tables,
        )

        self.assertEqual(
            [b.block_id for b in rmw_blocks],
            [b.block_id for b in reconstructed],
        )

        for rmw, rec in zip(
            rmw_blocks,
            reconstructed,
        ):
            self.assertEqual(
                rmw.data,
                rec.data,
            )

    def test_global_mixed_matches_full_reconstruction(
        self,
    ):
        frequencies = [
            np.array(
                [9, 1, 7, 3],
                dtype=np.int64,
            ),
            np.array(
                [8, 2, 6, 4],
                dtype=np.int64,
            ),
        ]

        new_to_old = [
            np.array(
                [0, 2, 3, 1],
                dtype=np.int64,
            ),
            np.array(
                [0, 2, 3, 1],
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
            table_sizes=[4, 4],
            mode=LAYOUT_GLOBAL_MIXED,
            global_layout=global_layout,
        )

        parent_tables = [
            make_table(
                4,
                100,
            ),
            make_table(
                4,
                200,
            ),
        ]

        current_tables = [
            parent_tables[
                0
            ].clone(),
            parent_tables[
                1
            ].clone(),
        ]

        dirty_rows = [
            np.array(
                [1, 3],
                dtype=np.int64,
            ),
            np.array(
                [2],
                dtype=np.int64,
            ),
        ]

        current_tables[
            0
        ][1].fill_(9001.0)

        current_tables[
            0
        ][3].fill_(9003.0)

        current_tables[
            1
        ][2].fill_(9102.0)

        plans = plan_dirty_blocks(
            layout,
            dirty_rows,
        )

        parent_image = build_checkpoint_image(
            layout,
            parent_tables,
        )

        with ParentFile(
            parent_image
        ) as parent:
            rmw_blocks, stats = parent_rmw_blocks(
                layout,
                plans,
                current_tables,
                parent.fd,
            )

        reconstructed, _ = reconstruct_blocks(
            layout,
            plans,
            current_tables,
        )

        self.assertEqual(
            len(rmw_blocks),
            len(reconstructed),
        )

        for rmw, rec in zip(
            rmw_blocks,
            reconstructed,
        ):
            self.assertEqual(
                rmw.block_id,
                rec.block_id,
            )
            self.assertEqual(
                rmw.data,
                rec.data,
            )

        self.assertEqual(
            stats.gathered_rows,
            3,
        )


class TestParentRMWValidation(
    unittest.TestCase
):
    def test_short_parent_read_is_rejected(
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

        with tempfile.TemporaryFile() as parent:
            # Deliberately shorter than one complete BRC block.
            parent.write(
                b"\x00" * 1024
            )
            parent.flush()

            with self.assertRaises(
                RuntimeError
            ):
                parent_rmw_blocks(
                    layout,
                    plans,
                    [table],
                    parent.fileno(),
                )

    def test_empty_plan_does_not_read_parent(
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

        blocks, stats = parent_rmw_blocks(
            layout,
            (),
            [table],
            -1,
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
            stats.gathered_rows,
            0,
        )
        self.assertEqual(
            stats.host_bytes,
            0,
        )
        self.assertEqual(
            stats.parent_read_bytes,
            0,
        )


if __name__ == "__main__":
    unittest.main()
