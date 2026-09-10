import tempfile
import unittest
from pathlib import Path

import numpy as np

from brc_checkpoint_blocks import (
    LAYOUT_GLOBAL_MIXED,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
    checkpoint_layout_fingerprint,
)
from brc_checkpoint_manager import (
    CheckpointResult,
)
from brc_checkpoint_publication import (
    CheckpointPublisher,
)
from brc_checkpoint_restore import (
    load_published_checkpoint,
)
from brc_global_layout import (
    build_global_storage_layout,
)


def make_global_layout(
    table_sizes,
    frequencies,
):
    new_to_old = [
        np.arange(
            size,
            dtype=np.int64,
        )
        for size in table_sizes
    ]

    global_layout = (
        build_global_storage_layout(
            [
                np.asarray(
                    values,
                    dtype=np.int64,
                )
                for values
                in frequencies
            ],
            new_to_old,
        )
    )

    return CheckpointLayout(
        table_sizes=table_sizes,
        mode=LAYOUT_GLOBAL_MIXED,
        global_layout=global_layout,
    )


def publish_c0(
    directory,
    layout,
):
    path = (
        Path(directory)
        / "C0.img"
    )

    path.write_bytes(
        bytes(
            layout.image_size
        )
    )

    checkpoint = CheckpointResult(
        generation=0,
        path=str(path),
        dirty_blocks=layout.num_blocks,
        dirty_rows=layout.total_rows,
        materialization=None,
    )

    publisher = CheckpointPublisher(
        directory,
        layout,
    )

    publisher.publish(
        checkpoint,
        {
            "step": 1,
        },
    )


class TestLayoutFingerprint(
    unittest.TestCase
):
    def test_fingerprint_is_deterministic(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[70, 58],
            mode=LAYOUT_LOCAL_ONLY,
        )

        self.assertEqual(
            checkpoint_layout_fingerprint(
                layout
            ),
            checkpoint_layout_fingerprint(
                layout
            ),
        )

    def test_local_table_boundaries_change_identity(
        self,
    ):
        a = CheckpointLayout(
            table_sizes=[64, 64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        b = CheckpointLayout(
            table_sizes=[70, 58],
            mode=LAYOUT_LOCAL_ONLY,
        )

        # Same aggregate geometry.
        self.assertEqual(
            a.num_tables,
            b.num_tables,
        )
        self.assertEqual(
            a.total_rows,
            b.total_rows,
        )
        self.assertEqual(
            a.image_size,
            b.image_size,
        )

        self.assertNotEqual(
            checkpoint_layout_fingerprint(a),
            checkpoint_layout_fingerprint(b),
        )

    def test_global_permutation_changes_identity(
        self,
    ):
        table_sizes = [4, 4]

        a = make_global_layout(
            table_sizes,
            [
                [100, 80, 60, 40],
                [90, 70, 50, 30],
            ],
        )

        b = make_global_layout(
            table_sizes,
            [
                [40, 30, 20, 10],
                [100, 90, 80, 70],
            ],
        )

        self.assertEqual(
            a.table_sizes.tolist(),
            b.table_sizes.tolist(),
        )

        self.assertEqual(
            a.image_size,
            b.image_size,
        )

        self.assertNotEqual(
            checkpoint_layout_fingerprint(a),
            checkpoint_layout_fingerprint(b),
        )


class TestVectorizedStorageSlots(
    unittest.TestCase
):
    def test_local_only_runtime_range(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64, 64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        slots = (
            layout.storage_slots_for_runtime_range(
                table_id=1,
                row_start=2,
                row_end=6,
            )
        )

        np.testing.assert_array_equal(
            slots,
            np.asarray(
                [66, 67, 68, 69],
                dtype=np.int64,
            ),
        )

    def test_global_runtime_range_matches_stage_b_mapping(
        self,
    ):
        table_sizes = [4, 4]

        frequencies = [
            np.asarray(
                [100, 80, 60, 40],
                dtype=np.int64,
            ),
            np.asarray(
                [90, 70, 50, 30],
                dtype=np.int64,
            ),
        ]

        new_to_old = [
            np.arange(
                4,
                dtype=np.int64,
            ),
            np.arange(
                4,
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
            table_sizes=table_sizes,
            mode=LAYOUT_GLOBAL_MIXED,
            global_layout=global_layout,
        )

        actual = (
            layout.storage_slots_for_runtime_range(
                table_id=1,
                row_start=1,
                row_end=4,
            )
        )

        # table 1 begins at flat-local offset 4.
        expected = np.asarray(
            global_layout[
                "global_slot_by_flat_local"
            ][5:8],
            dtype=np.int64,
        )

        np.testing.assert_array_equal(
            actual,
            expected,
        )


class TestRestoreRejectsWrongLayoutIdentity(
    unittest.TestCase
):
    def test_same_aggregate_but_wrong_table_boundaries_rejected(
        self,
    ):
        writer_layout = CheckpointLayout(
            table_sizes=[64, 64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        wrong_layout = CheckpointLayout(
            table_sizes=[70, 58],
            mode=LAYOUT_LOCAL_ONLY,
        )

        with tempfile.TemporaryDirectory() as tmp:
            publish_c0(
                tmp,
                writer_layout,
            )

            with self.assertRaises(
                ValueError
            ):
                load_published_checkpoint(
                    tmp,
                    wrong_layout,
                )

    def test_wrong_global_permutation_rejected(
        self,
    ):
        table_sizes = [4, 4]

        writer_layout = make_global_layout(
            table_sizes,
            [
                [100, 80, 60, 40],
                [90, 70, 50, 30],
            ],
        )

        wrong_layout = make_global_layout(
            table_sizes,
            [
                [40, 30, 20, 10],
                [100, 90, 80, 70],
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            publish_c0(
                tmp,
                writer_layout,
            )

            with self.assertRaises(
                ValueError
            ):
                load_published_checkpoint(
                    tmp,
                    wrong_layout,
                )


if __name__ == "__main__":
    unittest.main()
