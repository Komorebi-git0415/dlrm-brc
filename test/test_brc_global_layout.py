import unittest

import numpy as np

from brc_global_layout import (
    build_global_storage_layout,
    split_global_slots_by_table,
)


class TestBRCGlobalStorageLayout(
    unittest.TestCase
):
    @staticmethod
    def _inputs():
        # Table 0:
        # old frequencies = [5, 1, 5]
        #
        # After stage-1 local reorder:
        # new local 0 -> old 0 -> freq 5
        # new local 1 -> old 2 -> freq 5
        # new local 2 -> old 1 -> freq 1
        #
        # Table 1:
        # new local frequencies = [5, 2]
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

        return (
            frequencies,
            new_to_old,
        )

    def test_global_frequency_order_and_tie_break(
        self,
    ):
        frequencies, new_to_old = (
            self._inputs()
        )

        layout = build_global_storage_layout(
            frequencies,
            new_to_old,
        )

        # Flattened post-local rows:
        #
        # flat 0 = T0:L0 freq 5
        # flat 1 = T0:L1 freq 5
        # flat 2 = T0:L2 freq 1
        # flat 3 = T1:L0 freq 5
        # flat 4 = T1:L1 freq 2
        #
        # Global order must therefore be:
        #
        # T0:L0, T0:L1, T1:L0, T1:L1, T0:L2
        np.testing.assert_array_equal(
            layout[
                "flat_local_by_global_slot"
            ],
            np.array(
                [0, 1, 3, 4, 2],
                dtype=np.int64,
            ),
        )

        np.testing.assert_array_equal(
            layout[
                "frequency_by_global_slot"
            ],
            np.array(
                [5, 5, 5, 2, 1],
                dtype=np.int64,
            ),
        )

    def test_slot_map_is_bijection_and_inverse(
        self,
    ):
        frequencies, new_to_old = (
            self._inputs()
        )

        layout = build_global_storage_layout(
            frequencies,
            new_to_old,
        )

        slots = layout[
            "global_slot_by_flat_local"
        ]

        order = layout[
            "flat_local_by_global_slot"
        ]

        expected = np.arange(
            layout["total_rows"],
            dtype=np.int64,
        )

        np.testing.assert_array_equal(
            np.sort(slots),
            expected,
        )

        np.testing.assert_array_equal(
            slots[order],
            expected,
        )

        np.testing.assert_array_equal(
            order[slots],
            expected,
        )

    def test_split_slots_use_post_local_runtime_ids(
        self,
    ):
        frequencies, new_to_old = (
            self._inputs()
        )

        layout = build_global_storage_layout(
            frequencies,
            new_to_old,
        )

        per_table = (
            split_global_slots_by_table(
                layout
            )
        )

        self.assertEqual(
            len(per_table),
            2,
        )

        # T0:L0 -> slot 0
        # T0:L1 -> slot 1
        # T0:L2 -> slot 4
        np.testing.assert_array_equal(
            per_table[0],
            np.array(
                [0, 1, 4],
                dtype=np.int64,
            ),
        )

        # T1:L0 -> slot 2
        # T1:L1 -> slot 3
        np.testing.assert_array_equal(
            per_table[1],
            np.array(
                [2, 3],
                dtype=np.int64,
            ),
        )

    def test_invalid_new_to_old_is_rejected(
        self,
    ):
        frequencies, new_to_old = (
            self._inputs()
        )

        new_to_old[0] = np.array(
            [0, 0, 2],
            dtype=np.int64,
        )

        with self.assertRaises(
            ValueError
        ):
            build_global_storage_layout(
                frequencies,
                new_to_old,
            )

    def test_table_length_mismatch_is_rejected(
        self,
    ):
        frequencies, new_to_old = (
            self._inputs()
        )

        with self.assertRaises(
            ValueError
        ):
            build_global_storage_layout(
                frequencies,
                new_to_old[:1],
            )


if __name__ == "__main__":
    unittest.main()
