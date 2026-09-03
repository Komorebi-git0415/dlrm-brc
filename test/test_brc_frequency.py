#!/usr/bin/env python3

import unittest

import numpy as np

from brc_frequency import (
    build_frequency_permutation,
    count_frequencies,
    invert_permutation,
    remap_local_ids,
)


class TestBRCFrequency(unittest.TestCase):
    def test_basic_frequency_count(self):
        local_ids = np.array([0, 1, 2, 1, 2, 1], dtype=np.int64)

        frequencies = count_frequencies(local_ids, num_rows=3)

        np.testing.assert_array_equal(
            frequencies,
            np.array([1, 3, 2], dtype=np.int64),
        )

    def test_frequency_permutation(self):
        frequencies = np.array([1, 3, 2], dtype=np.int64)

        old_to_new, new_to_old = build_frequency_permutation(frequencies)

        np.testing.assert_array_equal(
            old_to_new,
            np.array([2, 0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            new_to_old,
            np.array([1, 2, 0], dtype=np.int64),
        )

    def test_remap_local_ids(self):
        local_ids = np.array([0, 1, 2, 1, 2, 1], dtype=np.int64)
        old_to_new = np.array([2, 0, 1], dtype=np.int64)

        remapped = remap_local_ids(local_ids, old_to_new)

        np.testing.assert_array_equal(
            remapped,
            np.array([2, 0, 1, 0, 1, 0], dtype=np.int64),
        )

    def test_deterministic_tie_break(self):
        # Rows 1, 3, and 4 all have frequency 5.  Their old IDs must decide
        # their relative order.
        frequencies = np.array([1, 5, 2, 5, 5], dtype=np.int64)

        old_to_new, new_to_old = build_frequency_permutation(frequencies)

        np.testing.assert_array_equal(
            new_to_old,
            np.array([1, 3, 4, 2, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            old_to_new,
            np.array([4, 0, 3, 1, 2], dtype=np.int64),
        )

    def test_unused_rows_are_preserved(self):
        local_ids = np.array([0, 0, 3], dtype=np.int64)

        frequencies = count_frequencies(local_ids, num_rows=5)
        old_to_new, new_to_old = build_frequency_permutation(frequencies)

        np.testing.assert_array_equal(
            frequencies,
            np.array([2, 0, 0, 1, 0], dtype=np.int64),
        )

        # Frequency order:
        # row 0 -> 2
        # row 3 -> 1
        # rows 1,2,4 -> 0, tie-broken by old ID
        np.testing.assert_array_equal(
            new_to_old,
            np.array([0, 3, 1, 2, 4], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            old_to_new,
            np.array([0, 2, 3, 1, 4], dtype=np.int64),
        )

    def test_single_row_table(self):
        local_ids = np.array([0, 0, 0], dtype=np.int64)

        frequencies = count_frequencies(local_ids, num_rows=1)
        old_to_new, new_to_old = build_frequency_permutation(frequencies)

        np.testing.assert_array_equal(frequencies, np.array([3]))
        np.testing.assert_array_equal(old_to_new, np.array([0]))
        np.testing.assert_array_equal(new_to_old, np.array([0]))

    def test_empty_access_stream_preserves_table_cardinality(self):
        frequencies = count_frequencies(
            np.array([], dtype=np.int64),
            num_rows=4,
        )

        old_to_new, new_to_old = build_frequency_permutation(frequencies)

        np.testing.assert_array_equal(
            frequencies,
            np.array([0, 0, 0, 0], dtype=np.int64),
        )

        # All rows tie at zero frequency, so old-ID ordering is preserved.
        np.testing.assert_array_equal(
            old_to_new,
            np.array([0, 1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            new_to_old,
            np.array([0, 1, 2, 3], dtype=np.int64),
        )

    def test_arbitrary_table_size(self):
        num_rows = 11
        local_ids = np.array([10, 5, 10, 7, 5, 10], dtype=np.int64)

        frequencies = count_frequencies(local_ids, num_rows)

        self.assertEqual(len(frequencies), num_rows)
        self.assertEqual(int(frequencies.sum()), len(local_ids))
        self.assertEqual(int(frequencies[10]), 3)
        self.assertEqual(int(frequencies[5]), 2)
        self.assertEqual(int(frequencies[7]), 1)

    def test_permutations_are_bijections_and_inverses(self):
        frequencies = np.array([8, 0, 3, 8, 1, 3, 0], dtype=np.int64)

        old_to_new, new_to_old = build_frequency_permutation(frequencies)

        expected = np.arange(len(frequencies), dtype=np.int64)

        np.testing.assert_array_equal(
            np.sort(old_to_new),
            expected,
        )
        np.testing.assert_array_equal(
            np.sort(new_to_old),
            expected,
        )
        np.testing.assert_array_equal(
            new_to_old[old_to_new],
            expected,
        )
        np.testing.assert_array_equal(
            old_to_new[new_to_old],
            expected,
        )

    def test_round_trip_remap(self):
        original = np.array(
            [4, 1, 6, 1, 0, 4, 4, 2],
            dtype=np.int64,
        )
        frequencies = count_frequencies(original, num_rows=7)

        old_to_new, new_to_old = build_frequency_permutation(frequencies)

        remapped = remap_local_ids(original, old_to_new)
        recovered = remap_local_ids(remapped, new_to_old)

        np.testing.assert_array_equal(recovered, original)

    def test_invert_permutation(self):
        permutation = np.array([3, 0, 2, 1], dtype=np.int64)

        inverse = invert_permutation(permutation)

        np.testing.assert_array_equal(
            inverse,
            np.array([1, 3, 2, 0], dtype=np.int64),
        )

    def test_out_of_range_id_rejected(self):
        with self.assertRaises(ValueError):
            count_frequencies(
                np.array([0, 1, 3], dtype=np.int64),
                num_rows=3,
            )

        with self.assertRaises(ValueError):
            count_frequencies(
                np.array([-1, 0], dtype=np.int64),
                num_rows=3,
            )

    def test_invalid_permutation_rejected(self):
        with self.assertRaises(ValueError):
            remap_local_ids(
                np.array([0, 1], dtype=np.int64),
                np.array([0, 0], dtype=np.int64),
            )

        with self.assertRaises(ValueError):
            invert_permutation(
                np.array([0, 2, 2], dtype=np.int64),
            )

    def test_frequency_reordering_preserves_access_semantics(self):
        canonical = np.array(
            [0, 1, 2, 1, 2, 1],
            dtype=np.int64,
        )

        frequencies = count_frequencies(canonical, num_rows=3)
        old_to_new, new_to_old = build_frequency_permutation(frequencies)
        remapped = remap_local_ids(canonical, old_to_new)

        # Translate every new ID back to its canonical row identity.
        canonical_recovered = new_to_old[remapped]

        np.testing.assert_array_equal(
            canonical_recovered,
            canonical,
        )


if __name__ == "__main__":
    unittest.main()
