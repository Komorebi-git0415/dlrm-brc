#!/usr/bin/env python3

import unittest

import numpy as np

from brc_frequency import count_table_frequencies


class TestBRCDatasetFrequency(unittest.TestCase):
    def test_arbitrary_number_of_tables(self):
        chunks = [
            np.array(
                [
                    [0, 1, 2],
                    [1, 1, 0],
                    [1, 0, 2],
                ],
                dtype=np.int64,
            ),
            np.array(
                [[2, 1, 1]],
                dtype=np.int64,
            ),
        ]

        frequencies = count_table_frequencies(
            chunks,
            table_sizes=np.array(
                [3, 2, 3],
                dtype=np.int64,
            ),
        )

        self.assertEqual(
            len(frequencies),
            3,
        )

        np.testing.assert_array_equal(
            frequencies[0],
            np.array(
                [1, 2, 1],
                dtype=np.int64,
            ),
        )

        np.testing.assert_array_equal(
            frequencies[1],
            np.array(
                [1, 3],
                dtype=np.int64,
            ),
        )

        np.testing.assert_array_equal(
            frequencies[2],
            np.array(
                [1, 1, 2],
                dtype=np.int64,
            ),
        )

    def test_unused_rows_are_preserved_across_chunks(self):
        chunks = [
            np.array(
                [[0], [3]],
                dtype=np.int64,
            ),
            np.array(
                [[0]],
                dtype=np.int64,
            ),
        ]

        frequencies = count_table_frequencies(
            chunks,
            table_sizes=np.array(
                [5],
                dtype=np.int64,
            ),
        )

        np.testing.assert_array_equal(
            frequencies[0],
            np.array(
                [2, 0, 0, 1, 0],
                dtype=np.int64,
            ),
        )

    def test_chunk_width_mismatch_is_rejected(self):
        chunks = [
            np.array(
                [
                    [0, 0],
                    [1, 1],
                ],
                dtype=np.int64,
            )
        ]

        with self.assertRaises(ValueError):
            count_table_frequencies(
                chunks,
                table_sizes=np.array(
                    [2, 2, 2],
                    dtype=np.int64,
                ),
            )


if __name__ == "__main__":
    unittest.main()
