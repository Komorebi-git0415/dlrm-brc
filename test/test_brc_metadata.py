#!/usr/bin/env python3

import os
import tempfile
import unittest

import numpy as np

from brc_metadata import (
    BRC_FREQUENCY_METADATA_VERSION,
    build_frequency_metadata,
    load_frequency_metadata,
    save_frequency_metadata,
)


class TestBRCFrequencyMetadata(
    unittest.TestCase
):
    def _frequencies(self):
        return (
            np.array(
                [1, 3, 2],
                dtype=np.int64,
            ),
            np.array(
                [2, 0, 4],
                dtype=np.int64,
            ),
        )

    def test_build_contains_deterministic_permutations(
        self
    ):
        payload = build_frequency_metadata(
            self._frequencies(),
            table_sizes=np.array(
                [3, 3],
                dtype=np.int64,
            ),
            sample_split="train",
        )

        self.assertEqual(
            int(
                payload[
                    "metadata_version"
                ]
            ),
            BRC_FREQUENCY_METADATA_VERSION,
        )

        self.assertEqual(
            int(payload["num_tables"]),
            2,
        )

        self.assertEqual(
            str(payload["sample_split"]),
            "train",
        )

        self.assertEqual(
            int(
                payload[
                    "num_profiled_samples"
                ]
            ),
            6,
        )

        np.testing.assert_array_equal(
            payload["old_to_new_0"],
            np.array(
                [2, 0, 1],
                dtype=np.int64,
            ),
        )

        np.testing.assert_array_equal(
            payload["new_to_old_0"],
            np.array(
                [1, 2, 0],
                dtype=np.int64,
            ),
        )

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(
                directory,
                "brc_frequency_metadata.npz",
            )

            save_frequency_metadata(
                path,
                self._frequencies(),
                table_sizes=np.array(
                    [3, 3],
                    dtype=np.int64,
                ),
                sample_split="train",
            )

            metadata = (
                load_frequency_metadata(
                    path
                )
            )

            self.assertEqual(
                metadata[
                    "metadata_version"
                ],
                1,
            )

            self.assertEqual(
                metadata["num_tables"],
                2,
            )

            self.assertEqual(
                metadata["sample_split"],
                "train",
            )

            self.assertEqual(
                metadata[
                    "num_profiled_samples"
                ],
                6,
            )

            np.testing.assert_array_equal(
                metadata[
                    "frequencies"
                ][1],
                np.array(
                    [2, 0, 4],
                    dtype=np.int64,
                ),
            )

    def test_same_frequency_input_builds_same_permutation(
        self
    ):
        left = build_frequency_metadata(
            self._frequencies(),
            table_sizes=np.array(
                [3, 3],
                dtype=np.int64,
            ),
            sample_split="train",
        )

        right = build_frequency_metadata(
            self._frequencies(),
            table_sizes=np.array(
                [3, 3],
                dtype=np.int64,
            ),
            sample_split="train",
        )

        for table_id in range(2):
            np.testing.assert_array_equal(
                left[
                    "old_to_new_{}".format(
                        table_id
                    )
                ],
                right[
                    "old_to_new_{}".format(
                        table_id
                    )
                ],
            )

            np.testing.assert_array_equal(
                left[
                    "new_to_old_{}".format(
                        table_id
                    )
                ],
                right[
                    "new_to_old_{}".format(
                        table_id
                    )
                ],
            )

    def test_frequency_length_mismatch_is_rejected(
        self
    ):
        frequencies = (
            np.array(
                [1, 2],
                dtype=np.int64,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            build_frequency_metadata(
                frequencies,
                table_sizes=np.array(
                    [3],
                    dtype=np.int64,
                ),
                sample_split="train",
            )

    def test_table_sample_count_mismatch_is_rejected(
        self
    ):
        frequencies = (
            np.array(
                [1, 3, 2],
                dtype=np.int64,
            ),
            np.array(
                [2, 0, 3],
                dtype=np.int64,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            build_frequency_metadata(
                frequencies,
                table_sizes=np.array(
                    [3, 3],
                    dtype=np.int64,
                ),
                sample_split="train",
            )

    def test_tampered_permutation_is_rejected(
        self
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(
                directory,
                "tampered.npz",
            )

            payload = (
                build_frequency_metadata(
                    self._frequencies(),
                    table_sizes=np.array(
                        [3, 3],
                        dtype=np.int64,
                    ),
                    sample_split="train",
                )
            )

            payload[
                "old_to_new_0"
            ] = np.array(
                [0, 1, 2],
                dtype=np.int64,
            )

            np.savez_compressed(
                path,
                **payload
            )

            with self.assertRaises(
                ValueError
            ):
                load_frequency_metadata(
                    path
                )


if __name__ == "__main__":
    unittest.main()
