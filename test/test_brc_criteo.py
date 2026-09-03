#!/usr/bin/env python3

import os
import tempfile
import unittest

import numpy as np

from brc_criteo import CriteoBRCAdapter


class TestCriteoBRCAdapter(unittest.TestCase):
    NUM_CRITEO_TABLES = 26

    def _make_dataset(
        self,
        directory,
        x_cat,
        counts=None,
    ):
        raw_path = os.path.join(
            directory,
            "day",
        )

        processed_file = os.path.join(
            directory,
            "terabyte_processed.npz",
        )

        if counts is None:
            counts = np.full(
                self.NUM_CRITEO_TABLES,
                8,
                dtype=np.int32,
            )

        total_per_file = np.array(
            [2, 2, 2],
            dtype=np.int64,
        )

        np.savez_compressed(
            raw_path + "_day_count.npz",
            total_per_file=total_per_file,
        )

        np.savez_compressed(
            processed_file,
            X_cat=x_cat,
            X_int=np.zeros(
                (len(x_cat), 13),
                dtype=np.int32,
            ),
            y=np.zeros(
                len(x_cat),
                dtype=np.int32,
            ),
            counts=counts,
        )

        np.savez_compressed(
            raw_path + "_fea_count.npz",
            counts=counts,
        )

        for day in range(3):
            start = day * 2
            end = start + 2

            np.savez_compressed(
                "{}_{}_reordered.npz".format(
                    raw_path,
                    day,
                ),
                X_cat=x_cat[start:end],
                X_int=np.zeros(
                    (2, 13),
                    dtype=np.int32,
                ),
                y=np.zeros(
                    2,
                    dtype=np.int32,
                ),
            )

        return raw_path, processed_file

    def _base_x_cat(self):
        x_cat = np.zeros(
            (
                6,
                self.NUM_CRITEO_TABLES,
            ),
            dtype=np.float64,
        )

        x_cat[:, 0] = np.array(
            [0, 1, 2, 1, 2, 1],
            dtype=np.float64,
        )

        x_cat[:, 1] = np.array(
            [2, 2, 1, 0, 1, 0],
            dtype=np.float64,
        )

        return x_cat

    def test_non_memory_map_train_and_all_frequency(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path, processed_file = (
                self._make_dataset(
                    directory,
                    self._base_x_cat(),
                )
            )

            adapter = CriteoBRCAdapter(
                dataset="terabyte",
                raw_path=raw_path,
                processed_data_file=processed_file,
                memory_map=False,
                max_ind_range=-1,
            )

            self.assertEqual(
                adapter.num_tables,
                self.NUM_CRITEO_TABLES,
            )

            train_frequency = (
                adapter.profile_frequencies(
                    "train"
                )
            )

            all_frequency = (
                adapter.profile_frequencies(
                    "all"
                )
            )

            np.testing.assert_array_equal(
                train_frequency[0],
                np.array(
                    [1, 2, 1, 0, 0, 0, 0, 0],
                    dtype=np.int64,
                ),
            )

            np.testing.assert_array_equal(
                all_frequency[0],
                np.array(
                    [1, 3, 2, 0, 0, 0, 0, 0],
                    dtype=np.int64,
                ),
            )

    def test_memory_map_train_matches_combined_train(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path, processed_file = (
                self._make_dataset(
                    directory,
                    self._base_x_cat(),
                )
            )

            combined = CriteoBRCAdapter(
                dataset="terabyte",
                raw_path=raw_path,
                processed_data_file=processed_file,
                memory_map=False,
                max_ind_range=-1,
            )

            memory_map = CriteoBRCAdapter(
                dataset="terabyte",
                raw_path=raw_path,
                processed_data_file=processed_file,
                memory_map=True,
                max_ind_range=-1,
            )

            combined_frequency = (
                combined.profile_frequencies(
                    "train"
                )
            )

            memory_map_frequency = (
                memory_map.profile_frequencies(
                    "train"
                )
            )

            self.assertEqual(
                len(combined_frequency),
                len(memory_map_frequency),
            )

            for left, right in zip(
                combined_frequency,
                memory_map_frequency,
            ):
                np.testing.assert_array_equal(
                    left,
                    right,
                )

    def test_final_day_test_and_val_split(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path, processed_file = (
                self._make_dataset(
                    directory,
                    self._base_x_cat(),
                )
            )

            adapter = CriteoBRCAdapter(
                dataset="terabyte",
                raw_path=raw_path,
                processed_data_file=processed_file,
                memory_map=True,
                max_ind_range=-1,
            )

            test_frequency = (
                adapter.profile_frequencies(
                    "test"
                )
            )

            val_frequency = (
                adapter.profile_frequencies(
                    "val"
                )
            )

            np.testing.assert_array_equal(
                test_frequency[0],
                np.array(
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    dtype=np.int64,
                ),
            )

            np.testing.assert_array_equal(
                val_frequency[0],
                np.array(
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    dtype=np.int64,
                ),
            )

    def test_max_ind_range_matches_stock_lookup_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            x_cat = self._base_x_cat()

            x_cat[:, 0] = np.array(
                [0, 5, 6, 7, 5, 1],
                dtype=np.float64,
            )

            counts = np.full(
                self.NUM_CRITEO_TABLES,
                8,
                dtype=np.int32,
            )

            raw_path, processed_file = (
                self._make_dataset(
                    directory,
                    x_cat,
                    counts=counts,
                )
            )

            adapter = CriteoBRCAdapter(
                dataset="terabyte",
                raw_path=raw_path,
                processed_data_file=processed_file,
                memory_map=False,
                max_ind_range=4,
            )

            self.assertEqual(
                int(adapter.table_sizes[0]),
                4,
            )

            frequency = (
                adapter.profile_frequencies(
                    "all"
                )
            )

            np.testing.assert_array_equal(
                frequency[0],
                np.array(
                    [1, 3, 1, 1],
                    dtype=np.int64,
                ),
            )


    def test_memory_map_reader_obeys_stream_chunk_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            x_cat = self._base_x_cat()

            raw_path, processed_file = (
                self._make_dataset(
                    directory,
                    x_cat,
                )
            )

            adapter = CriteoBRCAdapter(
                dataset="terabyte",
                raw_path=raw_path,
                processed_data_file=processed_file,
                memory_map=True,
                max_ind_range=-1,
                stream_chunk_rows=1,
            )

            chunks = list(
                adapter.iter_categorical_chunks(
                    "train"
                )
            )

            # Synthetic dataset contains:
            #
            #   day 0: 2 rows
            #   day 1: 2 rows
            #   day 2: 2 rows
            #
            # train excludes the final day. With one row
            # per chunk, four training rows must produce
            # exactly four chunks.
            self.assertEqual(
                [chunk.shape[0] for chunk in chunks],
                [1, 1, 1, 1],
            )

            for chunk in chunks:
                self.assertEqual(
                    chunk.dtype,
                    np.dtype(np.int64),
                )

            actual = np.concatenate(
                chunks,
                axis=0,
            )

            expected = x_cat[:4].astype(
                np.int64
            )

            np.testing.assert_array_equal(
                actual,
                expected,
            )

    def test_invalid_stream_chunk_rows_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path, processed_file = (
                self._make_dataset(
                    directory,
                    self._base_x_cat(),
                )
            )

            with self.assertRaises(ValueError):
                CriteoBRCAdapter(
                    dataset="terabyte",
                    raw_path=raw_path,
                    processed_data_file=processed_file,
                    memory_map=True,
                    max_ind_range=-1,
                    stream_chunk_rows=0,
                )



if __name__ == "__main__":
    unittest.main()
