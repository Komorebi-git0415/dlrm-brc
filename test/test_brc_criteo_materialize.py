import os
import tempfile
import unittest
import zipfile

import numpy as np

from brc_criteo_materialize import (
    materialize_criteo_day,
    validate_criteo_day_materialization,
)


class TestBRCMaterializeCriteoDay(
    unittest.TestCase
):
    @staticmethod
    def _permutations():
        return [
            np.array(
                [2, 0, 1],
                dtype=np.int64,
            ),
            np.array(
                [1, 3, 0, 2],
                dtype=np.int64,
            ),
            np.array(
                [1, 0],
                dtype=np.int64,
            ),
        ]

    @staticmethod
    def _arrays():
        x_cat = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 0],
                [0, 3, 1],
                [2, 1, 1],
                [1, 0, 0],
            ],
            dtype=np.float64,
        )

        x_int = np.arange(
            6 * 13,
            dtype=np.float64,
        ).reshape(
            6,
            13,
        )

        y = np.array(
            [0, 1, 0, 1, 1, 0],
            dtype=np.float64,
        )

        return (
            x_cat,
            x_int,
            y,
        )

    @classmethod
    def _write_source(
        cls,
        path,
        x_cat=None,
    ):
        (
            default_x_cat,
            x_int,
            y,
        ) = cls._arrays()

        if x_cat is None:
            x_cat = default_x_cat

        np.savez_compressed(
            path,
            X_cat=x_cat,
            X_int=x_int,
            y=y,
        )

    def test_materialize_maps_x_cat_and_preserves_other_arrays(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(
                directory,
                "source.npz",
            )

            derived = os.path.join(
                directory,
                "derived.npz",
            )

            self._write_source(
                source
            )

            permutations = (
                self._permutations()
            )

            result = materialize_criteo_day(
                source_path=source,
                output_path=derived,
                old_to_new=permutations,
                chunk_rows=2,
            )

            self.assertEqual(
                result,
                derived,
            )

            self.assertTrue(
                os.path.isfile(
                    derived
                )
            )

            with np.load(
                source,
                allow_pickle=False,
            ) as source_data:
                source_x_cat = np.asarray(
                    source_data["X_cat"]
                )

                source_x_int = np.asarray(
                    source_data["X_int"]
                )

                source_y = np.asarray(
                    source_data["y"]
                )

            with np.load(
                derived,
                allow_pickle=False,
            ) as derived_data:
                derived_x_cat = np.asarray(
                    derived_data["X_cat"]
                )

                derived_x_int = np.asarray(
                    derived_data["X_int"]
                )

                derived_y = np.asarray(
                    derived_data["y"]
                )

            expected = (
                source_x_cat.astype(
                    np.int64
                )
            )

            for table_id, permutation in enumerate(
                permutations
            ):
                expected[
                    :,
                    table_id,
                ] = permutation[
                    expected[
                        :,
                        table_id,
                    ]
                ]

            expected = expected.astype(
                np.float64
            )

            np.testing.assert_array_equal(
                derived_x_cat,
                expected,
            )

            np.testing.assert_array_equal(
                derived_x_int,
                source_x_int,
            )

            np.testing.assert_array_equal(
                derived_y,
                source_y,
            )

            self.assertEqual(
                derived_x_cat.dtype,
                source_x_cat.dtype,
            )

            self.assertEqual(
                derived_x_cat.shape,
                source_x_cat.shape,
            )

            self.assertTrue(
                derived_x_cat.flags.c_contiguous
            )

            with zipfile.ZipFile(
                derived,
                "r",
            ) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "X_cat.npy",
                        "X_int.npy",
                        "y.npy",
                    ],
                )

    def test_validator_accepts_valid_materialization(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(
                directory,
                "source.npz",
            )

            derived = os.path.join(
                directory,
                "derived.npz",
            )

            self._write_source(
                source
            )

            permutations = (
                self._permutations()
            )

            materialize_criteo_day(
                source,
                derived,
                permutations,
                chunk_rows=1,
            )

            self.assertTrue(
                validate_criteo_day_materialization(
                    source,
                    derived,
                    permutations,
                    chunk_rows=3,
                )
            )

    def test_existing_output_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(
                directory,
                "source.npz",
            )

            derived = os.path.join(
                directory,
                "derived.npz",
            )

            self._write_source(
                source
            )

            with open(
                derived,
                "wb",
            ) as stream:
                stream.write(
                    b"existing"
                )

            with self.assertRaises(
                FileExistsError
            ):
                materialize_criteo_day(
                    source,
                    derived,
                    self._permutations(),
                    chunk_rows=2,
                )

            with open(
                derived,
                "rb",
            ) as stream:
                self.assertEqual(
                    stream.read(),
                    b"existing",
                )

    def test_non_integral_id_is_rejected_without_publish(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(
                directory,
                "source.npz",
            )

            derived = os.path.join(
                directory,
                "derived.npz",
            )

            (
                x_cat,
                _,
                _,
            ) = self._arrays()

            x_cat = x_cat.copy()
            x_cat[0, 0] = 0.5

            self._write_source(
                source,
                x_cat=x_cat,
            )

            with self.assertRaises(
                ValueError
            ):
                materialize_criteo_day(
                    source,
                    derived,
                    self._permutations(),
                    chunk_rows=2,
                )

            self.assertFalse(
                os.path.exists(
                    derived
                )
            )

    def test_out_of_range_id_is_rejected_without_publish(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(
                directory,
                "source.npz",
            )

            derived = os.path.join(
                directory,
                "derived.npz",
            )

            (
                x_cat,
                _,
                _,
            ) = self._arrays()

            x_cat = x_cat.copy()

            # Table 0 has cardinality 3:
            # valid IDs are 0, 1, 2.
            x_cat[0, 0] = 3

            self._write_source(
                source,
                x_cat=x_cat,
            )

            with self.assertRaises(
                ValueError
            ):
                materialize_criteo_day(
                    source,
                    derived,
                    self._permutations(),
                    chunk_rows=2,
                )

            self.assertFalse(
                os.path.exists(
                    derived
                )
            )

    def test_invalid_permutation_is_rejected_without_publish(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(
                directory,
                "source.npz",
            )

            derived = os.path.join(
                directory,
                "derived.npz",
            )

            self._write_source(
                source
            )

            permutations = (
                self._permutations()
            )

            permutations[0] = np.array(
                [0, 0, 2],
                dtype=np.int64,
            )

            with self.assertRaises(
                ValueError
            ):
                materialize_criteo_day(
                    source,
                    derived,
                    permutations,
                    chunk_rows=2,
                )

            self.assertFalse(
                os.path.exists(
                    derived
                )
            )

    def test_validator_detects_tampered_x_cat(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(
                directory,
                "source.npz",
            )

            derived = os.path.join(
                directory,
                "derived.npz",
            )

            self._write_source(
                source
            )

            permutations = (
                self._permutations()
            )

            materialize_criteo_day(
                source,
                derived,
                permutations,
                chunk_rows=2,
            )

            with np.load(
                derived,
                allow_pickle=False,
            ) as data:
                x_cat = np.asarray(
                    data["X_cat"]
                ).copy()

                x_int = np.asarray(
                    data["X_int"]
                ).copy()

                y = np.asarray(
                    data["y"]
                ).copy()

            x_cat[0, 0] = (
                x_cat[0, 0] + 1
            ) % 3

            np.savez_compressed(
                derived,
                X_cat=x_cat,
                X_int=x_int,
                y=y,
            )

            with self.assertRaises(
                ValueError
            ):
                validate_criteo_day_materialization(
                    source,
                    derived,
                    permutations,
                    chunk_rows=2,
                )


if __name__ == "__main__":
    unittest.main()
