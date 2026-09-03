#!/usr/bin/env python3

"""
BRC adapter for the stock DLRM Criteo processed-data representation.

This module understands the file naming and split semantics used by the
upstream DLRM Criteo implementation. Frequency counting and permutation logic
remain dataset-independent.

Important lookup semantics mirrored from stock DLRM:

    effective table size =
        min(counts[t], max_ind_range)   if max_ind_range > 0
        counts[t]                       otherwise

    effective lookup ID =
        stored X_cat % max_ind_range    if max_ind_range > 0
        stored X_cat                    otherwise
"""

import os
import zipfile

import numpy as np

from brc_dataset import BRCDatasetAdapter, validate_sample_split


class CriteoBRCAdapter(BRCDatasetAdapter):
    def __init__(
        self,
        dataset,
        raw_path,
        processed_data_file,
        memory_map,
        max_ind_range,
        stream_chunk_rows=65536,
    ):
        if dataset not in ("kaggle", "terabyte"):
            raise ValueError(
                "unsupported Criteo dataset: {}".format(dataset)
            )

        if isinstance(max_ind_range, bool) or not isinstance(
            max_ind_range, (int, np.integer)
        ):
            raise TypeError("max_ind_range must be an integer")

        if isinstance(stream_chunk_rows, bool) or not isinstance(
            stream_chunk_rows, (int, np.integer)
        ):
            raise TypeError(
                "stream_chunk_rows must be an integer"
            )

        if int(stream_chunk_rows) <= 0:
            raise ValueError(
                "stream_chunk_rows must be positive"
            )

        self.dataset = dataset
        self.raw_path = str(raw_path)
        self.processed_data_file = (
            None
            if processed_data_file is None
            else str(processed_data_file)
        )
        self.memory_map = bool(memory_map)
        self.max_ind_range = int(max_ind_range)
        self.stream_chunk_rows = int(stream_chunk_rows)

        raw_dir = os.path.dirname(self.raw_path)
        raw_base = os.path.basename(self.raw_path)

        if dataset == "kaggle":
            self.d_file = raw_base.split(".")[0]
            npz_base = self.d_file + "_day"
        else:
            self.d_file = raw_base
            npz_base = self.d_file

        self.d_path = raw_dir
        self.npzfile = os.path.join(self.d_path, npz_base)

        self.day_count_file = os.path.join(
            self.d_path,
            self.d_file + "_day_count.npz",
        )

        self.feature_count_file = os.path.join(
            self.d_path,
            self.d_file + "_fea_count.npz",
        )

        self.total_per_file = self._load_total_per_file()
        self._table_sizes = self._load_effective_table_sizes()

    @property
    def table_sizes(self):
        return self._table_sizes.copy()

    @property
    def num_days(self):
        return len(self.total_per_file)

    def _load_total_per_file(self):
        if not os.path.exists(self.day_count_file):
            raise FileNotFoundError(self.day_count_file)

        with np.load(
            self.day_count_file,
            allow_pickle=False,
        ) as data:
            if "total_per_file" not in data:
                raise ValueError(
                    "missing total_per_file in {}".format(
                        self.day_count_file
                    )
                )

            total_per_file = np.asarray(
                data["total_per_file"],
                dtype=np.int64,
            )

        if total_per_file.ndim != 1:
            raise ValueError(
                "total_per_file must be one-dimensional"
            )

        if total_per_file.size == 0:
            raise ValueError(
                "dataset must contain at least one day"
            )

        if np.any(total_per_file < 0):
            raise ValueError(
                "total_per_file must be non-negative"
            )

        return total_per_file

    def _load_counts(self):
        if self.memory_map:
            counts_file = self.feature_count_file

            if not os.path.exists(counts_file):
                raise FileNotFoundError(counts_file)

            with np.load(
                counts_file,
                allow_pickle=False,
            ) as data:
                if "counts" not in data:
                    raise ValueError(
                        "missing counts in {}".format(
                            counts_file
                        )
                    )

                counts = np.asarray(
                    data["counts"],
                    dtype=np.int64,
                )

        else:
            if self.processed_data_file is None:
                raise ValueError(
                    "processed_data_file is required "
                    "when memory_map is false"
                )

            if not os.path.exists(
                self.processed_data_file
            ):
                raise FileNotFoundError(
                    self.processed_data_file
                )

            with np.load(
                self.processed_data_file,
                allow_pickle=False,
            ) as data:
                if "counts" not in data:
                    raise ValueError(
                        "missing counts in {}".format(
                            self.processed_data_file
                        )
                    )

                counts = np.asarray(
                    data["counts"],
                    dtype=np.int64,
                )

        if counts.ndim != 1:
            raise ValueError(
                "counts must be one-dimensional"
            )

        if counts.size == 0:
            raise ValueError(
                "counts must describe at least one table"
            )

        if np.any(counts <= 0):
            raise ValueError(
                "every embedding table must contain rows"
            )

        return counts

    def _load_effective_table_sizes(self):
        counts = self._load_counts()

        if self.max_ind_range > 0:
            return np.minimum(
                counts,
                self.max_ind_range,
            ).astype(np.int64, copy=False)

        return counts.astype(
            np.int64,
            copy=False,
        )

    def _normalize_categorical(self, x_cat):
        """
        Convert stock processed X_cat into effective
        EmbeddingBag lookup IDs.

        Stock processing can persist integer-valued
        categorical IDs using a floating dtype.
        """

        array = np.asarray(x_cat)

        if array.ndim != 2:
            raise ValueError(
                "X_cat must be two-dimensional"
            )

        if array.shape[1] != self.num_tables:
            raise ValueError(
                "X_cat table count {} does not match "
                "counts {}".format(
                    array.shape[1],
                    self.num_tables,
                )
            )

        if np.issubdtype(
            array.dtype,
            np.integer,
        ):
            ids = array.astype(
                np.int64,
                copy=False,
            )

        elif np.issubdtype(
            array.dtype,
            np.floating,
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    "X_cat contains non-finite values"
                )

            if not np.all(
                array == np.floor(array)
            ):
                raise ValueError(
                    "X_cat contains non-integral values"
                )

            ids = array.astype(np.int64)

        else:
            raise TypeError(
                "X_cat must contain numeric row IDs"
            )

        if np.any(ids < 0):
            raise ValueError(
                "X_cat contains negative row IDs"
            )

        # Match stock CriteoDataset.__getitem__().
        if self.max_ind_range > 0:
            ids = ids % self.max_ind_range

        for table_id, num_rows in enumerate(
            self._table_sizes
        ):
            table_ids = ids[:, table_id]

            if (
                table_ids.size
                and np.any(table_ids >= num_rows)
            ):
                raise ValueError(
                    "effective row ID outside embedding "
                    "table {} range".format(table_id)
                )

        return ids

    def _combined_split_bounds(
        self,
        sample_split,
    ):
        validate_sample_split(sample_split)

        total_samples = int(
            np.sum(self.total_per_file)
        )

        if sample_split == "all":
            return 0, total_samples

        train_end = int(
            np.sum(self.total_per_file[:-1])
        )

        final_day_size = int(
            self.total_per_file[-1]
        )

        # Match np.array_split(last_day, 2):
        # first half receives the extra item.
        test_size = (
            final_day_size + 1
        ) // 2

        if sample_split == "train":
            return 0, train_end

        if sample_split == "test":
            return (
                train_end,
                train_end + test_size,
            )

        return (
            train_end + test_size,
            total_samples,
        )

    @staticmethod
    def _read_npy_header(stream):
        """
        Read the NPY header embedded inside an NPZ member.

        Only normal C-order numeric arrays are consumed by the
        streaming reader. Unsupported formats fail closed.
        """
        version = np.lib.format.read_magic(stream)

        if version == (1, 0):
            shape, fortran_order, dtype = (
                np.lib.format.read_array_header_1_0(
                    stream
                )
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = (
                np.lib.format.read_array_header_2_0(
                    stream
                )
            )
        else:
            raise ValueError(
                "unsupported NPY version {}".format(
                    version
                )
            )

        return (
            tuple(shape),
            bool(fortran_order),
            np.dtype(dtype),
        )

    @staticmethod
    def _read_exact(stream, num_bytes):
        """Read exactly num_bytes from a sequential stream."""
        parts = []
        remaining = int(num_bytes)

        while remaining:
            piece = stream.read(remaining)

            if not piece:
                raise ValueError(
                    "truncated X_cat payload"
                )

            parts.append(piece)
            remaining -= len(piece)

        return b"".join(parts)

    @staticmethod
    def _skip_exact(stream, num_bytes):
        """
        Consume exactly num_bytes without retaining them in memory.
        """
        remaining = int(num_bytes)
        skip_chunk_bytes = 8 * 1024 * 1024

        while remaining:
            piece = stream.read(
                min(
                    remaining,
                    skip_chunk_bytes,
                )
            )

            if not piece:
                raise ValueError(
                    "truncated X_cat payload "
                    "while skipping"
                )

            remaining -= len(piece)

    def _iter_npz_x_cat_chunks(
        self,
        filename,
        expected_rows,
        start_row,
        end_row,
    ):
        """
        Stream row chunks directly from X_cat.npy inside an NPZ.

        This avoids materializing an entire Criteo day in RAM.
        """
        member_name = "X_cat.npy"

        with zipfile.ZipFile(
            filename,
            "r",
        ) as archive:
            if member_name not in archive.namelist():
                raise ValueError(
                    "missing X_cat in {}".format(
                        filename
                    )
                )

            with archive.open(
                member_name,
                "r",
            ) as stream:
                (
                    shape,
                    fortran_order,
                    dtype,
                ) = self._read_npy_header(
                    stream
                )

                if len(shape) != 2:
                    raise ValueError(
                        "X_cat must be two-dimensional"
                    )

                if shape[0] != expected_rows:
                    raise ValueError(
                        "{} contains {} rows, "
                        "expected {}".format(
                            filename,
                            shape[0],
                            expected_rows,
                        )
                    )

                if shape[1] != self.num_tables:
                    raise ValueError(
                        "X_cat table count {} "
                        "does not match counts "
                        "{}".format(
                            shape[1],
                            self.num_tables,
                        )
                    )

                if fortran_order:
                    raise ValueError(
                        "Fortran-ordered X_cat is not "
                        "supported by streaming reader"
                    )

                if dtype.hasobject:
                    raise TypeError(
                        "object-dtype X_cat is not supported"
                    )

                if (
                    start_row < 0
                    or end_row < start_row
                    or end_row > expected_rows
                ):
                    raise ValueError(
                        "invalid X_cat row bounds"
                    )

                row_bytes = (
                    self.num_tables
                    * dtype.itemsize
                )

                self._skip_exact(
                    stream,
                    start_row * row_bytes,
                )

                remaining_rows = (
                    end_row - start_row
                )

                while remaining_rows:
                    rows = min(
                        self.stream_chunk_rows,
                        remaining_rows,
                    )

                    payload = self._read_exact(
                        stream,
                        rows * row_bytes,
                    )

                    x_cat = np.frombuffer(
                        payload,
                        dtype=dtype,
                        count=(
                            rows
                            * self.num_tables
                        ),
                    ).reshape(
                        rows,
                        self.num_tables,
                    )

                    yield (
                        self._normalize_categorical(
                            x_cat
                        )
                    )

                    remaining_rows -= rows

    def _iter_memory_map_chunks(
        self,
        sample_split,
    ):
        validate_sample_split(sample_split)

        if sample_split == "all":
            day_indices = range(
                self.num_days
            )
        elif sample_split == "train":
            day_indices = range(
                self.num_days - 1
            )
        else:
            day_indices = [
                self.num_days - 1
            ]

        for day in day_indices:
            filename = (
                "{}_{}_reordered.npz".format(
                    self.npzfile,
                    day,
                )
            )

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    filename
                )

            expected_rows = int(
                self.total_per_file[day]
            )

            start_row = 0
            end_row = expected_rows

            if sample_split == "test":
                test_size = (
                    expected_rows + 1
                ) // 2
                end_row = test_size

            elif sample_split == "val":
                test_size = (
                    expected_rows + 1
                ) // 2
                start_row = test_size

            yield from self._iter_npz_x_cat_chunks(
                filename=filename,
                expected_rows=expected_rows,
                start_row=start_row,
                end_row=end_row,
            )

    def _iter_combined_chunks(
        self,
        sample_split,
    ):
        validate_sample_split(sample_split)

        if self.processed_data_file is None:
            raise ValueError(
                "processed_data_file is required"
            )

        with np.load(
            self.processed_data_file,
            allow_pickle=False,
        ) as data:
            if "X_cat" not in data:
                raise ValueError(
                    "missing X_cat in {}".format(
                        self.processed_data_file
                    )
                )

            x_cat = np.asarray(
                data["X_cat"]
            )

        expected_rows = int(
            np.sum(self.total_per_file)
        )

        if x_cat.shape[0] != expected_rows:
            raise ValueError(
                "{} contains {} rows, "
                "expected {}".format(
                    self.processed_data_file,
                    x_cat.shape[0],
                    expected_rows,
                )
            )

        start, end = (
            self._combined_split_bounds(
                sample_split
            )
        )

        yield self._normalize_categorical(
            x_cat[start:end]
        )

    def iter_categorical_chunks(
        self,
        sample_split,
    ):
        validate_sample_split(sample_split)

        if self.memory_map:
            yield from (
                self._iter_memory_map_chunks(
                    sample_split
                )
            )
        else:
            yield from (
                self._iter_combined_chunks(
                    sample_split
                )
            )
