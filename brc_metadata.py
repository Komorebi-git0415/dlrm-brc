#!/usr/bin/env python3

"""Generic metadata format for BRC table-frequency permutations."""

import os
import tempfile

import numpy as np

from brc_frequency import build_frequency_permutation


BRC_FREQUENCY_METADATA_VERSION = 1


def _require_integer_vector(values, name):
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(
            "{} must be one-dimensional".format(name)
        )

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            "{} must contain integers".format(name)
        )

    return array.astype(np.int64, copy=False)


def _require_table_sizes(table_sizes):
    sizes = _require_integer_vector(
        table_sizes,
        "table_sizes",
    )

    if sizes.size == 0:
        raise ValueError(
            "table_sizes must describe at least one table"
        )

    if np.any(sizes <= 0):
        raise ValueError(
            "every embedding table must contain rows"
        )

    return sizes


def _require_frequency(
    frequency,
    table_id,
    num_rows,
):
    name = "frequency_{}".format(table_id)

    freq = _require_integer_vector(
        frequency,
        name,
    )

    if freq.size != num_rows:
        raise ValueError(
            "{} length {} does not match table size {}".format(
                name,
                freq.size,
                num_rows,
            )
        )

    if np.any(freq < 0):
        raise ValueError(
            "{} must be non-negative".format(name)
        )

    return freq


def _require_permutation(
    values,
    name,
    num_rows,
):
    permutation = _require_integer_vector(
        values,
        name,
    )

    if permutation.size != num_rows:
        raise ValueError(
            "{} length {} does not match table size {}".format(
                name,
                permutation.size,
                num_rows,
            )
        )

    expected = np.arange(
        num_rows,
        dtype=np.int64,
    )

    if not np.array_equal(
        np.sort(permutation),
        expected,
    ):
        raise ValueError(
            "{} is not a permutation".format(name)
        )

    return permutation


def _require_scalar_int(value, name):
    array = np.asarray(value)

    if array.shape != ():
        raise ValueError(
            "{} must be a scalar".format(name)
        )

    if not np.issubdtype(
        array.dtype,
        np.integer,
    ):
        raise TypeError(
            "{} must be an integer".format(name)
        )

    return int(array.item())


def _require_scalar_string(value, name):
    array = np.asarray(value)

    if array.shape != ():
        raise ValueError(
            "{} must be a scalar".format(name)
        )

    result = str(array.item())

    if not result:
        raise ValueError(
            "{} must not be empty".format(name)
        )

    return result


def build_frequency_metadata(
    frequencies,
    table_sizes,
    sample_split,
):
    """
    Build one validated dataset-independent
    BRC frequency metadata payload.
    """

    sizes = _require_table_sizes(
        table_sizes
    )

    frequencies = tuple(frequencies)

    if len(frequencies) != sizes.size:
        raise ValueError(
            "frequency table count {} does not "
            "match table_sizes {}".format(
                len(frequencies),
                sizes.size,
            )
        )

    if (
        not isinstance(sample_split, str)
        or not sample_split
    ):
        raise ValueError(
            "sample_split must be a non-empty string"
        )

    payload = {
        "metadata_version": np.asarray(
            BRC_FREQUENCY_METADATA_VERSION,
            dtype=np.int64,
        ),
        "num_tables": np.asarray(
            sizes.size,
            dtype=np.int64,
        ),
        "sample_split": np.asarray(
            sample_split
        ),
        "table_sizes": sizes,
    }

    num_profiled_samples = None

    for table_id, num_rows in enumerate(
        sizes
    ):
        freq = _require_frequency(
            frequencies[table_id],
            table_id,
            int(num_rows),
        )

        table_samples = int(
            np.sum(
                freq,
                dtype=np.int64,
            )
        )

        if num_profiled_samples is None:
            num_profiled_samples = (
                table_samples
            )
        elif (
            table_samples
            != num_profiled_samples
        ):
            raise ValueError(
                "all tables must describe the "
                "same number of profiled samples"
            )

        old_to_new, new_to_old = (
            build_frequency_permutation(
                freq
            )
        )

        payload[
            "frequency_{}".format(table_id)
        ] = freq

        payload[
            "old_to_new_{}".format(table_id)
        ] = old_to_new

        payload[
            "new_to_old_{}".format(table_id)
        ] = new_to_old

    payload[
        "num_profiled_samples"
    ] = np.asarray(
        num_profiled_samples,
        dtype=np.int64,
    )

    return payload


def save_frequency_metadata(
    output_path,
    frequencies,
    table_sizes,
    sample_split,
):
    """
    Atomically save one compressed NPZ
    frequency metadata artifact.
    """

    payload = build_frequency_metadata(
        frequencies,
        table_sizes,
        sample_split,
    )

    output_path = os.path.abspath(
        os.fspath(output_path)
    )

    output_dir = os.path.dirname(
        output_path
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    fd, temporary_path = (
        tempfile.mkstemp(
            prefix=".brc-frequency-",
            suffix=".npz",
            dir=output_dir,
        )
    )

    os.close(fd)

    try:
        np.savez_compressed(
            temporary_path,
            **payload
        )

        os.replace(
            temporary_path,
            output_path,
        )

    except Exception:
        try:
            os.unlink(
                temporary_path
            )
        except FileNotFoundError:
            pass

        raise

    return output_path


def load_frequency_metadata(
    input_path,
):
    """
    Load and fail-closed validate one
    BRC frequency metadata artifact.
    """

    input_path = os.fspath(
        input_path
    )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as data:

        required_header = (
            "metadata_version",
            "num_tables",
            "sample_split",
            "table_sizes",
            "num_profiled_samples",
        )

        for key in required_header:
            if key not in data:
                raise ValueError(
                    "missing {} in {}".format(
                        key,
                        input_path,
                    )
                )

        version = _require_scalar_int(
            data["metadata_version"],
            "metadata_version",
        )

        if (
            version
            != BRC_FREQUENCY_METADATA_VERSION
        ):
            raise ValueError(
                "unsupported metadata_version "
                "{}".format(version)
            )

        num_tables = _require_scalar_int(
            data["num_tables"],
            "num_tables",
        )

        sample_split = (
            _require_scalar_string(
                data["sample_split"],
                "sample_split",
            )
        )

        sizes = _require_table_sizes(
            data["table_sizes"]
        )

        if num_tables != sizes.size:
            raise ValueError(
                "num_tables {} does not match "
                "table_sizes {}".format(
                    num_tables,
                    sizes.size,
                )
            )

        num_profiled_samples = (
            _require_scalar_int(
                data[
                    "num_profiled_samples"
                ],
                "num_profiled_samples",
            )
        )

        if num_profiled_samples < 0:
            raise ValueError(
                "num_profiled_samples "
                "must be non-negative"
            )

        frequencies = []
        old_to_new = []
        new_to_old = []

        for table_id, num_rows in enumerate(
            sizes
        ):
            frequency_key = (
                "frequency_{}".format(
                    table_id
                )
            )

            old_to_new_key = (
                "old_to_new_{}".format(
                    table_id
                )
            )

            new_to_old_key = (
                "new_to_old_{}".format(
                    table_id
                )
            )

            for key in (
                frequency_key,
                old_to_new_key,
                new_to_old_key,
            ):
                if key not in data:
                    raise ValueError(
                        "missing {} in {}".format(
                            key,
                            input_path,
                        )
                    )

            freq = _require_frequency(
                data[frequency_key],
                table_id,
                int(num_rows),
            )

            if (
                int(
                    np.sum(
                        freq,
                        dtype=np.int64,
                    )
                )
                != num_profiled_samples
            ):
                raise ValueError(
                    "frequency_{} sample count "
                    "does not match metadata".format(
                        table_id
                    )
                )

            stored_old_to_new = (
                _require_permutation(
                    data[old_to_new_key],
                    old_to_new_key,
                    int(num_rows),
                )
            )

            stored_new_to_old = (
                _require_permutation(
                    data[new_to_old_key],
                    new_to_old_key,
                    int(num_rows),
                )
            )

            (
                expected_old_to_new,
                expected_new_to_old,
            ) = build_frequency_permutation(
                freq
            )

            if not np.array_equal(
                stored_old_to_new,
                expected_old_to_new,
            ):
                raise ValueError(
                    "{} does not match frequency "
                    "ordering".format(
                        old_to_new_key
                    )
                )

            if not np.array_equal(
                stored_new_to_old,
                expected_new_to_old,
            ):
                raise ValueError(
                    "{} does not match frequency "
                    "ordering".format(
                        new_to_old_key
                    )
                )

            frequencies.append(
                freq.copy()
            )

            old_to_new.append(
                stored_old_to_new.copy()
            )

            new_to_old.append(
                stored_new_to_old.copy()
            )

    return {
        "metadata_version": version,
        "num_tables": num_tables,
        "sample_split": sample_split,
        "table_sizes": sizes.copy(),
        "num_profiled_samples":
            num_profiled_samples,
        "frequencies":
            tuple(frequencies),
        "old_to_new":
            tuple(old_to_new),
        "new_to_old":
            tuple(new_to_old),
    }
