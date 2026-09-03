#!/usr/bin/env python3

"""
Generic frequency-aware embedding-row reordering for BRC.

This module operates only on table-local embedding row IDs.  It has no
knowledge of any specific dataset, model implementation, table count, or checkpoint
layout.

For one embedding table:

    canonical local IDs
            |
            v
    count_frequencies()
            |
            v
    build_frequency_permutation()
            |
            v
    old_to_new / new_to_old
            |
            v
    remap_local_ids()

Rows are ordered by:

    1. descending access frequency;
    2. ascending original local row ID for deterministic tie breaking.
"""

from typing import Tuple

import numpy as np


def _require_integer_array(values, name):
    """Return *values* as a one-dimensional integer NumPy array."""
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError("{} must be one-dimensional".format(name))

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("{} must contain integer values".format(name))

    return array


def count_frequencies(local_ids, num_rows):
    """
    Count accesses to every row of one embedding table.

    Parameters
    ----------
    local_ids:
        One-dimensional sequence of canonical table-local row IDs.
    num_rows:
        Total number of rows in the embedding table.  This is explicit so
        rows that are never accessed are still represented.

    Returns
    -------
    numpy.ndarray
        int64 array of length ``num_rows`` where element r is the number of
        occurrences of canonical local row ID r.

    Raises
    ------
    ValueError
        If num_rows is invalid or any ID lies outside [0, num_rows).
    TypeError
        If local_ids does not contain integer values.
    """
    if isinstance(num_rows, bool) or not isinstance(
        num_rows, (int, np.integer)
    ):
        raise TypeError("num_rows must be an integer")

    num_rows = int(num_rows)

    if num_rows < 0:
        raise ValueError("num_rows must be non-negative")

    ids = _require_integer_array(local_ids, "local_ids")

    if ids.size:
        if num_rows == 0:
            raise ValueError("non-empty local_ids require num_rows > 0")

        if np.any(ids < 0) or np.any(ids >= num_rows):
            raise ValueError("local ID outside [0, num_rows)")

    frequencies = np.bincount(ids.astype(np.int64), minlength=num_rows)

    # bincount returns at least minlength entries.  With the range check above
    # it must therefore have exactly num_rows entries.
    return frequencies.astype(np.int64, copy=False)


def build_frequency_permutation(frequencies) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build deterministic old<->new embedding-row permutations.

    Rows with higher frequency receive smaller new IDs.  Equal-frequency
    rows are ordered by ascending old local ID.

    Parameters
    ----------
    frequencies:
        One-dimensional sequence containing one non-negative access count
        for every old local row ID.

    Returns
    -------
    (old_to_new, new_to_old):
        Two int64 arrays forming inverse permutations.
    """
    freq = _require_integer_array(frequencies, "frequencies").astype(
        np.int64, copy=False
    )

    if np.any(freq < 0):
        raise ValueError("frequencies must be non-negative")

    num_rows = freq.size
    old_ids = np.arange(num_rows, dtype=np.int64)

    # np.lexsort uses the last key as the primary key:
    #   primary   = -frequency  (descending frequency)
    #   secondary = old ID      (ascending deterministic tie break)
    new_to_old = np.lexsort((old_ids, -freq)).astype(np.int64, copy=False)

    old_to_new = np.empty(num_rows, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(num_rows, dtype=np.int64)

    return old_to_new, new_to_old


def remap_local_ids(local_ids, old_to_new):
    """
    Apply an old-local-ID -> new-local-ID permutation to an ID stream.
    """
    ids = _require_integer_array(local_ids, "local_ids")
    permutation = _require_integer_array(
        old_to_new, "old_to_new"
    ).astype(np.int64, copy=False)

    num_rows = permutation.size

    if num_rows:
        expected = np.arange(num_rows, dtype=np.int64)
        if not np.array_equal(np.sort(permutation), expected):
            raise ValueError("old_to_new must be a permutation")
    elif ids.size:
        raise ValueError("non-empty local_ids require a non-empty permutation")

    if ids.size and (np.any(ids < 0) or np.any(ids >= num_rows)):
        raise ValueError("local ID outside permutation range")

    return permutation[ids.astype(np.int64)]


def invert_permutation(permutation):
    """
    Return the inverse of a one-dimensional integer permutation.
    """
    perm = _require_integer_array(
        permutation, "permutation"
    ).astype(np.int64, copy=False)

    num_rows = perm.size
    expected = np.arange(num_rows, dtype=np.int64)

    if not np.array_equal(np.sort(perm), expected):
        raise ValueError("permutation must contain each row ID exactly once")

    inverse = np.empty(num_rows, dtype=np.int64)
    inverse[perm] = expected

    return inverse


def count_table_frequencies(categorical_chunks, table_sizes):
    """
    Count lookup frequencies for an arbitrary number of embedding tables.

    Parameters
    ----------
    categorical_chunks:
        Iterable of two-dimensional integer arrays. Each array has shape
        (num_samples, num_tables), and each element is an effective table-local
        embedding row ID.
    table_sizes:
        One-dimensional sequence containing the effective number of embedding
        rows for every table.

    Returns
    -------
    tuple of numpy.ndarray
        One int64 frequency array per table.

    Notes
    -----
    The number of tables is derived exclusively from table_sizes.
    This function has no dataset-specific table-count assumptions.
    """
    sizes = _require_integer_array(table_sizes, "table_sizes").astype(
        np.int64, copy=False
    )

    if np.any(sizes < 0):
        raise ValueError("table_sizes must be non-negative")

    num_tables = sizes.size
    frequencies = [
        np.zeros(int(num_rows), dtype=np.int64)
        for num_rows in sizes
    ]

    for chunk in categorical_chunks:
        array = np.asarray(chunk)

        if array.ndim != 2:
            raise ValueError(
                "categorical chunk must be two-dimensional"
            )

        if array.shape[1] != num_tables:
            raise ValueError(
                "categorical chunk table count does not match table_sizes"
            )

        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(
                "categorical chunks must contain integer values"
            )

        array = array.astype(np.int64, copy=False)

        for table_id, num_rows in enumerate(sizes):
            ids = array[:, table_id]

            if ids.size:
                if num_rows == 0:
                    raise ValueError(
                        "non-empty IDs cannot address a zero-row table"
                    )

                if np.any(ids < 0) or np.any(ids >= num_rows):
                    raise ValueError(
                        "local ID outside table range for table {}".format(
                            table_id
                        )
                    )

            frequencies[table_id] += np.bincount(
                ids,
                minlength=int(num_rows),
            ).astype(np.int64, copy=False)

    return tuple(frequencies)
