import numpy as np


def _validate_inputs(
    frequencies,
    new_to_old,
):
    if not isinstance(
        frequencies,
        (list, tuple),
    ):
        raise TypeError(
            "frequencies must be a list or tuple"
        )

    if not isinstance(
        new_to_old,
        (list, tuple),
    ):
        raise TypeError(
            "new_to_old must be a list or tuple"
        )

    if len(frequencies) != len(new_to_old):
        raise ValueError(
            "frequency/permutation table count mismatch"
        )

    if not frequencies:
        raise ValueError(
            "at least one table is required"
        )

    validated_frequencies = []
    validated_new_to_old = []

    for table_id, (
        frequency,
        permutation,
    ) in enumerate(
        zip(
            frequencies,
            new_to_old,
        )
    ):
        frequency = np.asarray(
            frequency
        )

        permutation = np.asarray(
            permutation
        )

        if frequency.ndim != 1:
            raise ValueError(
                "frequency {} must be "
                "one-dimensional".format(
                    table_id
                )
            )

        if permutation.ndim != 1:
            raise ValueError(
                "new_to_old {} must be "
                "one-dimensional".format(
                    table_id
                )
            )

        if len(frequency) != len(permutation):
            raise ValueError(
                "table {} frequency/permutation "
                "length mismatch".format(
                    table_id
                )
            )

        if len(frequency) <= 0:
            raise ValueError(
                "table {} must not be empty".format(
                    table_id
                )
            )

        if not np.issubdtype(
            frequency.dtype,
            np.integer,
        ):
            raise TypeError(
                "frequency {} must contain "
                "integers".format(
                    table_id
                )
            )

        if not np.issubdtype(
            permutation.dtype,
            np.integer,
        ):
            raise TypeError(
                "new_to_old {} must contain "
                "integers".format(
                    table_id
                )
            )

        frequency = frequency.astype(
            np.int64,
            copy=False,
        )

        permutation = permutation.astype(
            np.int64,
            copy=False,
        )

        if np.any(
            frequency < 0
        ):
            raise ValueError(
                "frequency {} contains negative "
                "values".format(
                    table_id
                )
            )

        expected = np.arange(
            len(permutation),
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(permutation),
            expected,
        ):
            raise ValueError(
                "new_to_old {} is not a "
                "permutation".format(
                    table_id
                )
            )

        validated_frequencies.append(
            frequency
        )

        validated_new_to_old.append(
            permutation
        )

    return (
        validated_frequencies,
        validated_new_to_old,
    )


def build_global_storage_layout(
    frequencies,
    new_to_old,
):
    """
    Build the second-stage cross-table storage layout.

    Runtime IDs are NOT changed here.

    Each table has already completed the first-stage
    local permutation. Therefore local row j in table t
    corresponds to:

        old_row = new_to_old[t][j]

    Its profile frequency is:

        frequencies[t][old_row]

    All post-local-reorder rows are then globally sorted
    by:

        1. frequency descending
        2. table_id ascending
        3. new_local_id ascending

    The result maps each runtime pair

        (table_id, new_local_id)

    to one global checkpoint storage slot.
    """
    (
        frequencies,
        new_to_old,
    ) = _validate_inputs(
        frequencies,
        new_to_old,
    )

    num_tables = len(
        frequencies
    )

    table_offsets = np.zeros(
        num_tables + 1,
        dtype=np.int64,
    )

    local_frequency_parts = []

    for table_id in range(
        num_tables
    ):
        frequency = frequencies[
            table_id
        ]

        permutation = new_to_old[
            table_id
        ]

        # Frequency indexed by the post-stage-1
        # runtime local row ID.
        local_frequency = frequency[
            permutation
        ]

        local_frequency_parts.append(
            local_frequency
        )

        table_offsets[
            table_id + 1
        ] = (
            table_offsets[table_id]
            + len(local_frequency)
        )

    flat_local_frequency = np.concatenate(
        local_frequency_parts
    )

    total_rows = len(
        flat_local_frequency
    )

    # Before sorting, flattened order is:
    #
    #   table 0 local row 0..N0-1
    #   table 1 local row 0..N1-1
    #   ...
    #
    # A stable descending-frequency sort therefore
    # gives the deterministic tie-break:
    #
    #   table_id ascending,
    #   then new_local_id ascending.
    global_order_flat_local = np.argsort(
        -flat_local_frequency,
        kind="stable",
    ).astype(
        np.int64,
        copy=False,
    )

    global_slot_by_flat_local = np.empty(
        total_rows,
        dtype=np.int64,
    )

    global_slot_by_flat_local[
        global_order_flat_local
    ] = np.arange(
        total_rows,
        dtype=np.int64,
    )

    frequency_by_global_slot = (
        flat_local_frequency[
            global_order_flat_local
        ]
    )

    if total_rows > 1:
        if not np.all(
            frequency_by_global_slot[:-1]
            >= frequency_by_global_slot[1:]
        ):
            raise AssertionError(
                "global frequency ordering failed"
            )

    return {
        "num_tables": num_tables,
        "total_rows": total_rows,
        "table_offsets": table_offsets,
        "global_slot_by_flat_local": (
            global_slot_by_flat_local
        ),
        "flat_local_by_global_slot": (
            global_order_flat_local
        ),
        "frequency_by_global_slot": (
            frequency_by_global_slot
        ),
    }


def split_global_slots_by_table(
    layout,
):
    """
    Return one global-slot vector for each runtime table.

    result[t][new_local_id] gives the global checkpoint
    storage slot of that post-stage-1 embedding row.
    """
    offsets = np.asarray(
        layout["table_offsets"],
        dtype=np.int64,
    )

    slots = np.asarray(
        layout["global_slot_by_flat_local"],
        dtype=np.int64,
    )

    num_tables = int(
        layout["num_tables"]
    )

    if len(offsets) != num_tables + 1:
        raise ValueError(
            "invalid table_offsets"
        )

    result = []

    for table_id in range(
        num_tables
    ):
        start = int(
            offsets[table_id]
        )

        end = int(
            offsets[table_id + 1]
        )

        result.append(
            slots[start:end]
        )

    return result
