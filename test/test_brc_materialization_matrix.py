import tempfile
import unittest

import numpy as np
import torch

from brc_block_materializer import reconstruct_blocks
from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    EMBEDDING_DIM,
    ROW_BYTES,
    LAYOUT_GLOBAL_MIXED,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
    plan_dirty_blocks,
)
from brc_global_layout import build_global_storage_layout
from brc_materialization import materialize_dirty_blocks
from brc_materialization_policy import (
    METHOD_PARENT_RMW,
    METHOD_RECONSTRUCT,
    MODE_HYBRID,
    MODE_ONLY_PARENT_RMW,
    MODE_ONLY_RECONSTRUCT,
    MaterializationPolicy,
)


TABLE_SIZES = [70, 67, 66]
TOTAL_ROWS = sum(TABLE_SIZES)


def make_layout(mode):
    if mode == LAYOUT_LOCAL_ONLY:
        return CheckpointLayout(
            table_sizes=TABLE_SIZES,
            mode=LAYOUT_LOCAL_ONLY,
        )

    if mode != LAYOUT_GLOBAL_MIXED:
        raise ValueError(
            f"unexpected layout mode: {mode}"
        )

    # Frequencies deliberately interleave tables in the
    # global checkpoint storage order:
    #
    # T0:L0 > T1:L0 > T2:L0 > T0:L1 > ...
    frequencies = []

    for table_id, size in enumerate(
        TABLE_SIZES
    ):
        frequencies.append(
            (
                1_000_000
                - np.arange(
                    size,
                    dtype=np.int64,
                )
                * len(TABLE_SIZES)
                - table_id
            )
        )

    # Identity stage-1 permutation is sufficient here.
    # Stage-B global layout still mixes rows across tables.
    new_to_old = [
        np.arange(
            size,
            dtype=np.int64,
        )
        for size in TABLE_SIZES
    ]

    global_layout = (
        build_global_storage_layout(
            frequencies,
            new_to_old,
        )
    )

    return CheckpointLayout(
        table_sizes=TABLE_SIZES,
        mode=LAYOUT_GLOBAL_MIXED,
        global_layout=global_layout,
    )


def make_tables():
    tables = []

    for table_id, size in enumerate(
        TABLE_SIZES
    ):
        weight = torch.empty(
            (
                size,
                EMBEDDING_DIM,
            ),
            dtype=torch.float32,
        )

        dimension_values = torch.arange(
            EMBEDDING_DIM,
            dtype=torch.float32,
        )

        for row_id in range(size):
            # Unique values for every table, row, and dimension.
            weight[
                row_id
            ].copy_(
                dimension_values
                + float(
                    table_id * 100_000
                    + row_id * 100
                )
            )

        tables.append(
            weight
        )

    return tables


def dirty_rows_from_storage_slots(
    layout,
    storage_slots,
):
    per_table = [
        set()
        for _ in range(
            layout.num_tables
        )
    ]

    for storage_slot in storage_slots:
        ref = layout.row_ref_for_slot(
            storage_slot
        )

        per_table[
            ref.table_id
        ].add(
            ref.row_id
        )

    return [
        np.asarray(
            sorted(rows),
            dtype=np.int64,
        )
        for rows in per_table
    ]


def apply_dirty_updates(
    layout,
    current_tables,
    storage_slots,
):
    for storage_slot in storage_slots:
        ref = layout.row_ref_for_slot(
            storage_slot
        )

        dimension_values = torch.arange(
            EMBEDDING_DIM,
            dtype=torch.float32,
        )

        current_tables[
            ref.table_id
        ][
            ref.row_id
        ].copy_(
            dimension_values
            + float(
                9_000_000
                + ref.table_id * 10_000
                + ref.row_id * 100
            )
        )


def build_checkpoint_image(
    layout,
    tables,
):
    image = bytearray(
        layout.image_size
    )

    for storage_slot in range(
        layout.total_rows
    ):
        ref = layout.row_ref_for_slot(
            storage_slot
        )

        payload = (
            tables[
                ref.table_id
            ][
                ref.row_id
            ]
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
            .tobytes(
                order="C"
            )
        )

        if len(payload) != ROW_BYTES:
            raise AssertionError(
                "unexpected embedding row size"
            )

        start = (
            storage_slot
            * ROW_BYTES
        )

        image[
            start:
            start + ROW_BYTES
        ] = payload

    return bytes(image)


class ParentFile:
    def __init__(
        self,
        payload,
    ):
        self._file = (
            tempfile.TemporaryFile()
        )

        self._file.write(
            payload
        )

        self._file.flush()

    @property
    def fd(self):
        return self._file.fileno()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self._file.close()


def policies():
    result = [
        (
            "only_reconstruct",
            MaterializationPolicy(
                MODE_ONLY_RECONSTRUCT
            ),
        ),
        (
            "only_parent_rmw",
            MaterializationPolicy(
                MODE_ONLY_PARENT_RMW
            ),
        ),
    ]

    for threshold in (
        0,
        1,
        4,
        8,
        16,
        32,
        64,
    ):
        result.append(
            (
                f"hybrid_{threshold}",
                MaterializationPolicy(
                    MODE_HYBRID,
                    threshold=threshold,
                ),
            )
        )

    return result


def dirty_cases():
    # 203 rows -> four checkpoint blocks:
    #
    # block 0: slots   0..63   (64 rows)
    # block 1: slots  64..127  (64 rows)
    # block 2: slots 128..191  (64 rows)
    # block 3: slots 192..202  (11 rows)

    return {
        # Four blocks, exactly one dirty row each.
        "sparse": [
            1,
            65,
            129,
            192,
        ],

        # Dirty counts:
        #
        # block 0 -> 2
        # block 1 -> 5
        # block 2 -> 17
        # block 3 -> 3
        "mixed": (
            [0, 1]
            + list(
                range(
                    64,
                    69,
                )
            )
            + list(
                range(
                    128,
                    145,
                )
            )
            + [
                192,
                194,
                202,
            ]
        ),

        # Dirty counts:
        #
        # block 0 -> 64
        # block 1 -> 64
        # block 2 -> 64
        # block 3 -> 11
        "dense": list(
            range(
                TOTAL_ROWS
            )
        ),

        # Exercise only the partial final block.
        "partial_final": [
            192,
            194,
            202,
        ],
    }


class TestMaterializationCorrectnessMatrix(
    unittest.TestCase
):
    def test_all_layout_policy_density_combinations(
        self,
    ):
        combinations = 0

        for layout_mode in (
            LAYOUT_LOCAL_ONLY,
            LAYOUT_GLOBAL_MIXED,
        ):
            layout = make_layout(
                layout_mode
            )

            self.assertEqual(
                layout.total_rows,
                TOTAL_ROWS,
            )

            self.assertEqual(
                layout.num_blocks,
                4,
            )

            for case_name, slots in (
                dirty_cases().items()
            ):
                dirty_rows = (
                    dirty_rows_from_storage_slots(
                        layout,
                        slots,
                    )
                )

                parent_tables = (
                    make_tables()
                )

                current_tables = [
                    table.clone()
                    for table in parent_tables
                ]

                apply_dirty_updates(
                    layout,
                    current_tables,
                    slots,
                )

                plans = plan_dirty_blocks(
                    layout,
                    dirty_rows,
                )

                parent_image = (
                    build_checkpoint_image(
                        layout,
                        parent_tables,
                    )
                )

                # Full current-state reconstruction is the
                # correctness oracle for every policy.
                oracle_blocks, _ = (
                    reconstruct_blocks(
                        layout,
                        plans,
                        current_tables,
                    )
                )

                oracle_by_id = {
                    block.block_id:
                    block.data
                    for block
                    in oracle_blocks
                }

                with ParentFile(
                    parent_image
                ) as parent:
                    for (
                        policy_name,
                        policy,
                    ) in policies():
                        with self.subTest(
                            layout=layout_mode,
                            case=case_name,
                            policy=policy_name,
                        ):
                            (
                                blocks,
                                stats,
                            ) = (
                                materialize_dirty_blocks(
                                    layout,
                                    plans,
                                    current_tables,
                                    policy,
                                    parent_fd=parent.fd,
                                )
                            )

                            combinations += 1

                            self.assertEqual(
                                [
                                    block.block_id
                                    for block
                                    in blocks
                                ],
                                [
                                    plan.block_id
                                    for plan
                                    in plans
                                ],
                            )

                            # Every selected strategy must
                            # produce exactly the same bytes
                            # as current-state reconstruction.
                            for block in blocks:
                                self.assertEqual(
                                    block.data,
                                    oracle_by_id[
                                        block.block_id
                                    ],
                                )

                            expected_reconstruct = [
                                plan
                                for plan in plans
                                if policy.select(
                                    plan
                                )
                                == METHOD_RECONSTRUCT
                            ]

                            expected_rmw = [
                                plan
                                for plan in plans
                                if policy.select(
                                    plan
                                )
                                == METHOD_PARENT_RMW
                            ]

                            self.assertEqual(
                                stats.num_blocks,
                                len(plans),
                            )

                            self.assertEqual(
                                stats.reconstruct_blocks,
                                len(
                                    expected_reconstruct
                                ),
                            )

                            self.assertEqual(
                                stats.parent_rmw_blocks,
                                len(
                                    expected_rmw
                                ),
                            )

                            expected_reconstruct_rows = sum(
                                len(
                                    layout.row_refs_for_block(
                                        plan.block_id
                                    )
                                )
                                for plan
                                in expected_reconstruct
                            )

                            expected_rmw_rows = sum(
                                plan.dirty_count
                                for plan
                                in expected_rmw
                            )

                            self.assertEqual(
                                stats.reconstruct_gathered_rows,
                                expected_reconstruct_rows,
                            )

                            self.assertEqual(
                                stats.parent_rmw_gathered_rows,
                                expected_rmw_rows,
                            )

                            self.assertEqual(
                                stats.reconstruct_host_bytes,
                                expected_reconstruct_rows
                                * ROW_BYTES,
                            )

                            self.assertEqual(
                                stats.parent_rmw_host_bytes,
                                expected_rmw_rows
                                * ROW_BYTES,
                            )

                            self.assertEqual(
                                stats.parent_read_bytes,
                                len(
                                    expected_rmw
                                )
                                * BLOCK_SIZE,
                            )

                            self.assertEqual(
                                stats.output_bytes,
                                len(plans)
                                * BLOCK_SIZE,
                            )

        # 2 layouts
        # x 4 dirty distributions
        # x 9 policy configurations
        self.assertEqual(
            combinations,
            72,
        )


if __name__ == "__main__":
    unittest.main()
