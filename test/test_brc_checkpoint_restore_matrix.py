import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from brc_checkpoint_blocks import (
    EMBEDDING_DIM,
    ROW_BYTES,
    LAYOUT_GLOBAL_MIXED,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
)
from brc_checkpoint_manager import CheckpointResult
from brc_checkpoint_publication import CheckpointPublisher
from brc_checkpoint_restore import restore_published_checkpoint
from brc_global_layout import build_global_storage_layout


TABLE_SIZES = [70, 67, 18]


def make_tables(generation):
    tables = []

    dims = torch.arange(
        EMBEDDING_DIM,
        dtype=torch.float32,
    )

    for table_id, size in enumerate(TABLE_SIZES):
        weight = torch.empty(
            (size, EMBEDDING_DIM),
            dtype=torch.float32,
        )

        for row_id in range(size):
            weight[row_id].copy_(
                dims
                + float(
                    generation * 1_000_000
                    + table_id * 100_000
                    + row_id * 100
                )
            )

        tables.append(weight)

    return tables


def make_layout(mode):
    if mode == LAYOUT_LOCAL_ONLY:
        return CheckpointLayout(
            table_sizes=TABLE_SIZES,
            mode=LAYOUT_LOCAL_ONLY,
        )

    if mode != LAYOUT_GLOBAL_MIXED:
        raise ValueError(
            f"unexpected layout: {mode}"
        )

    new_to_old = [
        np.arange(
            size,
            dtype=np.int64,
        )
        for size in TABLE_SIZES
    ]

    frequencies = []

    for table_id, size in enumerate(TABLE_SIZES):
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

    global_layout = build_global_storage_layout(
        frequencies,
        new_to_old,
    )

    return CheckpointLayout(
        table_sizes=TABLE_SIZES,
        mode=LAYOUT_GLOBAL_MIXED,
        global_layout=global_layout,
    )


def build_image(
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
                "unexpected row size"
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


def publish_generation(
    directory,
    layout,
    generation,
    tables,
):
    path = (
        Path(directory)
        / f"C{generation}.img"
    )

    path.write_bytes(
        build_image(
            layout,
            tables,
        )
    )

    checkpoint = CheckpointResult(
        generation=generation,
        path=str(path),
        dirty_blocks=layout.num_blocks,
        dirty_rows=layout.total_rows,
        materialization=None,
    )

    publisher = CheckpointPublisher(
        directory,
        layout,
    )

    publisher.publish(
        checkpoint,
        {
            "step": (
                1000
                + generation
            ),
            "epoch": generation,
            "marker": torch.tensor(
                [
                    generation,
                    123456,
                ],
                dtype=torch.int64,
            ),
        },
    )


class TestRestoreCorrectnessMatrix(
    unittest.TestCase
):
    def test_layout_generation_chunk_matrix(
        self,
    ):
        combinations = 0

        for mode in (
            LAYOUT_LOCAL_ONLY,
            LAYOUT_GLOBAL_MIXED,
        ):
            layout = make_layout(
                mode
            )

            # 155 real rows => final checkpoint block is partial.
            self.assertEqual(
                layout.total_rows,
                155,
            )

            for published_generation in (
                0,
                1,
            ):
                with tempfile.TemporaryDirectory() as tmp:
                    source0 = make_tables(
                        generation=0
                    )

                    publish_generation(
                        tmp,
                        layout,
                        generation=0,
                        tables=source0,
                    )

                    expected = source0

                    if published_generation == 1:
                        source1 = make_tables(
                            generation=1
                        )

                        publish_generation(
                            tmp,
                            layout,
                            generation=1,
                            tables=source1,
                        )

                        expected = source1

                    # Create a numerically newer image which is NOT
                    # application-published. Restore must ignore it.
                    unpublished = (
                        Path(tmp)
                        / f"C{published_generation + 1}.unpublished"
                    )

                    unpublished.write_bytes(
                        build_image(
                            layout,
                            make_tables(
                                generation=9
                            ),
                        )
                    )

                    for chunk_rows in (
                        1,
                        3,
                        7,
                        16,
                        64,
                        65536,
                    ):
                        with self.subTest(
                            layout=mode,
                            generation=published_generation,
                            chunk_rows=chunk_rows,
                        ):
                            destination = [
                                torch.full_like(
                                    table,
                                    -999.0,
                                )
                                for table
                                in expected
                            ]

                            restored = (
                                restore_published_checkpoint(
                                    tmp,
                                    layout,
                                    destination,
                                    chunk_rows=chunk_rows,
                                )
                            )

                            combinations += 1

                            self.assertEqual(
                                restored.generation,
                                published_generation,
                            )

                            self.assertEqual(
                                restored.training_state[
                                    "step"
                                ],
                                1000
                                + published_generation,
                            )

                            self.assertEqual(
                                restored.training_state[
                                    "epoch"
                                ],
                                published_generation,
                            )

                            torch.testing.assert_close(
                                restored.training_state[
                                    "marker"
                                ],
                                torch.tensor(
                                    [
                                        published_generation,
                                        123456,
                                    ],
                                    dtype=torch.int64,
                                ),
                            )

                            self.assertEqual(
                                restored.embedding.restored_rows,
                                layout.total_rows,
                            )

                            self.assertEqual(
                                restored.embedding.restored_payload_bytes,
                                layout.total_rows
                                * ROW_BYTES,
                            )

                            self.assertLessEqual(
                                restored.embedding.max_chunk_rows,
                                min(
                                    chunk_rows,
                                    max(
                                        TABLE_SIZES
                                    ),
                                ),
                            )

                            self.assertLessEqual(
                                restored.embedding.max_host_bytes,
                                chunk_rows
                                * ROW_BYTES,
                            )

                            for actual, wanted in zip(
                                destination,
                                expected,
                            ):
                                torch.testing.assert_close(
                                    actual,
                                    wanted,
                                )

        # 2 layouts
        # x 2 published generations
        # x 6 chunk sizes
        self.assertEqual(
            combinations,
            24,
        )


if __name__ == "__main__":
    unittest.main()
