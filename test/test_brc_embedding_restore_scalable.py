import math
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
from brc_checkpoint_restore import (
    restore_embedding_image,
)
from brc_global_layout import (
    build_global_storage_layout,
)


def make_table(
    num_rows,
    table_id,
):
    weight = torch.empty(
        (
            num_rows,
            EMBEDDING_DIM,
        ),
        dtype=torch.float32,
    )

    dims = torch.arange(
        EMBEDDING_DIM,
        dtype=torch.float32,
    )

    for row_id in range(
        num_rows
    ):
        weight[row_id].copy_(
            dims
            + float(
                table_id * 100_000
                + row_id * 100
            )
        )

    return weight


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

        start = (
            storage_slot
            * ROW_BYTES
        )

        image[
            start:
            start + ROW_BYTES
        ] = payload

    return bytes(image)


def make_global_layout(
    table_sizes,
):
    new_to_old = [
        np.arange(
            size,
            dtype=np.int64,
        )
        for size
        in table_sizes
    ]

    frequencies = []

    for table_id, size in enumerate(
        table_sizes
    ):
        frequencies.append(
            (
                1_000_000
                - np.arange(
                    size,
                    dtype=np.int64,
                )
                * len(table_sizes)
                - table_id
            )
        )

    global_layout = (
        build_global_storage_layout(
            frequencies,
            new_to_old,
        )
    )

    return CheckpointLayout(
        table_sizes=table_sizes,
        mode=LAYOUT_GLOBAL_MIXED,
        global_layout=global_layout,
    )


class TestChunkedLocalRestore(
    unittest.TestCase
):
    def test_restore_uses_bounded_chunks(
        self,
    ):
        table_sizes = [
            70,
            17,
        ]

        layout = CheckpointLayout(
            table_sizes=table_sizes,
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(
                size,
                table_id,
            )
            for table_id, size
            in enumerate(
                table_sizes
            )
        ]

        destination = [
            torch.zeros_like(
                table
            )
            for table in source
        ]

        chunk_rows = 16

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "C0.img"
            )

            path.write_bytes(
                build_image(
                    layout,
                    source,
                )
            )

            stats = (
                restore_embedding_image(
                    path,
                    layout,
                    destination,
                    chunk_rows=chunk_rows,
                )
            )

        for actual, expected in zip(
            destination,
            source,
        ):
            torch.testing.assert_close(
                actual,
                expected,
            )

        expected_chunks = sum(
            math.ceil(
                size
                / chunk_rows
            )
            for size
            in table_sizes
        )

        self.assertEqual(
            stats.num_chunks,
            expected_chunks,
        )

        self.assertLessEqual(
            stats.max_chunk_rows,
            chunk_rows,
        )

        self.assertLessEqual(
            stats.max_host_bytes,
            chunk_rows
            * ROW_BYTES,
        )

        self.assertEqual(
            stats.restored_rows,
            sum(
                table_sizes
            ),
        )


class TestChunkedGlobalRestore(
    unittest.TestCase
):
    def test_global_mixed_uses_vectorized_chunk_mapping(
        self,
    ):
        table_sizes = [
            20,
            19,
            18,
        ]

        layout = make_global_layout(
            table_sizes
        )

        source = [
            make_table(
                size,
                table_id,
            )
            for table_id, size
            in enumerate(
                table_sizes
            )
        ]

        destination = [
            torch.zeros_like(
                table
            )
            for table in source
        ]

        chunk_rows = 7

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "C4.img"
            )

            path.write_bytes(
                build_image(
                    layout,
                    source,
                )
            )

            stats = (
                restore_embedding_image(
                    path,
                    layout,
                    destination,
                    chunk_rows=chunk_rows,
                )
            )

        for actual, expected in zip(
            destination,
            source,
        ):
            torch.testing.assert_close(
                actual,
                expected,
            )

        expected_chunks = sum(
            math.ceil(
                size
                / chunk_rows
            )
            for size
            in table_sizes
        )

        self.assertEqual(
            stats.num_chunks,
            expected_chunks,
        )

        self.assertLessEqual(
            stats.max_host_bytes,
            chunk_rows
            * ROW_BYTES,
        )


class TestChunkedRestoreValidation(
    unittest.TestCase
):
    def test_invalid_chunk_rows_is_rejected(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(
                64,
                0,
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "C0.img"
            )

            path.write_bytes(
                build_image(
                    layout,
                    source,
                )
            )

            for chunk_rows in (
                0,
                -1,
                True,
                1.5,
            ):
                with self.subTest(
                    chunk_rows=chunk_rows
                ):
                    with self.assertRaises(
                        (TypeError, ValueError)
                    ):
                        restore_embedding_image(
                            path,
                            layout,
                            [
                                torch.zeros_like(
                                    source[0]
                                )
                            ],
                            chunk_rows=chunk_rows,
                        )

    def test_all_destinations_are_validated_before_write(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[
                64,
                64,
            ],
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(64, 0),
            make_table(64, 1),
        ]

        destination0 = (
            torch.zeros_like(
                source[0]
            )
        )

        # Invalid second table.
        destination1 = torch.zeros(
            (
                64,
                EMBEDDING_DIM,
            ),
            dtype=torch.float64,
        )

        before = (
            destination0.clone()
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "C0.img"
            )

            path.write_bytes(
                build_image(
                    layout,
                    source,
                )
            )

            with self.assertRaises(
                ValueError
            ):
                restore_embedding_image(
                    path,
                    layout,
                    [
                        destination0,
                        destination1,
                    ],
                    chunk_rows=8,
                )

        # Table 0 must still be untouched because destination
        # validation happens before the first restore copy.
        torch.testing.assert_close(
            destination0,
            before,
        )


if __name__ == "__main__":
    unittest.main()
