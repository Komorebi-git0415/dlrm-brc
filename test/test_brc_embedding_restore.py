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
    rows,
    table_id,
):
    weight = torch.empty(
        (
            rows,
            EMBEDDING_DIM,
        ),
        dtype=torch.float32,
    )

    dims = torch.arange(
        EMBEDDING_DIM,
        dtype=torch.float32,
    )

    for row_id in range(rows):
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

        row = (
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

        if len(row) != ROW_BYTES:
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
        ] = row

    return bytes(image)


def make_global_layout(
    table_sizes,
):
    new_to_old = [
        np.arange(
            size,
            dtype=np.int64,
        )
        for size in table_sizes
    ]

    # Deliberately interleave tables in global storage order.
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


class TestLocalOnlyEmbeddingRestore(
    unittest.TestCase
):
    def test_restores_multiple_tables(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[70, 17],
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(70, 0),
            make_table(17, 1),
        ]

        destination = [
            torch.zeros_like(
                source[0]
            ),
            torch.zeros_like(
                source[1]
            ),
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

            stats = (
                restore_embedding_image(
                    path,
                    layout,
                    destination,
                )
            )

        torch.testing.assert_close(
            destination[0],
            source[0],
        )

        torch.testing.assert_close(
            destination[1],
            source[1],
        )

        self.assertEqual(
            stats.num_tables,
            2,
        )

        self.assertEqual(
            stats.restored_rows,
            87,
        )

        self.assertEqual(
            stats.image_bytes,
            layout.image_size,
        )

        self.assertEqual(
            stats.restored_payload_bytes,
            87 * ROW_BYTES,
        )

    def test_final_block_padding_is_ignored(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[70],
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(70, 0)
        ]

        destination = [
            torch.zeros_like(
                source[0]
            )
        ]

        image = bytearray(
            build_image(
                layout,
                source,
            )
        )

        # Make final alignment padding deliberately non-zero.
        real_end = (
            layout.total_rows
            * ROW_BYTES
        )

        image[
            real_end:
        ] = bytes(
            [0xA5]
        ) * (
            len(image)
            - real_end
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "C0.img"
            )

            path.write_bytes(
                image
            )

            restore_embedding_image(
                path,
                layout,
                destination,
            )

        torch.testing.assert_close(
            destination[0],
            source[0],
        )


class TestGlobalMixedEmbeddingRestore(
    unittest.TestCase
):
    def test_global_storage_order_is_inverted(
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

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "C3.img"
            )

            path.write_bytes(
                build_image(
                    layout,
                    source,
                )
            )

            restore_embedding_image(
                path,
                layout,
                destination,
            )

        for actual, expected in zip(
            destination,
            source,
        ):
            torch.testing.assert_close(
                actual,
                expected,
            )


class TestEmbeddingBagCompatibility(
    unittest.TestCase
):
    def test_standard_embedding_bag_is_supported(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(64, 0)
        ]

        destination = torch.nn.EmbeddingBag(
            64,
            EMBEDDING_DIM,
            mode="sum",
            sparse=True,
        )

        with torch.no_grad():
            destination.weight.zero_()

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

            restore_embedding_image(
                path,
                layout,
                [destination],
            )

        torch.testing.assert_close(
            destination.weight,
            source[0],
        )


class TestEmbeddingRestoreValidation(
    unittest.TestCase
):
    def test_wrong_image_size_is_rejected(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "C0.img"
            )

            path.write_bytes(
                bytes(
                    layout.image_size
                    - 1
                )
            )

            with self.assertRaises(
                ValueError
            ):
                restore_embedding_image(
                    path,
                    layout,
                    [
                        torch.zeros(
                            (
                                64,
                                EMBEDDING_DIM,
                            ),
                            dtype=torch.float32,
                        )
                    ],
                )

    def test_table_count_is_checked(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(64, 0)
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

            with self.assertRaises(
                ValueError
            ):
                restore_embedding_image(
                    path,
                    layout,
                    [],
                )

    def test_wrong_embedding_dimension_is_rejected(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(64, 0)
        ]

        wrong = torch.zeros(
            (64, 8),
            dtype=torch.float32,
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
                    [wrong],
                )

    def test_wrong_dtype_is_rejected(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        source = [
            make_table(64, 0)
        ]

        wrong = torch.zeros(
            (
                64,
                EMBEDDING_DIM,
            ),
            dtype=torch.float64,
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
                    [wrong],
                )


if __name__ == "__main__":
    unittest.main()
