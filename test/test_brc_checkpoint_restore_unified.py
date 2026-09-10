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
from brc_checkpoint_manager import (
    CheckpointResult,
)
from brc_checkpoint_publication import (
    CheckpointPublisher,
)
from brc_checkpoint_restore import (
    restore_published_checkpoint,
)
from brc_global_layout import (
    build_global_storage_layout,
)


def make_table(
    num_rows,
    table_id,
    generation=0,
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

    for row_id in range(num_rows):
        weight[row_id].copy_(
            dims
            + float(
                generation * 1_000_000
                + table_id * 100_000
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


def publish_generation(
    directory,
    layout,
    generation,
    tables,
    step,
):
    checkpoint_path = (
        Path(directory)
        / f"C{generation}.img"
    )

    checkpoint_path.write_bytes(
        build_image(
            layout,
            tables,
        )
    )

    checkpoint = CheckpointResult(
        generation=generation,
        path=str(
            checkpoint_path
        ),
        dirty_blocks=layout.num_blocks,
        dirty_rows=layout.total_rows,
        materialization=None,
    )

    publisher = CheckpointPublisher(
        directory,
        layout,
    )

    return publisher.publish(
        checkpoint,
        {
            "step": step,
            "marker": torch.tensor(
                [generation, step],
                dtype=torch.int64,
            ),
        },
    )


class TestUnifiedLocalRestore(
    unittest.TestCase
):
    def test_restores_embedding_and_training_state(
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
                generation=0,
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
            publish_generation(
                tmp,
                layout,
                generation=0,
                tables=source,
                step=123,
            )

            restored = (
                restore_published_checkpoint(
                    tmp,
                    layout,
                    destination,
                    chunk_rows=16,
                )
            )

        self.assertEqual(
            restored.generation,
            0,
        )

        self.assertEqual(
            restored.training_state[
                "step"
            ],
            123,
        )

        self.assertEqual(
            restored.embedding.restored_rows,
            sum(
                table_sizes
            ),
        )

        self.assertLessEqual(
            restored.embedding.max_chunk_rows,
            16,
        )

        for actual, expected in zip(
            destination,
            source,
        ):
            torch.testing.assert_close(
                actual,
                expected,
            )


class TestUnifiedGlobalRestore(
    unittest.TestCase
):
    def test_restores_global_mixed_checkpoint(
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
                generation=0,
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
            publish_generation(
                tmp,
                layout,
                generation=0,
                tables=source,
                step=77,
            )

            restored = (
                restore_published_checkpoint(
                    tmp,
                    layout,
                    destination,
                    chunk_rows=7,
                )
            )

        self.assertEqual(
            restored.generation,
            0,
        )

        self.assertEqual(
            restored.training_state[
                "step"
            ],
            77,
        )

        for actual, expected in zip(
            destination,
            source,
        ):
            torch.testing.assert_close(
                actual,
                expected,
            )


class TestPublishedGenerationAuthority(
    unittest.TestCase
):
    def test_manifest_selects_latest_published_generation(
        self,
    ):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        table0 = [
            make_table(
                64,
                0,
                generation=0,
            )
        ]

        table1 = [
            make_table(
                64,
                0,
                generation=1,
            )
        ]

        destination = [
            torch.zeros_like(
                table0[0]
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            publish_generation(
                tmp,
                layout,
                generation=0,
                tables=table0,
                step=100,
            )

            publish_generation(
                tmp,
                layout,
                generation=1,
                tables=table1,
                step=200,
            )

            # An even newer C2.img exists, but it has NOT been
            # application-published and therefore must be ignored.
            (
                Path(tmp)
                / "C2.img"
            ).write_bytes(
                build_image(
                    layout,
                    [
                        make_table(
                            64,
                            0,
                            generation=2,
                        )
                    ],
                )
            )

            restored = (
                restore_published_checkpoint(
                    tmp,
                    layout,
                    destination,
                    chunk_rows=16,
                )
            )

        self.assertEqual(
            restored.generation,
            1,
        )

        self.assertEqual(
            restored.training_state[
                "step"
            ],
            200,
        )

        torch.testing.assert_close(
            destination[0],
            table1[0],
        )


class TestUnifiedRestoreValidation(
    unittest.TestCase
):
    def test_bad_metadata_does_not_modify_embedding(
        self,
    ):
        writer_layout = (
            CheckpointLayout(
                table_sizes=[
                    64,
                    64,
                ],
                mode=LAYOUT_LOCAL_ONLY,
            )
        )

        source = [
            make_table(64, 0),
            make_table(64, 1),
        ]

        wrong_layout = (
            CheckpointLayout(
                table_sizes=[
                    70,
                    58,
                ],
                mode=LAYOUT_LOCAL_ONLY,
            )
        )

        destination = [
            torch.full(
                (
                    70,
                    EMBEDDING_DIM,
                ),
                -1.0,
                dtype=torch.float32,
            ),
            torch.full(
                (
                    58,
                    EMBEDDING_DIM,
                ),
                -2.0,
                dtype=torch.float32,
            ),
        ]

        before = [
            table.clone()
            for table
            in destination
        ]

        with tempfile.TemporaryDirectory() as tmp:
            publish_generation(
                tmp,
                writer_layout,
                generation=0,
                tables=source,
                step=1,
            )

            with self.assertRaises(
                ValueError
            ):
                restore_published_checkpoint(
                    tmp,
                    wrong_layout,
                    destination,
                    chunk_rows=8,
                )

        for actual, expected in zip(
            destination,
            before,
        ):
            torch.testing.assert_close(
                actual,
                expected,
            )


if __name__ == "__main__":
    unittest.main()
