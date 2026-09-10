import json
import tempfile
import unittest
from pathlib import Path

import torch

from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
)
from brc_checkpoint_manager import (
    CheckpointResult,
)
from brc_checkpoint_publication import (
    MANIFEST_FILENAME,
    CheckpointPublisher,
)


def make_checkpoint(
    directory,
    generation,
    image_size,
    filename=None,
):
    if filename is None:
        filename = (
            f"C{generation}.img"
        )

    path = (
        Path(directory)
        / filename
    )

    path.write_bytes(
        bytes(image_size)
    )

    return CheckpointResult(
        generation=generation,
        path=str(path),
        dirty_blocks=0,
        dirty_rows=0,
        materialization=None,
    )


def load_manifest(directory):
    return json.loads(
        (
            Path(directory)
            / MANIFEST_FILENAME
        ).read_text(
            encoding="utf-8"
        )
    )


class TestCheckpointPublication(
    unittest.TestCase
):
    def setUp(self):
        self.tmp = (
            tempfile.TemporaryDirectory()
        )

        self.addCleanup(
            self.tmp.cleanup
        )

        self.directory = Path(
            self.tmp.name
        )

        self.layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        self.publisher = (
            CheckpointPublisher(
                self.directory,
                self.layout,
            )
        )

    def test_publish_c0_creates_sidecar_and_manifest(
        self,
    ):
        checkpoint = make_checkpoint(
            self.directory,
            generation=0,
            image_size=self.layout.image_size,
        )

        state = {
            "step": 17,
            "dense_weight": torch.arange(
                4,
                dtype=torch.float32,
            ),
        }

        published = self.publisher.publish(
            checkpoint,
            state,
        )

        sidecar_path = Path(
            published.sidecar_path
        )

        self.assertTrue(
            sidecar_path.is_file()
        )

        sidecar = torch.load(
            sidecar_path,
            map_location="cpu",
        )

        self.assertEqual(
            sidecar["generation"],
            0,
        )
        self.assertEqual(
            sidecar["state"]["step"],
            17,
        )

        torch.testing.assert_close(
            sidecar["state"]["dense_weight"],
            state["dense_weight"],
        )

        manifest = load_manifest(
            self.directory
        )

        self.assertEqual(
            manifest["generation"],
            0,
        )
        self.assertEqual(
            manifest["checkpoint_file"],
            "C0.img",
        )
        self.assertEqual(
            manifest["sidecar_file"],
            "C0.state.pt",
        )
        self.assertEqual(
            manifest["checkpoint_size"],
            BLOCK_SIZE,
        )

    def test_c1_replaces_manifest_but_keeps_c0_sidecar(
        self,
    ):
        c0 = make_checkpoint(
            self.directory,
            0,
            self.layout.image_size,
        )

        self.publisher.publish(
            c0,
            {"step": 10},
        )

        c1 = make_checkpoint(
            self.directory,
            1,
            self.layout.image_size,
        )

        self.publisher.publish(
            c1,
            {"step": 20},
        )

        manifest = load_manifest(
            self.directory
        )

        self.assertEqual(
            manifest["generation"],
            1,
        )
        self.assertEqual(
            manifest["checkpoint_file"],
            "C1.img",
        )
        self.assertEqual(
            manifest["sidecar_file"],
            "C1.state.pt",
        )

        self.assertTrue(
            (
                self.directory
                / "C0.state.pt"
            ).is_file()
        )
        self.assertTrue(
            (
                self.directory
                / "C1.state.pt"
            ).is_file()
        )

    def test_first_publication_must_be_generation_zero(
        self,
    ):
        c1 = make_checkpoint(
            self.directory,
            1,
            self.layout.image_size,
        )

        with self.assertRaises(
            ValueError
        ):
            self.publisher.publish(
                c1,
                {"step": 20},
            )

        self.assertFalse(
            (
                self.directory
                / "C1.state.pt"
            ).exists()
        )

    def test_publication_must_be_consecutive(
        self,
    ):
        c0 = make_checkpoint(
            self.directory,
            0,
            self.layout.image_size,
        )

        self.publisher.publish(
            c0,
            {"step": 10},
        )

        c2 = make_checkpoint(
            self.directory,
            2,
            self.layout.image_size,
        )

        with self.assertRaises(
            ValueError
        ):
            self.publisher.publish(
                c2,
                {"step": 30},
            )

        manifest = load_manifest(
            self.directory
        )

        self.assertEqual(
            manifest["generation"],
            0,
        )

    def test_existing_sidecar_is_not_overwritten(
        self,
    ):
        checkpoint = make_checkpoint(
            self.directory,
            0,
            self.layout.image_size,
        )

        self.publisher.publish(
            checkpoint,
            {"step": 1},
        )

        original = (
            self.directory
            / "C0.state.pt"
        ).read_bytes()

        # Remove manifest only to exercise the sidecar
        # no-overwrite rule directly.
        (
            self.directory
            / MANIFEST_FILENAME
        ).unlink()

        with self.assertRaises(
            FileExistsError
        ):
            self.publisher.publish(
                checkpoint,
                {"step": 999},
            )

        self.assertEqual(
            (
                self.directory
                / "C0.state.pt"
            ).read_bytes(),
            original,
        )

    def test_wrong_checkpoint_size_is_rejected(
        self,
    ):
        checkpoint = make_checkpoint(
            self.directory,
            0,
            BLOCK_SIZE - 1,
        )

        with self.assertRaises(
            ValueError
        ):
            self.publisher.publish(
                checkpoint,
                {"step": 1},
            )

        self.assertFalse(
            (
                self.directory
                / "C0.state.pt"
            ).exists()
        )

    def test_checkpoint_name_must_match_generation(
        self,
    ):
        checkpoint = make_checkpoint(
            self.directory,
            0,
            self.layout.image_size,
            filename="wrong.img",
        )

        with self.assertRaises(
            ValueError
        ):
            self.publisher.publish(
                checkpoint,
                {"step": 1},
            )

    def test_training_state_must_be_dict(
        self,
    ):
        checkpoint = make_checkpoint(
            self.directory,
            0,
            self.layout.image_size,
        )

        with self.assertRaises(
            TypeError
        ):
            self.publisher.publish(
                checkpoint,
                None,
            )


if __name__ == "__main__":
    unittest.main()
