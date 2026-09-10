import json
import tempfile
import unittest
from pathlib import Path

import torch

from brc_checkpoint_blocks import (
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
from brc_checkpoint_restore import (
    load_published_checkpoint,
)


class TestRestoreMetadata(unittest.TestCase):
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

    def publish_c0(self):
        checkpoint_path = (
            self.directory
            / "C0.img"
        )

        checkpoint_path.write_bytes(
            bytes(
                self.layout.image_size
            )
        )

        checkpoint = CheckpointResult(
            generation=0,
            path=str(checkpoint_path),
            dirty_blocks=1,
            dirty_rows=64,
            materialization=None,
        )

        publisher = CheckpointPublisher(
            self.directory,
            self.layout,
        )

        return publisher.publish(
            checkpoint,
            {
                "step": 123,
                "dense_weight": torch.arange(
                    4,
                    dtype=torch.float32,
                ),
            },
        )

    def load_manifest(self):
        path = (
            self.directory
            / MANIFEST_FILENAME
        )

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    def write_manifest(self, manifest):
        path = (
            self.directory
            / MANIFEST_FILENAME
        )

        path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_valid_publication_loads_training_state(
        self,
    ):
        self.publish_c0()

        restored = (
            load_published_checkpoint(
                self.directory,
                self.layout,
            )
        )

        self.assertEqual(
            restored.generation,
            0,
        )

        self.assertTrue(
            restored.checkpoint_path.endswith(
                "C0.img"
            )
        )

        self.assertTrue(
            restored.sidecar_path.endswith(
                "C0.state.pt"
            )
        )

        self.assertEqual(
            restored.training_state[
                "step"
            ],
            123,
        )

        torch.testing.assert_close(
            restored.training_state[
                "dense_weight"
            ],
            torch.arange(
                4,
                dtype=torch.float32,
            ),
        )

    def test_missing_manifest_is_rejected(
        self,
    ):
        with self.assertRaises(
            FileNotFoundError
        ):
            load_published_checkpoint(
                self.directory,
                self.layout,
            )

    def test_manifest_generation_filename_mismatch_is_rejected(
        self,
    ):
        self.publish_c0()

        manifest = self.load_manifest()
        manifest[
            "checkpoint_file"
        ] = "C1.img"

        self.write_manifest(
            manifest
        )

        with self.assertRaises(
            ValueError
        ):
            load_published_checkpoint(
                self.directory,
                self.layout,
            )

    def test_missing_sidecar_is_rejected(
        self,
    ):
        published = self.publish_c0()

        Path(
            published.sidecar_path
        ).unlink()

        with self.assertRaises(
            FileNotFoundError
        ):
            load_published_checkpoint(
                self.directory,
                self.layout,
            )

    def test_checkpoint_size_mismatch_is_rejected(
        self,
    ):
        published = self.publish_c0()

        Path(
            published.checkpoint_path
        ).write_bytes(
            bytes(
                self.layout.image_size
                - 1
            )
        )

        with self.assertRaises(
            ValueError
        ):
            load_published_checkpoint(
                self.directory,
                self.layout,
            )

    def test_layout_mismatch_is_rejected(
        self,
    ):
        self.publish_c0()

        different_layout = (
            CheckpointLayout(
                table_sizes=[128],
                mode=LAYOUT_LOCAL_ONLY,
            )
        )

        with self.assertRaises(
            ValueError
        ):
            load_published_checkpoint(
                self.directory,
                different_layout,
            )

    def test_sidecar_generation_mismatch_is_rejected(
        self,
    ):
        published = self.publish_c0()

        sidecar_path = Path(
            published.sidecar_path
        )

        sidecar = torch.load(
            sidecar_path,
            map_location="cpu",
        )

        sidecar[
            "generation"
        ] = 1

        torch.save(
            sidecar,
            sidecar_path,
        )

        # torch.save may change serialized length.
        manifest = self.load_manifest()
        manifest[
            "sidecar_size"
        ] = sidecar_path.stat().st_size

        self.write_manifest(
            manifest
        )

        with self.assertRaises(
            ValueError
        ):
            load_published_checkpoint(
                self.directory,
                self.layout,
            )


if __name__ == "__main__":
    unittest.main()
