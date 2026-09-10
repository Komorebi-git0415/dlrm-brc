import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import brc_checkpoint_publication
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


def make_checkpoint(
    directory,
    generation,
    image_size,
):
    path = (
        Path(directory)
        / f"C{generation}.img"
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


class TestPublicationFailureBoundary(
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

        # Establish one fully published generation C0.
        c0 = make_checkpoint(
            self.directory,
            generation=0,
            image_size=self.layout.image_size,
        )

        self.publisher.publish(
            c0,
            {
                "step": 100,
            },
        )

        self.assertEqual(
            load_manifest(
                self.directory
            )["generation"],
            0,
        )

    def test_sidecar_failure_does_not_advance_manifest(
        self,
    ):
        c1 = make_checkpoint(
            self.directory,
            generation=1,
            image_size=self.layout.image_size,
        )

        with mock.patch(
            "brc_checkpoint_publication."
            "_atomic_torch_save_no_overwrite",
            side_effect=OSError(
                "simulated sidecar failure"
            ),
        ):
            with self.assertRaises(
                OSError
            ):
                self.publisher.publish(
                    c1,
                    {
                        "step": 200,
                    },
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

        self.assertFalse(
            (
                self.directory
                / "C1.state.pt"
            ).exists()
        )

    def test_manifest_failure_does_not_publish_new_generation(
        self,
    ):
        c1 = make_checkpoint(
            self.directory,
            generation=1,
            image_size=self.layout.image_size,
        )

        with mock.patch(
            "brc_checkpoint_publication."
            "_atomic_write_json",
            side_effect=OSError(
                "simulated manifest failure"
            ),
        ):
            with self.assertRaises(
                OSError
            ):
                self.publisher.publish(
                    c1,
                    {
                        "step": 200,
                        "weight": torch.arange(
                            4,
                            dtype=torch.float32,
                        ),
                    },
                )

        # Sidecar must already have been completed before
        # manifest publication was attempted.
        sidecar_path = (
            self.directory
            / "C1.state.pt"
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
            1,
        )

        self.assertEqual(
            sidecar["state"]["step"],
            200,
        )

        # But C1 is NOT application-published because the
        # manifest publication point failed.
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

    def test_atomic_manifest_replace_failure_preserves_old_file(
        self,
    ):
        manifest_path = (
            self.directory
            / MANIFEST_FILENAME
        )

        old_bytes = (
            manifest_path.read_bytes()
        )

        new_manifest = {
            "format": "brc-checkpoint-manifest",
            "version": 1,
            "generation": 1,
        }

        real_replace = (
            brc_checkpoint_publication
            .os.replace
        )

        def fail_manifest_replace(
            source,
            destination,
        ):
            if Path(destination) == manifest_path:
                raise OSError(
                    "simulated os.replace failure"
                )

            return real_replace(
                source,
                destination,
            )

        with mock.patch(
            "brc_checkpoint_publication.os.replace",
            side_effect=fail_manifest_replace,
        ):
            with self.assertRaises(
                OSError
            ):
                (
                    brc_checkpoint_publication
                    ._atomic_write_json(
                        new_manifest,
                        manifest_path,
                    )
                )

        # Existing manifest must remain byte-for-byte intact.
        self.assertEqual(
            manifest_path.read_bytes(),
            old_bytes,
        )

        # Failed atomic write must clean its temporary file.
        leftovers = [
            path
            for path
            in self.directory.iterdir()
            if (
                path.name.startswith(
                    ".manifest.json.tmp-"
                )
            )
        ]

        self.assertEqual(
            leftovers,
            [],
        )


if __name__ == "__main__":
    unittest.main()
