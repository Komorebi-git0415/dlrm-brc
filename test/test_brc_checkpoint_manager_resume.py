import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brc_checkpoint_blocks import (
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
)
from brc_checkpoint_manager import BRCCkptManager
from brc_materialization_policy import (
    MODE_ONLY_RECONSTRUCT,
    MaterializationPolicy,
)


def make_table(num_rows):
    table = torch.empty(
        (num_rows, 16),
        dtype=torch.float32,
    )

    for row in range(num_rows):
        table[row].fill_(
            float(row)
        )

    return table


class TestCheckpointManagerResume(unittest.TestCase):
    def setUp(self):
        self.resume_patcher = mock.patch(
            "brc_checkpoint_manager.brc_uapi.lineage_resume",
            return_value=701,
        )

        self.create_patcher = mock.patch(
            "brc_checkpoint_manager.brc_uapi.create"
        )

        self.seal_patcher = mock.patch(
            "brc_checkpoint_manager.brc_uapi.seal"
        )

        self.close_patcher = mock.patch(
            "brc_checkpoint_manager.brc_uapi.close_session"
        )

        self.lineage_resume = (
            self.resume_patcher.start()
        )
        self.create = (
            self.create_patcher.start()
        )
        self.seal = (
            self.seal_patcher.start()
        )
        self.close_session = (
            self.close_patcher.start()
        )

        self.addCleanup(
            self.resume_patcher.stop
        )
        self.addCleanup(
            self.create_patcher.stop
        )
        self.addCleanup(
            self.seal_patcher.stop
        )
        self.addCleanup(
            self.close_patcher.stop
        )

    def make_manager(self, directory):
        layout = CheckpointLayout(
            table_sizes=[64],
            mode=LAYOUT_LOCAL_ONLY,
        )

        policy = MaterializationPolicy(
            MODE_ONLY_RECONSTRUCT
        )

        manager = BRCCkptManager(
            checkpoint_dir=directory,
            layout=layout,
            policy=policy,
        )

        return manager, layout

    def prepare_files(
        self,
        directory,
        image_size,
    ):
        ledger = (
            Path(directory)
            / "lineage.ledger"
        )

        ledger.write_bytes(
            bytes(64)
        )

        parent = (
            Path(directory)
            / "C1.img"
        )

        parent.write_bytes(
            bytes(image_size)
        )

        return parent

    def test_resume_sets_parent_and_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, layout = self.make_manager(
                tmpdir
            )

            parent = self.prepare_files(
                tmpdir,
                layout.image_size,
            )

            try:
                manager.resume(
                    generation=1,
                    parent_path=parent,
                )

                self.assertEqual(
                    manager.generation,
                    1,
                )

                self.assertEqual(
                    manager.parent_path,
                    str(parent),
                )

                self.lineage_resume.assert_called_once()

            finally:
                manager.close()

    def test_resume_allows_next_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, layout = self.make_manager(
                tmpdir
            )

            parent = self.prepare_files(
                tmpdir,
                layout.image_size,
            )

            table = make_table(
                64
            )

            table[2].fill_(
                9002.0
            )

            try:
                manager.resume(
                    generation=1,
                    parent_path="C1.img",
                )

                resumed_parent_fd = (
                    manager._parent_fd
                )

                result = manager.create_child(
                    [
                        np.asarray(
                            [2],
                            dtype=np.int64,
                        )
                    ],
                    [table],
                )

                self.assertEqual(
                    result.generation,
                    2,
                )

                self.assertTrue(
                    result.path.endswith(
                        "C2.img"
                    )
                )

                self.create.assert_called_once()

                self.assertEqual(
                    self.create.call_args.kwargs[
                        "parent_fd"
                    ],
                    resumed_parent_fd,
                )

            finally:
                manager.close()

    def test_wrong_parent_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, layout = self.make_manager(
                tmpdir
            )

            ledger = (
                Path(tmpdir)
                / "lineage.ledger"
            )

            ledger.write_bytes(
                bytes(64)
            )

            parent = (
                Path(tmpdir)
                / "C1.img"
            )

            parent.write_bytes(
                bytes(
                    layout.image_size - 1
                )
            )

            with self.assertRaises(
                ValueError
            ):
                manager.resume(
                    generation=1,
                    parent_path=parent,
                )

            self.lineage_resume.assert_not_called()

    def test_generation_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, layout = self.make_manager(
                tmpdir
            )

            parent = self.prepare_files(
                tmpdir,
                layout.image_size,
            )

            for generation in (
                -1,
                True,
                1.5,
                "1",
            ):
                with self.subTest(
                    generation=generation
                ):
                    with self.assertRaises(
                        ValueError
                    ):
                        manager.resume(
                            generation=generation,
                            parent_path=parent,
                        )


if __name__ == "__main__":
    unittest.main()
