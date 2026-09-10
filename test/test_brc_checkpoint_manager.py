import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
)
from brc_checkpoint_manager import BRCCkptManager
from brc_materialization_policy import (
    MODE_HYBRID,
    MODE_ONLY_RECONSTRUCT,
    MaterializationPolicy,
)


def make_table(num_rows, base):
    table = torch.empty(
        (num_rows, 16),
        dtype=torch.float32,
    )

    for row in range(num_rows):
        table[row].fill_(
            float(base + row)
        )

    return table


def read_file(path):
    with open(path, "rb") as stream:
        return stream.read()


class MockBRCManagerTest(unittest.TestCase):
    def setUp(self):
        self.lineage_begin_patcher = mock.patch(
            "brc_checkpoint_manager.brc_uapi.lineage_begin",
            return_value=700,
        )
        self.create_patcher = mock.patch(
            "brc_checkpoint_manager.brc_uapi.create"
        )
        self.seal_patcher = mock.patch(
            "brc_checkpoint_manager.brc_uapi.seal"
        )
        self.close_session_patcher = mock.patch(
            "brc_checkpoint_manager.brc_uapi.close_session"
        )

        self.lineage_begin = (
            self.lineage_begin_patcher.start()
        )
        self.create = self.create_patcher.start()
        self.seal = self.seal_patcher.start()
        self.close_session = (
            self.close_session_patcher.start()
        )

        self.addCleanup(
            self.lineage_begin_patcher.stop
        )
        self.addCleanup(
            self.create_patcher.stop
        )
        self.addCleanup(
            self.seal_patcher.stop
        )
        self.addCleanup(
            self.close_session_patcher.stop
        )

    def make_manager(
        self,
        directory,
        table_sizes,
        policy,
    ):
        layout = CheckpointLayout(
            table_sizes=table_sizes,
            mode=LAYOUT_LOCAL_ONLY,
        )

        manager = BRCCkptManager(
            checkpoint_dir=directory,
            layout=layout,
            policy=policy,
        )

        return manager, layout

    def test_root_checkpoint_is_complete_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MaterializationPolicy(
                MODE_ONLY_RECONSTRUCT
            )

            manager, layout = self.make_manager(
                tmpdir,
                [70],
                policy,
            )

            table = make_table(
                70,
                1000,
            )

            try:
                manager.start_new()

                result = manager.create_root(
                    [table]
                )

                self.assertEqual(
                    result.generation,
                    0,
                )
                self.assertEqual(
                    result.dirty_blocks,
                    2,
                )
                self.assertEqual(
                    result.dirty_rows,
                    70,
                )

                self.assertEqual(
                    manager.generation,
                    0,
                )

                self.assertEqual(
                    manager.parent_path,
                    str(
                        Path(tmpdir)
                        / "C0.img"
                    ),
                )

                payload = read_file(
                    result.path
                )

                self.assertEqual(
                    len(payload),
                    layout.image_size,
                )

                rows = np.frombuffer(
                    payload,
                    dtype=np.float32,
                ).reshape(
                    layout.num_blocks * 64,
                    16,
                )

                for row in range(70):
                    np.testing.assert_array_equal(
                        rows[row],
                        np.full(
                            16,
                            float(1000 + row),
                            dtype=np.float32,
                        ),
                    )

                # Final block padding.
                np.testing.assert_array_equal(
                    rows[70:],
                    np.zeros(
                        (58, 16),
                        dtype=np.float32,
                    ),
                )

                self.lineage_begin.assert_called_once()
                self.create.assert_not_called()

                self.assertEqual(
                    self.seal.call_count,
                    1,
                )

            finally:
                manager.close()

    def test_child_writes_only_dirty_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MaterializationPolicy(
                MODE_ONLY_RECONSTRUCT
            )

            manager, _layout = self.make_manager(
                tmpdir,
                [192],
                policy,
            )

            parent_table = make_table(
                192,
                1000,
            )

            current_table = (
                parent_table.clone()
            )

            current_table[
                70
            ].fill_(9000.0)

            try:
                manager.start_new()
                manager.create_root(
                    [parent_table]
                )

                result = manager.create_child(
                    [
                        np.array(
                            [70],
                            dtype=np.int64,
                        )
                    ],
                    [current_table],
                )

                self.assertEqual(
                    result.generation,
                    1,
                )
                self.assertEqual(
                    result.dirty_blocks,
                    1,
                )
                self.assertEqual(
                    result.dirty_rows,
                    1,
                )

                self.assertEqual(
                    result.materialization.reconstruct_blocks,
                    1,
                )
                self.assertEqual(
                    result.materialization.parent_rmw_blocks,
                    0,
                )

                payload = read_file(
                    result.path
                )

                self.assertEqual(
                    len(payload),
                    3 * BLOCK_SIZE,
                )

                # With the kernel mocked, clean holes remain zeros.
                # Therefore this directly proves that the application
                # wrote only dirty block 1.
                self.assertEqual(
                    payload[
                        0:
                        BLOCK_SIZE
                    ],
                    bytes(BLOCK_SIZE),
                )

                block1_expected = (
                    current_table[
                        64:128
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

                self.assertEqual(
                    payload[
                        BLOCK_SIZE:
                        2 * BLOCK_SIZE
                    ],
                    block1_expected,
                )

                self.assertEqual(
                    payload[
                        2 * BLOCK_SIZE:
                        3 * BLOCK_SIZE
                    ],
                    bytes(BLOCK_SIZE),
                )

                self.create.assert_called_once()

                self.assertEqual(
                    self.seal.call_count,
                    2,
                )

            finally:
                manager.close()

    def test_hybrid_policy_reaches_both_materializers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MaterializationPolicy(
                MODE_HYBRID,
                threshold=4,
            )

            manager, _layout = self.make_manager(
                tmpdir,
                [192],
                policy,
            )

            parent_table = make_table(
                192,
                1000,
            )

            current_table = (
                parent_table.clone()
            )

            # block 0: 1 dirty row  -> RMW
            # block 1: 5 dirty rows -> reconstruction
            # block 2: 10 dirty rows -> reconstruction
            dirty = (
                [1]
                + list(range(64, 69))
                + list(range(128, 138))
            )

            for row in dirty:
                current_table[
                    row
                ].fill_(
                    float(
                        9000 + row
                    )
                )

            try:
                manager.start_new()
                manager.create_root(
                    [parent_table]
                )

                result = manager.create_child(
                    [
                        np.array(
                            dirty,
                            dtype=np.int64,
                        )
                    ],
                    [current_table],
                )

                stats = result.materialization

                self.assertEqual(
                    stats.num_blocks,
                    3,
                )
                self.assertEqual(
                    stats.parent_rmw_blocks,
                    1,
                )
                self.assertEqual(
                    stats.reconstruct_blocks,
                    2,
                )

                self.assertEqual(
                    stats.parent_rmw_gathered_rows,
                    1,
                )

                self.assertEqual(
                    stats.reconstruct_gathered_rows,
                    128,
                )

                self.assertEqual(
                    stats.parent_read_bytes,
                    BLOCK_SIZE,
                )

                # All three blocks are dirty in this case, so the
                # mocked-kernel child file is a complete current image.
                expected = (
                    current_table
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

                self.assertEqual(
                    read_file(
                        result.path
                    ),
                    expected,
                )

            finally:
                manager.close()

    def test_generation_and_parent_roll_forward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MaterializationPolicy(
                MODE_ONLY_RECONSTRUCT
            )

            manager, _layout = self.make_manager(
                tmpdir,
                [64],
                policy,
            )

            table = make_table(
                64,
                100,
            )

            try:
                manager.start_new()

                root = manager.create_root(
                    [table]
                )

                self.assertEqual(
                    root.generation,
                    0,
                )

                table1 = table.clone()
                table1[1].fill_(9001.0)

                child1 = manager.create_child(
                    [
                        np.array(
                            [1],
                            dtype=np.int64,
                        )
                    ],
                    [table1],
                )

                self.assertEqual(
                    child1.generation,
                    1,
                )

                parent_fd_before_c2 = (
                    manager._parent_fd
                )

                table2 = table1.clone()
                table2[2].fill_(9002.0)

                child2 = manager.create_child(
                    [
                        np.array(
                            [2],
                            dtype=np.int64,
                        )
                    ],
                    [table2],
                )

                self.assertEqual(
                    child2.generation,
                    2,
                )

                self.assertEqual(
                    manager.generation,
                    2,
                )

                self.assertTrue(
                    manager.parent_path.endswith(
                        "C2.img"
                    )
                )

                self.assertEqual(
                    self.create.call_count,
                    2,
                )

                second_create = (
                    self.create
                    .call_args_list[1]
                )

                self.assertEqual(
                    second_create.kwargs[
                        "parent_fd"
                    ],
                    parent_fd_before_c2,
                )

            finally:
                manager.close()

    def test_child_requires_root_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MaterializationPolicy(
                MODE_ONLY_RECONSTRUCT
            )

            manager, _layout = self.make_manager(
                tmpdir,
                [64],
                policy,
            )

            table = make_table(
                64,
                0,
            )

            try:
                manager.start_new()

                with self.assertRaises(
                    RuntimeError
                ):
                    manager.create_child(
                        [
                            np.array(
                                [1],
                                dtype=np.int64,
                            )
                        ],
                        [table],
                    )

                self.create.assert_not_called()

            finally:
                manager.close()

    def test_failure_after_create_marks_manager_broken(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MaterializationPolicy(
                MODE_ONLY_RECONSTRUCT
            )

            manager, _layout = self.make_manager(
                tmpdir,
                [64],
                policy,
            )

            table = make_table(
                64,
                0,
            )

            # Root SEAL succeeds; child SEAL fails.
            self.seal.side_effect = [
                None,
                OSError(
                    "simulated child seal failure"
                ),
            ]

            try:
                manager.start_new()
                manager.create_root(
                    [table]
                )

                current = table.clone()
                current[1].fill_(1234.0)

                with self.assertRaises(
                    OSError
                ):
                    manager.create_child(
                        [
                            np.array(
                                [1],
                                dtype=np.int64,
                            )
                        ],
                        [current],
                    )

                # brc3 has no Abort. Once BRC_CREATE succeeded and
                # publication failed, this manager must not continue.
                with self.assertRaises(
                    RuntimeError
                ):
                    manager.create_child(
                        [
                            np.array(
                                [2],
                                dtype=np.int64,
                            )
                        ],
                        [current],
                    )

                self.assertEqual(
                    self.create.call_count,
                    1,
                )

            finally:
                manager.close()

    def test_empty_dirty_generation_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MaterializationPolicy(
                MODE_ONLY_RECONSTRUCT
            )

            manager, layout = self.make_manager(
                tmpdir,
                [64],
                policy,
            )

            table = make_table(
                64,
                0,
            )

            try:
                manager.start_new()
                manager.create_root(
                    [table]
                )

                result = manager.create_child(
                    [
                        np.array(
                            [],
                            dtype=np.int64,
                        )
                    ],
                    [table],
                )

                self.assertEqual(
                    result.generation,
                    1,
                )
                self.assertEqual(
                    result.dirty_blocks,
                    0,
                )
                self.assertEqual(
                    result.dirty_rows,
                    0,
                )

                self.assertEqual(
                    result.materialization.num_blocks,
                    0,
                )

                self.assertEqual(
                    os.stat(
                        result.path
                    ).st_size,
                    layout.image_size,
                )

                # Kernel is mocked, so this remains an all-hole file.
                # Real brc3 SEAL inheritance is tested separately in C3.
                self.assertEqual(
                    read_file(
                        result.path
                    ),
                    bytes(
                        layout.image_size
                    ),
                )

            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
