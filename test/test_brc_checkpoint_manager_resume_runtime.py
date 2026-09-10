#!/usr/bin/env python3
"""Real brc3 resume test for BRCCkptManager.

Sequence:

    manager A:
        LINEAGE_BEGIN -> C0 -> C1 -> close

    manager B:
        LINEAGE_RESUME(C1, generation=1)
        -> C2 -> close

C2 exercises:
    block 0: Parent-RMW
    block 1: Reconstruction
    block 2: BRC inheritance
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    EMBEDDING_DIM,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
)
from brc_checkpoint_manager import BRCCkptManager
from brc_materialization_policy import (
    MODE_HYBRID,
    MaterializationPolicy,
)


TEST_ROOT = "/mnt/brc-test/c5"
NUM_ROWS = 192


def make_table(base):
    weight = torch.empty(
        (NUM_ROWS, EMBEDDING_DIM),
        dtype=torch.float32,
    )

    dimensions = torch.arange(
        EMBEDDING_DIM,
        dtype=torch.float32,
    )

    for row_id in range(NUM_ROWS):
        weight[row_id].copy_(
            dimensions
            + float(
                base + row_id * 100
            )
        )

    return weight


def checkpoint_image(table):
    payload = (
        table
        .detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
        .tobytes(order="C")
    )

    if len(payload) != 3 * BLOCK_SIZE:
        raise AssertionError(
            "unexpected checkpoint image size"
        )

    return payload


def read_file(path):
    with open(path, "rb") as stream:
        return stream.read()


def main():
    kernel = os.uname().release

    if kernel != "6.12.103-brc3":
        raise RuntimeError(
            f"wrong kernel: {kernel}; "
            "expected 6.12.103-brc3"
        )

    if not os.path.isdir(TEST_ROOT):
        raise RuntimeError(
            f"missing test directory: {TEST_ROOT}"
        )

    if not os.access(TEST_ROOT, os.W_OK):
        raise RuntimeError(
            f"test directory is not writable: {TEST_ROOT}"
        )

    layout = CheckpointLayout(
        table_sizes=[NUM_ROWS],
        mode=LAYOUT_LOCAL_ONLY,
    )

    if layout.num_blocks != 3:
        raise AssertionError(
            "expected exactly three blocks"
        )

    if layout.image_size != 3 * BLOCK_SIZE:
        raise AssertionError(
            "unexpected image size"
        )

    policy = MaterializationPolicy(
        MODE_HYBRID,
        threshold=4,
    )

    run_dir = tempfile.mkdtemp(
        prefix="brc-c5-resume-",
        dir=TEST_ROOT,
    )

    print(f"run_dir={run_dir}")
    print(f"kernel={kernel}")
    print("layout=local-only")
    print("policy=hybrid threshold=4")

    # ==============================================================
    # Manager A: new lineage -> C0 -> C1
    # ==============================================================

    table0 = make_table(
        1000
    )

    manager_a = BRCCkptManager(
        checkpoint_dir=run_dir,
        layout=layout,
        policy=policy,
    )

    try:
        manager_a.start_new()

        c0 = manager_a.create_root(
            [table0]
        )

        if c0.generation != 0:
            raise AssertionError(
                "unexpected C0 generation"
            )

        if read_file(c0.path) != checkpoint_image(table0):
            raise AssertionError(
                "C0 content mismatch"
            )

        print("[1] manager A: C0 published")

        # C1:
        # block 0 -> one dirty row -> Parent-RMW
        # blocks 1,2 -> inherited
        table1 = table0.clone()

        table1[1].fill_(
            9001.0
        )

        c1 = manager_a.create_child(
            [
                np.asarray(
                    [1],
                    dtype=np.int64,
                )
            ],
            [table1],
        )

        if c1.generation != 1:
            raise AssertionError(
                "unexpected C1 generation"
            )

        if c1.dirty_blocks != 1:
            raise AssertionError(
                "C1 should have one dirty block"
            )

        if c1.dirty_rows != 1:
            raise AssertionError(
                "C1 should have one dirty row"
            )

        if (
            c1.materialization.parent_rmw_blocks
            != 1
        ):
            raise AssertionError(
                "C1 should use one Parent-RMW block"
            )

        if (
            c1.materialization.reconstruct_blocks
            != 0
        ):
            raise AssertionError(
                "C1 should use no reconstruction blocks"
            )

        if read_file(c1.path) != checkpoint_image(table1):
            raise AssertionError(
                "C1 content mismatch"
            )

        ledger_size = os.stat(
            manager_a.ledger_path
        ).st_size

        if ledger_size != 96:
            raise AssertionError(
                f"unexpected ledger size after C1: "
                f"{ledger_size}"
            )

        print(
            "[2] manager A: C1 published "
            "(1 RMW + 2 inherited)"
        )

    finally:
        manager_a.close()

    print("[3] manager A closed at published C1")

    # ==============================================================
    # Manager B: LINEAGE_RESUME from published C1
    # ==============================================================

    manager_b = BRCCkptManager(
        checkpoint_dir=run_dir,
        layout=layout,
        policy=policy,
    )

    try:
        manager_b.resume(
            generation=1,
            parent_path="C1.img",
        )

        if manager_b.generation != 1:
            raise AssertionError(
                "resume generation mismatch"
            )

        expected_parent = str(
            Path(run_dir)
            / "C1.img"
        )

        if manager_b.parent_path != expected_parent:
            raise AssertionError(
                "resume parent path mismatch"
            )

        print(
            "[4] manager B: LINEAGE_RESUME at C1 PASS"
        )

        # C2:
        #
        # block 0:
        #   one newly dirty row -> Parent-RMW
        #
        # block 1:
        #   eight newly dirty rows -> Reconstruction
        #
        # block 2:
        #   clean -> BRC inheritance
        table2 = table1.clone()

        dirty_rows = (
            [2]
            + list(
                range(
                    64,
                    72,
                )
            )
        )

        table2[2].fill_(
            9002.0
        )

        for row_id in range(
            64,
            72,
        ):
            table2[row_id].fill_(
                float(
                    10000 + row_id
                )
            )

        c2 = manager_b.create_child(
            [
                np.asarray(
                    dirty_rows,
                    dtype=np.int64,
                )
            ],
            [table2],
        )

        if c2.generation != 2:
            raise AssertionError(
                "unexpected C2 generation"
            )

        if c2.dirty_blocks != 2:
            raise AssertionError(
                "C2 should have two dirty blocks"
            )

        if c2.dirty_rows != 9:
            raise AssertionError(
                "C2 should have nine dirty rows"
            )

        stats = c2.materialization

        if stats.parent_rmw_blocks != 1:
            raise AssertionError(
                "C2 should use one Parent-RMW block"
            )

        if stats.reconstruct_blocks != 1:
            raise AssertionError(
                "C2 should use one reconstruction block"
            )

        if stats.parent_rmw_gathered_rows != 1:
            raise AssertionError(
                "C2 RMW should gather one row"
            )

        if stats.reconstruct_gathered_rows != 64:
            raise AssertionError(
                "C2 reconstruction should gather 64 rows"
            )

        if stats.parent_read_bytes != BLOCK_SIZE:
            raise AssertionError(
                "C2 RMW should read one parent block"
            )

        if stats.output_bytes != 2 * BLOCK_SIZE:
            raise AssertionError(
                "C2 should write two materialized blocks"
            )

        if read_file(c2.path) != checkpoint_image(table2):
            raise AssertionError(
                "C2 content mismatch after resume"
            )

        ledger_size = os.stat(
            manager_b.ledger_path
        ).st_size

        if ledger_size != 112:
            raise AssertionError(
                f"unexpected ledger size after C2: "
                f"{ledger_size}"
            )

        print(
            "[5] manager B: C2 published after resume "
            "(1 RMW + 1 reconstruct + 1 inherited)"
        )

        print(
            "[6] persistent ledger: PASS "
            "(64 + 3*16 = 112 bytes)"
        )

        print()
        print("=============================================")
        print("C5 BRC MANAGER RESUME RUNTIME: PASS")
        print("=============================================")
        print(f"artifacts kept at: {run_dir}")
        print(
            "published lineage across sessions: "
            "C0 -> C1 -> [RESUME] -> C2"
        )

    finally:
        manager_b.close()


if __name__ == "__main__":
    main()
