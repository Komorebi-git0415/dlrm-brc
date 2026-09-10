#!/usr/bin/env python3
"""Real brc3 runtime test for BRCCkptManager.

Requires:
    kernel: 6.12.103-brc3
    filesystem: /mnt/brc-test
    writable directory: /mnt/brc-test/c5

The test exercises:

    LINEAGE_BEGIN
        -> C0 full root
        -> C1 hybrid child
             block 0: Parent-RMW
             block 1: Reconstruction
             block 2: clean / inherited by BRC
        -> C2 empty-dirty child
             all blocks inherited
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    EMBEDDING_DIM,
    ROW_BYTES,
    LAYOUT_GLOBAL_MIXED,
    CheckpointLayout,
)
from brc_checkpoint_manager import BRCCkptManager
from brc_global_layout import build_global_storage_layout
from brc_materialization_policy import (
    MODE_HYBRID,
    MaterializationPolicy,
)


TEST_ROOT = "/mnt/brc-test/c5"
TABLE_SIZES = [64, 64, 64]
TOTAL_ROWS = sum(TABLE_SIZES)


def make_layout():
    # Identity stage-1 permutation.
    new_to_old = [
        np.arange(size, dtype=np.int64)
        for size in TABLE_SIZES
    ]

    # Interleave rows from different tables in global storage order:
    #
    # T0:L0, T1:L0, T2:L0,
    # T0:L1, T1:L1, T2:L1, ...
    frequencies = []

    for table_id, size in enumerate(TABLE_SIZES):
        frequencies.append(
            (
                1_000_000
                - np.arange(size, dtype=np.int64) * len(TABLE_SIZES)
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


def make_tables():
    tables = []

    dimensions = torch.arange(
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
                dimensions
                + float(
                    table_id * 100_000
                    + row_id * 100
                )
            )

        tables.append(weight)

    return tables


def build_checkpoint_image(layout, tables):
    image = bytearray(layout.image_size)

    for storage_slot in range(layout.total_rows):
        ref = layout.row_ref_for_slot(
            storage_slot
        )

        payload = (
            tables[ref.table_id][ref.row_id]
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.float32, copy=False)
            .tobytes(order="C")
        )

        if len(payload) != ROW_BYTES:
            raise AssertionError(
                "unexpected embedding row byte size"
            )

        start = storage_slot * ROW_BYTES
        image[start:start + ROW_BYTES] = payload

    return bytes(image)


def dirty_rows_from_slots(layout, slots):
    dirty = [
        set()
        for _ in range(layout.num_tables)
    ]

    for storage_slot in slots:
        ref = layout.row_ref_for_slot(
            storage_slot
        )

        dirty[ref.table_id].add(
            ref.row_id
        )

    return [
        np.asarray(
            sorted(rows),
            dtype=np.int64,
        )
        for rows in dirty
    ]


def apply_updates(layout, tables, slots):
    dimensions = torch.arange(
        EMBEDDING_DIM,
        dtype=torch.float32,
    )

    for storage_slot in slots:
        ref = layout.row_ref_for_slot(
            storage_slot
        )

        tables[ref.table_id][ref.row_id].copy_(
            dimensions
            + float(
                9_000_000
                + ref.table_id * 10_000
                + ref.row_id * 100
            )
        )


def read_exact_file(path):
    with open(path, "rb") as stream:
        payload = stream.read()

    return payload


def main():
    kernel = os.uname().release

    if kernel != "6.12.103-brc3":
        raise RuntimeError(
            f"wrong kernel: {kernel}; expected 6.12.103-brc3"
        )

    if not os.path.isdir(TEST_ROOT):
        raise RuntimeError(
            f"missing test directory: {TEST_ROOT}"
        )

    if not os.access(TEST_ROOT, os.W_OK):
        raise RuntimeError(
            f"test directory is not writable: {TEST_ROOT}"
        )

    layout = make_layout()

    if layout.total_rows != TOTAL_ROWS:
        raise AssertionError(
            "unexpected total row count"
        )

    if layout.num_blocks != 3:
        raise AssertionError(
            "expected exactly three checkpoint blocks"
        )

    if layout.image_size != 3 * BLOCK_SIZE:
        raise AssertionError(
            "unexpected checkpoint image size"
        )

    policy = MaterializationPolicy(
        MODE_HYBRID,
        threshold=4,
    )

    run_dir = tempfile.mkdtemp(
        prefix="brc-c5-",
        dir=TEST_ROOT,
    )

    print(f"run_dir={run_dir}")
    print(f"kernel={kernel}")
    print(f"layout={layout.mode}")
    print(f"total_rows={layout.total_rows}")
    print(f"image_size={layout.image_size}")
    print("policy=hybrid threshold=4")

    parent_tables = make_tables()

    manager = BRCCkptManager(
        checkpoint_dir=run_dir,
        layout=layout,
        policy=policy,
    )

    try:
        # ------------------------------------------------------------
        # C0: complete root checkpoint.
        # ------------------------------------------------------------
        manager.start_new()

        root = manager.create_root(
            parent_tables
        )

        if root.generation != 0:
            raise AssertionError(
                "unexpected root generation"
            )

        if root.dirty_blocks != 3:
            raise AssertionError(
                "root must contain all three blocks"
            )

        if root.dirty_rows != TOTAL_ROWS:
            raise AssertionError(
                "root must contain all embedding rows"
            )

        expected_c0 = build_checkpoint_image(
            layout,
            parent_tables,
        )

        actual_c0 = read_exact_file(
            root.path
        )

        if actual_c0 != expected_c0:
            raise AssertionError(
                "C0 content mismatch"
            )

        print("[1] C0 root: PASS")

        # ------------------------------------------------------------
        # C1:
        #
        # block 0: 1 dirty row  -> Parent-RMW
        # block 1: 8 dirty rows -> Reconstruction
        # block 2: 0 dirty rows -> kernel inheritance
        # ------------------------------------------------------------
        current_tables = [
            table.clone()
            for table in parent_tables
        ]

        dirty_slots = (
            [7]
            + list(range(64, 72))
        )

        dirty_rows = dirty_rows_from_slots(
            layout,
            dirty_slots,
        )

        apply_updates(
            layout,
            current_tables,
            dirty_slots,
        )

        child1 = manager.create_child(
            dirty_rows,
            current_tables,
        )

        if child1.generation != 1:
            raise AssertionError(
                "unexpected C1 generation"
            )

        if child1.dirty_blocks != 2:
            raise AssertionError(
                "C1 should contain two dirty blocks"
            )

        if child1.dirty_rows != 9:
            raise AssertionError(
                "C1 should contain nine dirty rows"
            )

        stats = child1.materialization

        if stats.parent_rmw_blocks != 1:
            raise AssertionError(
                "C1 block 0 should use Parent-RMW"
            )

        if stats.reconstruct_blocks != 1:
            raise AssertionError(
                "C1 block 1 should use Reconstruction"
            )

        if stats.parent_rmw_gathered_rows != 1:
            raise AssertionError(
                "Parent-RMW should gather one live row"
            )

        if stats.reconstruct_gathered_rows != 64:
            raise AssertionError(
                "Reconstruction should gather one complete block"
            )

        if stats.parent_read_bytes != BLOCK_SIZE:
            raise AssertionError(
                "Parent-RMW should read exactly one parent block"
            )

        if stats.output_bytes != 2 * BLOCK_SIZE:
            raise AssertionError(
                "application should write exactly two dirty blocks"
            )

        expected_c1 = build_checkpoint_image(
            layout,
            current_tables,
        )

        actual_c1 = read_exact_file(
            child1.path
        )

        if actual_c1 != expected_c1:
            raise AssertionError(
                "C1 content mismatch after BRC SEAL"
            )

        print(
            "[2] C1 hybrid: PASS "
            "(1 RMW + 1 reconstruct + 1 inherited)"
        )

        # ------------------------------------------------------------
        # C2:
        #
        # No embedding row changed.
        # Application writes no blocks; BRC SEAL must inherit all three
        # blocks from C1.
        # ------------------------------------------------------------
        empty_dirty = [
            np.asarray([], dtype=np.int64)
            for _ in TABLE_SIZES
        ]

        child2 = manager.create_child(
            empty_dirty,
            current_tables,
        )

        if child2.generation != 2:
            raise AssertionError(
                "unexpected C2 generation"
            )

        if child2.dirty_blocks != 0:
            raise AssertionError(
                "C2 must contain zero dirty blocks"
            )

        if child2.dirty_rows != 0:
            raise AssertionError(
                "C2 must contain zero dirty rows"
            )

        if child2.materialization.num_blocks != 0:
            raise AssertionError(
                "C2 materializer must emit no blocks"
            )

        actual_c2 = read_exact_file(
            child2.path
        )

        if actual_c2 != expected_c1:
            raise AssertionError(
                "C2 inheritance content mismatch"
            )

        print(
            "[3] C2 empty-dirty inheritance: PASS"
        )

        ledger_size = os.stat(
            manager.ledger_path
        ).st_size

        # 64-byte header + three 16-byte published entries.
        if ledger_size != 112:
            raise AssertionError(
                f"unexpected ledger size: {ledger_size}"
            )

        print("[4] persistent ledger: PASS (112 bytes)")

        print()
        print("========================================")
        print("C5 BRC CHECKPOINT MANAGER RUNTIME: PASS")
        print("========================================")
        print(f"artifacts kept at: {run_dir}")
        print("published checkpoints: C0 -> C1 -> C2")

    finally:
        manager.close()


if __name__ == "__main__":
    main()
