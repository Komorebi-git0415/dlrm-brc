#!/usr/bin/env python3
"""Real brc3 runtime test for application checkpoint restore.

Sequence:

    real BRC lineage:
        C0 -> publish
        C1 -> publish
        close manager

    fresh embedding modules:
        manifest -> C1
        restore C1.img
        restore C1.state.pt

C1 exercises:
    block 0: Parent-RMW
    block 1: Reconstruction
    block 2: BRC inheritance

The checkpoint layout is global-mixed and has a partial final block.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from brc_checkpoint_blocks import (
    EMBEDDING_DIM,
    LAYOUT_GLOBAL_MIXED,
    CheckpointLayout,
)
from brc_checkpoint_manager import (
    BRCCkptManager,
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
from brc_materialization_policy import (
    MODE_HYBRID,
    MaterializationPolicy,
)


TEST_ROOT = "/mnt/brc-test/c7"

TABLE_SIZES = [
    70,
    67,
    18,
]


def make_layout():
    new_to_old = [
        np.arange(
            size,
            dtype=np.int64,
        )
        for size in TABLE_SIZES
    ]

    # Deliberately interleave tables in checkpoint storage order.
    frequencies = []

    for table_id, size in enumerate(
        TABLE_SIZES
    ):
        frequencies.append(
            (
                1_000_000
                - np.arange(
                    size,
                    dtype=np.int64,
                )
                * len(TABLE_SIZES)
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
        table_sizes=TABLE_SIZES,
        mode=LAYOUT_GLOBAL_MIXED,
        global_layout=global_layout,
    )


def make_tables(
    generation,
):
    tables = []

    dimensions = torch.arange(
        EMBEDDING_DIM,
        dtype=torch.float32,
    )

    for table_id, size in enumerate(
        TABLE_SIZES
    ):
        weight = torch.empty(
            (
                size,
                EMBEDDING_DIM,
            ),
            dtype=torch.float32,
        )

        for row_id in range(
            size
        ):
            weight[
                row_id
            ].copy_(
                dimensions
                + float(
                    generation * 1_000_000
                    + table_id * 100_000
                    + row_id * 100
                )
            )

        tables.append(
            weight
        )

    return tables


def dirty_rows_from_storage_slots(
    layout,
    storage_slots,
):
    per_table = [
        set()
        for _ in range(
            layout.num_tables
        )
    ]

    for storage_slot in storage_slots:
        ref = layout.row_ref_for_slot(
            storage_slot
        )

        per_table[
            ref.table_id
        ].add(
            ref.row_id
        )

    return [
        np.asarray(
            sorted(rows),
            dtype=np.int64,
        )
        for rows in per_table
    ]


def apply_updates(
    layout,
    tables,
    storage_slots,
):
    dimensions = torch.arange(
        EMBEDDING_DIM,
        dtype=torch.float32,
    )

    for storage_slot in storage_slots:
        ref = layout.row_ref_for_slot(
            storage_slot
        )

        tables[
            ref.table_id
        ][
            ref.row_id
        ].copy_(
            dimensions
            + float(
                9_000_000
                + ref.table_id * 10_000
                + ref.row_id * 100
            )
        )


def make_empty_embedding_bags():
    modules = []

    for size in TABLE_SIZES:
        module = torch.nn.EmbeddingBag(
            size,
            EMBEDDING_DIM,
            mode="sum",
            sparse=True,
        )

        with torch.no_grad():
            module.weight.fill_(
                -777.0
            )

        modules.append(
            module
        )

    return modules


def load_manifest(
    run_dir,
):
    path = (
        Path(run_dir)
        / "manifest.json"
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main():
    kernel = os.uname().release

    if kernel != "6.12.103-brc3":
        raise RuntimeError(
            f"wrong kernel: {kernel}; "
            "expected 6.12.103-brc3"
        )

    if not os.path.isdir(
        TEST_ROOT
    ):
        raise RuntimeError(
            f"missing test directory: "
            f"{TEST_ROOT}"
        )

    if not os.access(
        TEST_ROOT,
        os.W_OK,
    ):
        raise RuntimeError(
            f"test directory is not writable: "
            f"{TEST_ROOT}"
        )

    layout = make_layout()

    if layout.total_rows != 155:
        raise AssertionError(
            "expected 155 real rows"
        )

    if layout.num_blocks != 3:
        raise AssertionError(
            "expected three checkpoint blocks"
        )

    policy = MaterializationPolicy(
        MODE_HYBRID,
        threshold=4,
    )

    run_dir = tempfile.mkdtemp(
        prefix="brc-c7-restore-",
        dir=TEST_ROOT,
    )

    print(
        f"run_dir={run_dir}"
    )
    print(
        f"kernel={kernel}"
    )
    print(
        "layout=global-mixed"
    )
    print(
        f"total_rows={layout.total_rows}"
    )
    print(
        f"image_size={layout.image_size}"
    )
    print(
        "policy=hybrid threshold=4"
    )

    manager = BRCCkptManager(
        checkpoint_dir=run_dir,
        layout=layout,
        policy=policy,
    )

    publisher = CheckpointPublisher(
        checkpoint_dir=run_dir,
        layout=layout,
    )

    # ==============================================================
    # Produce a real published BRC C0.
    # ==============================================================

    table0 = make_tables(
        generation=0
    )

    try:
        manager.start_new()

        c0 = manager.create_root(
            table0
        )

        publisher.publish(
            c0,
            {
                "step": 100,
                "epoch": 0,
                "dense_marker": torch.tensor(
                    [10, 20, 30],
                    dtype=torch.float32,
                ),
            },
        )

        print(
            "[1] real BRC C0 published"
        )

        # ==========================================================
        # Produce C1.
        #
        # storage block 0:
        #     1 dirty row -> Parent-RMW
        #
        # storage block 1:
        #     8 dirty rows -> Reconstruction
        #
        # storage block 2:
        #     clean -> inherited by BRC
        # ==========================================================

        table1 = [
            table.clone()
            for table in table0
        ]

        dirty_slots = (
            [7]
            + list(
                range(
                    64,
                    72,
                )
            )
        )

        apply_updates(
            layout,
            table1,
            dirty_slots,
        )

        dirty_rows = (
            dirty_rows_from_storage_slots(
                layout,
                dirty_slots,
            )
        )

        c1 = manager.create_child(
            dirty_rows,
            table1,
        )

        stats = c1.materialization

        if c1.generation != 1:
            raise AssertionError(
                "unexpected C1 generation"
            )

        if c1.dirty_blocks != 2:
            raise AssertionError(
                "C1 should have two dirty blocks"
            )

        if c1.dirty_rows != 9:
            raise AssertionError(
                "C1 should have nine dirty rows"
            )

        if stats.parent_rmw_blocks != 1:
            raise AssertionError(
                "C1 should contain one "
                "Parent-RMW block"
            )

        if stats.reconstruct_blocks != 1:
            raise AssertionError(
                "C1 should contain one "
                "reconstruction block"
            )

        publisher.publish(
            c1,
            {
                "step": 200,
                "epoch": 1,
                "dense_marker": torch.tensor(
                    [110, 120, 130],
                    dtype=torch.float32,
                ),
                "optimizer_marker": {
                    "lr": 0.005,
                    "counter": 17,
                },
                "rng_state": (
                    torch.get_rng_state().clone()
                ),
            },
        )

        print(
            "[2] real BRC C1 published "
            "(1 RMW + 1 reconstruct + 1 inherited)"
        )

    finally:
        manager.close()

    print(
        "[3] original checkpoint manager closed"
    )

    # ==============================================================
    # Fresh runtime objects.
    # ==============================================================

    destination = (
        make_empty_embedding_bags()
    )

    for module in destination:
        if not torch.all(
            module.weight
            == -777.0
        ):
            raise AssertionError(
                "fresh destination was not initialized "
                "to sentinel value"
            )

    print(
        "[4] fresh EmbeddingBag modules created"
    )

    # ==============================================================
    # Restore exclusively from application publication metadata.
    # ==============================================================

    restored = (
        restore_published_checkpoint(
            run_dir,
            layout,
            destination,
            chunk_rows=7,
        )
    )

    if restored.generation != 1:
        raise AssertionError(
            "restore did not select published C1"
        )

    if restored.training_state[
        "step"
    ] != 200:
        raise AssertionError(
            "training step was not restored"
        )

    if restored.training_state[
        "epoch"
    ] != 1:
        raise AssertionError(
            "training epoch was not restored"
        )

    torch.testing.assert_close(
        restored.training_state[
            "dense_marker"
        ],
        torch.tensor(
            [110, 120, 130],
            dtype=torch.float32,
        ),
    )

    if (
        restored.training_state[
            "optimizer_marker"
        ][
            "counter"
        ]
        != 17
    ):
        raise AssertionError(
            "optimizer marker mismatch"
        )

    print(
        "[5] published C1 metadata + sidecar restored"
    )

    # ==============================================================
    # Exact embedding correctness.
    # ==============================================================

    for table_id, (
        module,
        expected,
    ) in enumerate(
        zip(
            destination,
            table1,
        )
    ):
        try:
            torch.testing.assert_close(
                module.weight,
                expected,
                rtol=0,
                atol=0,
            )
        except AssertionError as exc:
            raise AssertionError(
                "restored embedding mismatch for "
                f"table {table_id}"
            ) from exc

    if (
        restored.embedding.restored_rows
        != layout.total_rows
    ):
        raise AssertionError(
            "restored row count mismatch"
        )

    if (
        restored.embedding.max_chunk_rows
        > 7
    ):
        raise AssertionError(
            "chunk bound was violated"
        )

    print(
        "[6] all embedding rows restored "
        "exactly from real BRC C1"
    )

    # ==============================================================
    # Manifest-v2 identity sanity check.
    # ==============================================================

    manifest = load_manifest(
        run_dir
    )

    if manifest[
        "version"
    ] != 2:
        raise AssertionError(
            "expected manifest version 2"
        )

    if manifest[
        "generation"
    ] != 1:
        raise AssertionError(
            "manifest generation mismatch"
        )

    if manifest[
        "layout"
    ][
        "table_sizes"
    ] != TABLE_SIZES:
        raise AssertionError(
            "manifest table_sizes mismatch"
        )

    fingerprint = manifest[
        "layout"
    ].get(
        "fingerprint"
    )

    if (
        not isinstance(
            fingerprint,
            str,
        )
        or len(fingerprint) != 64
    ):
        raise AssertionError(
            "manifest layout fingerprint missing"
        )

    print(
        "[7] manifest-v2 layout identity: PASS"
    )

    print()
    print(
        "======================================="
    )
    print(
        "C7 REAL BRC CHECKPOINT RESTORE: PASS"
    )
    print(
        "======================================="
    )
    print(
        f"artifacts kept at: {run_dir}"
    )
    print(
        "restored generation: C1"
    )
    print(
        "source: real brc3 sealed/inherited checkpoint"
    )
    print(
        "manager lineage resume intentionally not tested here"
    )


if __name__ == "__main__":
    main()
