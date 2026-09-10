#!/usr/bin/env python3
"""Real brc3 integration test for application checkpoint publication.

Sequence:

    BRC C0 SEAL
        -> C0.state.pt
        -> manifest -> generation 0

    BRC C1 SEAL
        -> C1.state.pt
        -> atomic manifest -> generation 1

Final consistency requirement:

    kernel ledger tail
        ==
    BRCCkptManager generation
        ==
    manifest generation
        ==
    sidecar generation
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from brc_checkpoint_blocks import (
    EMBEDDING_DIM,
    LAYOUT_LOCAL_ONLY,
    CheckpointLayout,
)
from brc_checkpoint_manager import (
    BRCCkptManager,
)
from brc_checkpoint_publication import (
    CheckpointPublisher,
)
from brc_materialization_policy import (
    MODE_HYBRID,
    MaterializationPolicy,
)


TEST_ROOT = "/mnt/brc-test/c6"
NUM_ROWS = 128

LEDGER_HEADER = struct.Struct(
    "<IHHHHIQQQQQQ"
)


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
                base
                + row_id * 100
            )
        )

    return weight


def load_manifest(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as stream:
        return json.load(
            stream
        )


def load_sidecar(path):
    return torch.load(
        path,
        map_location="cpu",
    )


def ledger_published_tail(
    ledger_path,
):
    with open(
        ledger_path,
        "rb",
    ) as stream:
        header_bytes = stream.read(
            LEDGER_HEADER.size
        )

    if len(header_bytes) != 64:
        raise AssertionError(
            "short BRC ledger header"
        )

    fields = LEDGER_HEADER.unpack(
        header_bytes
    )

    base_generation = fields[8]
    nr_entries = fields[10]

    if nr_entries == 0:
        return None

    return (
        base_generation
        + nr_entries
        - 1
    )


def assert_no_temp_files(run_dir):
    leftovers = [
        path.name
        for path in Path(
            run_dir
        ).iterdir()
        if (
            path.name.startswith(".")
            and ".tmp-" in path.name
        )
    ]

    if leftovers:
        raise AssertionError(
            "temporary publication files remain: "
            f"{leftovers}"
        )


def main():
    kernel = os.uname().release

    if kernel != "6.12.103-brc3":
        raise RuntimeError(
            f"wrong kernel: {kernel}; "
            "expected 6.12.103-brc3"
        )

    if not os.path.isdir(TEST_ROOT):
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

    layout = CheckpointLayout(
        table_sizes=[NUM_ROWS],
        mode=LAYOUT_LOCAL_ONLY,
    )

    policy = MaterializationPolicy(
        MODE_HYBRID,
        threshold=4,
    )

    run_dir = tempfile.mkdtemp(
        prefix="brc-c6-",
        dir=TEST_ROOT,
    )

    print(
        f"run_dir={run_dir}"
    )
    print(
        f"kernel={kernel}"
    )
    print(
        "layout=local-only"
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

    try:
        # ==========================================================
        # Generation 0
        # ==========================================================
        table0 = make_table(
            1000
        )

        manager.start_new()

        c0 = manager.create_root(
            [table0]
        )

        if c0.generation != 0:
            raise AssertionError(
                "unexpected C0 generation"
            )

        print(
            "[1] C0 BRC_SEAL: PASS"
        )

        state0 = {
            "step": 100,
            "epoch": 0,
            "dense_model": {
                "weight": torch.arange(
                    8,
                    dtype=torch.float32,
                ),
            },
            "optimizer": {
                "lr": 0.01,
                "momentum_buffer": torch.arange(
                    4,
                    dtype=torch.float32,
                ),
            },
            "rng_state": (
                torch.get_rng_state().clone()
            ),
        }

        published0 = (
            publisher.publish(
                c0,
                state0,
            )
        )

        manifest0 = load_manifest(
            published0.manifest_path
        )

        sidecar0 = load_sidecar(
            published0.sidecar_path
        )

        if manifest0[
            "generation"
        ] != 0:
            raise AssertionError(
                "manifest did not publish C0"
            )

        if sidecar0[
            "generation"
        ] != 0:
            raise AssertionError(
                "C0 sidecar generation mismatch"
            )

        if sidecar0[
            "state"
        ][
            "step"
        ] != 100:
            raise AssertionError(
                "C0 training step mismatch"
            )

        if manifest0[
            "checkpoint_file"
        ] != "C0.img":
            raise AssertionError(
                "C0 manifest checkpoint mismatch"
            )

        if manifest0[
            "sidecar_file"
        ] != "C0.state.pt":
            raise AssertionError(
                "C0 manifest sidecar mismatch"
            )

        assert_no_temp_files(
            run_dir
        )

        print(
            "[2] C0 application publication: PASS"
        )

        # ==========================================================
        # Generation 1
        # ==========================================================
        table1 = table0.clone()

        # One sparse update. With threshold=4 this block uses
        # Parent-RMW. The second block remains clean and is inherited.
        table1[
            1
        ].fill_(
            9001.0
        )

        c1 = manager.create_child(
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

        if (
            c1.materialization
            .parent_rmw_blocks
            != 1
        ):
            raise AssertionError(
                "C1 should use one "
                "Parent-RMW block"
            )

        if (
            c1.materialization
            .reconstruct_blocks
            != 0
        ):
            raise AssertionError(
                "C1 should not use "
                "reconstruction"
            )

        print(
            "[3] C1 BRC_SEAL: PASS"
        )

        state1 = {
            "step": 200,
            "epoch": 1,
            "dense_model": {
                "weight": torch.arange(
                    8,
                    dtype=torch.float32,
                )
                + 100.0,
            },
            "optimizer": {
                "lr": 0.005,
                "momentum_buffer": (
                    torch.arange(
                        4,
                        dtype=torch.float32,
                    )
                    + 10.0
                ),
            },
            "rng_state": (
                torch.get_rng_state().clone()
            ),
        }

        published1 = (
            publisher.publish(
                c1,
                state1,
            )
        )

        manifest1 = load_manifest(
            published1.manifest_path
        )

        sidecar1 = load_sidecar(
            published1.sidecar_path
        )

        if manifest1[
            "generation"
        ] != 1:
            raise AssertionError(
                "manifest did not advance to C1"
            )

        if manifest1[
            "checkpoint_file"
        ] != "C1.img":
            raise AssertionError(
                "manifest checkpoint is not C1"
            )

        if manifest1[
            "sidecar_file"
        ] != "C1.state.pt":
            raise AssertionError(
                "manifest sidecar is not C1"
            )

        if sidecar1[
            "generation"
        ] != 1:
            raise AssertionError(
                "C1 sidecar generation mismatch"
            )

        if sidecar1[
            "state"
        ][
            "step"
        ] != 200:
            raise AssertionError(
                "C1 training step mismatch"
            )

        # Historical C0 sidecar must still exist and remain unchanged.
        old_sidecar = load_sidecar(
            Path(run_dir)
            / "C0.state.pt"
        )

        if old_sidecar[
            "state"
        ][
            "step"
        ] != 100:
            raise AssertionError(
                "historical C0 sidecar changed"
            )

        if not (
            Path(run_dir)
            / "C1.state.pt"
        ).is_file():
            raise AssertionError(
                "C1 sidecar missing"
            )

        assert_no_temp_files(
            run_dir
        )

        print(
            "[4] C1 application publication: PASS"
        )

        # ==========================================================
        # Cross-layer publication consistency.
        # ==========================================================
        ledger_path = (
            Path(run_dir)
            / "lineage.ledger"
        )

        kernel_tail = (
            ledger_published_tail(
                ledger_path
            )
        )

        manager_generation = (
            manager.generation
        )

        manifest_generation = (
            manifest1[
                "generation"
            ]
        )

        sidecar_generation = (
            sidecar1[
                "generation"
            ]
        )

        if not (
            kernel_tail
            == manager_generation
            == manifest_generation
            == sidecar_generation
            == 1
        ):
            raise AssertionError(
                "cross-layer generation mismatch: "
                f"kernel={kernel_tail}, "
                f"manager={manager_generation}, "
                f"manifest={manifest_generation}, "
                f"sidecar={sidecar_generation}"
            )

        ledger_size = (
            ledger_path.stat().st_size
        )

        if ledger_size != 96:
            raise AssertionError(
                f"unexpected ledger size: "
                f"{ledger_size}"
            )

        if manifest1[
            "checkpoint_size"
        ] != layout.image_size:
            raise AssertionError(
                "manifest checkpoint size mismatch"
            )

        actual_sidecar_size = (
            Path(
                published1.sidecar_path
            ).stat().st_size
        )

        if manifest1[
            "sidecar_size"
        ] != actual_sidecar_size:
            raise AssertionError(
                "manifest sidecar size mismatch"
            )

        print(
            "[5] kernel/manager/manifest/sidecar "
            "generation alignment: PASS"
        )

        print(
            "[6] persistent ledger: "
            "PASS (64 + 2*16 = 96 bytes)"
        )

        print()
        print(
            "==========================================="
        )
        print(
            "C6 BRC CHECKPOINT PUBLICATION RUNTIME: PASS"
        )
        print(
            "==========================================="
        )
        print(
            f"artifacts kept at: {run_dir}"
        )
        print(
            "application published generation: C1"
        )
        print(
            "safe interruption boundary reached: yes"
        )

    finally:
        manager.close()


if __name__ == "__main__":
    main()
