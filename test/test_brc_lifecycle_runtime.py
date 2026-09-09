#!/usr/bin/env python3
"""Runtime lifecycle test for ext4 BRC on the dedicated test filesystem.

This is intentionally separate from normal unit tests because it requires:
  * the brc3 kernel
  * a BRC-capable ext4 filesystem
  * /mnt/brc-test mounted read-write

Lifecycle exercised:

    LINEAGE_BEGIN
        -> full C0
        -> SEAL C0
        -> CREATE sparse C1
        -> write dirty blocks
        -> SEAL C1
        -> verify inherited clean blocks
        -> close session
        -> LINEAGE_RESUME
        -> CREATE sparse C2
        -> SEAL C2
        -> verify inheritance after resume
"""

from __future__ import annotations

import os
import tempfile

import brc_uapi


TEST_ROOT = "/mnt/brc-test/c3"
BLOCK_SIZE = 4096
NR_BLOCKS = 4
IMAGE_SIZE = BLOCK_SIZE * NR_BLOCKS

LEDGER_HEADER_SIZE = 64
LEDGER_ENTRY_SIZE = 16


def block(byte_value: int) -> bytes:
    return bytes([byte_value]) * BLOCK_SIZE


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0

    while written < len(view):
        n = os.pwrite(fd, view[written:], offset + written)
        if n <= 0:
            raise RuntimeError("pwrite made no progress")
        written += n


def read_block(fd: int, block_index: int) -> bytes:
    data = os.pread(fd, BLOCK_SIZE, block_index * BLOCK_SIZE)
    if len(data) != BLOCK_SIZE:
        raise RuntimeError(
            f"short read for block {block_index}: "
            f"{len(data)} != {BLOCK_SIZE}"
        )
    return data


def verify_image(path: str, expected: list[bytes]) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        if os.fstat(fd).st_size != IMAGE_SIZE:
            raise AssertionError(
                f"{path}: unexpected size {os.fstat(fd).st_size}"
            )

        for index, expected_block in enumerate(expected):
            actual = read_block(fd, index)
            if actual != expected_block:
                raise AssertionError(
                    f"{path}: block {index} content mismatch"
                )
    finally:
        os.close(fd)


def verify_ledger_size(path: str, nr_entries: int) -> None:
    expected = LEDGER_HEADER_SIZE + nr_entries * LEDGER_ENTRY_SIZE
    actual = os.stat(path).st_size

    if actual != expected:
        raise AssertionError(
            f"ledger size mismatch: actual={actual}, expected={expected}"
        )


def safe_close(fd: int | None) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def main() -> None:
    if os.uname().release != "6.12.103-brc3":
        raise RuntimeError(
            f"wrong runtime kernel: {os.uname().release}; "
            "expected 6.12.103-brc3"
        )

    statvfs = os.statvfs(TEST_ROOT)
    if statvfs.f_bsize != BLOCK_SIZE:
        raise RuntimeError(
            f"unexpected filesystem block size: {statvfs.f_bsize}"
        )

    run_dir = tempfile.mkdtemp(prefix="brc-c3-", dir=TEST_ROOT)

    ledger_path = os.path.join(run_dir, "lineage.ledger")
    c0_path = os.path.join(run_dir, "C0.img")
    c1_path = os.path.join(run_dir, "C1.img")
    c2_path = os.path.join(run_dir, "C2.img")

    print(f"run_dir={run_dir}")
    print(f"kernel={os.uname().release}")
    print(f"block_size={statvfs.f_bsize}")

    ledger_fd = None
    session_fd = None
    c0_fd = None
    c1_fd = None
    c2_fd = None

    try:
        # ------------------------------------------------------------
        # 1. Persistent lineage begin
        # ------------------------------------------------------------
        ledger_fd = os.open(
            ledger_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )

        session_fd = brc_uapi.lineage_begin(ledger_fd)

        print(f"[1] LINEAGE_BEGIN: session_fd={session_fd}")

        if os.stat(ledger_path).st_size != LEDGER_HEADER_SIZE:
            raise AssertionError(
                "LINEAGE_BEGIN did not publish a 64-byte ledger header"
            )

        # ------------------------------------------------------------
        # 2. Build and seal root C0
        # ------------------------------------------------------------
        c0_expected = [
            block(0x10),
            block(0x20),
            block(0x30),
            block(0x40),
        ]

        c0_fd = os.open(
            c0_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )

        os.ftruncate(c0_fd, IMAGE_SIZE)

        for index, data in enumerate(c0_expected):
            pwrite_all(c0_fd, data, index * BLOCK_SIZE)

        brc_uapi.seal(c0_fd, session_fd)

        verify_image(c0_path, c0_expected)
        verify_ledger_size(ledger_path, 1)

        print("[2] C0: full root sealed and published")

        # ------------------------------------------------------------
        # 3. CREATE sparse C1
        #
        # Dirty:
        #   block 1
        #   block 3
        #
        # Clean:
        #   block 0 -> inherit C0
        #   block 2 -> inherit C0
        # ------------------------------------------------------------
        c1_fd = os.open(
            c1_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )

        if os.fstat(c1_fd).st_size != 0:
            raise AssertionError("C1 must be empty before BRC_CREATE")

        brc_uapi.create(
            child_fd=c1_fd,
            parent_fd=c0_fd,
            session_fd=session_fd,
        )

        os.ftruncate(c1_fd, IMAGE_SIZE)

        c1_dirty_1 = block(0xB1)
        c1_dirty_3 = block(0xD1)

        pwrite_all(c1_fd, c1_dirty_1, 1 * BLOCK_SIZE)
        pwrite_all(c1_fd, c1_dirty_3, 3 * BLOCK_SIZE)

        # Do not read clean ranges while the child is BUILDING.
        # The production checkpoint path only writes dirty blocks
        # before SEAL; it does not read clean child holes.
        print("[3] C1: CREATE succeeded; dirty blocks written")

        brc_uapi.seal(c1_fd, session_fd)

        c1_expected = [
            c0_expected[0],
            c1_dirty_1,
            c0_expected[2],
            c1_dirty_3,
        ]

        verify_image(c1_path, c1_expected)
        verify_ledger_size(ledger_path, 2)

        print("[4] C1: sealed; clean blocks inherited from C0")

        # ------------------------------------------------------------
        # 4. Simulate process/session close at a committed boundary.
        # ------------------------------------------------------------
        safe_close(c0_fd)
        c0_fd = None

        safe_close(c1_fd)
        c1_fd = None

        brc_uapi.close_session(session_fd)
        session_fd = None

        safe_close(ledger_fd)
        ledger_fd = None

        print("[5] original session closed after published C1")

        # ------------------------------------------------------------
        # 5. Reopen ledger and LINEAGE_RESUME
        # ------------------------------------------------------------
        ledger_fd = os.open(ledger_path, os.O_RDWR)
        session_fd = brc_uapi.lineage_resume(ledger_fd)

        print(f"[6] LINEAGE_RESUME: session_fd={session_fd}")

        # Published tail must be C1. Use C1 as parent for C2.
        c1_fd = os.open(c1_path, os.O_RDONLY)

        # ------------------------------------------------------------
        # 6. CREATE C2 after resume
        #
        # Only block 2 is dirty.
        # ------------------------------------------------------------
        c2_fd = os.open(
            c2_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )

        if os.fstat(c2_fd).st_size != 0:
            raise AssertionError("C2 must be empty before BRC_CREATE")

        brc_uapi.create(
            child_fd=c2_fd,
            parent_fd=c1_fd,
            session_fd=session_fd,
        )

        os.ftruncate(c2_fd, IMAGE_SIZE)

        c2_dirty_2 = block(0xC2)
        pwrite_all(c2_fd, c2_dirty_2, 2 * BLOCK_SIZE)

        brc_uapi.seal(c2_fd, session_fd)

        c2_expected = [
            c1_expected[0],
            c1_expected[1],
            c2_dirty_2,
            c1_expected[3],
        ]

        verify_image(c2_path, c2_expected)
        verify_ledger_size(ledger_path, 3)

        print("[7] C2: sealed after resume; inheritance verified")

        print()
        print("========================================")
        print("C3 BRC LIFECYCLE TEST: PASS")
        print("========================================")
        print(f"artifacts kept at: {run_dir}")
        print("ledger entries: 3")
        print("published checkpoints: C0 -> C1 -> C2")

    finally:
        safe_close(c2_fd)
        safe_close(c1_fd)
        safe_close(c0_fd)

        if session_fd is not None:
            try:
                brc_uapi.close_session(session_fd)
            except OSError:
                pass

        safe_close(ledger_fd)


if __name__ == "__main__":
    main()
