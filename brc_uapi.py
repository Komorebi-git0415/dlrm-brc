"""Low-level Python wrapper for the ext4 BRC ioctl UAPI.

ABI baseline:
    kernel commit 1983e6e246c9
    runtime kernel 6.12.103-brc3

This module intentionally contains only the low-level ioctl interface.
It does not implement checkpoint policy, dirty tracking, manifests,
restore logic, or DLRM integration.
"""

from __future__ import annotations

import fcntl
import os
import struct
from dataclasses import dataclass


# Linux generic ioctl encoding used by the current x86-64 experiment host.
_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = 8
_IOC_SIZESHIFT = 16
_IOC_DIRSHIFT = 30

_EXT4_IOC_TYPE = ord("f")


def _ioc(direction: int, nr: int, size: int = 0) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (_EXT4_IOC_TYPE << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _io(nr: int) -> int:
    return _ioc(_IOC_NONE, nr)


def _iow(nr: int, size: int) -> int:
    return _ioc(_IOC_WRITE, nr, size)


def _iowr(nr: int, size: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, nr, size)


# Audited brc3 UAPI structure layouts.
#
# struct ext4_brc_create {
#     __s32 parent_fd;
#     __s32 session_fd;
#     __u32 flags;
#     __u32 reserved[3];
# };
_BRC_CREATE = struct.Struct("=iiI3I")
#
# struct ext4_brc_control {
#     __s32 session_fd;
#     __u32 flags;
#     __u32 reserved[4];
# };
_BRC_CONTROL = struct.Struct("=iI4I")
#
# struct ext4_brc_reclaim {
#     __u64 through_generation;
#     __u32 flags;
#     __u32 reserved;
# };
_BRC_RECLAIM = struct.Struct("=QII")
#
# struct ext4_brc_classify {
#     __u64 generation;
#     __u64 shared_blocks;
#     __u64 dead_unique_blocks;
#     __u64 old_hole_blocks;
#     __u64 compared_blocks;
#     __u32 flags;
#     __u32 reserved;
# };
_BRC_CLASSIFY = struct.Struct("=QQQQQII")


BRC_CREATE_SIZE = _BRC_CREATE.size
BRC_CONTROL_SIZE = _BRC_CONTROL.size
BRC_RECLAIM_SIZE = _BRC_RECLAIM.size
BRC_CLASSIFY_SIZE = _BRC_CLASSIFY.size


# include/uapi/linux/ext4.h @ 1983e6e246c9
EXT4_IOC_BRC_TEST = _io(45)
EXT4_IOC_BRC_CREATE = _iow(46, BRC_CREATE_SIZE)
EXT4_IOC_BRC_SEAL = _iow(47, BRC_CONTROL_SIZE)
EXT4_IOC_BRC_SESSION_BEGIN = _io(48)
EXT4_IOC_BRC_LINEAGE_BEGIN = _io(49)
EXT4_IOC_BRC_RECLAIM_THROUGH = _iow(50, BRC_RECLAIM_SIZE)
EXT4_IOC_BRC_CLASSIFY_PAIR = _iowr(51, BRC_CLASSIFY_SIZE)
EXT4_IOC_BRC_RECLAIM_PHYSICAL = _io(52)
EXT4_IOC_BRC_LINEAGE_RESUME = _io(53)


def _check_u64(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{name} is outside uint64 range")


@dataclass(frozen=True)
class BRCClassifyResult:
    generation: int
    shared_blocks: int
    dead_unique_blocks: int
    old_hole_blocks: int
    compared_blocks: int


def test(fd: int) -> None:
    """Issue EXT4_IOC_BRC_TEST."""
    fcntl.ioctl(fd, EXT4_IOC_BRC_TEST)


def session_begin(directory_fd: int) -> int:
    """Begin a legacy non-persistent BRC session.

    Stage C normally uses lineage_begin()/lineage_resume() instead.
    """
    return fcntl.ioctl(directory_fd, EXT4_IOC_BRC_SESSION_BEGIN)


def lineage_begin(lineage_fd: int) -> int:
    """Create a persistent lineage and return its anonymous session fd.

    lineage_fd must refer to an empty writable regular file.
    """
    return fcntl.ioctl(lineage_fd, EXT4_IOC_BRC_LINEAGE_BEGIN)


def lineage_resume(lineage_fd: int) -> int:
    """Resume an existing persistent lineage and return a session fd."""
    return fcntl.ioctl(lineage_fd, EXT4_IOC_BRC_LINEAGE_RESUME)


def create(child_fd: int, parent_fd: int, session_fd: int) -> None:
    """Prepare an empty child checkpoint.

    The ioctl is issued on child_fd. parent_fd and session_fd are passed
    through struct ext4_brc_create.
    """
    arg = bytearray(
        _BRC_CREATE.pack(
            parent_fd,
            session_fd,
            0,  # flags
            0,
            0,
            0,  # reserved[3]
        )
    )
    fcntl.ioctl(child_fd, EXT4_IOC_BRC_CREATE, arg, True)


def seal(checkpoint_fd: int, session_fd: int) -> None:
    """Seal and publish a root or child checkpoint."""
    arg = bytearray(
        _BRC_CONTROL.pack(
            session_fd,
            0,  # flags
            0,
            0,
            0,
            0,  # reserved[4]
        )
    )
    fcntl.ioctl(checkpoint_fd, EXT4_IOC_BRC_SEAL, arg, True)


def reclaim_through(lineage_fd: int, through_generation: int) -> None:
    """Logically retire checkpoints through the specified generation."""
    _check_u64(through_generation, "through_generation")

    arg = bytearray(
        _BRC_RECLAIM.pack(
            through_generation,
            0,  # flags
            0,  # reserved
        )
    )
    fcntl.ioctl(lineage_fd, EXT4_IOC_BRC_RECLAIM_THROUGH, arg, True)


def reclaim_physical(lineage_fd: int) -> None:
    """Advance physical reclamation by one retired generation."""
    fcntl.ioctl(lineage_fd, EXT4_IOC_BRC_RECLAIM_PHYSICAL)


def classify_pair(lineage_fd: int, generation: int) -> BRCClassifyResult:
    """Classify the retired checkpoint generation against its successor."""
    _check_u64(generation, "generation")

    arg = bytearray(
        _BRC_CLASSIFY.pack(
            generation,
            0,  # shared_blocks
            0,  # dead_unique_blocks
            0,  # old_hole_blocks
            0,  # compared_blocks
            0,  # flags
            0,  # reserved
        )
    )

    fcntl.ioctl(lineage_fd, EXT4_IOC_BRC_CLASSIFY_PAIR, arg, True)

    (
        returned_generation,
        shared_blocks,
        dead_unique_blocks,
        old_hole_blocks,
        compared_blocks,
        _flags,
        _reserved,
    ) = _BRC_CLASSIFY.unpack(arg)

    if returned_generation != generation:
        raise RuntimeError(
            "kernel returned an unexpected generation: "
            f"requested={generation}, returned={returned_generation}"
        )

    return BRCClassifyResult(
        generation=returned_generation,
        shared_blocks=shared_blocks,
        dead_unique_blocks=dead_unique_blocks,
        old_hole_blocks=old_hole_blocks,
        compared_blocks=compared_blocks,
    )


def close_session(session_fd: int) -> None:
    """Close a BRC anonymous session fd."""
    os.close(session_fd)
