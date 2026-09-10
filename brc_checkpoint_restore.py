"""Restore helpers for application-published BRC checkpoints.

C7 restore begins from manifest.json, never by scanning C*.img files.

This module currently implements publication metadata validation and
training-state sidecar loading. Embedding image restoration is added in
the next C7 step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from pathlib import Path

import torch

from brc_checkpoint_blocks import (
    BLOCK_SIZE,
    EMBEDDING_DIM,
    ROW_BYTES,
    ROWS_PER_BLOCK,
    CheckpointLayout,
    checkpoint_layout_fingerprint,
)
from brc_checkpoint_publication import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MANIFEST_VERSION,
    SIDECAR_FORMAT,
    SIDECAR_VERSION,
)


@dataclass(frozen=True)
class RestoreMetadata:
    """Validated application-published checkpoint metadata."""

    generation: int
    checkpoint_path: str
    sidecar_path: str
    training_state: dict


def _load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        return json.load(stream)


def load_published_checkpoint(
    checkpoint_dir,
    layout: CheckpointLayout,
) -> RestoreMetadata:
    """Load and validate the latest application-published generation."""

    if not isinstance(
        layout,
        CheckpointLayout,
    ):
        raise TypeError(
            "layout must be a CheckpointLayout"
        )

    checkpoint_dir = Path(
        checkpoint_dir
    ).resolve()

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            "checkpoint directory does not exist: "
            f"{checkpoint_dir}"
        )

    manifest_path = (
        checkpoint_dir
        / MANIFEST_FILENAME
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest does not exist: {manifest_path}"
        )

    manifest = _load_json(
        manifest_path
    )

    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            "manifest has unexpected format"
        )

    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            "manifest has unsupported version"
        )

    generation = manifest.get(
        "generation"
    )

    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
    ):
        raise ValueError(
            "manifest has invalid generation"
        )

    expected_checkpoint_name = (
        f"C{generation}.img"
    )

    expected_sidecar_name = (
        f"C{generation}.state.pt"
    )

    if (
        manifest.get("checkpoint_file")
        != expected_checkpoint_name
    ):
        raise ValueError(
            "manifest checkpoint filename does not "
            "match generation"
        )

    if (
        manifest.get("sidecar_file")
        != expected_sidecar_name
    ):
        raise ValueError(
            "manifest sidecar filename does not "
            "match generation"
        )

    checkpoint_path = (
        checkpoint_dir
        / expected_checkpoint_name
    )

    sidecar_path = (
        checkpoint_dir
        / expected_sidecar_name
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint image is missing: {checkpoint_path}"
        )

    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"training-state sidecar is missing: {sidecar_path}"
        )

    checkpoint_size = (
        checkpoint_path.stat().st_size
    )

    if checkpoint_size != layout.image_size:
        raise ValueError(
            "checkpoint image size does not match "
            "current layout"
        )

    if (
        manifest.get("checkpoint_size")
        != checkpoint_size
    ):
        raise ValueError(
            "manifest checkpoint size does not match file"
        )

    sidecar_size = (
        sidecar_path.stat().st_size
    )

    if (
        manifest.get("sidecar_size")
        != sidecar_size
    ):
        raise ValueError(
            "manifest sidecar size does not match file"
        )

    layout_meta = manifest.get(
        "layout"
    )

    if not isinstance(
        layout_meta,
        dict,
    ):
        raise ValueError(
            "manifest layout metadata is missing"
        )

    expected_layout = {
        "mode": layout.mode,
        "num_tables": layout.num_tables,
        "total_rows": layout.total_rows,
        "image_size": layout.image_size,
        "block_size": BLOCK_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "row_bytes": ROW_BYTES,
        "rows_per_block": ROWS_PER_BLOCK,
        "table_sizes": [
            int(value)
            for value
            in layout.table_sizes
        ],
        "fingerprint": (
            checkpoint_layout_fingerprint(
                layout
            )
        ),
    }

    for key, expected in expected_layout.items():
        if layout_meta.get(key) != expected:
            raise ValueError(
                "manifest layout mismatch for "
                f"{key}: expected {expected!r}, "
                f"got {layout_meta.get(key)!r}"
            )

    sidecar = torch.load(
        sidecar_path,
        map_location="cpu",
    )

    if not isinstance(
        sidecar,
        dict,
    ):
        raise ValueError(
            "sidecar payload must be a dict"
        )

    if sidecar.get("format") != SIDECAR_FORMAT:
        raise ValueError(
            "sidecar has unexpected format"
        )

    if sidecar.get("version") != SIDECAR_VERSION:
        raise ValueError(
            "sidecar has unsupported version"
        )

    if sidecar.get("generation") != generation:
        raise ValueError(
            "sidecar generation does not match manifest"
        )

    training_state = sidecar.get(
        "state"
    )

    if not isinstance(
        training_state,
        dict,
    ):
        raise ValueError(
            "sidecar training state must be a dict"
        )

    return RestoreMetadata(
        generation=generation,
        checkpoint_path=str(
            checkpoint_path
        ),
        sidecar_path=str(
            sidecar_path
        ),
        training_state=training_state,
    )


@dataclass(frozen=True)
class EmbeddingRestoreStats:
    """Statistics for restoring one checkpoint embedding image."""

    num_tables: int
    restored_rows: int
    image_bytes: int
    restored_payload_bytes: int
    num_chunks: int
    max_chunk_rows: int
    max_host_bytes: int


def _restore_embedding_weight(table):
    """Return one destination FP32 embedding weight tensor."""

    if isinstance(table, torch.Tensor):
        weight = table

    elif (
        hasattr(table, "weight")
        and isinstance(
            table.weight,
            torch.Tensor,
        )
    ):
        weight = table.weight

    else:
        raise TypeError(
            "embedding table must be a Tensor or "
            "an object with a Tensor .weight"
        )

    if weight.ndim != 2:
        raise ValueError(
            "embedding weight must be 2-dimensional"
        )

    if weight.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            "embedding dimension mismatch: "
            f"expected {EMBEDDING_DIM}, "
            f"got {weight.shape[1]}"
        )

    if weight.dtype != torch.float32:
        raise ValueError(
            "embedding weight must use float32"
        )

    return weight


def _validate_restore_destinations(
    layout: CheckpointLayout,
    embedding_tables,
):
    """Validate every destination before modifying any live weight."""

    try:
        embedding_tables = tuple(
            embedding_tables
        )
    except TypeError as exc:
        raise TypeError(
            "embedding_tables must be iterable"
        ) from exc

    if (
        len(embedding_tables)
        != layout.num_tables
    ):
        raise ValueError(
            "embedding table count does not match layout"
        )

    table_sizes = layout.table_sizes
    weights = []

    for table_id, table in enumerate(
        embedding_tables
    ):
        weight = _restore_embedding_weight(
            table
        )

        expected_rows = int(
            table_sizes[
                table_id
            ]
        )

        if weight.shape[0] != expected_rows:
            raise ValueError(
                "embedding row count mismatch for "
                f"table {table_id}: "
                f"expected {expected_rows}, "
                f"got {weight.shape[0]}"
            )

        weights.append(
            weight
        )

    return tuple(weights)


def restore_embedding_image(
    checkpoint_path,
    layout: CheckpointLayout,
    embedding_tables,
    *,
    chunk_rows: int = 65536,
) -> EmbeddingRestoreStats:
    """Restore checkpoint rows into live embedding tables.

    The checkpoint file is memory-mapped instead of being read into one
    giant Python bytes object.

    Restoration proceeds table by table and chunk by chunk.  For each
    runtime-row chunk, CheckpointLayout provides the corresponding
    checkpoint storage slots in vectorized NumPy form.

    This supports both local-only and global-mixed layouts without
    constructing per-row Python dictionaries.

    Destination weights may reside on CPU or CUDA.  CUDA correctness is
    intentionally left for validation on the later GPU platform.
    """

    if not isinstance(
        layout,
        CheckpointLayout,
    ):
        raise TypeError(
            "layout must be a CheckpointLayout"
        )

    if (
        isinstance(chunk_rows, (bool, np.bool_))
        or not isinstance(
            chunk_rows,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "chunk_rows must be an integer"
        )

    chunk_rows = int(
        chunk_rows
    )

    if chunk_rows <= 0:
        raise ValueError(
            "chunk_rows must be positive"
        )

    # Validate ALL live destinations before copying the first row.
    # This avoids a half-restored model caused by discovering an invalid
    # later table after earlier tables were already modified.
    weights = _validate_restore_destinations(
        layout,
        embedding_tables,
    )

    checkpoint_path = Path(
        checkpoint_path
    ).resolve(
        strict=True
    )

    if not checkpoint_path.is_file():
        raise ValueError(
            "checkpoint image must be a regular file"
        )

    image_size = (
        checkpoint_path.stat().st_size
    )

    if image_size != layout.image_size:
        raise ValueError(
            "checkpoint image size does not match layout"
        )

    restored_rows = 0
    num_chunks = 0
    max_chunk_rows = 0
    max_host_bytes = 0

    # Map only the real embedding rows.  Alignment padding after the
    # final real row is deliberately excluded from the array shape.
    storage_rows = np.memmap(
        checkpoint_path,
        mode="r",
        dtype=np.float32,
        offset=0,
        shape=(
            layout.total_rows,
            EMBEDDING_DIM,
        ),
        order="C",
    )

    try:
        table_sizes = (
            layout.table_sizes
        )

        with torch.no_grad():
            for table_id, weight in enumerate(
                weights
            ):
                table_size = int(
                    table_sizes[
                        table_id
                    ]
                )

                for row_start in range(
                    0,
                    table_size,
                    chunk_rows,
                ):
                    row_end = min(
                        row_start
                        + chunk_rows,
                        table_size,
                    )

                    rows_in_chunk = (
                        row_end
                        - row_start
                    )

                    storage_slots = (
                        layout
                        .storage_slots_for_runtime_range(
                            table_id,
                            row_start,
                            row_end,
                        )
                    )

                    if storage_slots.shape != (
                        rows_in_chunk,
                    ):
                        raise RuntimeError(
                            "checkpoint layout returned "
                            "unexpected storage-slot shape"
                        )

                    if rows_in_chunk:
                        if (
                            np.any(
                                storage_slots < 0
                            )
                            or np.any(
                                storage_slots
                                >= layout.total_rows
                            )
                        ):
                            raise RuntimeError(
                                "checkpoint layout returned "
                                "out-of-range storage slot"
                            )

                    if (
                        layout.mode
                        == "local-only"
                    ):
                        # Local-only storage is contiguous in runtime
                        # row order.  A direct slice avoids unnecessary
                        # fancy indexing.
                        first_slot = int(
                            storage_slots[0]
                        )

                        host_rows = np.array(
                            storage_rows[
                                first_slot:
                                first_slot
                                + rows_in_chunk
                            ],
                            dtype=np.float32,
                            copy=True,
                            order="C",
                        )

                    else:
                        # Global-mixed storage may scatter one runtime
                        # table across arbitrary checkpoint slots.
                        # NumPy gathers the whole chunk in one vectorized
                        # operation, with no per-row Python objects.
                        host_rows = np.array(
                            storage_rows[
                                storage_slots
                            ],
                            dtype=np.float32,
                            copy=True,
                            order="C",
                        )

                    if host_rows.shape != (
                        rows_in_chunk,
                        EMBEDDING_DIM,
                    ):
                        raise RuntimeError(
                            "unexpected restored chunk shape"
                        )

                    restored = torch.from_numpy(
                        host_rows
                    )

                    if (
                        restored.device
                        != weight.device
                    ):
                        restored = restored.to(
                            device=weight.device,
                            dtype=torch.float32,
                        )

                    weight[
                        row_start:
                        row_end
                    ].copy_(
                        restored
                    )

                    restored_rows += (
                        rows_in_chunk
                    )

                    num_chunks += 1

                    max_chunk_rows = max(
                        max_chunk_rows,
                        rows_in_chunk,
                    )

                    max_host_bytes = max(
                        max_host_bytes,
                        int(
                            host_rows.nbytes
                        ),
                    )

    finally:
        # np.memmap has no public close() method.  Close the underlying
        # mmap explicitly so repeated restores do not retain descriptors.
        mmap_handle = getattr(
            storage_rows,
            "_mmap",
            None,
        )

        if mmap_handle is not None:
            mmap_handle.close()

    if restored_rows != layout.total_rows:
        raise RuntimeError(
            "restored row count does not match layout"
        )

    return EmbeddingRestoreStats(
        num_tables=layout.num_tables,
        restored_rows=restored_rows,
        image_bytes=image_size,
        restored_payload_bytes=(
            layout.total_rows
            * ROW_BYTES
        ),
        num_chunks=num_chunks,
        max_chunk_rows=max_chunk_rows,
        max_host_bytes=max_host_bytes,
    )


@dataclass(frozen=True)
class CheckpointRestoreResult:
    """Result of restoring one application-published checkpoint."""

    generation: int
    checkpoint_path: str
    sidecar_path: str
    training_state: dict
    embedding: EmbeddingRestoreStats


def restore_published_checkpoint(
    checkpoint_dir,
    layout: CheckpointLayout,
    embedding_tables,
    *,
    chunk_rows: int = 65536,
) -> CheckpointRestoreResult:
    """Restore the latest application-published checkpoint.

    Restore order:

        manifest / sidecar validation
            ->
        embedding image restoration
            ->
        return restored training state

    The manifest is the sole authority for selecting the generation.
    The function never scans C*.img files for the largest generation.

    This function restores embedding weights but deliberately does not
    apply dense-model, optimizer, scheduler, or RNG state itself.  Those
    objects are training-loop specific and are returned through
    ``training_state`` for the later integration stage.
    """

    metadata = load_published_checkpoint(
        checkpoint_dir,
        layout,
    )

    embedding_stats = (
        restore_embedding_image(
            metadata.checkpoint_path,
            layout,
            embedding_tables,
            chunk_rows=chunk_rows,
        )
    )

    return CheckpointRestoreResult(
        generation=metadata.generation,
        checkpoint_path=metadata.checkpoint_path,
        sidecar_path=metadata.sidecar_path,
        training_state=metadata.training_state,
        embedding=embedding_stats,
    )
