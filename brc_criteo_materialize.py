import hashlib
import os
import tempfile
import zipfile

import numpy as np


_EXPECTED_MEMBERS = (
    "X_cat.npy",
    "X_int.npy",
    "y.npy",
)


def _read_exact(stream, num_bytes):
    parts = []
    remaining = int(num_bytes)

    while remaining:
        piece = stream.read(remaining)

        if not piece:
            raise ValueError(
                "truncated NPY payload"
            )

        parts.append(piece)
        remaining -= len(piece)

    return b"".join(parts)


def _read_npy_header(stream):
    version = np.lib.format.read_magic(stream)

    if version == (1, 0):
        shape, fortran_order, dtype = (
            np.lib.format.read_array_header_1_0(
                stream
            )
        )
    elif version == (2, 0):
        shape, fortran_order, dtype = (
            np.lib.format.read_array_header_2_0(
                stream
            )
        )
    else:
        raise ValueError(
            "unsupported NPY version {}".format(
                version
            )
        )

    return (
        tuple(shape),
        bool(fortran_order),
        np.dtype(dtype),
    )


def _write_npy_header(
    stream,
    shape,
    dtype,
):
    header = {
        "descr": np.lib.format.dtype_to_descr(
            np.dtype(dtype)
        ),
        "fortran_order": False,
        "shape": tuple(shape),
    }

    np.lib.format.write_array_header_1_0(
        stream,
        header,
    )


def _normalize_categorical(array):
    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            "X_cat chunk must be two-dimensional"
        )

    if np.issubdtype(
        array.dtype,
        np.floating,
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(
                "X_cat contains non-finite values"
            )

        ids = array.astype(
            np.int64
        )

        if not np.all(array == ids):
            raise ValueError(
                "X_cat contains non-integral values"
            )

        return ids

    if np.issubdtype(
        array.dtype,
        np.integer,
    ):
        if np.issubdtype(
            array.dtype,
            np.unsignedinteger,
        ):
            if (
                array.size
                and int(array.max())
                > np.iinfo(np.int64).max
            ):
                raise ValueError(
                    "X_cat cannot be represented "
                    "as int64"
                )

        return array.astype(
            np.int64,
            copy=False,
        )

    raise TypeError(
        "X_cat must have numeric integer-like dtype"
    )


def _validate_permutations(old_to_new):
    if not isinstance(
        old_to_new,
        (list, tuple),
    ):
        raise TypeError(
            "old_to_new must be a list or tuple"
        )

    if not old_to_new:
        raise ValueError(
            "old_to_new must not be empty"
        )

    result = []

    for table_id, permutation in enumerate(
        old_to_new
    ):
        permutation = np.asarray(
            permutation
        )

        if permutation.ndim != 1:
            raise ValueError(
                "permutation {} must be "
                "one-dimensional".format(
                    table_id
                )
            )

        if not np.issubdtype(
            permutation.dtype,
            np.integer,
        ):
            raise TypeError(
                "permutation {} must contain "
                "integers".format(
                    table_id
                )
            )

        permutation = permutation.astype(
            np.int64,
            copy=False,
        )

        num_rows = len(permutation)

        if num_rows <= 0:
            raise ValueError(
                "permutation {} must not be "
                "empty".format(
                    table_id
                )
            )

        expected = np.arange(
            num_rows,
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(permutation),
            expected,
        ):
            raise ValueError(
                "permutation {} is invalid".format(
                    table_id
                )
            )

        result.append(permutation)

    return result


def _copy_zip_member(
    source_archive,
    target_archive,
    member_name,
    chunk_bytes=8 * 1024 * 1024,
):
    with source_archive.open(
        member_name,
        "r",
    ) as source:
        with target_archive.open(
            member_name,
            "w",
            force_zip64=True,
        ) as target:
            while True:
                piece = source.read(
                    chunk_bytes
                )

                if not piece:
                    break

                target.write(piece)


def _hash_zip_member(
    archive,
    member_name,
):
    digest = hashlib.sha256()

    with archive.open(
        member_name,
        "r",
    ) as stream:
        while True:
            piece = stream.read(
                8 * 1024 * 1024
            )

            if not piece:
                break

            digest.update(piece)

    return digest.digest()


def _check_member_layout(archive):
    names = tuple(
        archive.namelist()
    )

    if names != _EXPECTED_MEMBERS:
        raise ValueError(
            "unexpected NPZ member layout: "
            "{}".format(names)
        )


def _remap_x_cat_member(
    source_archive,
    target_archive,
    old_to_new,
    chunk_rows,
):
    with source_archive.open(
        "X_cat.npy",
        "r",
    ) as source:
        (
            shape,
            fortran_order,
            dtype,
        ) = _read_npy_header(
            source
        )

        if len(shape) != 2:
            raise ValueError(
                "X_cat must be two-dimensional"
            )

        if fortran_order:
            raise ValueError(
                "Fortran-ordered X_cat is not "
                "supported"
            )

        num_samples, num_tables = shape

        if num_tables != len(old_to_new):
            raise ValueError(
                "X_cat table count {} does not "
                "match permutation count {}".format(
                    num_tables,
                    len(old_to_new),
                )
            )

        if dtype.hasobject:
            raise TypeError(
                "object-dtype X_cat is not supported"
            )

        row_bytes = (
            num_tables
            * dtype.itemsize
        )

        with target_archive.open(
            "X_cat.npy",
            "w",
            force_zip64=True,
        ) as target:
            _write_npy_header(
                target,
                shape=shape,
                dtype=dtype,
            )

            consumed_rows = 0

            while consumed_rows < num_samples:
                rows = min(
                    chunk_rows,
                    num_samples - consumed_rows,
                )

                payload = _read_exact(
                    source,
                    rows * row_bytes,
                )

                source_chunk = np.frombuffer(
                    payload,
                    dtype=dtype,
                    count=rows * num_tables,
                ).reshape(
                    rows,
                    num_tables,
                )

                ids = _normalize_categorical(
                    source_chunk
                )

                for table_id in range(
                    num_tables
                ):
                    column = ids[
                        :,
                        table_id,
                    ]

                    num_rows = len(
                        old_to_new[table_id]
                    )

                    if (
                        np.any(column < 0)
                        or np.any(
                            column >= num_rows
                        )
                    ):
                        raise ValueError(
                            "X_cat ID out of range "
                            "for table {}".format(
                                table_id
                            )
                        )

                    ids[
                        :,
                        table_id,
                    ] = old_to_new[
                        table_id
                    ][column]

                output_chunk = ids.astype(
                    dtype,
                    copy=False,
                )

                target.write(
                    output_chunk.tobytes(
                        order="C"
                    )
                )

                consumed_rows += rows

            if source.read(1):
                raise ValueError(
                    "unexpected trailing X_cat bytes"
                )


def validate_criteo_day_materialization(
    source_path,
    derived_path,
    old_to_new,
    chunk_rows=65536,
):
    old_to_new = _validate_permutations(
        old_to_new
    )

    if isinstance(chunk_rows, bool) or not isinstance(
        chunk_rows,
        (int, np.integer),
    ):
        raise TypeError(
            "chunk_rows must be an integer"
        )

    chunk_rows = int(chunk_rows)

    if chunk_rows <= 0:
        raise ValueError(
            "chunk_rows must be positive"
        )

    with zipfile.ZipFile(
        source_path,
        "r",
    ) as source_archive:
        with zipfile.ZipFile(
            derived_path,
            "r",
        ) as derived_archive:
            _check_member_layout(
                source_archive
            )
            _check_member_layout(
                derived_archive
            )

            for member_name in (
                "X_int.npy",
                "y.npy",
            ):
                source_hash = (
                    _hash_zip_member(
                        source_archive,
                        member_name,
                    )
                )

                derived_hash = (
                    _hash_zip_member(
                        derived_archive,
                        member_name,
                    )
                )

                if source_hash != derived_hash:
                    raise ValueError(
                        "{} changed during "
                        "materialization".format(
                            member_name
                        )
                    )

            with source_archive.open(
                "X_cat.npy",
                "r",
            ) as source:
                with derived_archive.open(
                    "X_cat.npy",
                    "r",
                ) as derived:
                    (
                        source_shape,
                        source_fortran,
                        source_dtype,
                    ) = _read_npy_header(
                        source
                    )

                    (
                        derived_shape,
                        derived_fortran,
                        derived_dtype,
                    ) = _read_npy_header(
                        derived
                    )

                    if (
                        source_shape
                        != derived_shape
                    ):
                        raise ValueError(
                            "X_cat shape changed"
                        )

                    if (
                        source_dtype
                        != derived_dtype
                    ):
                        raise ValueError(
                            "X_cat dtype changed"
                        )

                    if (
                        source_fortran
                        or derived_fortran
                    ):
                        raise ValueError(
                            "Fortran-ordered X_cat "
                            "is not supported"
                        )

                    if len(source_shape) != 2:
                        raise ValueError(
                            "X_cat must be "
                            "two-dimensional"
                        )

                    (
                        num_samples,
                        num_tables,
                    ) = source_shape

                    if (
                        num_tables
                        != len(old_to_new)
                    ):
                        raise ValueError(
                            "X_cat table count does "
                            "not match metadata"
                        )

                    row_bytes = (
                        num_tables
                        * source_dtype.itemsize
                    )

                    consumed_rows = 0

                    while (
                        consumed_rows
                        < num_samples
                    ):
                        rows = min(
                            chunk_rows,
                            num_samples
                            - consumed_rows,
                        )

                        source_payload = (
                            _read_exact(
                                source,
                                rows * row_bytes,
                            )
                        )

                        derived_payload = (
                            _read_exact(
                                derived,
                                rows * row_bytes,
                            )
                        )

                        source_chunk = (
                            np.frombuffer(
                                source_payload,
                                dtype=source_dtype,
                            ).reshape(
                                rows,
                                num_tables,
                            )
                        )

                        derived_chunk = (
                            np.frombuffer(
                                derived_payload,
                                dtype=derived_dtype,
                            ).reshape(
                                rows,
                                num_tables,
                            )
                        )

                        source_ids = (
                            _normalize_categorical(
                                source_chunk
                            )
                        )

                        derived_ids = (
                            _normalize_categorical(
                                derived_chunk
                            )
                        )

                        for table_id in range(
                            num_tables
                        ):
                            source_column = (
                                source_ids[
                                    :,
                                    table_id,
                                ]
                            )

                            num_rows = len(
                                old_to_new[
                                    table_id
                                ]
                            )

                            if (
                                np.any(
                                    source_column
                                    < 0
                                )
                                or np.any(
                                    source_column
                                    >= num_rows
                                )
                            ):
                                raise ValueError(
                                    "source X_cat ID "
                                    "out of range for "
                                    "table {}".format(
                                        table_id
                                    )
                                )

                            expected = (
                                old_to_new[
                                    table_id
                                ][
                                    source_column
                                ]
                            )

                            if not np.array_equal(
                                derived_ids[
                                    :,
                                    table_id,
                                ],
                                expected,
                            ):
                                raise ValueError(
                                    "derived X_cat "
                                    "mapping mismatch "
                                    "for table {}".format(
                                        table_id
                                    )
                                )

                        consumed_rows += rows

                    if source.read(1):
                        raise ValueError(
                            "unexpected trailing "
                            "source X_cat bytes"
                        )

                    if derived.read(1):
                        raise ValueError(
                            "unexpected trailing "
                            "derived X_cat bytes"
                        )

    return True


def _fsync_directory(path):
    directory_fd = os.open(
        path,
        os.O_RDONLY,
    )

    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def materialize_criteo_day(
    source_path,
    output_path,
    old_to_new,
    chunk_rows=65536,
):
    old_to_new = _validate_permutations(
        old_to_new
    )

    if isinstance(chunk_rows, bool) or not isinstance(
        chunk_rows,
        (int, np.integer),
    ):
        raise TypeError(
            "chunk_rows must be an integer"
        )

    chunk_rows = int(chunk_rows)

    if chunk_rows <= 0:
        raise ValueError(
            "chunk_rows must be positive"
        )

    source_path = os.path.abspath(
        source_path
    )

    output_path = os.path.abspath(
        output_path
    )

    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            source_path
        )

    output_dir = os.path.dirname(
        output_path
    )

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            output_dir
        )

    if os.path.exists(output_path):
        raise FileExistsError(
            "refusing to overwrite existing "
            "output: {}".format(
                output_path
            )
        )

    temp_fd, temp_path = tempfile.mkstemp(
        prefix="." + os.path.basename(
            output_path
        ) + ".",
        suffix=".tmp",
        dir=output_dir,
    )

    os.close(temp_fd)

    try:
        with zipfile.ZipFile(
            source_path,
            "r",
        ) as source_archive:
            _check_member_layout(
                source_archive
            )

            with zipfile.ZipFile(
                temp_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as target_archive:
                _remap_x_cat_member(
                    source_archive,
                    target_archive,
                    old_to_new=old_to_new,
                    chunk_rows=chunk_rows,
                )

                _copy_zip_member(
                    source_archive,
                    target_archive,
                    "X_int.npy",
                )

                _copy_zip_member(
                    source_archive,
                    target_archive,
                    "y.npy",
                )

        with open(
            temp_path,
            "rb",
        ) as stream:
            os.fsync(
                stream.fileno()
            )

        validate_criteo_day_materialization(
            source_path=source_path,
            derived_path=temp_path,
            old_to_new=old_to_new,
            chunk_rows=chunk_rows,
        )

        os.replace(
            temp_path,
            output_path,
        )

        _fsync_directory(
            output_dir
        )

    except Exception:
        try:
            os.unlink(
                temp_path
            )
        except FileNotFoundError:
            pass

        raise

    return output_path
