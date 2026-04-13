"""Helpers for the isolated Rust retrieval prototype."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from holovec.backends.base import Array
from holovec.models.base import VSAModel
from holovec.utils.search import PreparedSearchIndex, prepare_search_index

REPO_ROOT = Path(__file__).resolve().parent.parent
RUST_SEARCH_MANIFEST = REPO_ROOT / "prototypes" / "rust_search" / "Cargo.toml"


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libholovec_rust_search.dylib"
    if sys.platform.startswith("win"):
        return "holovec_rust_search.dll"
    return "libholovec_rust_search.so"


def rust_search_library_path(*, release: bool = True) -> Path:
    profile = "release" if release else "debug"
    return REPO_ROOT / "prototypes" / "rust_search" / "target" / profile / _library_filename()


def build_rust_search_library(*, release: bool = True) -> Path:
    cmd = ["cargo", "build", "--manifest-path", str(RUST_SEARCH_MANIFEST)]
    if release:
        cmd.append("--release")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return rust_search_library_path(release=release)


_LIBRARY: ctypes.CDLL | None = None


def _load_library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY

    path = rust_search_library_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Rust search library not built: {path}. Run with --build-rust first."
        )

    library = ctypes.CDLL(str(path))

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int8_p = ctypes.POINTER(ctypes.c_int8)
    c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
    c_uint32_p = ctypes.POINTER(ctypes.c_uint32)
    c_size_t_p = ctypes.POINTER(ctypes.c_size_t)

    library.holovec_continuous_index_new.argtypes = [c_float_p, ctypes.c_size_t, ctypes.c_size_t]
    library.holovec_continuous_index_new.restype = ctypes.c_void_p
    library.holovec_continuous_index_query.argtypes = [
        ctypes.c_void_p,
        c_float_p,
        ctypes.c_size_t,
        c_size_t_p,
        c_float_p,
    ]
    library.holovec_continuous_index_query.restype = ctypes.c_int
    library.holovec_continuous_index_free.argtypes = [ctypes.c_void_p]
    library.holovec_continuous_index_free.restype = None

    library.holovec_discrete_index_new.argtypes = [c_int8_p, ctypes.c_size_t, ctypes.c_size_t]
    library.holovec_discrete_index_new.restype = ctypes.c_void_p
    library.holovec_discrete_index_query.argtypes = [
        ctypes.c_void_p,
        c_int8_p,
        ctypes.c_size_t,
        c_size_t_p,
        c_float_p,
    ]
    library.holovec_discrete_index_query.restype = ctypes.c_int
    library.holovec_discrete_index_free.argtypes = [ctypes.c_void_p]
    library.holovec_discrete_index_free.restype = None

    library.holovec_sparse_index_new.argtypes = [c_uint8_p, ctypes.c_size_t, ctypes.c_size_t]
    library.holovec_sparse_index_new.restype = ctypes.c_void_p
    library.holovec_sparse_index_query.argtypes = [
        ctypes.c_void_p,
        c_uint8_p,
        ctypes.c_size_t,
        c_size_t_p,
        c_float_p,
    ]
    library.holovec_sparse_index_query.restype = ctypes.c_int
    library.holovec_sparse_index_free.argtypes = [ctypes.c_void_p]
    library.holovec_sparse_index_free.restype = None

    library.holovec_segment_index_new.argtypes = [c_uint32_p, ctypes.c_size_t, ctypes.c_size_t]
    library.holovec_segment_index_new.restype = ctypes.c_void_p
    library.holovec_segment_index_query.argtypes = [
        ctypes.c_void_p,
        c_uint32_p,
        ctypes.c_size_t,
        c_size_t_p,
        c_float_p,
    ]
    library.holovec_segment_index_query.restype = ctypes.c_int
    library.holovec_segment_index_free.argtypes = [ctypes.c_void_p]
    library.holovec_segment_index_free.restype = None

    _LIBRARY = library
    return library


EncodeQuery = Callable[[object], NDArray[Any]]


class RustPreparedIndex:
    """Python wrapper around a built Rust prepared-search index."""

    def __init__(
        self,
        *,
        labels: list[str],
        handle: int,
        query_fn: Any,
        free_fn: Any,
        query_dtype: type[Any],
        encode_query: EncodeQuery,
    ) -> None:
        self.labels = labels
        self._handle = handle
        self._query_fn = query_fn
        self._query_dtype = query_dtype
        self._encode_query = encode_query
        self._finalizer = weakref.finalize(self, free_fn, ctypes.c_void_p(handle))

    def query(self, query: object, *, k: int) -> tuple[list[str], list[float]]:
        encoded_query = self._encode_query(query)
        out_indices: NDArray[np.uintp] = np.empty(k, dtype=np.uintp)
        out_scores: NDArray[np.float32] = np.empty(k, dtype=np.float32)
        status = self._query_fn(
            ctypes.c_void_p(self._handle),
            encoded_query.ctypes.data_as(ctypes.POINTER(self._query_dtype)),
            ctypes.c_size_t(k),
            out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status != 0:
            raise RuntimeError(f"Rust query failed with status {status}")
        labels = [self.labels[int(index)] for index in out_indices.tolist()]
        return labels, [float(score) for score in out_scores.tolist()]


def _continuous_query_array(query: object) -> NDArray[Any]:
    return cast(NDArray[Any], np.ascontiguousarray(np.asarray(query, dtype=np.float32)))


def _discrete_query_array(query: object) -> NDArray[Any]:
    return cast(NDArray[Any], np.ascontiguousarray(np.asarray(query, dtype=np.int8)))


def _sparse_query_array(query: object) -> NDArray[Any]:
    return cast(NDArray[Any], np.ascontiguousarray(np.asarray(query, dtype=np.uint8)))


def _segment_query_array(model: VSAModel, query: object) -> NDArray[Any]:
    pattern = model.space.segment_argmax(query)
    return cast(NDArray[Any], np.ascontiguousarray(np.asarray(pattern, dtype=np.uint32)))


def prepare_rust_search_index(
    codebook: dict[str, Array],
    model: VSAModel,
) -> RustPreparedIndex:
    return prepare_rust_search_from_index(prepare_search_index(codebook, model), model)


def prepare_rust_search_from_index(
    index: PreparedSearchIndex,
    model: VSAModel,
) -> RustPreparedIndex:
    library = _load_library()

    if index.mode == "continuous":
        matrix = np.ascontiguousarray(
            np.asarray(model.backend.to_numpy(index.matrix), dtype=np.float32)
        )
        handle = library.holovec_continuous_index_new(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(matrix.shape[0]),
            ctypes.c_size_t(matrix.shape[1]),
        )
        if not handle:
            raise RuntimeError("Failed to build Rust continuous index")
        return RustPreparedIndex(
            labels=index.labels,
            handle=int(handle),
            query_fn=library.holovec_continuous_index_query,
            free_fn=library.holovec_continuous_index_free,
            query_dtype=ctypes.c_float,
            encode_query=lambda query: _continuous_query_array(model.backend.to_numpy(query)),
        )

    if index.mode == "discrete":
        matrix = np.ascontiguousarray(
            np.asarray(model.backend.to_numpy(index.matrix), dtype=np.int8)
        )
        handle = library.holovec_discrete_index_new(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ctypes.c_size_t(matrix.shape[0]),
            ctypes.c_size_t(matrix.shape[1]),
        )
        if not handle:
            raise RuntimeError("Failed to build Rust discrete index")
        return RustPreparedIndex(
            labels=index.labels,
            handle=int(handle),
            query_fn=library.holovec_discrete_index_query,
            free_fn=library.holovec_discrete_index_free,
            query_dtype=ctypes.c_int8,
            encode_query=lambda query: _discrete_query_array(model.backend.to_numpy(query)),
        )

    if index.mode == "sparse":
        matrix = np.ascontiguousarray(
            np.asarray(model.backend.to_numpy(index.matrix), dtype=np.uint8)
        )
        handle = library.holovec_sparse_index_new(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(matrix.shape[0]),
            ctypes.c_size_t(matrix.shape[1]),
        )
        if not handle:
            raise RuntimeError("Failed to build Rust sparse index")
        return RustPreparedIndex(
            labels=index.labels,
            handle=int(handle),
            query_fn=library.holovec_sparse_index_query,
            free_fn=library.holovec_sparse_index_free,
            query_dtype=ctypes.c_uint8,
            encode_query=lambda query: _sparse_query_array(model.backend.to_numpy(query)),
        )

    if index.mode == "sparse_segment":
        if index.segment_patterns is None:
            raise ValueError("Prepared segmented index is missing segment patterns")
        patterns = np.ascontiguousarray(np.asarray(index.segment_patterns, dtype=np.uint32))
        handle = library.holovec_segment_index_new(
            patterns.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_size_t(patterns.shape[0]),
            ctypes.c_size_t(patterns.shape[1]),
        )
        if not handle:
            raise RuntimeError("Failed to build Rust segmented index")
        return RustPreparedIndex(
            labels=index.labels,
            handle=int(handle),
            query_fn=library.holovec_segment_index_query,
            free_fn=library.holovec_segment_index_free,
            query_dtype=ctypes.c_uint32,
            encode_query=lambda query: _segment_query_array(model, query),
        )

    raise NotImplementedError(f"Rust prototype does not support prepared mode {index.mode!r}")
