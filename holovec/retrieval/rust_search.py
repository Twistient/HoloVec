"""Optional Rust-backed prepared retrieval kernels.

This module provides a narrow runtime bridge to the cargo-based retrieval
prototype under ``prototypes/rust_search/``. It is intentionally optional:

- the default retrieval path remains exact prepared NumPy
- callers opt in by requesting ``search_backend="rust"``
- unsupported modes or missing native artifacts fall back to NumPy
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..backends.base import Array
from ..models.base import VSAModel
from ..spaces.spaces import SparseSegmentSpace
from ..utils.search import PreparedSearchIndex, prepare_search_index

RUST_SEARCH_LIBRARY_ENV = "HOLOVEC_RUST_SEARCH_LIBRARY"
RUST_SEARCH_MANIFEST_ENV = "HOLOVEC_RUST_SEARCH_MANIFEST"
PRODUCTION_SUPPORTED_MODES = frozenset({"discrete", "sparse", "sparse_segment"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libholovec_rust_search.dylib"
    if sys.platform.startswith("win"):
        return "holovec_rust_search.dll"
    return "libholovec_rust_search.so"


def rust_search_manifest_path() -> Path:
    configured = os.getenv(RUST_SEARCH_MANIFEST_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return _repo_root() / "prototypes" / "rust_search" / "Cargo.toml"


def rust_search_library_path(*, release: bool = True) -> Path:
    configured = os.getenv(RUST_SEARCH_LIBRARY_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    profile = "release" if release else "debug"
    return _repo_root() / "prototypes" / "rust_search" / "target" / profile / _library_filename()


def build_rust_search_library(*, release: bool = True) -> Path:
    """Build the optional Rust retrieval library and return its path."""
    manifest_path = rust_search_manifest_path()
    cmd = ["cargo", "build", "--manifest-path", str(manifest_path)]
    if release:
        cmd.append("--release")
    subprocess.run(cmd, cwd=manifest_path.parent.parent, check=True)
    return rust_search_library_path(release=release)


def supports_rust_search_mode(mode: str, *, production_only: bool = False) -> bool:
    """Return whether the Rust bridge supports a prepared-search mode."""
    if production_only:
        return mode in PRODUCTION_SUPPORTED_MODES
    return mode in {"continuous", *PRODUCTION_SUPPORTED_MODES}


_LIBRARY: ctypes.CDLL | None = None


def _load_library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY

    path = rust_search_library_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Rust search library not built: {path}. "
            "Build it with holovec.retrieval.rust_search.build_rust_search_library()."
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


EncodeQuery = Callable[[Array], NDArray[Any]]


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

    def query(self, query: Array, *, k: int) -> tuple[list[str], list[float]]:
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


def query_rust_prepared_index(
    query: Array,
    index: RustPreparedIndex,
    *,
    k: int,
    return_similarities: bool = True,
) -> tuple[list[str], list[float] | None]:
    """Query a cached Rust prepared-search index."""
    labels, scores = index.query(query, k=k)
    if return_similarities:
        return labels, scores
    return labels, None


def _continuous_query_array(query: Array) -> NDArray[Any]:
    return np.ascontiguousarray(np.asarray(query, dtype=np.float32))


def _discrete_query_array(query: Array) -> NDArray[Any]:
    return np.ascontiguousarray(np.asarray(query, dtype=np.int8))


def _sparse_query_array(query: Array) -> NDArray[Any]:
    return np.ascontiguousarray(np.asarray(query, dtype=np.uint8))


def _segment_query_array(model: VSAModel, query: Array) -> NDArray[Any]:
    pattern = cast(SparseSegmentSpace, model.space).segment_argmax(query)
    return np.ascontiguousarray(np.asarray(pattern, dtype=np.uint32))


def prepare_rust_search_index(
    codebook: dict[str, Array],
    model: VSAModel,
) -> RustPreparedIndex:
    """Prepare a Rust-backed exact-search index from a codebook."""
    return prepare_rust_search_from_index(prepare_search_index(codebook, model), model)


def prepare_rust_search_from_index(
    index: PreparedSearchIndex,
    model: VSAModel,
) -> RustPreparedIndex:
    """Prepare a Rust-backed exact-search index from prepared search state."""
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
