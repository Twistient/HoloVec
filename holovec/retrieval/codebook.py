from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..backends.base import Array
from ..backends import get_backend, Backend


class Codebook:
    """Thin wrapper for label→vector mappings with convenience methods.

    Keeps insertion order of labels. Vectors are backend arrays.
    """

    def __init__(self, items: Optional[Dict[str, Array]] = None, backend: Optional[Backend] = None):
        self._items: Dict[str, Array] = {}
        self._backend: Backend = backend if backend is not None else get_backend("numpy")
        if items:
            self.extend(items)

    # Basic operations
    def add(self, label: str, vector: Array) -> None:
        self._items[label] = vector

    def extend(self, items: Dict[str, Array]) -> None:
        for k, v in items.items():
            self.add(k, v)

    @property
    def labels(self) -> List[str]:
        return list(self._items.keys())

    @property
    def size(self) -> int:
        return len(self._items)

    def as_list(self) -> List[Tuple[str, Array]]:
        return list(self._items.items())

    def as_matrix(self, backend: Optional[Backend] = None) -> Tuple[List[str], Array]:
        """Return (labels, matrix) where matrix has shape (L, D)."""
        be = backend or self._backend
        if self.size == 0:
            return [], be.zeros((0,), dtype="float32")
        labels = self.labels
        stacked = be.stack([self._items[lbl] for lbl in labels], axis=0)
        return labels, stacked

    # Persistence (npz)
    def save(self, path: str) -> None:
        labels, mat = self.as_matrix()
        mat_np = self._backend.to_numpy(mat)
        np.savez(path, labels=np.array(labels, dtype=object), matrix=mat_np)

    @classmethod
    def load(cls, path: str, backend: Optional[Backend] = None) -> "Codebook":
        be = backend or get_backend("numpy")
        data = np.load(path, allow_pickle=True)
        labels = [str(x) for x in data["labels"].tolist()]
        mat = data["matrix"]
        items: Dict[str, Array] = {}
        for i, lbl in enumerate(labels):
            items[lbl] = be.from_numpy(mat[i])
        return cls(items=items, backend=be)

