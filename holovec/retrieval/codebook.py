from collections.abc import ItemsView, Iterator, KeysView, ValuesView

import numpy as np

from ..backends import Backend, get_backend
from ..backends.base import Array


class Codebook:
    """Thin wrapper for label→vector mappings with convenience methods.

    Keeps insertion order of labels. Vectors are backend arrays.
    """

    FORMAT_VERSION = 2

    def __init__(self, items: dict[str, Array] | None = None, backend: Backend | None = None):
        self._items: dict[str, Array] = {}
        self._backend: Backend = backend if backend is not None else get_backend("numpy")
        self._version = 0
        if items:
            self.extend(items)

    # Basic operations
    def add(self, label: str, vector: Array) -> None:
        self._items[label] = vector
        self._version += 1

    def extend(self, items: dict[str, Array]) -> None:
        for k, v in items.items():
            self.add(k, v)

    @property
    def labels(self) -> list[str]:
        return list(self._items.keys())

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def version(self) -> int:
        """Return the current mutation counter for cache invalidation."""
        return self._version

    # Dict-like interface
    def __getitem__(self, label: str) -> Array:
        """Get vector by label. Raises KeyError if not found."""
        return self._items[label]

    def __contains__(self, label: str) -> bool:
        """Check if label exists in codebook."""
        return label in self._items

    def __len__(self) -> int:
        """Return number of items in codebook."""
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        """Iterate over labels."""
        return iter(self._items)

    def items(self) -> ItemsView[str, Array]:
        """Return iterator over (label, vector) pairs."""
        return self._items.items()

    def keys(self) -> KeysView[str]:
        """Return iterator over labels."""
        return self._items.keys()

    def values(self) -> ValuesView[Array]:
        """Return iterator over vectors."""
        return self._items.values()

    def get(self, label: str, default: Array | None = None) -> Array | None:
        """Get vector by label, returning default if not found."""
        return self._items.get(label, default)

    def as_list(self) -> list[tuple[str, Array]]:
        return list(self._items.items())

    def as_matrix(self, backend: Backend | None = None) -> tuple[list[str], Array]:
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
        np.savez(
            path,
            format_version=np.array(self.FORMAT_VERSION, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.str_),
            matrix=mat_np,
        )

    @classmethod
    def load(
        cls,
        path: str,
        backend: Backend | None = None,
        *,
        allow_unsafe_legacy: bool = False,
    ) -> "Codebook":
        be = backend or get_backend("numpy")
        try:
            with np.load(path, allow_pickle=False) as data:
                format_version = int(np.asarray(data["format_version"]).item())
                if format_version != cls.FORMAT_VERSION:
                    raise ValueError(
                        f"Unsupported codebook format version {format_version}; "
                        f"expected {cls.FORMAT_VERSION}"
                    )
                labels = [str(x) for x in data["labels"].tolist()]
                mat = data["matrix"]
        except KeyError as exc:
            if not allow_unsafe_legacy:
                raise ValueError(
                    "Legacy codebook files require unsafe pickle-backed loading. "
                    "Re-run with allow_unsafe_legacy=True, then re-save the file."
                ) from exc
            with np.load(path, allow_pickle=True) as data:
                labels = [str(x) for x in data["labels"].tolist()]
                mat = data["matrix"]
        except ValueError as exc:
            message = str(exc)
            is_pickle_error = "Object arrays cannot be loaded when allow_pickle=False" in message
            if not is_pickle_error or not allow_unsafe_legacy:
                if is_pickle_error:
                    raise ValueError(
                        "Legacy codebook files require unsafe pickle-backed loading. "
                        "Re-run with allow_unsafe_legacy=True, then re-save the file."
                    ) from exc
                raise
            with np.load(path, allow_pickle=True) as data:
                labels = [str(x) for x in data["labels"].tolist()]
                mat = data["matrix"]

        items: dict[str, Array] = {}
        for i, lbl in enumerate(labels):
            items[lbl] = be.from_numpy(mat[i])
        return cls(items=items, backend=be)
