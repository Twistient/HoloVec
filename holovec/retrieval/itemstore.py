from collections.abc import Callable

from ..backends.base import Array
from ..models.base import VSAModel
from ..utils.cleanup import BruteForceCleanup, CleanupStrategy
from ..utils.search import (
    PreparedSearchIndex,
    nearest_neighbors,
    prepare_search_index,
    query_prepared_index,
)
from .codebook import Codebook
from .rust_search import (
    PRODUCTION_SUPPORTED_MODES,
    RustPreparedIndex,
    prepare_rust_search_from_index,
    query_rust_prepared_index,
)

SUPPORTED_SEARCH_BACKENDS = frozenset({"numpy", "rust"})


def _validate_search_backend(search_backend: str) -> str:
    if search_backend not in SUPPORTED_SEARCH_BACKENDS:
        raise ValueError(
            f"search_backend must be one of {sorted(SUPPORTED_SEARCH_BACKENDS)}, "
            f"got {search_backend!r}"
        )
    return search_backend


class ItemStore:
    """Thin retrieval wrapper around a Codebook and a CleanupStrategy.

    Provides nearest-neighbor queries and multi-factor factorization via
    the configured cleanup strategy.
    """

    def __init__(
        self,
        model: VSAModel,
        cleanup: CleanupStrategy | None = None,
        *,
        search_backend: str = "numpy",
    ) -> None:
        self.model = model
        self.cleanup: CleanupStrategy = cleanup if cleanup is not None else BruteForceCleanup()
        self.search_backend = _validate_search_backend(search_backend)
        self.codebook = Codebook(backend=model.backend)
        self._prepared_index: PreparedSearchIndex | None = None
        self._prepared_index_version = -1
        self._rust_index: RustPreparedIndex | None = None
        self._rust_index_version = -1

    def _invalidate_search_cache(self) -> None:
        self._prepared_index = None
        self._prepared_index_version = -1
        self._rust_index = None
        self._rust_index_version = -1

    def _get_prepared_index(self) -> PreparedSearchIndex:
        if self._prepared_index is None or self._prepared_index_version != self.codebook.version:
            self._prepared_index = prepare_search_index(self.codebook._items, self.model)
            self._prepared_index_version = self.codebook.version
        return self._prepared_index

    def _get_rust_index(self) -> RustPreparedIndex:
        prepared_index = self._get_prepared_index()
        if prepared_index.mode not in PRODUCTION_SUPPORTED_MODES:
            raise NotImplementedError(
                f"Rust search_backend does not support prepared mode {prepared_index.mode!r}"
            )
        if self._rust_index is None or self._rust_index_version != self.codebook.version:
            self._rust_index = prepare_rust_search_from_index(prepared_index, self.model)
            self._rust_index_version = self.codebook.version
        return self._rust_index

    def _format_query_results(
        self,
        labels: list[str],
        similarities: list[float] | None,
        *,
        return_similarities: bool,
    ) -> list[tuple[str, float]]:
        return (
            list(zip(labels, similarities or [], strict=True))
            if return_similarities
            else [(lbl, 0.0) for lbl in labels]
        )

    def _query_prepared(
        self,
        query_fn: Callable[[Array, int, bool], tuple[list[str], list[float] | None]],
        vec: Array,
        *,
        k: int,
        return_similarities: bool,
    ) -> list[tuple[str, float]]:
        labels, sims = query_fn(vec, k, return_similarities)
        return self._format_query_results(
            labels,
            sims,
            return_similarities=return_similarities,
        )

    def fit(self, items: dict[str, Array] | Codebook) -> "ItemStore":
        if isinstance(items, Codebook):
            self.codebook = items
        else:
            self.codebook = Codebook(items, backend=self.model.backend)
        self._invalidate_search_cache()
        return self

    def add(self, label: str, vector: Array) -> None:
        self.codebook.add(label, vector)
        self._invalidate_search_cache()

    def extend(self, items: dict[str, Array]) -> None:
        self.codebook.extend(items)
        self._invalidate_search_cache()

    def query(
        self,
        vec: Array,
        k: int = 1,
        return_similarities: bool = True,
        fast: bool = True,
    ) -> list[tuple[str, float]]:
        """Query top-k nearest items.

        If fast=True, uses a batched matrix routine when possible, otherwise
        falls back to scalar nearest_neighbors.
        """
        if fast and self.codebook.size > 0:
            if self.search_backend == "rust":
                try:
                    return self._query_prepared(
                        lambda query, k, return_similarities: query_rust_prepared_index(
                            query,
                            self._get_rust_index(),
                            k=k,
                            return_similarities=return_similarities,
                        ),
                        vec,
                        k=k,
                        return_similarities=return_similarities,
                    )
                except (
                    FileNotFoundError,
                    NotImplementedError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    # Fall back to the exact prepared NumPy path when Rust is unavailable.
                    pass
            try:
                return self._query_prepared(
                    lambda query, k, return_similarities: query_prepared_index(
                        query,
                        self._get_prepared_index(),
                        self.model,
                        k=k,
                        return_similarities=return_similarities,
                    ),
                    vec,
                    k=k,
                    return_similarities=return_similarities,
                )
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                # Fallback to scalar path on any backend issues
                pass

        labels, sims = nearest_neighbors(
            vec, self.codebook._items, self.model, k=k, return_similarities=True
        )
        return (
            list(zip(labels, sims or [], strict=True))
            if return_similarities
            else [(lbl, 0.0) for lbl in labels]
        )

    def factorize(
        self,
        vec: Array,
        n_factors: int,
        max_iterations: int | None = None,
        threshold: float | None = None,
        temperature: float = 20.0,
        top_k: int = 1,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = "hard",
    ) -> tuple[list[str], list[float]]:
        return self.cleanup.factorize(
            vec,
            self.codebook._items,
            self.model,
            n_factors=n_factors,
            max_iterations=max_iterations,
            threshold=threshold,
            temperature=temperature,
            top_k=top_k,
            patience=patience,
            min_delta=min_delta,
            mode=mode,
        )

    # Persistence delegates to Codebook
    def save(self, path: str) -> None:
        self.codebook.save(path)

    @classmethod
    def load(
        cls,
        model: VSAModel,
        path: str,
        cleanup: CleanupStrategy | None = None,
        *,
        search_backend: str = "numpy",
        allow_unsafe_legacy: bool = False,
    ) -> "ItemStore":
        store = cls(model=model, cleanup=cleanup, search_backend=search_backend)
        store.codebook = Codebook.load(
            path,
            backend=model.backend,
            allow_unsafe_legacy=allow_unsafe_legacy,
        )
        return store
