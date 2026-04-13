from ..backends.base import Array
from ..models.base import VSAModel
from ..utils.search import PreparedSearchIndex, prepare_search_index, query_prepared_index
from .codebook import Codebook
from .itemstore import _validate_search_backend
from .rust_search import (
    PRODUCTION_SUPPORTED_MODES,
    RustPreparedIndex,
    prepare_rust_search_from_index,
    query_rust_prepared_index,
)


class AssocStore:
    """Lean heteroassociative store: keys → values via aligned codebooks.

    Stores two codebooks with aligned label order. Query by a key vector returns
    the best-matching key label and its corresponding value label/vector.
    """

    def __init__(self, model: VSAModel, *, search_backend: str = "numpy") -> None:
        self.model = model
        self.search_backend = _validate_search_backend(search_backend)
        self.keys = Codebook(backend=model.backend)
        self.values = Codebook(backend=model.backend)
        self._prepared_key_index: PreparedSearchIndex | None = None
        self._prepared_key_index_version = -1
        self._rust_key_index: RustPreparedIndex | None = None
        self._rust_key_index_version = -1

    def _invalidate_search_cache(self) -> None:
        self._prepared_key_index = None
        self._prepared_key_index_version = -1
        self._rust_key_index = None
        self._rust_key_index_version = -1

    def _get_prepared_key_index(self) -> PreparedSearchIndex:
        if (
            self._prepared_key_index is None
            or self._prepared_key_index_version != self.keys.version
        ):
            self._prepared_key_index = prepare_search_index(self.keys._items, self.model)
            self._prepared_key_index_version = self.keys.version
        return self._prepared_key_index

    def _get_rust_key_index(self) -> RustPreparedIndex:
        prepared_index = self._get_prepared_key_index()
        if prepared_index.mode not in PRODUCTION_SUPPORTED_MODES:
            raise NotImplementedError(
                f"Rust search_backend does not support prepared mode {prepared_index.mode!r}"
            )
        if self._rust_key_index is None or self._rust_key_index_version != self.keys.version:
            self._rust_key_index = prepare_rust_search_from_index(prepared_index, self.model)
            self._rust_key_index_version = self.keys.version
        return self._rust_key_index

    def fit(self, key_items: dict[str, Array], value_items: dict[str, Array]) -> "AssocStore":
        # Intersect labels and preserve deterministic order
        labels = [lbl for lbl in key_items.keys() if lbl in value_items]
        self.keys = Codebook({lbl: key_items[lbl] for lbl in labels}, backend=self.model.backend)
        self.values = Codebook(
            {lbl: value_items[lbl] for lbl in labels}, backend=self.model.backend
        )
        self._invalidate_search_cache()
        return self

    def add(self, label: str, key_vec: Array, value_vec: Array) -> None:
        self.keys.add(label, key_vec)
        self.values.add(label, value_vec)
        self._invalidate_search_cache()

    def query_label(self, key_vec: Array, k: int = 1) -> list[tuple[str, float]]:
        if self.search_backend == "rust":
            try:
                labels, sims = query_rust_prepared_index(
                    key_vec,
                    self._get_rust_key_index(),
                    k=k,
                    return_similarities=True,
                )
                return list(zip(labels, sims or [], strict=True))
            except (
                FileNotFoundError,
                NotImplementedError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                pass

        labels, sims = query_prepared_index(
            key_vec,
            self._get_prepared_key_index(),
            self.model,
            k=k,
            return_similarities=True,
        )
        return list(zip(labels, sims or [], strict=True))

    def query_value(self, key_vec: Array) -> tuple[str, Array]:
        lbls = self.query_label(key_vec, k=1)
        if not lbls:
            raise ValueError("No items in store")
        lbl = lbls[0][0]
        return lbl, self.values._items[lbl]

    def save(self, keys_path: str, values_path: str) -> None:
        self.keys.save(keys_path)
        self.values.save(values_path)

    @classmethod
    def load(
        cls,
        model: VSAModel,
        keys_path: str,
        values_path: str,
        *,
        search_backend: str = "numpy",
        allow_unsafe_legacy: bool = False,
    ) -> "AssocStore":
        st = cls(model, search_backend=search_backend)
        st.keys = Codebook.load(
            keys_path,
            backend=model.backend,
            allow_unsafe_legacy=allow_unsafe_legacy,
        )
        st.values = Codebook.load(
            values_path,
            backend=model.backend,
            allow_unsafe_legacy=allow_unsafe_legacy,
        )
        return st
