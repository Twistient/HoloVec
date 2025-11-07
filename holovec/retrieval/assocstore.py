from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..backends.base import Array
from ..models.base import VSAModel
from ..utils.search import nearest_neighbors
from .codebook import Codebook


class AssocStore:
    """Lean heteroassociative store: keys → values via aligned codebooks.

    Stores two codebooks with aligned label order. Query by a key vector returns
    the best-matching key label and its corresponding value label/vector.
    """

    def __init__(self, model: VSAModel) -> None:
        self.model = model
        self.keys = Codebook(backend=model.backend)
        self.values = Codebook(backend=model.backend)
        self._label_order: List[str] = []

    def fit(self, key_items: Dict[str, Array], value_items: Dict[str, Array]) -> "AssocStore":
        # Intersect labels and preserve deterministic order
        labels = [lbl for lbl in key_items.keys() if lbl in value_items]
        self._label_order = labels
        self.keys = Codebook({lbl: key_items[lbl] for lbl in labels}, backend=self.model.backend)
        self.values = Codebook({lbl: value_items[lbl] for lbl in labels}, backend=self.model.backend)
        return self

    def add(self, label: str, key_vec: Array, value_vec: Array) -> None:
        self.keys.add(label, key_vec)
        self.values.add(label, value_vec)
        if label not in self._label_order:
            self._label_order.append(label)

    def query_label(self, key_vec: Array, k: int = 1) -> List[Tuple[str, float]]:
        labels, sims = nearest_neighbors(key_vec, self.keys._items, self.model, k=k, return_similarities=True)
        return list(zip(labels, sims or []))

    def query_value(self, key_vec: Array, top: int = 1) -> Tuple[str, Array]:
        lbls = self.query_label(key_vec, k=1)
        if not lbls:
            raise ValueError("No items in store")
        lbl = lbls[0][0]
        return lbl, self.values._items[lbl]

    def save(self, keys_path: str, values_path: str) -> None:
        self.keys.save(keys_path)
        self.values.save(values_path)

    @classmethod
    def load(cls, model: VSAModel, keys_path: str, values_path: str) -> "AssocStore":
        st = cls(model)
        st.keys = Codebook.load(keys_path, backend=model.backend)
        st.values = Codebook.load(values_path, backend=model.backend)
        st._label_order = st.keys.labels
        return st

