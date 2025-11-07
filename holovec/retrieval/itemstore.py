from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..backends.base import Array
from ..models.base import VSAModel
from ..utils.cleanup import CleanupStrategy, BruteForceCleanup, ResonatorCleanup
from ..utils.search import nearest_neighbors
from .codebook import Codebook


class ItemStore:
    """Thin retrieval wrapper around a Codebook and a CleanupStrategy.

    Provides nearest-neighbor queries and multi-factor factorization via
    the configured cleanup strategy.
    """

    def __init__(
        self,
        model: VSAModel,
        cleanup: Optional[CleanupStrategy] = None,
    ) -> None:
        self.model = model
        self.cleanup: CleanupStrategy = cleanup if cleanup is not None else BruteForceCleanup()
        self.codebook = Codebook(backend=model.backend)

    def fit(self, items: Dict[str, Array] | Codebook) -> "ItemStore":
        if isinstance(items, Codebook):
            self.codebook = items
        else:
            self.codebook = Codebook(items, backend=self.model.backend)
        return self

    def add(self, label: str, vector: Array) -> None:
        self.codebook.add(label, vector)

    def extend(self, items: Dict[str, Array]) -> None:
        self.codebook.extend(items)

    def query(
        self,
        vec: Array,
        k: int = 1,
        return_similarities: bool = True,
        fast: bool = True,
    ) -> List[Tuple[str, float]]:
        """Query top-k nearest items.

        If fast=True, uses a batched matrix routine when possible, otherwise
        falls back to scalar nearest_neighbors.
        """
        if fast and self.codebook.size > 0:
            labels, mat = self.codebook.as_matrix(self.model.backend)
            be = self.model.backend
            # Continuous spaces: cosine-like; ComplexSpace handled specially
            space_name = self.model.space.space_name
            try:
                if space_name.startswith("complex"):
                    # sim = Re(conj(C) @ v) / D
                    v = vec
                    conjC = be.conjugate(mat)
                    dots = be.matmul(conjC, v)  # (L,)
                    sims_arr = be.real(dots)
                    sims_np = be.to_numpy(sims_arr) / float(self.model.dimension)
                else:
                    # cosine: (C v) / (||C_i|| * ||v||)
                    dots = be.matmul(mat, vec)  # (L,)
                    # norms per row
                    # norm(C_i) = sqrt(sum(C_i^2)) → use l2 along axis=1
                    row_norms = be.norm(mat, ord=2, axis=1)
                    v_norm = be.norm(vec, ord=2)
                    denom = be.multiply(row_norms, v_norm)
                    sims_arr = be.divide(dots, denom)
                    sims_np = be.to_numpy(sims_arr)
                # Prepare top-k
                import numpy as _np
                sims_np = sims_np.astype(float)
                if k >= len(labels):
                    order = _np.argsort(-sims_np)
                else:
                    # partial sort then full sort within top-k
                    idx_part = _np.argpartition(-sims_np, kth=k-1)[:k]
                    order = idx_part[_np.argsort(-sims_np[idx_part])]
                out = [(labels[i], float(sims_np[i])) for i in order[:k]]
                if return_similarities:
                    return out
                else:
                    return [(lbl, 0.0) for lbl, _ in out]
            except Exception:
                # Fallback to scalar path on any backend issues
                pass

        labels, sims = nearest_neighbors(vec, self.codebook._items, self.model, k=k, return_similarities=True)
        return list(zip(labels, sims or [])) if return_similarities else [(lbl, 0.0) for lbl in labels]

    def factorize(
        self,
        vec: Array,
        n_factors: int,
        **kwargs,
    ) -> Tuple[List[str], List[float]]:
        return self.cleanup.factorize(
            vec,
            self.codebook._items,
            self.model,
            n_factors=n_factors,
            **kwargs,
        )

    # Persistence delegates to Codebook
    def save(self, path: str) -> None:
        self.codebook.save(path)

    @classmethod
    def load(
        cls,
        model: VSAModel,
        path: str,
        cleanup: Optional[CleanupStrategy] = None,
    ) -> "ItemStore":
        store = cls(model=model, cleanup=cleanup)
        store.codebook = Codebook.load(path, backend=model.backend)
        return store
