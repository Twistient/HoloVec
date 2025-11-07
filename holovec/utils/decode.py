from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..backends.base import Array
from ..models.base import VSAModel
from .search import nearest_neighbors, threshold_search


def decode_nearest(hv: Array, codebook: Dict[str, Array], model: VSAModel, k: int = 1) -> List[Tuple[str, float]]:
    labels, sims = nearest_neighbors(hv, codebook, model, k=k, return_similarities=True)
    sims = sims or []
    return list(zip(labels, sims))


def decode_threshold(hv: Array, codebook: Dict[str, Array], model: VSAModel, threshold: float = 0.8) -> List[Tuple[str, float]]:
    labels, sims = threshold_search(hv, codebook, model, threshold=threshold, return_similarities=True)
    sims = sims or []
    return list(zip(labels, sims))


def decode_multilabel(
    hv: Array,
    codebook: Dict[str, Array],
    model: VSAModel,
    method: str = "threshold",
    k: int = 5,
    threshold: float = 0.7,
) -> List[Tuple[str, float]]:
    """Decode multi-label sets using either threshold or top-k strategy."""
    method = (method or "threshold").lower()
    if method == "threshold":
        return decode_threshold(hv, codebook, model, threshold=threshold)
    elif method == "topk" or method == "top-k":
        return decode_nearest(hv, codebook, model, k=k)
    else:
        raise ValueError("method must be one of {'threshold','topk'}")

