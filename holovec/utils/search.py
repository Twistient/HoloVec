"""Search utilities for VSA codebook operations.

This module provides search functions for finding hypervectors in codebooks,
including k-nearest neighbors, threshold-based search, and batch similarity
computation.

Key Features:
    - K-nearest neighbors (K-NN) search
    - Threshold-based retrieval
    - Vectorized batch similarity computation
    - Efficient codebook operations

Based on:
    Standard VSA search operations for associative memory
    and content-addressable storage.

References:
    Kanerva (2009): Hyperdimensional Computing
    Plate (2003): Holographic Reduced Representations
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..backends.base import Array
from ..models.base import VSAModel
from ..spaces.spaces import SparseSegmentSpace


def nearest_neighbors(
    query: Array,
    codebook: Dict[str, Array],
    model: VSAModel,
    k: int = 5,
    return_similarities: bool = True,
) -> Tuple[List[str], Optional[List[float]]]:
    """Find k-nearest neighbors in codebook.

    Computes similarity between query and all codebook entries,
    returning the k entries with highest similarity.

    Args:
        query: Query hypervector
        codebook: Dictionary mapping labels to hypervectors
        model: VSA model for similarity computation
        k: Number of neighbors to return (default: 5)
        return_similarities: If True, return similarities (default: True)

    Returns:
        Tuple of:
            - labels: List of k labels sorted by similarity (highest first)
            - similarities: List of k similarities (if return_similarities=True),
                          otherwise None

    Raises:
        TypeError: If arguments are not correct types
        ValueError: If k < 1, k > codebook size, or codebook is empty

    Examples:
        >>> # Find 5 nearest neighbors
        >>> labels, sims = nearest_neighbors(query, codebook, model, k=5)
        >>> for label, sim in zip(labels, sims):
        ...     print(f"{label}: {sim:.3f}")
        >>>
        >>> # Get only labels
        >>> labels, _ = nearest_neighbors(
        ...     query, codebook, model, k=3, return_similarities=False
        ... )

    References:
        Kanerva (2009): Hyperdimensional computing and associative memory
    """
    # Type validation
    if not isinstance(codebook, dict):
        raise TypeError(f"codebook must be dict, got {type(codebook)}")
    if not isinstance(model, VSAModel):
        raise TypeError(f"model must be VSAModel, got {type(model)}")
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k)}")
    if not isinstance(return_similarities, bool):
        raise TypeError(f"return_similarities must be bool, got {type(return_similarities)}")

    # Value validation
    if len(codebook) == 0:
        raise ValueError("codebook must not be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(codebook):
        raise ValueError(
            f"k ({k}) cannot exceed codebook size ({len(codebook)})"
        )

    # Compute all similarities
    similarities_dict = {}
    for label, vector in codebook.items():
        similarity = model.similarity(query, vector)
        similarities_dict[label] = float(similarity)

    # Sort by similarity (descending) and take top k
    sorted_items = sorted(
        similarities_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    labels = [label for label, _ in sorted_items]
    similarities = [sim for _, sim in sorted_items] if return_similarities else None

    return labels, similarities


def threshold_search(
    query: Array,
    codebook: Dict[str, Array],
    model: VSAModel,
    threshold: float = 0.8,
    return_similarities: bool = True,
) -> Tuple[List[str], Optional[List[float]]]:
    """Find all codebook entries above similarity threshold.

    Returns all entries where similarity(query, entry) >= threshold,
    sorted by similarity (highest first).

    Args:
        query: Query hypervector
        codebook: Dictionary mapping labels to hypervectors
        model: VSA model for similarity computation
        threshold: Minimum similarity threshold (default: 0.8)
        return_similarities: If True, return similarities (default: True)

    Returns:
        Tuple of:
            - labels: List of labels above threshold, sorted by similarity
            - similarities: List of similarities (if return_similarities=True),
                          otherwise None

    Raises:
        TypeError: If arguments are not correct types
        ValueError: If threshold not in [0.0, 1.0] or codebook is empty

    Examples:
        >>> # Find all matches above 0.9 similarity
        >>> labels, sims = threshold_search(
        ...     query, codebook, model, threshold=0.9
        ... )
        >>> print(f"Found {len(labels)} matches")
        >>>
        >>> # Lenient threshold
        >>> labels, _ = threshold_search(
        ...     query, codebook, model, threshold=0.5,
        ...     return_similarities=False
        ... )

    References:
        Standard associative memory retrieval operation
    """
    # Type validation
    if not isinstance(codebook, dict):
        raise TypeError(f"codebook must be dict, got {type(codebook)}")
    if not isinstance(model, VSAModel):
        raise TypeError(f"model must be VSAModel, got {type(model)}")
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be numeric, got {type(threshold)}")
    if not isinstance(return_similarities, bool):
        raise TypeError(f"return_similarities must be bool, got {type(return_similarities)}")

    # Value validation
    if len(codebook) == 0:
        raise ValueError("codebook must not be empty")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    # Compute similarities and filter by threshold
    filtered_items = []
    for label, vector in codebook.items():
        similarity = model.similarity(query, vector)
        if similarity >= threshold:
            filtered_items.append((label, float(similarity)))

    # Sort by similarity (descending)
    sorted_items = sorted(filtered_items, key=lambda x: x[1], reverse=True)

    labels = [label for label, _ in sorted_items]
    similarities = [sim for _, sim in sorted_items] if return_similarities else None

    return labels, similarities


def batch_similarity(
    queries: List[Array],
    codebook: Dict[str, Array],
    model: VSAModel,
) -> List[Dict[str, float]]:
    """Compute similarities between multiple queries and codebook.

    Efficiently computes similarity between each query and all codebook
    entries, returning results as a list of dictionaries.

    Args:
        queries: List of query hypervectors
        codebook: Dictionary mapping labels to hypervectors
        model: VSA model for similarity computation

    Returns:
        List of dictionaries, one per query, mapping labels to similarities

    Raises:
        TypeError: If arguments are not correct types
        ValueError: If queries is empty or codebook is empty

    Examples:
        >>> # Batch process multiple queries
        >>> results = batch_similarity([q1, q2, q3], codebook, model)
        >>> for i, sims in enumerate(results):
        ...     print(f"Query {i}:")
        ...     best_label = max(sims, key=sims.get)
        ...     print(f"  Best: {best_label} ({sims[best_label]:.3f})")
        >>>
        >>> # Find best match for each query
        >>> for query_sims in results:
        ...     best = max(query_sims.items(), key=lambda x: x[1])
        ...     print(f"Best: {best[0]} with similarity {best[1]:.3f}")

    References:
        Vectorized operations for efficient batch processing
    """
    # Type validation
    if not isinstance(queries, list):
        raise TypeError(f"queries must be list, got {type(queries)}")
    if not isinstance(codebook, dict):
        raise TypeError(f"codebook must be dict, got {type(codebook)}")
    if not isinstance(model, VSAModel):
        raise TypeError(f"model must be VSAModel, got {type(model)}")

    # Value validation
    if len(queries) == 0:
        raise ValueError("queries must not be empty")
    if len(codebook) == 0:
        raise ValueError("codebook must not be empty")

    # Compute similarities for each query
    results = []
    for query in queries:
        similarities = {}
        for label, vector in codebook.items():
            similarity = model.similarity(query, vector)
            similarities[label] = float(similarity)
        results.append(similarities)

    return results


# ===== Segment-wise search utilities (for BSDC-SEG) =====

def segment_pattern(vec: Array, space: SparseSegmentSpace) -> List[int]:
    """Return per-segment argmax indices (length S) for a vector.

    Projects vec to the nearest valid segment pattern via space.normalize(), then
    returns the index of the active bit per segment.
    """
    idx = space.segment_argmax(vec)
    return [int(i) for i in idx.tolist()] if hasattr(idx, 'tolist') else list(idx)


def find_by_segment_pattern(
    codebook: Dict[str, Array],
    space: SparseSegmentSpace,
    pattern: List[Optional[int]],
    match_mode: str = 'exact',
    min_fraction: float = 1.0,
) -> List[Tuple[str, float]]:
    """Find entries whose segment pattern matches the query pattern.

    - pattern: list of length S with segment indices or None/-1 as wildcards.
    - match_mode:
        - 'exact': all specified segments must match; returns [(label, 1.0), ...]
        - 'fraction': return fraction of matching specified segments, filter by min_fraction
    Returns a list of (label, score) sorted by score desc.
    """
    import numpy as _np

    S = space.segments
    if len(pattern) != S:
        raise ValueError(f"pattern length ({len(pattern)}) must equal segments ({S})")
    # Normalize pattern: -1/None = wildcard
    pat = []
    for p in pattern:
        if p is None or (isinstance(p, int) and p < 0):
            pat.append(None)
        else:
            pat.append(int(p))

    results: List[Tuple[str, float]] = []
    specified = [i for i, v in enumerate(pat) if v is not None]
    denom = max(1, len(specified))

    for label, vec in codebook.items():
        idx = segment_pattern(vec, space)
        matches = 0
        ok = True
        for s in specified:
            if idx[s] == pat[s]:
                matches += 1
            elif match_mode == 'exact':
                ok = False
                break
        if match_mode == 'exact':
            if ok:
                results.append((label, 1.0))
        else:
            frac = matches / denom
            if frac >= min_fraction:
                results.append((label, frac))

    # Sort by score desc
    results.sort(key=lambda t: t[1], reverse=True)
    return results
