"""General utility operations for VSA systems.

This module provides general-purpose operations for hypervector manipulation
and analysis, including top-k selection, noise injection, and similarity
matrix computation.

Key Features:
    - Top-k selection from scored collections
    - Controlled noise injection for robustness testing
    - Pairwise similarity matrix computation
    - Support for various VSA operations

References:
    Kanerva (2009): Hyperdimensional Computing
    Plate (2003): Holographic Reduced Representations
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..backends.base import Array
from ..models.base import VSAModel


def select_top_k(
    items: Dict[str, float],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Select top-k items by score.

    Sorts items by score (descending) and returns the top k items
    as (label, score) tuples.

    Args:
        items: Dictionary mapping labels to scores
        k: Number of items to select (default: 5)

    Returns:
        List of (label, score) tuples sorted by score (highest first)

    Raises:
        TypeError: If arguments are not correct types
        ValueError: If k < 1, k > items size, or items is empty

    Examples:
        >>> # Select top 3 by similarity
        >>> scores = {'a': 0.95, 'b': 0.87, 'c': 0.92, 'd': 0.75}
        >>> top = select_top_k(scores, k=3)
        >>> print(top)
        [('a', 0.95), ('c', 0.92), ('b', 0.87)]
        >>>
        >>> # Get just the labels
        >>> labels = [label for label, _ in select_top_k(scores, k=2)]
        >>> print(labels)
        ['a', 'c']

    References:
        Standard selection operation for ranked retrieval
    """
    # Type validation
    if not isinstance(items, dict):
        raise TypeError(f"items must be dict, got {type(items)}")
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k)}")

    # Value validation
    if len(items) == 0:
        raise ValueError("items must not be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(items):
        raise ValueError(f"k ({k}) cannot exceed items size ({len(items)})")

    # Sort by score (descending) and take top k
    sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)[:k]

    return sorted_items


def add_noise(
    vector: Array,
    model: VSAModel,
    noise_level: float = 0.1,
    seed: int = None,
) -> Array:
    """Add controlled noise to a hypervector.

    Adds noise by bundling the original vector with a random vector,
    weighted by noise_level. Useful for testing robustness and
    approximate matching.

    Args:
        vector: Original hypervector
        model: VSA model for random generation and bundling
        noise_level: Proportion of noise to add (0.0 = none, 1.0 = full)
                    (default: 0.1)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Noisy hypervector

    Raises:
        TypeError: If arguments are not correct types
        ValueError: If noise_level not in [0.0, 1.0]

    Examples:
        >>> # Add 10% noise
        >>> noisy = add_noise(original, model, noise_level=0.1)
        >>> sim = model.similarity(original, noisy)
        >>> print(f"Similarity after noise: {sim:.3f}")
        >>>
        >>> # Heavy noise for stress testing
        >>> very_noisy = add_noise(original, model, noise_level=0.5)
        >>>
        >>> # Reproducible noise
        >>> noisy1 = add_noise(original, model, noise_level=0.2, seed=42)
        >>> noisy2 = add_noise(original, model, noise_level=0.2, seed=42)
        >>> # noisy1 and noisy2 are identical

    References:
        Robustness testing in Kanerva (2009) and related work
    """
    # Type validation
    if not isinstance(model, VSAModel):
        raise TypeError(f"model must be VSAModel, got {type(model)}")
    if not isinstance(noise_level, (int, float)):
        raise TypeError(f"noise_level must be numeric, got {type(noise_level)}")
    if seed is not None and not isinstance(seed, int):
        raise TypeError(f"seed must be int or None, got {type(seed)}")

    # Value validation
    if not (0.0 <= noise_level <= 1.0):
        raise ValueError(f"noise_level must be in [0.0, 1.0], got {noise_level}")

    # Generate random noise
    noise = model.random(seed=seed)

    # Simple approach: bundle original with scaled noise
    # For discrete models like MAP, bundling doesn't allow fine-grained mixing
    # So we use a simple threshold-based approach
    if noise_level == 0.0:
        return vector
    elif noise_level == 1.0:
        return noise
    elif noise_level >= 0.5:
        # High noise: bundle equal parts
        return model.bundle([vector, noise])
    else:
        # Low noise: bundle with more signal
        # Use 3:1 ratio for low noise, 2:1 for medium
        ratio = 3 if noise_level < 0.3 else 2
        vectors = [vector] * ratio + [noise]
        return model.bundle(vectors)


def similarity_matrix(
    vectors: List[Array],
    model: VSAModel,
    labels: List[str] = None,
) -> np.ndarray:
    """Compute pairwise similarity matrix.

    Computes similarity between all pairs of vectors, returning an
    n×n similarity matrix where entry (i,j) is similarity(vectors[i], vectors[j]).

    Args:
        vectors: List of hypervectors
        model: VSA model for similarity computation
        labels: Optional labels for vectors (for reference, not used in computation)

    Returns:
        NxN numpy array of pairwise similarities

    Raises:
        TypeError: If arguments are not correct types
        ValueError: If vectors is empty or labels length doesn't match

    Examples:
        >>> # Compute similarity matrix
        >>> vectors = [model.random(seed=i) for i in range(5)]
        >>> sim_matrix = similarity_matrix(vectors, model)
        >>> print(f"Shape: {sim_matrix.shape}")
        (5, 5)
        >>> print(f"Diagonal (self-similarity): {np.diag(sim_matrix)}")
        >>>
        >>> # With labels for interpretation
        >>> labels = ['cat', 'dog', 'bird', 'fish', 'snake']
        >>> sim_matrix = similarity_matrix(vectors, model, labels)
        >>> # Most similar pair (excluding self-similarity)
        >>> np.fill_diagonal(sim_matrix, -np.inf)
        >>> i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        >>> print(f"Most similar: {labels[i]} - {labels[j]}")

    References:
        Standard analysis tool for VSA systems
    """
    # Type validation
    if not isinstance(vectors, list):
        raise TypeError(f"vectors must be list, got {type(vectors)}")
    if not isinstance(model, VSAModel):
        raise TypeError(f"model must be VSAModel, got {type(model)}")
    if labels is not None and not isinstance(labels, list):
        raise TypeError(f"labels must be list or None, got {type(labels)}")

    # Value validation
    if len(vectors) == 0:
        raise ValueError("vectors must not be empty")
    if labels is not None and len(labels) != len(vectors):
        raise ValueError(
            f"labels length ({len(labels)}) must match vectors length ({len(vectors)})"
        )

    # Compute pairwise similarities
    n = len(vectors)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity = model.similarity(vectors[i], vectors[j])
            matrix[i, j] = float(similarity)

    return matrix
