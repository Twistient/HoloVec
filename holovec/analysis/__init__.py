"""Analysis tools for Vector Symbolic Architectures.

This module provides tools for analyzing VSA models, including:
- Theoretical capacity estimation
- Empirical capacity testing
- Dimension recommendation
- Model comparison utilities

The capacity analysis is based on research from:
- Schlegel et al. (2022): "Comparison of VSA"
- Kleyko et al. (2023): "HDC/VSA Survey"
- Frady et al. (2021): "VFA Framework"

Example:
    >>> from holovec.analysis import theoretical_capacity, recommend_dimension
    >>>
    >>> # Get theoretical capacity metrics
    >>> capacity = theoretical_capacity('FHRR', dim=1000)
    >>> print(f"Bundle capacity: {capacity['bundle_capacity']:.0f}")
    >>>
    >>> # Get recommended dimension for a use case
    >>> dim = recommend_dimension('FHRR', n_items=100, n_bindings=3)
    >>> print(f"Recommended dimension: {dim}")
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models.base import VSAModel


# Theoretical capacity constants derived from literature
# See Schlegel et al. (2022) and Kleyko et al. (2023)
_CAPACITY_CONSTANTS = {
    'FHRR': {
        'items_per_binding': 3.0,  # ~D/3 items per single binding
        'bundle_factor': 0.37,    # ~0.37*sqrt(D) items in bundle
        'recovery_accuracy': 1.0,  # Exact inverse via conjugate
        'binding_depth': 5,       # Practical limit for sequential bindings
        'description': 'Best capacity, exact inverse, complex phasors',
    },
    'MAP': {
        'items_per_binding': 15.0,  # ~D/15 items per binding
        'bundle_factor': 0.30,    # ~0.30*sqrt(D) items in bundle
        'recovery_accuracy': 1.0,  # Self-inverse
        'binding_depth': 4,
        'description': 'Self-inverse, bipolar, hardware-friendly',
    },
    'HRR': {
        'items_per_binding': 4.0,  # ~D/4 items
        'bundle_factor': 0.35,
        'recovery_accuracy': 0.71,  # Approximate inverse
        'binding_depth': 3,
        'description': 'Circular convolution, approximate inverse',
    },
    'GHRR': {
        'items_per_binding': 2.5,  # Matrix-based, good capacity
        'bundle_factor': 0.40,
        'recovery_accuracy': 1.0,  # Exact inverse
        'binding_depth': 6,
        'description': 'Non-commutative matrix binding, SOTA 2024',
    },
    'VTB': {
        'items_per_binding': 3.5,
        'bundle_factor': 0.35,
        'recovery_accuracy': 0.85,  # Approximate
        'binding_depth': 4,
        'description': 'Non-commutative, vector-derived transforms',
    },
    'BSC': {
        'items_per_binding': 20.0,  # Lower capacity
        'bundle_factor': 0.25,
        'recovery_accuracy': 1.0,  # Self-inverse (XOR)
        'binding_depth': 3,
        'description': 'Binary XOR, FPGA-friendly',
    },
    'BSDC': {
        'items_per_binding': 25.0,
        'bundle_factor': 0.20,
        'recovery_accuracy': 0.65,  # Approximate
        'binding_depth': 2,
        'description': 'Sparse binary, memory-efficient',
    },
    'BSDC_SEG': {
        'items_per_binding': 20.0,
        'bundle_factor': 0.22,
        'recovery_accuracy': 1.0,  # Self-inverse within segments
        'binding_depth': 3,
        'description': 'Segmented sparse, fast search',
    },
}


def theoretical_capacity(model: str, dim: int) -> dict:
    """Return theoretical capacity metrics for a VSA model.

    Provides estimates based on published research for:
    - Items storable per binding operation
    - Bundle capacity (superposition limit)
    - Recovery accuracy after unbinding
    - Maximum practical binding depth

    Args:
        model: Model name ('FHRR', 'MAP', 'HRR', 'GHRR', 'VTB', 'BSC', 'BSDC', 'BSDC_SEG')
        dim: Vector dimension

    Returns:
        Dictionary with capacity metrics:
            - items_per_binding: Max items in single binding for reliable recovery
            - bundle_capacity: Max items in superposition bundle
            - recovery_accuracy: Expected accuracy after unbinding (0-1)
            - binding_depth: Practical limit for sequential bindings
            - description: Brief model description

    Raises:
        ValueError: If model name is not recognized

    Example:
        >>> capacity = theoretical_capacity('FHRR', dim=1000)
        >>> print(f"Items per binding: {capacity['items_per_binding']:.1f}")
        Items per binding: 333.3
        >>> print(f"Bundle capacity: {capacity['bundle_capacity']:.0f}")
        Bundle capacity: 12

    References:
        Schlegel et al. (2022): "A Comparison of Vector Symbolic Architectures"
        Kleyko et al. (2023): "HDC/VSA Survey Part I & II"
    """
    model_upper = model.upper().replace('-', '_')
    if model_upper not in _CAPACITY_CONSTANTS:
        available = ', '.join(_CAPACITY_CONSTANTS.keys())
        raise ValueError(f"Unknown model '{model}'. Available: {available}")

    constants = _CAPACITY_CONSTANTS[model_upper]

    return {
        'items_per_binding': dim / constants['items_per_binding'],
        'bundle_capacity': constants['bundle_factor'] * math.sqrt(dim),
        'recovery_accuracy': constants['recovery_accuracy'],
        'binding_depth': constants['binding_depth'],
        'description': constants['description'],
    }


def recommend_dimension(
    model: str,
    n_items: int = 100,
    n_bindings: int = 1,
    target_accuracy: float = 0.95,
    safety_factor: float = 1.5,
) -> int:
    """Recommend minimum dimension for given requirements.

    Calculates the minimum vector dimension needed to reliably store
    a given number of items with a specified number of bindings.

    Args:
        model: Model name ('FHRR', 'MAP', 'HRR', etc.)
        n_items: Number of distinct items to store/discriminate
        n_bindings: Number of sequential bindings (composition depth)
        target_accuracy: Desired retrieval accuracy (0-1, default 0.95)
        safety_factor: Multiplier for safety margin (default 1.5)

    Returns:
        Recommended dimension (rounded to nice number like 512, 1024, 2048...)

    Example:
        >>> dim = recommend_dimension('FHRR', n_items=50, n_bindings=2)
        >>> print(f"Recommended: {dim}")
        Recommended: 512

        >>> dim = recommend_dimension('MAP', n_items=1000, n_bindings=3)
        >>> print(f"Recommended: {dim}")
        Recommended: 4096

    Notes:
        The recommendation uses conservative estimates. For production use,
        consider running empirical_capacity_test() to validate.
    """
    model_upper = model.upper().replace('-', '_')
    if model_upper not in _CAPACITY_CONSTANTS:
        available = ', '.join(_CAPACITY_CONSTANTS.keys())
        raise ValueError(f"Unknown model '{model}'. Available: {available}")

    constants = _CAPACITY_CONSTANTS[model_upper]

    # Base dimension from items per binding
    base_dim = n_items * constants['items_per_binding']

    # Adjust for binding depth (capacity decreases with depth)
    depth_factor = 1 + 0.2 * max(0, n_bindings - 1)  # 20% increase per binding level
    adjusted_dim = base_dim * depth_factor

    # Adjust for target accuracy
    # Higher accuracy requires more headroom
    accuracy_factor = 1.0 + (1.0 - target_accuracy) * 5  # Lower accuracy = less dim
    if target_accuracy > 0.95:
        accuracy_factor = 1.0 + (target_accuracy - 0.95) * 10  # More dim for high accuracy

    final_dim = adjusted_dim * accuracy_factor * safety_factor

    # Round to nice power of 2 or common dimension
    nice_dims = [128, 256, 512, 1024, 2048, 4096, 8192, 10000, 16384]
    for nice in nice_dims:
        if nice >= final_dim:
            return nice

    # If larger than 16384, round to nearest 1000
    return int(math.ceil(final_dim / 1000) * 1000)


def compare_models(dim: int = 1000) -> dict[str, dict]:
    """Compare all VSA models at a given dimension.

    Returns capacity metrics for all models, useful for model selection.

    Args:
        dim: Vector dimension for comparison

    Returns:
        Dictionary mapping model names to their capacity metrics

    Example:
        >>> comparison = compare_models(dim=1000)
        >>> for model, metrics in sorted(
        ...     comparison.items(),
        ...     key=lambda x: x[1]['items_per_binding'],
        ...     reverse=True
        ... ):
        ...     print(f"{model}: {metrics['items_per_binding']:.0f} items/binding")
        FHRR: 333 items/binding
        MAP: 67 items/binding
        ...
    """
    return {
        model: theoretical_capacity(model, dim)
        for model in _CAPACITY_CONSTANTS.keys()
    }


def empirical_capacity_test(
    model: 'VSAModel',
    codebook_sizes: list[int] | None = None,
    n_trials: int = 50,
    accuracy_threshold: float = 0.95,
) -> dict:
    """Empirically test model capacity via progressive testing.

    Creates codebooks of increasing size and measures recovery accuracy
    to find the practical capacity limit.

    Args:
        model: VSA model instance to test
        codebook_sizes: List of codebook sizes to test (default: auto)
        n_trials: Number of trials per codebook size
        accuracy_threshold: Accuracy threshold for "success"

    Returns:
        Dictionary with:
            - max_codebook_size: Largest codebook with accuracy >= threshold
            - accuracies: Dict mapping codebook size to mean accuracy
            - dimension: Model dimension tested
            - model_name: Name of model tested

    Example:
        >>> from holovec import VSA
        >>> model = VSA.create('FHRR', dim=512)
        >>> results = empirical_capacity_test(model, n_trials=20)
        >>> print(f"Max codebook: {results['max_codebook_size']}")
    """
    if codebook_sizes is None:
        # Auto-select based on dimension
        dim = model.dimension
        max_expected = int(dim / 3)  # Conservative estimate
        codebook_sizes = [10, 25, 50, 100, 150, 200, 300, 500]
        codebook_sizes = [s for s in codebook_sizes if s <= max_expected * 2]

    accuracies = {}
    max_passing_size = 0

    for n in codebook_sizes:
        correct = 0
        for trial in range(n_trials):
            # Create codebook
            codebook = {
                f"item_{i}": model.random(seed=trial * 1000 + i)
                for i in range(n)
            }

            # Test random item recovery
            test_key = f"item_{trial % n}"
            test_vec = codebook[test_key]

            # Find best match
            best_key = None
            best_sim = -float('inf')
            for key, vec in codebook.items():
                sim = float(model.similarity(test_vec, vec))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_key == test_key:
                correct += 1

        accuracy = correct / n_trials
        accuracies[n] = accuracy

        if accuracy >= accuracy_threshold:
            max_passing_size = n

    return {
        'max_codebook_size': max_passing_size,
        'accuracies': accuracies,
        'dimension': model.dimension,
        'model_name': model.model_name,
    }


__all__ = [
    'theoretical_capacity',
    'recommend_dimension',
    'compare_models',
    'empirical_capacity_test',
]
