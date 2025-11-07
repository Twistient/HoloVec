"""CPSE/CPSD utilities for context-preserving compositional encoding.

This module provides utilities for Context-Preserving SDR Encoding (CPSE) and
Context-Preserving SDR Decoding (CPSD), which represent a superior evolution
of Context-Dependent Thinning (CDT).

Key Features:
    - Order preservation via position permutations
    - Stable convergence (1.95% ± 0.15% error)
    - Fast convergence (4-5 iterations for M≥4 components)
    - Practical decoding methods (basic CPSD + Triadic Memory)

Based on:
    Malits & Mendelson (2025) "Context-Preserving Encoding/Decoding
    of Compositional Structures"

References:
    Paper: Malits & Mendelson (2025) - CPSE/CPSD specifications
    GitHub: https://github.com/PeterOvermann/TriadicMemory

Mathematical Foundation:
    - Additive iterations: K ≈ log(1 - 1/M) / log(1 - M·p)  [Eq. 8]
    - Subtractive iterations: Complex formula [Eq. 15]
    - Total: 4-5 iterations for M ≥ 4 (near-constant)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple, Any

from ..backends.base import Array
from ..models.base import VSAModel


class CPSEMetadata:
    """Metadata for CPSE encoding operations.

    Tracks permutation patterns, component structure, and encoding parameters
    for context-preserving operations. This metadata is essential for decoding
    and should be stored alongside encoded vectors.

    The metadata enables:
        - Reconstruction of position permutations for decoding
        - Validation of convergence in encoding/decoding cycles
        - Deterministic reproduction of encoding operations

    Attributes:
        n_components: Number of components in composition (M)
        permutation_seeds: Seeds for generating position-specific permutations
        base_seed: Base seed for reproducibility

    Examples:
        >>> # Create metadata for 5-component composition
        >>> metadata = CPSEMetadata(
        ...     n_components=5,
        ...     permutation_seeds=[42, 43, 44, 45, 46],
        ...     base_seed=42
        ... )
        >>>
        >>> # Serialize for storage
        >>> metadata.to_json('cpse_metadata.json')
        >>>
        >>> # Later, reload for decoding
        >>> metadata = CPSEMetadata.from_json('cpse_metadata.json')

    References:
        Malits & Mendelson (2025), Section 3.1: Position Encoding
    """

    def __init__(
        self,
        n_components: int,
        permutation_seeds: List[int],
        base_seed: int = 42
    ):
        """Initialize CPSE metadata.

        Args:
            n_components: Number of components in composition (must be >= 2)
            permutation_seeds: Seed for each position permutation
                              (must have length == n_components)
            base_seed: Base seed for reproducibility (default: 42)

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If n_components < 2 or permutation_seeds length mismatch

        Examples:
            >>> # Minimal valid metadata
            >>> metadata = CPSEMetadata(2, [42, 43])
            >>>
            >>> # Typical usage with 5 components
            >>> seeds = generate_permutation_patterns(n_patterns=5)
            >>> metadata = CPSEMetadata(5, seeds, base_seed=42)
        """
        # Type validation
        if not isinstance(n_components, int):
            raise TypeError(f"n_components must be int, got {type(n_components)}")
        if not isinstance(permutation_seeds, list):
            raise TypeError(f"permutation_seeds must be list, got {type(permutation_seeds)}")
        if not isinstance(base_seed, int):
            raise TypeError(f"base_seed must be int, got {type(base_seed)}")

        # Value validation
        if n_components < 2:
            raise ValueError(
                f"n_components must be >= 2 (need at least 2 components for composition), "
                f"got {n_components}"
            )
        if len(permutation_seeds) != n_components:
            raise ValueError(
                f"Expected {n_components} permutation seeds, got {len(permutation_seeds)}"
            )

        # Assignment after validation
        self.n_components = n_components
        self.permutation_seeds = permutation_seeds
        self.base_seed = base_seed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary.

        Returns:
            Dictionary with all metadata fields

        Examples:
            >>> metadata = CPSEMetadata(3, [42, 43, 44])
            >>> data = metadata.to_dict()
            >>> print(data)
            {'n_components': 3, 'permutation_seeds': [42, 43, 44], 'base_seed': 42}
        """
        return {
            'n_components': self.n_components,
            'permutation_seeds': self.permutation_seeds,
            'base_seed': self.base_seed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CPSEMetadata':
        """Deserialize metadata from dictionary.

        Args:
            data: Dictionary with metadata fields

        Returns:
            CPSEMetadata instance

        Raises:
            KeyError: If required fields are missing
            TypeError/ValueError: If field values are invalid

        Examples:
            >>> data = {'n_components': 3, 'permutation_seeds': [42, 43, 44], 'base_seed': 42}
            >>> metadata = CPSEMetadata.from_dict(data)
            >>> print(metadata.n_components)
            3
        """
        return cls(
            n_components=data['n_components'],
            permutation_seeds=data['permutation_seeds'],
            base_seed=data['base_seed']
        )

    def to_json(self, path: str):
        """Save metadata to JSON file.

        Args:
            path: File path for saving

        Examples:
            >>> metadata = CPSEMetadata(3, [42, 43, 44])
            >>> metadata.to_json('my_cpse_metadata.json')
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'CPSEMetadata':
        """Load metadata from JSON file.

        Args:
            path: File path for loading

        Returns:
            CPSEMetadata instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing

        Examples:
            >>> metadata = CPSEMetadata.from_json('my_cpse_metadata.json')
            >>> print(metadata.n_components)
            3
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation of metadata."""
        return (
            f"CPSEMetadata(n_components={self.n_components}, "
            f"permutation_seeds={self.permutation_seeds}, "
            f"base_seed={self.base_seed})"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another CPSEMetadata instance."""
        if not isinstance(other, CPSEMetadata):
            return False
        return (
            self.n_components == other.n_components and
            self.permutation_seeds == other.permutation_seeds and
            self.base_seed == other.base_seed
        )


def generate_permutation_patterns(
    n_patterns: int,
    base_seed: int = 42
) -> List[int]:
    """Generate permutation seeds for CPSE encoding.

    Creates deterministic permutation seeds for position-dependent
    thinning operations. Each seed generates a unique permutation
    matrix used to encode position information.

    The seeds are generated as: [base_seed, base_seed+1, ..., base_seed+n-1]

    Args:
        n_patterns: Number of permutation patterns to generate
        base_seed: Base random seed (default: 42)

    Returns:
        List of permutation seeds (length == n_patterns)

    Raises:
        TypeError: If arguments are not correct types
        ValueError: If n_patterns < 1

    Examples:
        >>> # Generate seeds for 5-component composition
        >>> seeds = generate_permutation_patterns(n_patterns=5)
        >>> print(seeds)
        [42, 43, 44, 45, 46]
        >>>
        >>> # Generate with custom base seed
        >>> seeds = generate_permutation_patterns(n_patterns=3, base_seed=100)
        >>> print(seeds)
        [100, 101, 102]

    References:
        Malits & Mendelson (2025), Section 3.1: Position Encoding
        - Each position i gets permutation p̃ᵢ derived from seed[i]
        - Deterministic generation ensures reproducibility
    """
    # Type validation
    if not isinstance(n_patterns, int):
        raise TypeError(f"n_patterns must be int, got {type(n_patterns)}")
    if not isinstance(base_seed, int):
        raise TypeError(f"base_seed must be int, got {type(base_seed)}")

    # Value validation
    if n_patterns < 1:
        raise ValueError(f"n_patterns must be >= 1, got {n_patterns}")

    # Generate sequential seeds
    return [base_seed + i for i in range(n_patterns)]


def validate_cpse_convergence(
    original_components: List[Array],
    decoded_components: List[Array],
    model: VSAModel,
    threshold: float = 0.95
) -> Tuple[bool, List[float]]:
    """Validate CPSE decoding convergence.

    Checks if decoded components are sufficiently similar to originals
    by computing pairwise similarities and comparing against a threshold.
    This is essential for verifying that the encoding-decoding cycle
    preserves information.

    Typical convergence rates (Malits & Mendelson 2025, Table 1):
        - Basic CPSD: 95-98% similarity for M=2-5 components
        - With Triadic Memory: 97-99% similarity
        - Target threshold: 0.95 (95%) is conservative

    Args:
        original_components: Original component hypervectors (length M)
        decoded_components: Decoded component hypervectors (length M)
        model: VSA model for similarity computation
        threshold: Minimum acceptable similarity (default: 0.95)

    Returns:
        Tuple of:
            - converged (bool): True if all similarities >= threshold
            - similarities (List[float]): Similarity for each component pair

    Raises:
        TypeError: If arguments are not correct types
        ValueError: If component lists have different lengths

    Examples:
        >>> # Validate decoding with strict threshold
        >>> converged, sims = validate_cpse_convergence(
        ...     original_components=originals,
        ...     decoded_components=decoded,
        ...     model=model,
        ...     threshold=0.95
        ... )
        >>> if converged:
        ...     print(f"Converged! Avg similarity: {np.mean(sims):.3f}")
        ... else:
        ...     print(f"Failed to converge. Min similarity: {min(sims):.3f}")
        >>>
        >>> # More lenient threshold for noisy conditions
        >>> converged, sims = validate_cpse_convergence(
        ...     originals, decoded, model, threshold=0.90
        ... )

    References:
        Malits & Mendelson (2025), Section 4: Experimental Results
        - Table 1 shows typical convergence rates for different M
        - Figure 3 demonstrates convergence behavior
    """
    # Type validation
    if not isinstance(original_components, list):
        raise TypeError(
            f"original_components must be list, got {type(original_components)}"
        )
    if not isinstance(decoded_components, list):
        raise TypeError(
            f"decoded_components must be list, got {type(decoded_components)}"
        )
    if not isinstance(model, VSAModel):
        raise TypeError(f"model must be VSAModel, got {type(model)}")
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be numeric, got {type(threshold)}")

    # Value validation
    if len(original_components) != len(decoded_components):
        raise ValueError(
            f"Component lists must have same length: "
            f"{len(original_components)} vs {len(decoded_components)}"
        )

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in range [0.0, 1.0], got {threshold}"
        )

    # Compute similarities for each component pair
    similarities = [
        float(model.similarity(orig, dec))
        for orig, dec in zip(original_components, decoded_components)
    ]

    # Check if all similarities meet threshold
    converged = all(sim >= threshold for sim in similarities)

    return converged, similarities
