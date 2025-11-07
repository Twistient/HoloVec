"""Base classes for vector spaces in holovec.

This module defines the abstract interface for vector spaces used in
hyperdimensional computing. Each space defines how random vectors are
generated and what similarity metric is appropriate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..backends import Backend, get_backend
from ..backends.base import Array


class VectorSpace(ABC):
    """Abstract base class for vector spaces.

    A vector space defines:
    - How random vectors are generated
    - What similarity metric is appropriate
    - How vectors are normalized
    - What algebraic operations are natural
    """

    def __init__(self, dimension: int, backend: Optional[Backend] = None, seed: Optional[int] = None):
        """Initialize vector space.

        Args:
            dimension: Dimensionality of vectors
            backend: Computational backend to use (defaults to auto-detect)
            seed: Random seed for reproducibility
        """
        if dimension < 1:
            raise ValueError(f"Dimension must be positive, got {dimension}")

        self.dimension = dimension
        self.backend = backend if backend is not None else get_backend()
        self.seed = seed

    @abstractmethod
    def random(self, seed: Optional[int] = None) -> Array:
        """Generate a random vector in this space.

        Args:
            seed: Optional seed for this specific vector

        Returns:
            Random vector from the appropriate distribution
        """
        pass

    @abstractmethod
    def similarity(self, a: Array, b: Array) -> float:
        """Compute similarity between two vectors.

        The similarity measure is space-specific:
        - Cosine similarity for real/complex spaces
        - Hamming distance for binary/bipolar spaces

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (higher means more similar)
        """
        pass

    @abstractmethod
    def normalize(self, vec: Array) -> Array:
        """Normalize a vector according to space conventions.

        Args:
            vec: Vector to normalize

        Returns:
            Normalized vector
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> str:
        """Return the appropriate dtype for this space."""
        pass

    @property
    @abstractmethod
    def space_name(self) -> str:
        """Return the name of this vector space."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self.dimension}, backend={self.backend.name})"


class DiscreteSpace(VectorSpace):
    """Base class for discrete vector spaces (binary, bipolar)."""

    def similarity(self, a: Array, b: Array) -> float:
        """Use Hamming-based similarity for discrete spaces.

        Returns normalized similarity in [0, 1] where:
        - 1.0 means identical
        - 0.0 means maximally different
        """
        hamming = self.backend.hamming_distance(a, b)
        # Normalize to [0, 1] where 1 is most similar
        return 1.0 - (hamming / self.dimension)


class ContinuousSpace(VectorSpace):
    """Base class for continuous vector spaces (real, complex)."""

    def similarity(self, a: Array, b: Array) -> float:
        """Use cosine similarity for continuous spaces.

        Returns similarity in [-1, 1] where:
        - 1.0 means same direction
        - -1.0 means opposite direction
        - 0.0 means orthogonal
        """
        return self.backend.cosine_similarity(a, b)

    def normalize(self, vec: Array) -> Array:
        """L2 normalization for continuous spaces."""
        return self.backend.normalize(vec, ord=2)
