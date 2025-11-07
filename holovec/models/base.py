"""Base class for VSA (Vector Symbolic Architecture) models.

This module defines the abstract interface that all VSA models must implement.
Each model provides binding, unbinding, bundling, and permutation operations
with different algebraic properties.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from ..backends import Backend, get_backend
from ..backends.base import Array
from ..spaces.base import VectorSpace


class VSAModel(ABC):
    """Abstract base class for VSA models.

    A VSA model defines the core operations:
    - bind: Associate two vectors (creates dissimilar result)
    - unbind: Recover one vector given the other and their binding
    - bundle: Combine multiple vectors (preserves similarity)
    - permute: Reorder vector to represent position/sequence

    Different models have different algebraic properties:
    - Self-inverse binding: bind(a, b) = unbind(a, b)
    - Exact vs approximate inverse
    - Commutativity of binding
    """

    def __init__(
        self,
        space: VectorSpace,
        backend: Optional[Backend] = None
    ):
        """Initialize VSA model.

        Args:
            space: Vector space defining the representation
            backend: Computational backend (defaults to space's backend)
        """
        self.space = space
        self.backend = backend if backend is not None else space.backend
        self.dimension = space.dimension

    # ===== Core VSA Operations =====

    @abstractmethod
    def bind(self, a: Array, b: Array) -> Array:
        """Bind two vectors to create an association.

        Binding creates a new vector that is dissimilar to both inputs
        but preserves structured similarity (similar inputs → similar bindings).

        Args:
            a: First vector
            b: Second vector

        Returns:
            Bound vector representing the association of a and b
        """
        pass

    @abstractmethod
    def unbind(self, a: Array, b: Array) -> Array:
        """Unbind to recover one vector given the other.

        For self-inverse models: unbind(a, b) = bind(a, b)
        For others: approximately recovers a from bind(a, b) and b

        Args:
            a: Bound vector or first operand
            b: Second operand

        Returns:
            Recovered vector (exact or approximate depending on model)
        """
        pass

    @abstractmethod
    def bundle(self, vectors: Sequence[Array]) -> Array:
        """Bundle (superpose) multiple vectors.

        Bundling combines vectors while preserving similarity to all inputs.
        The result is similar to each input vector.

        Args:
            vectors: Sequence of vectors to bundle

        Returns:
            Bundled vector representing the superposition

        Raises:
            ValueError: If vectors is empty
        """
        pass

    @abstractmethod
    def permute(self, vec: Array, k: int = 1) -> Array:
        """Permute vector to represent position or sequence.

        Permutation reorders coordinates and is used to encode position
        or create sequences. It's invertible and preserves similarity.

        Args:
            vec: Vector to permute
            k: Number of positions to shift (default: 1)

        Returns:
            Permuted vector
        """
        pass

    def unpermute(self, vec: Array, k: int = 1) -> Array:
        """Inverse permutation.

        Args:
            vec: Vector to unpermute
            k: Number of positions to shift back (default: 1)

        Returns:
            Unpermuted vector
        """
        return self.permute(vec, -k)

    # ===== Similarity and Cleanup =====

    def similarity(self, a: Array, b: Array) -> float:
        """Compute similarity between two vectors.

        Delegates to the vector space's similarity metric.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (space-dependent metric)
        """
        return self.space.similarity(a, b)

    def normalize(self, vec: Array) -> Array:
        """Normalize vector according to space conventions.

        Args:
            vec: Vector to normalize

        Returns:
            Normalized vector
        """
        return self.space.normalize(vec)

    # ===== Vector Generation =====

    def random(self, seed: Optional[int] = None) -> Array:
        """Generate a random vector from the space.

        Args:
            seed: Optional random seed

        Returns:
            Random vector
        """
        return self.space.random(seed=seed)

    def random_sequence(self, n: int, seed: Optional[int] = None) -> List[Array]:
        """Generate n random vectors.

        Args:
            n: Number of vectors to generate
            seed: Optional base seed (each vector gets seed + i)

        Returns:
            List of random vectors
        """
        if seed is not None:
            return [self.random(seed=seed + i) for i in range(n)]
        return [self.random() for _ in range(n)]

    # ===== Compositional Operations =====

    def bind_multiple(self, vectors: Sequence[Array]) -> Array:
        """Bind multiple vectors sequentially.

        For n vectors: bind(bind(bind(v1, v2), v3), ...)

        Args:
            vectors: Sequence of vectors to bind

        Returns:
            Result of sequential binding

        Raises:
            ValueError: If fewer than 2 vectors provided
        """
        if len(vectors) < 2:
            raise ValueError("Need at least 2 vectors to bind")

        result = vectors[0]
        for vec in vectors[1:]:
            result = self.bind(result, vec)
        return result

    # ===== Model Properties =====

    @property
    @abstractmethod
    def is_self_inverse(self) -> bool:
        """Whether binding is self-inverse (bind = unbind)."""
        pass

    @property
    @abstractmethod
    def is_commutative(self) -> bool:
        """Whether binding is commutative (bind(a, b) = bind(b, a))."""
        pass

    @property
    @abstractmethod
    def is_exact_inverse(self) -> bool:
        """Whether unbinding gives exact recovery (no approximation error)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name (e.g., 'MAP', 'FHRR', 'HRR')."""
        pass

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(dimension={self.dimension}, "
                f"space={self.space.space_name}, backend={self.backend.name})")
