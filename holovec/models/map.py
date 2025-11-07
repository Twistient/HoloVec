"""MAP (Multiply-Add-Permute) VSA model.

MAP uses element-wise multiplication for binding, which is self-inverse.
It's one of the simplest VSA models and works with both bipolar {-1, +1}
and continuous real values.

Properties:
- Self-inverse: bind(a, b) = unbind(a, b)
- Commutative: bind(a, b) = bind(b, a)
- Exact inverse: unbind(bind(a, b), b) = a (for bipolar)
- Simple hardware: Only XOR for bipolar, multiply for continuous

References:
- Kanerva (2009): "Hyperdimensional Computing: An Introduction"
- Schlegel et al. (2021): Comparison of VSA models
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..backends import Backend
from ..backends.base import Array
from ..spaces import BipolarSpace, VectorSpace
from .base import VSAModel


class MAPModel(VSAModel):
    """MAP (Multiply-Add-Permute) model.

    Binding: element-wise multiplication
    Unbinding: element-wise multiplication (self-inverse)
    Bundling: element-wise addition + normalization
    Permutation: circular shift

    Best used with BipolarSpace or RealSpace.
    """

    def __init__(
        self,
        dimension: int = 10000,
        space: Optional[VectorSpace] = None,
        backend: Optional[Backend] = None,
        seed: Optional[int] = None
    ):
        """Initialize MAP model.

        Args:
            dimension: Dimensionality of hypervectors
            space: Vector space (defaults to BipolarSpace)
            backend: Computational backend
            seed: Random seed for space
        """
        if space is None:
            from ..backends import get_backend
            backend = backend if backend is not None else get_backend()
            space = BipolarSpace(dimension, backend=backend, seed=seed)

        super().__init__(space, backend)

        # Pre-compute permutation indices for efficiency
        self._permutation_indices = list(range(self.dimension))

    @property
    def model_name(self) -> str:
        return "MAP"

    @property
    def is_self_inverse(self) -> bool:
        return True

    @property
    def is_commutative(self) -> bool:
        return True

    @property
    def is_exact_inverse(self) -> bool:
        # Exact for bipolar, approximate for continuous
        return self.space.space_name == "bipolar"

    def bind(self, a: Array, b: Array) -> Array:
        """Bind using element-wise multiplication.

        For bipolar: XOR when represented as {0,1}
        For real: Hadamard product

        Args:
            a: First vector
            b: Second vector

        Returns:
            Bound vector c = a ⊙ b
        """
        result = self.backend.multiply(a, b)
        # Normalize to maintain unit norm for continuous spaces
        if self.space.space_name != "bipolar":
            result = self.normalize(result)
        return result

    def unbind(self, a: Array, b: Array) -> Array:
        """Unbind using element-wise multiplication (self-inverse).

        Since binding is self-inverse: unbind(c, b) = c ⊙ b

        Args:
            a: Bound vector (or first operand)
            b: Second operand

        Returns:
            Unbound vector (exact for bipolar, approximate for continuous)
        """
        # For MAP, binding = unbinding
        return self.bind(a, b)

    def bundle(self, vectors: Sequence[Array]) -> Array:
        """Bundle using element-wise addition.

        For bipolar: majority vote after summing
        For real: sum and normalize

        Args:
            vectors: Sequence of vectors to bundle

        Returns:
            Bundled vector

        Raises:
            ValueError: If vectors is empty
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")

        vectors = list(vectors)

        # Sum all vectors
        result = self.backend.sum(self.backend.stack(vectors, axis=0), axis=0)

        # Normalize according to space
        if self.space.space_name == "bipolar":
            # Majority vote: sign of sum
            result = self.backend.sign(result)
            # Handle zeros (shouldn't happen in practice, but be safe)
            # If sum is 0, randomly choose ±1
            zeros_mask = (result == 0)
            if self.backend.to_numpy(zeros_mask).any():
                # For any zeros, use the first vector's value
                first_vec = vectors[0]
                result = self.backend.where(zeros_mask, first_vec, result)
        else:
            # For continuous spaces, L2 normalize
            result = self.normalize(result)

        return result

    def permute(self, vec: Array, k: int = 1) -> Array:
        """Permute using circular shift.

        Shifts vector elements by k positions to the right.
        Negative k shifts left.

        Args:
            vec: Vector to permute
            k: Number of positions to shift

        Returns:
            Permuted vector
        """
        return self.backend.roll(vec, shift=k)

    def __repr__(self) -> str:
        return (f"MAPModel(dimension={self.dimension}, "
                f"space={self.space.space_name}, "
                f"backend={self.backend.name})")
