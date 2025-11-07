"""FHRR (Fourier Holographic Reduced Representations) VSA model.

FHRR uses complex-valued vectors (unit phasors) with element-wise complex
multiplication for binding. It achieves the best capacity among VSA models
and has exact inverse through complex conjugation.

Properties:
- Exact inverse: unbind using complex conjugate
- Commutative: bind(a, b) = bind(b, a)
- Best capacity: ~330 dimensions for 15 vectors (Schlegel et al.)
- Natural fractional power encoding for continuous data
- Element-wise operations (very efficient)

References:
- Plate (2003): "Holographic Reduced Representations"
- Schlegel et al. (2021): Comparison showing FHRR's superior capacity
- Frady et al. (2021): VFA framework and fractional power encoding
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..backends import Backend
from ..backends.base import Array
from ..spaces import ComplexSpace, VectorSpace
from .base import VSAModel


class FHRRModel(VSAModel):
    """FHRR (Fourier HRR) model using complex phasors.

    Binding: element-wise complex multiplication (phase addition)
    Unbinding: element-wise multiplication with conjugate (phase subtraction)
    Bundling: element-wise addition + normalization to unit magnitude
    Permutation: circular shift (can also use phase rotation)

    Uses ComplexSpace with unit-magnitude phasors.
    """

    def __init__(
        self,
        dimension: int = 512,
        space: Optional[VectorSpace] = None,
        backend: Optional[Backend] = None,
        seed: Optional[int] = None
    ):
        """Initialize FHRR model.

        Args:
            dimension: Dimensionality of hypervectors
                      (can be smaller than MAP due to better capacity)
            space: Vector space (defaults to ComplexSpace)
            backend: Computational backend
            seed: Random seed for space
        """
        if space is None:
            from ..backends import get_backend
            backend = backend if backend is not None else get_backend()
            space = ComplexSpace(dimension, backend=backend, seed=seed)

        super().__init__(space, backend)

    @property
    def model_name(self) -> str:
        return "FHRR"

    @property
    def is_self_inverse(self) -> bool:
        return False  # Requires conjugate, not same operation

    @property
    def is_commutative(self) -> bool:
        return True  # Complex multiplication is commutative

    @property
    def is_exact_inverse(self) -> bool:
        return True  # Conjugate provides exact inverse

    def bind(self, a: Array, b: Array) -> Array:
        """Bind using element-wise complex multiplication.

        For unit phasors: (a * b)[i] = a[i] * b[i]
        This adds phase angles: ∠(a*b) = ∠a + ∠b

        Args:
            a: First vector (unit phasors)
            b: Second vector (unit phasors)

        Returns:
            Bound vector c = a ⊙ b (element-wise product)
        """
        result = self.backend.multiply(a, b)
        # Normalize to unit magnitude
        return self.normalize(result)

    def unbind(self, a: Array, b: Array) -> Array:
        """Unbind using element-wise multiplication with conjugate.

        To recover original from c = a ⊙ b:
        unbind(c, b) = c ⊙ b* = (a ⊙ b) ⊙ b* = a ⊙ (b ⊙ b*) = a ⊙ 1 = a

        Args:
            a: Bound vector (or first operand)
            b: Second operand

        Returns:
            Unbound vector (exact recovery)
        """
        b_conj = self.backend.conjugate(b)
        result = self.backend.multiply(a, b_conj)
        return self.normalize(result)

    def bundle(self, vectors: Sequence[Array]) -> Array:
        """Bundle using element-wise addition.

        Sum phasors and normalize back to unit magnitude.
        The result points in the "average" direction of inputs.

        Args:
            vectors: Sequence of vectors to bundle

        Returns:
            Bundled vector (normalized to unit magnitude)

        Raises:
            ValueError: If vectors is empty
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")

        vectors = list(vectors)

        # Sum all vectors (phasors add vectorially)
        result = self.backend.sum(self.backend.stack(vectors, axis=0), axis=0)

        # Normalize to unit magnitude
        return self.normalize(result)

    def permute(self, vec: Array, k: int = 1) -> Array:
        """Permute using circular shift.

        For FHRR, permutation can be done as:
        1. Circular shift (coordinate permutation)
        2. Phase rotation (multiply by exp(i*2πk/D))

        We use circular shift for consistency with other models.

        Args:
            vec: Vector to permute
            k: Number of positions to shift

        Returns:
            Permuted vector
        """
        return self.backend.roll(vec, shift=k)

    def fractional_power(self, vec: Array, exponent: float) -> Array:
        """Raise phasor to a fractional power.

        For unit phasor z = exp(iθ): z^α = exp(iαθ)
        This is useful for encoding continuous values.

        Args:
            vec: Vector of unit phasors
            exponent: Power to raise to

        Returns:
            Vector with phases scaled by exponent

        Example:
            >>> base = model.random()
            >>> # Encode value 2.5 using fractional power
            >>> encoded = model.fractional_power(base, 2.5)
        """
        # For unit phasors z = exp(iθ), we want: z^α = exp(iαθ)
        # This is exact and avoids branch cuts from complex logarithms.
        #
        # Implementation:
        # 1. Extract phase θ = arg(z)
        # 2. Scale by exponent: αθ
        # 3. Create new phasor: exp(iαθ)

        # Get phase angles using backend operation
        angles = self.backend.angle(vec)

        # Scale angles by exponent
        scaled_angles = self.backend.multiply_scalar(angles, exponent)

        # Create new phasors: exp(i * scaled_angles)
        # exp(iθ) = cos(θ) + i*sin(θ)
        result = self.backend.exp(1j * scaled_angles)

        # Renormalize to unit magnitude (handle numerical errors)
        return self.normalize(result)

    def __repr__(self) -> str:
        return (f"FHRRModel(dimension={self.dimension}, "
                f"space={self.space.space_name}, "
                f"backend={self.backend.name})")
