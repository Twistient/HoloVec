"""HRR (Holographic Reduced Representations) VSA model.

HRR uses circular convolution for binding, which provides approximate
inverse through circular correlation. This is the classic VSA model
developed by Tony Plate.

Properties:
- Binding: Circular convolution
- Unbinding: Circular correlation (approximate inverse)
- Commutative: Yes
- Exact inverse: No (approximate)
- Complexity: O(D log D) via FFT

References:
- Plate (1991): "Holographic Reduced Representations"
- Plate (1995): "Holographic reduced representations: Distributed
  representation for cognitive structures"
- Plate (2003): Full book on HRR
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..backends import Backend
from ..backends.base import Array
from ..spaces import RealSpace, VectorSpace
from .base import VSAModel


class HRRModel(VSAModel):
    """HRR (Holographic Reduced Representations) model.

    Binding: circular convolution (via FFT)
    Unbinding: circular correlation (via FFT)
    Bundling: element-wise addition + normalization
    Permutation: circular shift

    Uses RealSpace with Gaussian distribution N(0, 1/D).
    """

    def __init__(
        self,
        dimension: int = 10000,
        space: Optional[VectorSpace] = None,
        backend: Optional[Backend] = None,
        seed: Optional[int] = None
    ):
        """Initialize HRR model.

        Args:
            dimension: Dimensionality of hypervectors (recommend 1000-10000)
            space: Vector space (defaults to RealSpace)
            backend: Computational backend
            seed: Random seed for space
        """
        if space is None:
            from ..backends import get_backend
            backend = backend if backend is not None else get_backend()
            space = RealSpace(dimension, backend=backend, seed=seed)

        super().__init__(space, backend)

    @property
    def model_name(self) -> str:
        return "HRR"

    @property
    def is_self_inverse(self) -> bool:
        return False  # Requires correlation, not same operation

    @property
    def is_commutative(self) -> bool:
        return True  # Convolution is commutative

    @property
    def is_exact_inverse(self) -> bool:
        return False  # Correlation gives approximate inverse

    def bind(self, a: Array, b: Array) -> Array:
        """Bind using circular convolution.

        Implemented via FFT: conv(a, b) = IFFT(FFT(a) * FFT(b))

        Args:
            a: First vector
            b: Second vector

        Returns:
            Bound vector c = a ⊛ b (circular convolution)
        """
        # Circular convolution in frequency domain
        result = self.backend.circular_convolve(a, b)

        # Do NOT normalize - preserves magnitude for proper unbinding via
        # circular correlation. Normalization would interfere with the
        # mathematical relationship required for unbind recovery.
        return result

    def unbind(self, a: Array, b: Array) -> Array:
        """Unbind using circular correlation (approximate inverse of convolution).

        This is the classic HRR unbinding operation that uses circular correlation
        to approximately recover the original vector from a bound pair.

        Args:
            a: Bound vector c = x ⊛ b (result of circular convolution)
            b: Key vector (second operand in binding)

        Returns:
            Approximate recovery of x (original vector), normalized to unit length

        Notes
        -----
        **Mathematical Foundation:**

        HRR binding via circular convolution:
            c = x ⊛ b

        In frequency domain (Fourier):
            C(ω) = X(ω) · B(ω)

        Unbinding via circular correlation:
            x̂ = c ⋆ b = IFFT(C(ω) · B*(ω))

        Where B*(ω) is the complex conjugate of B(ω).

        Substituting C(ω) = X(ω) · B(ω):
            x̂ = IFFT(X(ω) · B(ω) · B*(ω))
              = IFFT(X(ω) · |B(ω)|²)

        For random vectors with approximately uniform power spectrum (|B(ω)|² ≈ 1),
        this gives x̂ ≈ x.

        **Approximation Quality:**

        Recovery similarity depends on:
        - Dimension D: Higher D → better recovery
        - Noise level: Clean binding → better unbind
        - Bundle size: More items → more interference

        Empirical performance (D=10000):
        - Clean unbind: similarity ≈ 0.99
        - After bundling 2 items: similarity ≈ 0.57
        - After bundling 10 items: similarity ≈ 0.30
        - After bundling 100 items: similarity decreases further

        References
        ----------
        - Plate (1995): "Holographic Reduced Representations"
        - Plate (2003): "Holographic Reduced Representations" (full book)

        Examples
        --------
        >>> model = VSA.create('HRR', dim=10000)
        >>> x = model.random(seed=1)
        >>> b = model.random(seed=2)
        >>> c = model.bind(x, b)
        >>> x_recovered = model.unbind(c, b)
        >>> similarity = model.similarity(x, x_recovered)
        >>> print(f"Recovery similarity: {similarity:.3f}")  # ~0.99
        """
        # Transform to frequency domain
        fa = self.backend.fft(a)
        fb = self.backend.fft(b)

        # Circular correlation: C(ω) * conj(B(ω))
        # This is the classic HRR unbinding operation (Plate, 1995)
        fr = self.backend.multiply(fa, self.backend.conjugate(fb))

        # Transform back to time domain
        time = self.backend.ifft(fr)

        # Take real part (imaginary part should be near zero due to real inputs)
        result = self.backend.real(time)

        # Normalize to unit length for consistent comparison with other vectors
        return self.normalize(result)

    def bundle(self, vectors: Sequence[Array]) -> Array:
        """Bundle using element-wise addition (superposition).

        For HRR, bundling is simple vector addition without normalization.
        This preserves the magnitude relationships needed for proper unbinding.

        Args:
            vectors: Sequence of vectors to bundle

        Returns:
            Bundled vector (unnormalized sum)

        Raises:
            ValueError: If vectors is empty

        Notes:
            Unlike some VSA models, HRR does NOT normalize after bundling.
            Normalization would interfere with the circular correlation unbinding
            operation. The unbind() method handles normalization of its output.
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")

        vectors = list(vectors)

        # Sum all vectors (simple superposition, no normalization)
        result = self.backend.sum(self.backend.stack(vectors, axis=0), axis=0)

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
        return (f"HRRModel(dimension={self.dimension}, "
                f"space={self.space.space_name}, "
                f"backend={self.backend.name})")
