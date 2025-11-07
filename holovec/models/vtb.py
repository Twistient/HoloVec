"""VTB (Vector-derived Transformation Binding) VSA model (MBAT-style).

VTB implements non-commutative binding by applying a transformation derived
from one vector to the other:

    c = M(a) @ b

In this implementation, M(a) is constructed implicitly as a weighted sum of
fixed non-commuting basis transformations applied to b (no D×D matrices are
materialized):

    M(a) @ b ≈ Σ_k w_k(a) · R_k(b)

where R_k are fixed basis transformations (circular shifts here) and w_k(a)
are content-dependent weights (softmax of projections of a onto K code vectors).

Properties:
- Binding: weighted sum of transformed b (non-commutative)
- Unbinding: approximate inverse via weighted inverse transforms
- Non-commutative: a ⊗ b ≠ b ⊗ a
- Exact inverse: No (approximate)

References:
- Gallant & Okaywe (2013): Matrix Binding of Additive Terms (MBAT)
- Gayler (2003); Plate (2003); Schlegel et al. (2022)
"""

from __future__ import annotations

from typing import Optional, Sequence, List

import numpy as np

from ..backends import Backend
from ..backends.base import Array
from ..spaces import RealSpace, VectorSpace
from .base import VSAModel


class VTBModel(VSAModel):
    """VTB (Vector-derived Transformation Binding) model.

    Binding (MBAT-style): c = Σ_k w_k(a) · roll(b, s_k)
    Unbinding (approximate): b̂ = Σ_k w_k(a) · roll(c, -s_k)
    Bundling: element-wise addition + normalization
    Permutation: circular shift

    Uses RealSpace with L2-normalized real-valued vectors.
    """

    def __init__(
        self,
        dimension: int = 10000,
        space: Optional[VectorSpace] = None,
        backend: Optional[Backend] = None,
        seed: Optional[int] = None,
        n_bases: int = 4,
        shifts: Optional[List[int]] = None,
        temperature: float = 100.0,
    ):
        """Initialize VTB model.

        Args:
            dimension: Dimensionality of hypervectors
            space: Vector space (defaults to RealSpace)
            backend: Computational backend
            seed: Random seed for space
        """
        if space is None:
            from ..backends import get_backend
            backend = backend if backend is not None else get_backend()
            space = RealSpace(dimension, backend=backend, seed=seed)

        super().__init__(space, backend)

        # MBAT parameters
        self.n_bases = int(n_bases)
        if self.n_bases < 2:
            raise ValueError("n_bases must be >= 2")
        self.temperature = float(temperature)
        if self.temperature <= 0:
            self.temperature = 1.0

        # Basis transformations: use integer circular shifts as R_k
        if shifts is None:
            # choose distinct small shifts spread across dimension
            step = max(1, self.dimension // (self.n_bases + 1))
            self.shifts = [((i + 1) * step) % self.dimension for i in range(self.n_bases)]
            # ensure non-zero and unique
            self.shifts = [s if s != 0 else 1 for s in self.shifts]
            self.shifts = list(dict.fromkeys(self.shifts))
            while len(self.shifts) < self.n_bases:
                # fill with incremental shifts
                self.shifts.append((self.shifts[-1] + 1) % self.dimension)
        else:
            if len(shifts) != self.n_bases:
                raise ValueError("len(shifts) must equal n_bases")
            self.shifts = [int(s) % self.dimension for s in shifts]

        # Code vectors U_k to produce weights w_k(a) = softmax(τ · <a, U_k>)
        # Stack shape (K, D)
        self._U = self.backend.stack([
            self.backend.normalize(self.backend.random_normal(self.dimension, seed=(seed or 0) + k))
            for k in range(self.n_bases)
        ], axis=0)

    @property
    def model_name(self) -> str:
        return "VTB"

    @property
    def is_self_inverse(self) -> bool:
        return False

    @property
    def is_commutative(self) -> bool:
        return False

    @property
    def is_exact_inverse(self) -> bool:
        return False

    def _weights(self, a: Array) -> Array:
        """Compute softmax weights over bases from vector a.

        w_k(a) = softmax(τ · <a, U_k>)
        Returns shape (K,)
        """
        # scores: (K,)
        scores = []
        for k in range(self.n_bases):
            uk = self._U[k]
            scores.append(self.backend.dot(a, uk))
        scores = self.backend.stack(scores, axis=0)
        # scale by temperature then softmax
        scaled = self.backend.multiply_scalar(scores, self.temperature)
        return self.backend.softmax(scaled, axis=0)

    def _vector_to_circulant(self, vec: Array) -> Array:
        """Convert vector to circulant matrix.

        A circulant matrix is a special matrix where each row is a circular
        shift of the previous row. For vector [a, b, c]:
            [[a, b, c],
             [c, a, b],
             [b, c, a]]

        This construction enables efficient matrix-vector multiplication
        via circular convolution in the frequency domain.

        Args:
            vec: Vector of shape (D,)

        Returns:
            Circulant matrix of shape (D, D)
        """
        vec_np = self.backend.to_numpy(vec)
        D = len(vec_np)

        # Build circulant matrix: each row is a circular shift
        matrix = np.zeros((D, D), dtype=vec_np.dtype)
        for i in range(D):
            matrix[i] = np.roll(vec_np, i)

        return self.backend.from_numpy(matrix)

    def bind(self, a: Array, b: Array) -> Array:
        """Bind using MBAT-style weighted basis transforms.

        c = Σ_k w_k(a) · roll(b, s_k)
        """
        # Derive transform from a to act on b
        w = self._weights(a)  # (K,)
        # accumulate weighted shifts
        parts = []
        for k, shift in enumerate(self.shifts):
            wk = w[k]
            rb = self.backend.roll(b, shift=shift)
            parts.append(self.backend.multiply_scalar(rb, float(self.backend.to_numpy(wk))))
        result = self.backend.sum(self.backend.stack(parts, axis=0), axis=0)
        return self.normalize(result)

    def unbind(self, c: Array, b: Array) -> Array:
        """Approximate unbinding using weighted inverse transforms.

        b̂ = Σ_k w_k(a) · roll(c, -s_k)
        """
        # Use same transform derived from b
        w = self._weights(b)
        parts = []
        for k, shift in enumerate(self.shifts):
            wk = w[k]
            rc = self.backend.roll(c, shift=-shift)
            parts.append(self.backend.multiply_scalar(rc, float(self.backend.to_numpy(wk))))
        num = self.backend.sum(self.backend.stack(parts, axis=0), axis=0)
        # Denominator as sum of squared weights (scalar)
        w_np = self.backend.to_numpy(w)
        denom = float((w_np ** 2).sum()) + 1e-8
        result = self.backend.multiply_scalar(num, 1.0 / denom)
        return self.normalize(result)

    def bundle(self, vectors: Sequence[Array]) -> Array:
        """Bundle using element-wise addition.

        Sum all hypervectors element-wise and normalize.

        Args:
            vectors: Sequence of hypervectors to bundle

        Returns:
            Bundled hypervector

        Raises:
            ValueError: If vectors is empty
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")

        vectors = list(vectors)

        # Sum all vectors
        result = self.backend.sum(self.backend.stack(vectors, axis=0), axis=0)

        # Normalize to unit length
        return self.normalize(result)

    def permute(self, vec: Array, k: int = 1) -> Array:
        """Permute using circular shift.

        Shifts vector elements by k positions. Combined with binding,
        this can encode position in sequences.

        Args:
            vec: Hypervector to permute
            k: Number of positions to shift (default: 1)

        Returns:
            Permuted hypervector
        """
        return self.backend.roll(vec, shift=k)

    def test_non_commutativity(self, a: Array, b: Array) -> float:
        """Test degree of non-commutativity for two hypervectors.

        Computes: similarity(a ⊗ b, b ⊗ a)

        A similarity of 1.0 means commutative, close to 0 means non-commutative.

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Similarity between a⊗b and b⊗a (should be low for VTB)
        """
        ab = self.bind(a, b)
        ba = self.bind(b, a)
        return self.similarity(ab, ba)

    def bind_sequence(self, items: Sequence[Array], use_permute: bool = True) -> Array:
        """Bind a sequence of items with positional encoding.

        Two strategies:
        1. With permutation: c = a₁ ⊗ ρ⁰(pos) + a₂ ⊗ ρ¹(pos) + ...
        2. Without permutation: c = (...((a₁ ⊗ a₂) ⊗ a₃)...) (nested binding)

        Args:
            items: Sequence of hypervectors to bind
            use_permute: If True, use permutation strategy; else nested binding

        Returns:
            Sequence hypervector

        Raises:
            ValueError: If items is empty
        """
        if not items:
            raise ValueError("Cannot bind empty sequence")

        items = list(items)

        if use_permute:
            # Strategy 1: Bind each item with permuted position vector
            pos = self.random(seed=42)  # Fixed position vector
            bound_items = []

            for i, item in enumerate(items):
                permuted_pos = self.permute(pos, k=i)
                bound_items.append(self.bind(item, permuted_pos))

            return self.bundle(bound_items)
        else:
            # Strategy 2: Nested binding (naturally non-commutative)
            result = items[0]
            for item in items[1:]:
                result = self.bind(result, item)
            return result

    def __repr__(self) -> str:
        return (f"VTBModel(dimension={self.dimension}, "
                f"space={self.space.space_name}, "
                f"backend={self.backend.name})")


# Helper function to compare VTB vs HRR unbinding quality
def compare_vtb_hrr_unbinding(dimension: int = 1000, trials: int = 10) -> dict:
    """Compare unbinding quality between VTB and HRR.

    Both models use approximate inverse, but with different mechanisms.
    This function empirically measures recovery quality.

    Args:
        dimension: Dimensionality of hypervectors
        trials: Number of random trials

    Returns:
        Dictionary with comparison statistics
    """
    from ..backends import get_backend
    from .hrr import HRRModel

    backend = get_backend('numpy')

    vtb = VTBModel(dimension=dimension, backend=backend)
    hrr = HRRModel(dimension=dimension, backend=backend)

    vtb_similarities = []
    hrr_similarities = []

    for i in range(trials):
        # Generate random vectors
        a_vtb = vtb.random(seed=i*2)
        b_vtb = vtb.random(seed=i*2+1)

        a_hrr = hrr.random(seed=i*2)
        b_hrr = hrr.random(seed=i*2+1)

        # Bind
        c_vtb = vtb.bind(a_vtb, b_vtb)
        c_hrr = hrr.bind(a_hrr, b_hrr)

        # Unbind (recover the second operand for VTB, first for HRR)
        b_recovered_vtb = vtb.unbind(c_vtb, a_vtb)
        a_recovered_hrr = hrr.unbind(c_hrr, b_hrr)

        # Measure similarity
        sim_vtb = vtb.similarity(b_vtb, b_recovered_vtb)
        sim_hrr = hrr.similarity(a_hrr, a_recovered_hrr)

        vtb_similarities.append(sim_vtb)
        hrr_similarities.append(sim_hrr)

    return {
        'vtb_mean': float(np.mean(vtb_similarities)),
        'vtb_std': float(np.std(vtb_similarities)),
        'hrr_mean': float(np.mean(hrr_similarities)),
        'hrr_std': float(np.std(hrr_similarities)),
        'vtb_better': np.mean(vtb_similarities) > np.mean(hrr_similarities),
    }
