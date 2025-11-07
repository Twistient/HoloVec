"""BSDC (Binary Sparse Distributed Codes) VSA model.

BSDC uses sparse binary vectors where only a small fraction of bits are set to 1.
This makes them memory-efficient and biologically plausible, while maintaining
the key properties of hyperdimensional computing.

Properties:
- Binding: XOR (self-inverse)
- Bundling: Majority voting with sparsity preservation
- Sparsity: Typically p = 1/√D (optimal for capacity)
- Memory efficient: Can use sparse data structures
- Biologically plausible: Similar to sparse neural codes
- Self-inverse binding

Key Advantages:
- Very memory efficient for high dimensions (D > 10000)
- Biological plausibility (cortical neurons ~1% active)
- Fast operations on sparse representations
- Good capacity with much lower memory footprint

Optimal Sparsity:
- p = 1/√D maximizes capacity (from information theory)
- For D=10,000: p ≈ 0.01 (1% of bits are 1)
- For D=100,000: p ≈ 0.003 (0.3% of bits are 1)

References:
- Kanerva (1988): "Sparse Distributed Memory" (foundational work)
- Rachkovskij & Kussul (2001): "Binding and normalization of binary sparse codes"
- Kleyko et al. (2023): HDC/VSA Survey (BSDC comparison)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..backends import Backend
from ..backends.base import Array
from ..spaces import SparseSpace, VectorSpace
from .base import VSAModel


class BSDCModel(VSAModel):
    """BSDC (Binary Sparse Distributed Codes) model.

    Binding: XOR (element-wise, self-inverse)
    Unbinding: XOR (same as binding, self-inverse)
    Bundling: Majority voting with sparsity preservation
    Permutation: circular shift

    Uses SparseSpace with optimal sparsity p = 1/√D.
    """

    def __init__(
        self,
        dimension: int = 10000,
        sparsity: Optional[float] = None,
        space: Optional[VectorSpace] = None,
        backend: Optional[Backend] = None,
        seed: Optional[int] = None
    ):
        """Initialize BSDC model.

        Args:
            dimension: Dimensionality of hypervectors (typically > 1000)
            sparsity: Fraction of 1s (default: 1/√D which is optimal)
            space: Vector space (defaults to SparseSpace with optimal sparsity)
            backend: Computational backend
            seed: Random seed for space
        """
        if space is None:
            from ..backends import get_backend
            backend = backend if backend is not None else get_backend()
            space = SparseSpace(dimension, sparsity=sparsity, backend=backend, seed=seed)

        super().__init__(space, backend)

        # Store sparsity for easy access
        if isinstance(space, SparseSpace):
            self.sparsity = space.sparsity
        else:
            # Fallback if using non-sparse space
            import math
            self.sparsity = sparsity if sparsity is not None else 1.0 / math.sqrt(dimension)

    @property
    def model_name(self) -> str:
        return "BSDC"

    @property
    def is_self_inverse(self) -> bool:
        return True  # XOR is self-inverse

    @property
    def is_commutative(self) -> bool:
        return True  # XOR is commutative

    @property
    def is_exact_inverse(self) -> bool:
        return True  # XOR is exact inverse of itself

    def bind(self, a: Array, b: Array) -> Array:
        """Bind using XOR.

        For sparse binary codes, XOR preserves sparsity on average.
        Expected sparsity of result: p(1-p) + (1-p)p = 2p(1-p)

        For optimal p = 1/√D, result sparsity ≈ 2/√D (slightly increased).

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Bound hypervector c = a XOR b
        """
        result = self.backend.xor(a, b)

        # Optional: re-sparsify if needed to maintain sparsity
        # For now, let XOR naturally handle it
        return result

    def unbind(self, a: Array, b: Array) -> Array:
        """Unbind using XOR (self-inverse).

        Since XOR is self-inverse: unbind(bind(a, b), b) = a

        Args:
            a: Bound hypervector (or first operand)
            b: Second operand

        Returns:
            Unbound hypervector (exact recovery)
        """
        # XOR is self-inverse
        return self.bind(a, b)

    def bundle(self, vectors: Sequence[Array], maintain_sparsity: bool = True) -> Array:
        """Bundle using majority voting.

        For sparse codes, bundling requires careful handling to maintain sparsity:
        1. Sum all vectors element-wise
        2. Apply threshold to get binary result
        3. Optionally re-sparsify to maintain target sparsity

        Args:
            vectors: Sequence of hypervectors to bundle
            maintain_sparsity: If True, enforce target sparsity (default: True)

        Returns:
            Bundled hypervector

        Raises:
            ValueError: If vectors is empty
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")

        vectors = list(vectors)

        # Sum all vectors (counts how many 1s at each position)
        sum_result = self.backend.sum(self.backend.stack(vectors, axis=0), axis=0)

        if maintain_sparsity:
            # Strategy: Take top-k positions with highest counts
            # where k ≈ sparsity * dimension
            sum_np = self.backend.to_numpy(sum_result)
            target_ones = int(self.sparsity * self.dimension)

            # Get indices of top-k values
            if target_ones > 0:
                # Use argpartition for efficiency (O(n) instead of O(n log n))
                threshold_idx = max(0, len(sum_np) - target_ones)
                threshold = np.partition(sum_np, threshold_idx)[threshold_idx]

                # Set positions >= threshold to 1, rest to 0
                result_np = (sum_np >= threshold).astype(np.int32)

                # If we have ties at the threshold, we might have slightly more
                # than target_ones. This is acceptable for maintaining sparsity.
                return self.backend.from_numpy(result_np)
            else:
                # No ones in result (edge case)
                return self.backend.zeros(self.dimension, dtype='int32')
        else:
            # Simple majority voting: threshold at N/2
            threshold = len(vectors) / 2.0
            result = self.backend.threshold(sum_result, threshold=threshold, above=1.0, below=0.0)
            return result.astype('int32')

    def permute(self, vec: Array, k: int = 1) -> Array:
        """Permute using circular shift.

        Shifts vector elements by k positions. For sparse codes,
        this maintains sparsity perfectly.

        Args:
            vec: Hypervector to permute
            k: Number of positions to shift (default: 1)

        Returns:
            Permuted hypervector
        """
        return self.backend.roll(vec, shift=k, axis=0)

    def measure_sparsity(self, vec: Array) -> float:
        """Measure actual sparsity of a vector.

        Args:
            vec: Hypervector to measure

        Returns:
            Fraction of 1s in the vector
        """
        vec_np = self.backend.to_numpy(vec)
        count_ones = np.sum(vec_np)
        return float(count_ones) / len(vec_np)

    def rehash(self, vec: Array) -> Array:
        """Rehash vector to restore optimal sparsity.

        Useful after multiple operations that may have changed sparsity.
        Randomly selects positions to maintain target sparsity while
        preserving as much similarity as possible.

        Args:
            vec: Hypervector to rehash

        Returns:
            Rehashed hypervector with target sparsity
        """
        vec_np = self.backend.to_numpy(vec)
        target_ones = int(self.sparsity * self.dimension)

        # Get current 1 positions
        current_ones = np.where(vec_np == 1)[0]
        current_count = len(current_ones)

        if current_count == target_ones:
            # Already at target sparsity
            return vec
        elif current_count > target_ones:
            # Too many 1s: randomly remove some
            keep_indices = np.random.choice(
                current_ones, size=target_ones, replace=False
            )
            result = np.zeros_like(vec_np)
            result[keep_indices] = 1
        else:
            # Too few 1s: randomly add some
            current_zeros = np.where(vec_np == 0)[0]
            add_count = target_ones - current_count
            add_indices = np.random.choice(
                current_zeros, size=add_count, replace=False
            )
            result = vec_np.copy()
            result[add_indices] = 1

        return self.backend.from_numpy(result.astype(np.int32))

    def encode_sequence(
        self,
        items: Sequence[Array],
        use_ngrams: bool = False,
        n: int = 2
    ) -> Array:
        """Encode sequence of items.

        Two strategies:
        1. Position binding: item_i ⊗ ρⁱ(position)
        2. N-grams: Bundle all n-grams in sequence

        Args:
            items: Sequence of hypervectors
            use_ngrams: If True, use n-gram encoding (default: False)
            n: N-gram size (default: 2 for bigrams)

        Returns:
            Sequence hypervector

        Raises:
            ValueError: If items is empty
        """
        if not items:
            raise ValueError("Cannot encode empty sequence")

        items = list(items)

        if use_ngrams:
            # N-gram encoding
            if len(items) < n:
                # Sequence too short for n-grams, fall back to simple bundle
                return self.bundle(items)

            ngrams = []
            for i in range(len(items) - n + 1):
                # Create n-gram by binding n consecutive items
                ngram = items[i]
                for j in range(1, n):
                    ngram = self.bind(ngram, items[i + j])
                ngrams.append(ngram)

            return self.bundle(ngrams)
        else:
            # Position binding encoding
            pos = self.random(seed=42)  # Fixed position vector
            bound_items = []

            for i, item in enumerate(items):
                permuted_pos = self.permute(pos, k=i)
                bound_items.append(self.bind(item, permuted_pos))

            return self.bundle(bound_items)

    def __repr__(self) -> str:
        return (f"BSDCModel(dimension={self.dimension}, "
                f"sparsity={self.sparsity:.4f}, "
                f"space={self.space.space_name}, "
                f"backend={self.backend.name})")


def optimal_sparsity(dimension: int) -> float:
    """Calculate optimal sparsity for given dimension.

    The optimal sparsity p = 1/√D maximizes the capacity of sparse
    distributed codes.

    Args:
        dimension: Dimensionality of hypervectors

    Returns:
        Optimal sparsity value

    Examples:
        >>> optimal_sparsity(10000)
        0.01
        >>> optimal_sparsity(100000)
        0.00316...
    """
    import math
    return 1.0 / math.sqrt(dimension)


def expected_ones(dimension: int, sparsity: Optional[float] = None) -> int:
    """Calculate expected number of 1s for given dimension and sparsity.

    Args:
        dimension: Dimensionality of hypervectors
        sparsity: Sparsity level (default: optimal = 1/√D)

    Returns:
        Expected number of 1 bits

    Examples:
        >>> expected_ones(10000)
        100
        >>> expected_ones(10000, sparsity=0.05)
        500
    """
    if sparsity is None:
        sparsity = optimal_sparsity(dimension)
    return int(dimension * sparsity)


def compare_sparse_vs_dense(
    dimension: int = 10000,
    trials: int = 10
) -> dict:
    """Compare BSDC (sparse) vs BSC (dense) performance.

    Compares memory usage, binding preservation, and capacity.

    Args:
        dimension: Dimensionality of hypervectors
        trials: Number of random trials

    Returns:
        Dictionary with comparison statistics
    """
    from ..backends import get_backend
    from .bsc import BSCModel

    backend = get_backend('numpy')

    bsdc = BSDCModel(dimension=dimension, backend=backend, seed=42)
    bsc = BSCModel(dimension=dimension, backend=backend, seed=42)

    # Measure memory efficiency
    a_bsdc = bsdc.random(seed=1)
    a_bsc = bsc.random(seed=1)

    ones_bsdc = float(backend.sum(a_bsdc))
    ones_bsc = float(backend.sum(a_bsc))

    memory_ratio = ones_bsdc / ones_bsc

    # Measure binding quality
    bsdc_sims = []
    bsc_sims = []

    for i in range(trials):
        a_bsdc = bsdc.random(seed=i*2)
        b_bsdc = bsdc.random(seed=i*2+1)

        a_bsc = bsc.random(seed=i*2)
        b_bsc = bsc.random(seed=i*2+1)

        # Bind and unbind
        c_bsdc = bsdc.bind(a_bsdc, b_bsdc)
        recovered_bsdc = bsdc.unbind(c_bsdc, b_bsdc)

        c_bsc = bsc.bind(a_bsc, b_bsc)
        recovered_bsc = bsc.unbind(c_bsc, b_bsc)

        # Measure recovery
        sim_bsdc = bsdc.similarity(a_bsdc, recovered_bsdc)
        sim_bsc = bsc.similarity(a_bsc, recovered_bsc)

        bsdc_sims.append(sim_bsdc)
        bsc_sims.append(sim_bsc)

    return {
        'memory_ratio': memory_ratio,  # BSDC/BSC (should be ~0.01 for D=10000)
        'bsdc_recovery_mean': float(np.mean(bsdc_sims)),
        'bsc_recovery_mean': float(np.mean(bsc_sims)),
        'bsdc_sparsity': bsdc.sparsity,
        'bsdc_expected_ones': expected_ones(dimension),
        'bsc_expected_ones': dimension // 2,  # ~50% for dense binary
    }
