"""BSDC (Binary Sparse Distributed Codes) VSA model.

BSDC uses sparse binary vectors where only a small fraction of bits are set to 1.
This makes them memory-efficient and biologically plausible, while maintaining
the key properties of hyperdimensional computing.

Properties:
- Binding: XOR (self-inverse) or CDT (context-dependent thinning)
- Bundling: Majority voting with sparsity preservation
- Sparsity: Typically p = 1/√D (optimal for capacity)
- Memory efficient: Can use sparse data structures
- Biologically plausible: Similar to sparse neural codes

Binding Modes:
- 'xor': Traditional XOR binding (self-inverse, result dissimilar to inputs)
- 'cdt': Context-Dependent Thinning (preserves similarity to components)

Key Advantages:
- Very memory efficient for high dimensions (D > 10000)
- Biological plausibility (cortical neurons ~1% active)
- Fast operations on sparse representations
- Good capacity with much lower memory footprint
- CDT mode preserves both structured and unstructured similarity

Optimal Sparsity:
- p = 1/√D maximizes capacity (from information theory)
- For D=10,000: p ≈ 0.01 (1% of bits are 1)
- For D=100,000: p ≈ 0.003 (0.3% of bits are 1)

References:
- Kanerva (1988): "Sparse Distributed Memory" (foundational work)
- Rachkovskij (2001): "Binary Sparse Distributed Codes for Structures" (CDT)
- Kleyko et al. (2023): HDC/VSA Survey (BSDC comparison)
"""

from collections.abc import Sequence

import numpy as np

from ..backends import Backend
from ..backends.base import Array
from ..spaces import SparseSpace, VectorSpace
from .base import VSAModel


class BSDCModel(VSAModel):
    """BSDC (Binary Sparse Distributed Codes) model.

    Binding: XOR (element-wise, self-inverse) or CDT (context-dependent thinning)
    Unbinding: XOR (same as binding) or similarity-based (CDT)
    Bundling: Majority voting with sparsity preservation
    Permutation: circular shift

    Uses SparseSpace with optimal sparsity p = 1/√D.

    Binding Modes:
        - 'xor': Traditional XOR binding. Self-inverse, result dissimilar to inputs.
        - 'cdt': Context-Dependent Thinning (Rachkovskij 2001). Preserves both
          structured similarity (similar inputs → similar outputs) and unstructured
          similarity (result similar to its components).

    Example:
        >>> # Default XOR mode
        >>> model = BSDCModel(dimension=10000)
        >>>
        >>> # CDT mode for analogical reasoning
        >>> model = BSDCModel(dimension=10000, binding_mode='cdt')
    """

    def __init__(
        self,
        dimension: int = 10000,
        sparsity: float | None = None,
        space: VectorSpace | None = None,
        backend: Backend | None = None,
        seed: int | None = None,
        binding_mode: str = 'xor',
    ):
        """Initialize BSDC model.

        Args:
            dimension: Dimensionality of hypervectors (typically > 1000)
            sparsity: Fraction of 1s (default: 1/√D which is optimal)
            space: Vector space (defaults to SparseSpace with optimal sparsity)
            backend: Computational backend
            seed: Random seed for space
            binding_mode: 'xor' (default) or 'cdt' for context-dependent thinning
        """
        if binding_mode not in ('xor', 'cdt'):
            raise ValueError(f"binding_mode must be 'xor' or 'cdt', got '{binding_mode}'")

        if space is None:
            from ..backends import get_backend
            backend = backend if backend is not None else get_backend()
            space = SparseSpace(dimension, sparsity=sparsity, backend=backend, seed=seed)

        super().__init__(space, backend)

        self.binding_mode = binding_mode
        self._seed = seed

        # Store sparsity for easy access
        if isinstance(space, SparseSpace):
            self.sparsity = space.sparsity
        else:
            # Fallback if using non-sparse space
            import math
            self.sparsity = sparsity if sparsity is not None else 1.0 / math.sqrt(dimension)

        # Pre-generate permutation patterns for CDT
        if binding_mode == 'cdt':
            self._cdt_permutations = self._generate_cdt_permutations()

    @property
    def model_name(self) -> str:
        return "BSDC"

    @property
    def is_self_inverse(self) -> bool:
        return self.binding_mode == 'xor'  # Only XOR is self-inverse

    @property
    def is_commutative(self) -> bool:
        return True  # Both XOR and CDT are commutative

    @property
    def is_exact_inverse(self) -> bool:
        return self.binding_mode == 'xor'  # Only XOR has exact inverse

    def _generate_cdt_permutations(self, n_permutations: int = 20) -> list:
        """Generate fixed permutation patterns for CDT thinning.

        Args:
            n_permutations: Number of permutation patterns to generate

        Returns:
            List of permutation index arrays
        """
        rng = np.random.default_rng(self._seed if self._seed is not None else 42)
        return [rng.permutation(self.dimension) for _ in range(n_permutations)]

    def _compute_thinning_iterations(
        self,
        n_components: int,
        current_density: float,
    ) -> int:
        """Compute K iterations needed to reach target sparsity.

        The CDT algorithm thins a superposition by applying permuted self-conjunction.
        After OR of S components: p(Z) ≈ 1 - (1-p)^S ≈ p*S (for small p)
        We need K iterations to reduce back to target sparsity.

        Args:
            n_components: Number of components in superposition
            current_density: Current density after OR superposition

        Returns:
            Number of thinning iterations K
        """
        import math

        if current_density <= self.sparsity:
            return 0

        # From Rachkovskij 2001:
        # p(Z ∧ Z^~) ≈ p(Z)^2 for random permutations
        # After K iterations with OR of permutations:
        # Expected density ≈ current_density * (density of OR of K permuted copies)
        # We want: current_density * OR_density ≈ target_sparsity

        # Simplified: K ≈ target_sparsity / current_density^2
        K = max(1, int(math.ceil(self.sparsity / (current_density ** 2))))
        return min(K, len(self._cdt_permutations))

    def context_dependent_thinning(
        self,
        components: Sequence[Array],
    ) -> Array:
        """Bind components using context-dependent thinning (CDT).

        Algorithm (Rachkovskij 2001):
            1. Superpose components via OR: Z = X₁ ∨ X₂ ∨ ... ∨ Xₛ
            2. Thin via permuted self-conjunction:
               ⟨Z⟩ = Z ∧ (Z^~(1) ∨ Z^~(2) ∨ ... ∨ Z^~(K))

        Properties:
            - Preserves unstructured similarity: result is similar to each component
            - Preserves structured similarity: similar inputs → similar outputs
            - Maintains target sparsity automatically

        Args:
            components: Sequence of hypervectors to bind together

        Returns:
            Bound hypervector with preserved similarity to components

        Example:
            >>> model = BSDCModel(dimension=10000, binding_mode='cdt')
            >>> a, b, c = model.random(), model.random(), model.random()
            >>> bound = model.context_dependent_thinning([a, b, c])
            >>> # bound is similar to a, b, and c (unstructured similarity)
        """
        if not components:
            raise ValueError("Cannot bind empty sequence")

        components = list(components)

        if len(components) == 1:
            return components[0].copy() if hasattr(components[0], 'copy') else components[0]

        # Convert to numpy for efficient logical operations
        components_np = [self.backend.to_numpy(c) for c in components]

        # Step 1: Superpose via OR
        z = components_np[0].astype(bool)
        for c in components_np[1:]:
            z = np.logical_or(z, c.astype(bool))

        # Step 2: Compute required thinning iterations
        current_density = float(np.sum(z)) / self.dimension
        K = self._compute_thinning_iterations(len(components), current_density)

        if K == 0:
            # Already at or below target sparsity
            result = z.astype(np.int32)
            return self.backend.from_numpy(result)

        # Step 3: Thin via permuted self-conjunction
        # ⟨Z⟩ = Z ∧ (Z^~(1) ∨ Z^~(2) ∨ ... ∨ Z^~(K))
        permuted_or = np.zeros(self.dimension, dtype=bool)
        for k in range(K):
            perm_idx = k % len(self._cdt_permutations)
            z_permuted = z[self._cdt_permutations[perm_idx]]
            permuted_or = np.logical_or(permuted_or, z_permuted)

        result = np.logical_and(z, permuted_or).astype(np.int32)
        return self.backend.from_numpy(result)

    def bind(self, a: Array, b: Array) -> Array:
        """Bind two hypervectors.

        Behavior depends on binding_mode:
        - 'xor': XOR binding (self-inverse, result dissimilar to inputs)
        - 'cdt': Context-dependent thinning (preserves similarity to inputs)

        For XOR mode:
            - Preserves sparsity on average: p(1-p) + (1-p)p = 2p(1-p)
            - For optimal p = 1/√D, result sparsity ≈ 2/√D

        For CDT mode:
            - Result is similar to both a and b (unstructured similarity)
            - Similar inputs produce similar outputs (structured similarity)

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Bound hypervector
        """
        if self.binding_mode == 'cdt':
            return self.context_dependent_thinning([a, b])
        else:
            # XOR binding (default)
            return self.backend.xor(a, b)

    def unbind(self, a: Array, b: Array) -> Array:
        """Unbind to recover value.

        Behavior depends on binding_mode:
        - 'xor': XOR is self-inverse, exact recovery: unbind(bind(a, b), b) = a
        - 'cdt': No inverse exists; returns the bound vector itself since it's
          already similar to the components (use similarity search for retrieval)

        Args:
            a: Bound hypervector (or first operand)
            b: Second operand (key for XOR mode, ignored for CDT mode)

        Returns:
            For XOR: Exact unbound hypervector
            For CDT: The bound vector (use similarity search to find components)
        """
        if self.binding_mode == 'cdt':
            # CDT doesn't have an inverse operation
            # The bound vector is already similar to its components,
            # so return it for similarity-based retrieval
            return a
        else:
            # XOR is self-inverse
            return self.backend.xor(a, b)

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
                f"binding_mode='{self.binding_mode}', "
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


def expected_ones(dimension: int, sparsity: float | None = None) -> int:
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
