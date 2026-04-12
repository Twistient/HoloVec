"""Concrete vector space implementations.

This module implements the specific vector spaces used in various VSA models:
- BipolarSpace: {-1, +1} for MAP
- BinarySpace: {0, 1} for BSC
- RealSpace: Gaussian N(0, 1/D) for HRR, VTB
- ComplexSpace: Unit phasors for FHRR
- SparseSpace: Sparse binary for BSDC, SBC
"""


from collections.abc import Iterator

import numpy as np

from ..backends import Backend
from ..backends.base import Array
from .base import ContinuousSpace, DiscreteSpace


class BipolarSpace(DiscreteSpace):
    """Bipolar vector space with values in {-1, +1}.

    Used by MAP (Multiply-Add-Permute) model. Binding is element-wise
    multiplication, which is self-inverse.

    Properties:
    - Self-inverse binding
    - Simple hardware implementation
    - XOR equivalent when represented as {0, 1}
    """

    @property
    def space_name(self) -> str:
        return "bipolar"

    @property
    def dtype(self) -> str:
        return "float32"

    def random(self, seed: int | None = None) -> Array:
        """Generate random bipolar vector with equal probability for ±1.

        Args:
            seed: Optional seed for this specific vector

        Returns:
            Bipolar vector with values in {-1, +1}
        """
        return self.backend.random_bipolar(self.dimension, p=0.5, dtype=self.dtype, seed=seed)

    def normalize(self, vec: Array) -> Array:
        """Clip to bipolar values {-1, +1}.

        For MAP, normalization often means projecting back to ±1.
        """
        return self.backend.sign(vec)


class BinarySpace(DiscreteSpace):
    """Binary vector space with values in {0, 1}.

    Used by BSC (Binary Spatter Codes). Binding is XOR operation,
    which is self-inverse.

    Properties:
    - Self-inverse binding (XOR)
    - Mathematically equivalent to bipolar MAP
    - Simple bit operations
    """

    @property
    def space_name(self) -> str:
        return "binary"

    @property
    def dtype(self) -> str:
        return "int32"

    def random(self, seed: int | None = None) -> Array:
        """Generate random binary vector with equal probability for 0 and 1.

        Args:
            seed: Optional seed for this specific vector

        Returns:
            Binary vector with values in {0, 1}
        """
        return self.backend.random_binary(self.dimension, p=0.5, dtype=self.dtype, seed=seed)

    def normalize(self, vec: Array) -> Array:
        """Threshold to binary values {0, 1}.

        For BSC, normalization means majority vote: values >= threshold → 1, else 0.
        """
        binary = self.backend.threshold(vec, threshold=0.5, above=1, below=0)
        return self.backend.astype(binary, self.dtype)


class RealSpace(ContinuousSpace):
    """Real-valued vector space with Gaussian distribution.

    Used by HRR (Holographic Reduced Representations) and VTB.
    Vectors are drawn from N(0, 1/D) and L2-normalized.

    Properties:
    - Approximate inverse binding
    - Circular convolution for binding
    - Continuous similarity measures
    """

    @property
    def space_name(self) -> str:
        return "real"

    @property
    def dtype(self) -> str:
        return "float32"

    def random(self, seed: int | None = None) -> Array:
        """Generate random real vector from N(0, 1/D) and normalize.

        Args:
            seed: Optional seed for this specific vector

        Returns:
            L2-normalized real vector
        """
        # Sample from N(0, 1/D) where D is dimension
        import math
        std = 1.0 / math.sqrt(self.dimension)
        vec = self.backend.random_normal(
            self.dimension,
            mean=0.0,
            std=std,
            dtype=self.dtype,
            seed=seed
        )
        return self.normalize(vec)


class ComplexSpace(ContinuousSpace):
    """Complex vector space with unit phasors.

    Used by FHRR (Fourier HRR). Vectors are unit-magnitude complex numbers
    with uniformly distributed phase angles.

    Properties:
    - Exact inverse binding (complex conjugate)
    - Best capacity among VSA models
    - Natural for frequency-domain operations
    - Fractional power encoding for continuous data
    """

    @property
    def space_name(self) -> str:
        return "complex"

    @property
    def dtype(self) -> str:
        return "complex64"

    def random(self, seed: int | None = None) -> Array:
        """Generate random unit phasor with uniform phase angle.

        Args:
            seed: Optional seed for this specific vector

        Returns:
            Complex vector with magnitude 1 and random phase
        """
        return self.backend.random_phasor(self.dimension, dtype=self.dtype, seed=seed)

    def normalize(self, vec: Array) -> Array:
        """Normalize to unit magnitude.

        For complex vectors, this means |z| = 1 for all components.
        """
        magnitudes = self.backend.abs(vec)
        # Avoid division by zero
        magnitudes = self.backend.clip(magnitudes, 1e-12, float('inf'))
        return vec / magnitudes

    def similarity(self, a: Array, b: Array) -> float:
        """Cosine similarity for complex vectors.

        For unit phasors, this is equivalent to the real part of the
        normalized dot product.
        """
        # For unit phasors: <a, b*> / (|a| |b|) = <a, b*>
        dot_product = self.backend.dot(a, self.backend.conjugate(b))
        # Take real part robustly across backends and normalize by dimension
        real_part = self.backend.real(dot_product)
        real_value = float(self.backend.to_numpy(real_part)) / self.dimension
        # Numerical safety: clamp to [-1, 1]
        if real_value > 1.0:
            return 1.0
        if real_value < -1.0:
            return -1.0
        return real_value


class SparseSpace(DiscreteSpace):
    """Sparse binary vector space.

    Used by BSDC (Binary Sparse Distributed Codes) and SBC.
    Vectors have a small fraction of 1s, rest are 0s.

    Properties:
    - Memory efficient
    - Biologically plausible
    - Optimal sparsity: p = 1/√D
    - Block-based variants for efficiency
    """

    def __init__(
        self,
        dimension: int,
        sparsity: float | None = None,
        backend: Backend | None = None,
        seed: int | None = None
    ):
        """Initialize sparse space.

        Args:
            dimension: Dimensionality of vectors
            sparsity: Fraction of 1s (default: 1/√D which is optimal)
            backend: Computational backend
            seed: Random seed
        """
        super().__init__(dimension, backend, seed)

        if sparsity is None:
            # Optimal sparsity from literature
            import math
            self.sparsity = 1.0 / math.sqrt(dimension)
        else:
            if not 0 < sparsity < 1:
                raise ValueError(f"Sparsity must be in (0, 1), got {sparsity}")
            self.sparsity = sparsity

    @property
    def space_name(self) -> str:
        return "sparse"

    @property
    def dtype(self) -> str:
        return "int32"

    def random(self, seed: int | None = None) -> Array:
        """Generate random sparse binary vector.

        Args:
            seed: Optional seed for this specific vector

        Returns:
            Sparse binary vector with specified sparsity
        """
        return self.backend.random_binary(
            self.dimension,
            p=self.sparsity,
            dtype=self.dtype,
            seed=seed
        )

    def normalize(self, vec: Array) -> Array:
        """Threshold and maintain sparsity.

        For sparse vectors, we threshold to {0, 1} and optionally
        enforce sparsity constraint.
        """
        # Simple threshold to binary
        binary = self.backend.threshold(vec, threshold=0.5, above=1, below=0)
        return self.backend.astype(binary, self.dtype)

    def similarity(self, a: Array, b: Array) -> float:
        """Overlap-based similarity for sparse vectors.

        For sparse vectors, Jaccard similarity or overlap coefficient
        is often more appropriate than Hamming distance.

        Returns:
            Overlap coefficient in [0, 1]
        """
        # Compute overlap (intersection)
        intersection = float(self.backend.sum(self.backend.multiply(a, b)))

        # Compute minimum of the two counts (conservative similarity)
        count_a = float(self.backend.sum(a))
        count_b = float(self.backend.sum(b))

        min_count = min(count_a, count_b)
        if min_count == 0:
            return 0.0

        return intersection / min_count


class SparseSegmentSpace(DiscreteSpace):
    """Segmented sparse binary vector space with exactly 1-hot per segment.

    Dimension D is split into S equal segments (D % S == 0). Each segment has
    length L = D / S and contains exactly one active bit. This yields a total of
    S ones across the vector. Similarity is the fraction of segments where the
    active index matches (in [0,1]).

    This space underlies BSDC-SEG style codes.
    """

    def __init__(
        self,
        dimension: int,
        segments: int,
        backend: Backend | None = None,
        seed: int | None = None,
    ):
        if segments < 1:
            raise ValueError(f"segments must be >= 1, got {segments}")
        if dimension % segments != 0:
            raise ValueError(
                f"dimension ({dimension}) must be divisible by segments ({segments})"
            )
        self.segments = int(segments)
        self.segment_length = dimension // segments
        super().__init__(dimension, backend, seed)

    @property
    def space_name(self) -> str:
        return f"sparse_segment(s={self.segments})"

    @property
    def dtype(self) -> str:
        return "int32"

    def _segment_ranges(self) -> Iterator[tuple[int, int, int]]:
        L = self.segment_length
        for s in range(self.segments):
            start = s * L
            end = start + L
            yield s, start, end

    def _segment_one_hot(self, indices: Array) -> Array:
        positions = self.backend.array(list(range(self.segment_length)), dtype='int32')
        one_hot = indices[:, None] == positions[None, :]
        return self.backend.astype(one_hot, self.dtype)

    def random(self, seed: int | None = None) -> Array:
        indices = self.backend.randint(
            (self.segments,),
            low=0,
            high=self.segment_length,
            dtype='int32',
            seed=seed,
        )
        one_hot = self._segment_one_hot(indices)
        return self.backend.reshape(one_hot, (self.dimension,))

    def normalize(self, vec: Array) -> Array:
        """Project to nearest valid pattern: 1-hot per segment.

        For each segment, pick argmax index; set it to 1 and others to 0.
        Works for binary or real-valued inputs.
        """
        if self.backend.shape(vec) != (self.dimension,):
            raise ValueError("Expected 1D vector of length D for normalize()")
        segments = self.backend.reshape(vec, (self.segments, self.segment_length))
        indices = self.backend.argmax(segments, axis=1)
        normalized = self._segment_one_hot(indices)
        return self.backend.reshape(normalized, (self.dimension,))

    def similarity(self, a: Array, b: Array) -> float:
        """Segment-wise match fraction in [0,1].

        Converts inputs to nearest valid 1-hot per segment, then compares
        the active index per segment.
        """
        a_segments = self.backend.reshape(self.normalize(a), (self.segments, self.segment_length))
        b_segments = self.backend.reshape(self.normalize(b), (self.segments, self.segment_length))
        a_idx = self.backend.argmax(a_segments, axis=1)
        b_idx = self.backend.argmax(b_segments, axis=1)
        matches = self.backend.astype(a_idx == b_idx, 'float32')
        return float(self.backend.to_numpy(self.backend.mean(matches)))

    # ===== Segment-aware helpers =====
    def segment_argmax(self, vec: Array) -> np.ndarray:
        """Return indices (length S) of active positions per segment after normalize()."""
        segments = self.backend.reshape(self.normalize(vec), (self.segments, self.segment_length))
        indices = self.backend.argmax(segments, axis=1)
        return np.asarray(self.backend.to_numpy(indices), dtype=np.int32)

    def mask_segments(self, vec: Array, keep: np.ndarray | list[int]) -> Array:
        """Zero out all segments not in keep.

        keep: iterable of segment indices to retain (0..S-1).
        """
        keep_set = {int(x) for x in keep}
        segments = self.backend.reshape(vec, (self.segments, self.segment_length))
        masked_segments = []
        for s in range(self.segments):
            if s in keep_set:
                masked_segments.append(segments[s])
            else:
                masked_segments.append(self.backend.zeros(self.segment_length, dtype=self.dtype))
        return self.backend.concatenate(masked_segments, axis=0)

    def select_segments(self, vec: Array, select: np.ndarray | list[int]) -> Array:
        """Return a compacted vector consisting of concatenated selected segments.

        This is often useful for analysis; not a fixed-length holovec vector.
        """
        sel = [int(x) for x in select]
        segments = self.backend.reshape(vec, (self.segments, self.segment_length))
        parts = [segments[s] for s in sel]
        return self.backend.concatenate(parts, axis=0)

    def block_rotate(self, vec: Array, k: int = 1) -> Array:
        """Rotate indices within each segment by k positions (cyclic)."""
        segments = self.backend.reshape(vec, (self.segments, self.segment_length))
        kk = int(k) % self.segment_length
        rotated = [self.backend.roll(segments[s], shift=kk, axis=0) for s in range(self.segments)]
        return self.backend.concatenate(rotated, axis=0)

    def block_permute(self, vec: Array, perm: np.ndarray | list[int]) -> Array:
        """Apply a fixed within-segment permutation (length L) to every segment."""
        p = np.asarray(perm, dtype=np.int64)
        if p.shape[0] != self.segment_length:
            raise ValueError(
                f"perm length must equal segment_length ({self.segment_length})"
            )
        perm_indices = self.backend.array(p.tolist(), dtype='int64')
        segments = self.backend.reshape(vec, (self.segments, self.segment_length))
        permuted = [self.backend.permute(segments[s], perm_indices) for s in range(self.segments)]
        return self.backend.concatenate(permuted, axis=0)

class MatrixSpace(ContinuousSpace):
    """Matrix-valued vector space for GHRR.

    Each "dimension" is actually an m×m unitary matrix. Used by GHRR
    (Generalized Holographic Reduced Representations) for non-commutative
    binding and better compositional structure encoding.

    Properties:
    - Non-commutative binding (controlled by matrix structure)
    - Better capacity for bound hypervectors
    - Tunable commutativity via diagonality
    - Adaptive kernels via input-dependent Q matrices
    """

    def __init__(
        self,
        dimension: int,
        matrix_size: int = 3,
        backend: Backend | None = None,
        seed: int | None = None,
        diagonality: float | None = None,
    ):
        """Initialize matrix space.

        Args:
            dimension: Number of matrices in the hypervector
            matrix_size: Size m of each m×m matrix (default: 3)
            backend: Computational backend
            seed: Random seed
        """
        if matrix_size < 1:
            raise ValueError(f"Matrix size must be positive, got {matrix_size}")

        self.matrix_size = matrix_size
        self.diagonality = diagonality
        super().__init__(dimension, backend, seed)

    @property
    def space_name(self) -> str:
        return f"matrix_{self.matrix_size}x{self.matrix_size}"

    @property
    def dtype(self) -> str:
        return "complex64"

    def random(self, seed: int | None = None) -> Array:
        """Generate random GHRR hypervector with QΛ structure.

        Returns a stack of D matrices of size (m×m): H_j = Q_j @ Λ_j
        where Q_j ∈ U(m) and Λ_j = diag(e^{iθ₁}, ..., e^{iθₘ}).

        The "diagonality" parameter (if set in [0,1]) interpolates Q_j
        toward identity: 0.0 → Haar-random unitary, 1.0 → identity.
        """
        m = self.matrix_size
        D = self.dimension
        real_seed = seed
        imag_seed = None if seed is None else seed + 1
        phase_seed = None if seed is None else seed + 2

        real = self.backend.random_normal((D, m, m), dtype='float32', seed=real_seed)
        imag = self.backend.random_normal((D, m, m), dtype='float32', seed=imag_seed)
        ginibre = self.backend.astype(real + 1j * imag, self.dtype)

        # U @ Vh projects the random complex batch onto the unitary group.
        U, _S, Vh = self.backend.svd(ginibre, full_matrices=True)
        Q = self.backend.astype(self.backend.matmul(U, Vh), self.dtype)

        alpha = self.diagonality if self.diagonality is not None else 0.0
        identity = self.backend.eye(m, dtype=self.dtype)
        identity_batch = self.backend.stack([identity for _ in range(D)], axis=0)

        if alpha >= 1.0:
            Q_eff = identity_batch
        elif alpha <= 0.0:
            Q_eff = Q
        else:
            blended = (1.0 - alpha) * Q + alpha * identity_batch
            U2, _S2, Vh2 = self.backend.svd(blended, full_matrices=True)
            Q_eff = self.backend.astype(self.backend.matmul(U2, Vh2), self.dtype)

        Lambda = self.backend.random_phasor((D, m, m), dtype=self.dtype, seed=phase_seed)
        return self.backend.matmul(Q_eff, Lambda)

    def similarity(self, a: Array, b: Array) -> float:
        """GHRR similarity: δ(H₁, H₂) = (1/mD) Re(tr(Σⱼ aⱼbⱼ†)).

        Args:
            a: First hypervector (D, m, m)
            b: Second hypervector (D, m, m)

        Returns:
            Similarity score in [-1, 1]
        """
        m = self.matrix_size
        D = self.dimension

        # Compute b^† (conjugate transpose of each matrix)
        b_conj_t = self.backend.conjugate(self.backend.matrix_transpose(b))

        # Element-wise matrix multiply: aⱼ @ bⱼ†
        products = self.backend.matmul(a, b_conj_t)

        # Compute trace of each product
        traces = self.backend.matrix_trace(products)

        # Sum all traces
        total = self.backend.sum(traces)

        # Normalize and take real part
        similarity = self.backend.to_numpy(total).real / (m * D)

        return float(similarity)

    def normalize(self, vec: Array) -> Array:
        """Normalize to ensure each matrix is unitary via polar decomposition.

        For GHRR, normalization means ensuring each m×m matrix in the hypervector
        is unitary (U†U = I). This is critical for maintaining quasi-orthogonality
        properties after bundling operations.

        Uses polar decomposition via SVD: for matrix H = U·Σ·Vh, the closest
        unitary matrix is U·Vh (discarding the singular values Σ).

        Reference:
            Yeung et al. (2024): "Generalized Holographic Reduced Representations"
            Section 4.1 - Unitarity requirement for quasi-orthogonality

        Args:
            vec: Array of matrices (D, m, m)

        Returns:
            Normalized array where each matrix is unitary

        Note:
            This uses SVD which is O(m³) per matrix. For typical GHRR with
            small m (e.g., m=2-5) and moderate D (100-1000), this is efficient.
        """
        # Use SVD to compute polar decomposition: H = U·Σ·Vh
        # The unitary part of the polar decomposition is U·Vh
        U, S, Vh = self.backend.svd(vec, full_matrices=True)

        # Compute unitary part: U_polar = U @ Vh
        # This is the closest unitary matrix to vec in Frobenius norm
        unitary = self.backend.matmul(U, Vh)

        return unitary


# Convenience factory function
def create_space(
    space_type: str,
    dimension: int,
    backend: Backend | None = None,
    seed: int | None = None,
    **kwargs: object,
) -> DiscreteSpace | ContinuousSpace:
    """Factory function to create vector spaces.

    Args:
        space_type: One of 'bipolar', 'binary', 'real', 'complex', 'sparse'
        dimension: Dimensionality of vectors
        backend: Computational backend
        seed: Random seed
        **kwargs: Space-specific arguments (e.g., sparsity for SparseSpace)

    Returns:
        Vector space instance

    Examples:
        >>> space = create_space('bipolar', 10000)
        >>> space = create_space('complex', 512)
        >>> space = create_space('sparse', 10000, sparsity=0.01)
    """
    space_map: dict[str, type[DiscreteSpace] | type[ContinuousSpace]] = {
        'bipolar': BipolarSpace,
        'binary': BinarySpace,
        'real': RealSpace,
        'complex': ComplexSpace,
        'sparse': SparseSpace,
        'sparse_segment': SparseSegmentSpace,
        'matrix': MatrixSpace,
    }

    space_type = space_type.lower()
    if space_type not in space_map:
        raise ValueError(f"Unknown space type '{space_type}'. Available: {list(space_map.keys())}")

    space_class = space_map[space_type]
    return space_class(dimension, backend=backend, seed=seed, **kwargs)
