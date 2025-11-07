"""Mathematical and numerical constants for HoloVec.

This module centralizes all magic numbers, default parameters, and numerical
constants used throughout the library. Centralizing these values:
1. Makes tuning and experimentation easier
2. Ensures consistency across modules
3. Documents the rationale for chosen values
4. Facilitates research reproducibility

All constants are documented with:
- Mathematical justification
- Typical use cases
- Literature references where applicable
"""

from __future__ import annotations

# =============================================================================
# Numerical Stability
# =============================================================================

# Epsilon for numerical stability in division and normalization
# Used to prevent division by zero and numerical overflow
EPSILON_DEFAULT = 1e-8
"""Default epsilon for numerical stability.

Used in operations like division, logarithms, and normalizations where
zero values could cause numerical issues. Value chosen to be:
- Small enough to not affect results (< machine precision significance)
- Large enough to prevent catastrophic cancellation
- Compatible with float32 precision (smallest normal: ~1.17e-38)

References:
    - IEEE 754 single precision: ~1.2e-38 (smallest normal)
    - Typical conditioning: epsilon should be << 1e-6 for stability
"""

EPSILON_NORMALIZATION = 1e-12
"""Epsilon for L2 normalization operations.

Used specifically in vector normalization to prevent division by zero
when normalizing zero vectors or very small vectors. Stricter than
EPSILON_DEFAULT because normalization is sensitive to scale.

References:
    - Numerical recipes recommend 1e-12 for double precision normalization
    - Ensures normalized vectors have magnitude close to 1.0 (within 1e-12)
"""

EPSILON_WIENER = 1e-8
"""Epsilon for Wiener deconvolution (HRR unbinding).

Regularization parameter in Wiener filter formula:
    H^-1 = conj(H) / (|H|^2 + epsilon)

Prevents amplification of noise in frequency domain. Value represents
signal-to-noise ratio assumption.

References:
    - Plate (2003): Holographic Reduced Representations
    - Wiener filter theory: epsilon ~ (noise_power / signal_power)
    - Empirically: 1e-8 provides good balance for D>=512
"""

# =============================================================================
# Convergence Thresholds
# =============================================================================

CONVERGENCE_THRESHOLD_HIGH = 0.99
"""High-fidelity convergence threshold (99% similarity).

Used for operations requiring high precision:
- Cleanup convergence in resonator networks
- Decoder convergence in gradient descent
- Quality checks in test suites

Typical use: Accept result if cosine similarity >= 0.99
"""

CONVERGENCE_THRESHOLD_MEDIUM = 0.9
"""Medium-fidelity convergence threshold (90% similarity).

Used for operations accepting moderate error:
- Fast cleanup algorithms
- Approximate decoders
- Interactive applications

Typical use: Accept result if cosine similarity >= 0.9
"""

CONVERGENCE_THRESHOLD_LOW = 0.7
"""Low-fidelity convergence threshold (70% similarity).

Used for operations requiring only rough approximation:
- Exploratory analysis
- Initialization heuristics
- Sanity checks

Typical use: Accept result if cosine similarity >= 0.7
"""

# =============================================================================
# Default Hyperparameters - Iterations
# =============================================================================

DEFAULT_ITERATIONS_CLEANUP = 20
"""Default iterations for cleanup algorithms.

Empirically determined across multiple VSA models:
- Resonator cleanup typically converges in 5-15 iterations
- 20 iterations provides safety margin
- Beyond 20 iterations, additional improvement is minimal

References:
    - Kymn et al. (2024): Resonator Networks
    - Empirical analysis shows diminishing returns after iteration 15
"""

DEFAULT_ITERATIONS_GRADIENT = 100
"""Default iterations for gradient-based optimization.

Used in decoders and learning algorithms. Value chosen to balance:
- Convergence quality (100 iterations usually sufficient)
- Computational cost (linear in iterations)
- Early stopping (many algorithms converge in 20-50 iterations)

Typical convergence:
- FractionalPowerEncoder decoder: 20-50 iterations
- Mixture weight learning: 50-100 iterations
"""

# =============================================================================
# Default Hyperparameters - Dimensions
# =============================================================================

DEFAULT_DIM_FHRR = 512
"""Default dimensionality for FHRR (Fourier HRR).

FHRR has best capacity due to exact inverses via complex conjugation.
Empirical capacity for D=512:
- ~330 items can be bundled with 90% similarity threshold
- Significantly better than MAP/HRR at same dimension

References:
    - Schlegel et al. (2022): Comparison of VSAs
    - Table 3: FHRR capacity analysis
"""

DEFAULT_DIM_MAP = 10000
"""Default dimensionality for MAP (Multiply-Add-Permute).

MAP requires higher dimensions due to approximate inverses.
Empirical capacity for D=10000:
- ~510 items can be bundled with 90% similarity threshold
- Lower capacity-per-dimension than FHRR but simpler operations

References:
    - Schlegel et al. (2022): Comparison of VSAs
    - Table 3: MAP capacity analysis
"""

DEFAULT_DIM_HRR = 10000
"""Default dimensionality for HRR (Holographic RR).

HRR uses circular convolution with approximate inverses.
Similar capacity to MAP:
- ~510 items at D=10000
- Circular convolution more expensive than MAP element-wise multiply

References:
    - Plate (2003): Holographic Reduced Representations
    - Schlegel et al. (2022): Capacity comparison
"""

DEFAULT_DIM_BSC = 10000
"""Default dimensionality for BSC (Binary Spatter Codes).

Binary vectors with XOR binding. Requires high dimensions:
- XOR is self-inverse (exact)
- But bundling degrades quickly with binary addition
- D=10000 provides reasonable capacity

References:
    - Kanerva (2009): Hyperdimensional Computing Introduction
"""

DEFAULT_DIM_BSDC = 10000
"""Default dimensionality for BSDC (Binary Sparse Distributed Codes).

Sparse binary vectors. High dimensions needed despite sparsity:
- Sparsity parameter separate from dimension
- Typical sparsity: 1-5% active bits
- D=10000 with 2% sparsity = 200 active bits

References:
    - Kanerva (1988): Sparse Distributed Memory
"""

# =============================================================================
# Encoder Parameters
# =============================================================================

DEFAULT_RESOLUTION_DECODER = 1000
"""Default resolution for grid search in decoders.

Used in scalar decoders when performing grid search over value range:
- Higher resolution = better initial guess
- 1000 points provides good balance
- Gradient descent refines from grid search

Trade-off:
- Resolution 100: Fast but coarse (~0.01 precision)
- Resolution 1000: Balanced (~0.001 precision)
- Resolution 10000: Slow but precise (~0.0001 precision)
"""

DEFAULT_GRADIENT_STEP = 0.01
"""Initial step size for gradient descent decoders.

Step size for gradient descent in value decoders:
- Too large: Oscillation, may overshoot
- Too small: Slow convergence
- 0.01 empirically works well for normalized [0,1] ranges

Typical adjustment: Multiply by 0.95 each iteration (decay)
"""

DEFAULT_GRADIENT_DECAY = 0.95
"""Decay factor for gradient descent step size.

Multiplicative decay per iteration: step = step * decay
- 0.95 provides smooth convergence
- Reaches ~0.006 after 20 iterations
- Prevents oscillation in final iterations

Alternative strategies:
- No decay (0.95^0 = 1.0): Faster but may oscillate
- Fast decay (0.90): More stable but slower
- Adaptive (0.95): Good balance
"""

DEFAULT_TOLERANCE = 1e-6
"""Convergence tolerance for iterative algorithms.

Stop iteration when |change| < tolerance:
- For gradient descent: |gradient| < tolerance
- For resonator: |similarity_change| < tolerance
- Value 1e-6 ensures high precision without over-iteration

Typical convergence:
- Gradient descent: Reaches 1e-6 in 30-80 iterations
- Resonator cleanup: Reaches 1e-6 in 10-30 iterations
"""

# =============================================================================
# BSDC (Sparse Binary) Parameters
# =============================================================================

DEFAULT_SPARSITY = 0.02
"""Default sparsity for BSDC vectors (2% active).

Fraction of dimensions that are active (1) vs inactive (0):
- 2% sparsity at D=10000 → 200 active dimensions
- Balances capacity and computational efficiency
- Matches Sparse Distributed Memory literature

References:
    - Kanerva (1988): SDM typically uses 1-5% sparsity
    - Empirical: 2% provides good discriminability
"""

DEFAULT_SEGMENT_SIZE = 100
"""Default segment size for BSDC-SEG.

Block size for segmented sparse codes:
- Each segment has local sparsity
- 100-dimensional segments are computationally efficient
- At D=10000: 100 segments of 100 dims each

Trade-offs:
- Small segments (50): More fine-grained, higher variance
- Medium segments (100): Balanced
- Large segments (500): Less variance, less expressive
"""

# =============================================================================
# Cleanup Parameters
# =============================================================================

DEFAULT_CLEANUP_K = 10
"""Default K for K-NN cleanup.

Number of nearest neighbors to consider:
- K=1: Most similar item only (deterministic)
- K=10: Smooth over top 10 (robust to noise)
- K=50: Very smooth (may include dissimilar items)

Typical use:
- Clean query: K=1 sufficient
- Noisy query: K=10 recommended
- Very noisy: K=20-50
"""

DEFAULT_SIMILARITY_THRESHOLD = 0.7
"""Default similarity threshold for cleanup.

Minimum similarity to accept match:
- > 0.9: High confidence, few false positives
- > 0.7: Medium confidence, balanced
- > 0.5: Low confidence, may include noise

Typical use:
- Production: 0.9 threshold
- Exploration: 0.7 threshold
- Sanity checks: 0.5 threshold
"""

# =============================================================================
# Resonator Network Parameters
# =============================================================================

DEFAULT_RESONATOR_ITERATIONS = 20
"""Default iterations for resonator cleanup.

Iterative factorization typically converges in 10-20 iterations:
- Iteration 5: Rough approximation
- Iteration 10: Good approximation
- Iteration 20: Near-optimal (diminishing returns beyond)

References:
    - Kymn et al. (2024): Resonator Networks for VSA
    - Figure 3: Convergence typically plateaus at iteration 15-20
"""

DEFAULT_RESONATOR_TEMPERATURE = 0.1
"""Default temperature for soft resonator.

Controls softmax sharpness in soft resonator:
- T → 0: Approaches hard resonator (winner-take-all)
- T = 0.1: Balanced (default)
- T → ∞: Uniform distribution (no selection)

Mathematical form:
    softmax(similarities / T)

Typical use:
- Hard cleanup: Use hard resonator (no temperature)
- Smooth cleanup: T = 0.1-1.0
- Very smooth: T = 1.0-10.0
"""

# =============================================================================
# Mathematical Constants
# =============================================================================

# Pi is provided by backend-specific implementations
# (NumPy, PyTorch, JAX all provide their own pi constants)
# No need to define here to avoid backend dependency

# =============================================================================
# Export Control
# =============================================================================

__all__ = [
    # Numerical stability
    'EPSILON_DEFAULT',
    'EPSILON_NORMALIZATION',
    'EPSILON_WIENER',

    # Convergence
    'CONVERGENCE_THRESHOLD_HIGH',
    'CONVERGENCE_THRESHOLD_MEDIUM',
    'CONVERGENCE_THRESHOLD_LOW',

    # Iterations
    'DEFAULT_ITERATIONS_CLEANUP',
    'DEFAULT_ITERATIONS_GRADIENT',

    # Dimensions
    'DEFAULT_DIM_FHRR',
    'DEFAULT_DIM_MAP',
    'DEFAULT_DIM_HRR',
    'DEFAULT_DIM_BSC',
    'DEFAULT_DIM_BSDC',

    # Encoder parameters
    'DEFAULT_RESOLUTION_DECODER',
    'DEFAULT_GRADIENT_STEP',
    'DEFAULT_GRADIENT_DECAY',
    'DEFAULT_TOLERANCE',

    # BSDC parameters
    'DEFAULT_SPARSITY',
    'DEFAULT_SEGMENT_SIZE',

    # Cleanup parameters
    'DEFAULT_CLEANUP_K',
    'DEFAULT_SIMILARITY_THRESHOLD',

    # Resonator parameters
    'DEFAULT_RESONATOR_ITERATIONS',
    'DEFAULT_RESONATOR_TEMPERATURE',
]
