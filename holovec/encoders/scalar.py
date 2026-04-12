"""
Scalar encoders for mapping continuous values to hypervectors.

This module implements various methods for encoding scalar values,
preserving locality: similar scalars map to similar hypervectors.
"""

from holovec.backends.base import Array
from holovec.encoders.base import ScalarEncoder
from holovec.models.base import VSAModel


class FractionalPowerEncoder(ScalarEncoder):
    """
    Fractional Power Encoding (FPE) for continuous scalars.

    Based on Frady et al. (2021) "Computing on Functions Using Randomized
    Vector Representations". Encodes scalars by exponentiating a random
    phasor base vector: encode(x) = φ^x.

    The inner product between encoded vectors approximates a similarity
    kernel (sinc for uniform phase distribution). This encoding preserves
    linearity and enables precise decoding via sinc kernel reconstruction.

    Works best with FHRR (complex domain) but also supports HRR (real domain).

    References:
        Frady et al. (2021): https://arxiv.org/abs/2109.03429
        Verges et al. (2025): Learning encoding phasors with FPE

    Attributes:
        bandwidth: Controls kernel width (lower = wider kernel)
        base_phasor: Random phasor vector φ = [e^(iφ₁), ..., e^(iφₙ)]
    """

    def __init__(
        self,
        model: VSAModel,
        min_val: float,
        max_val: float,
        bandwidth: float = 1.0,
        seed: int | None = None,
        phase_dist: str = "uniform",
        mixture_bandwidths: list[float] | None = None,
        mixture_weights: list[float] | None = None,
    ):
        """
        Initialize FractionalPowerEncoder.

        Parameters
        ----------
        model : VSAModel
            VSA model (FHRR or HRR). FHRR (complex-valued) is preferred for
            exact fractional powers. HRR (real-valued) uses cosine projection.
        min_val : float
            Minimum value of encoding range. Values below this will be clipped.
        max_val : float
            Maximum value of encoding range. Values above this will be clipped.
        bandwidth : float, optional
            Bandwidth parameter β controlling kernel width (default: 1.0).

            **Mathematical Role:**
            - Encoding: z(x) = φ^(β·x_normalized)
            - Kernel: K(x₁, x₂) ≈ sinc(β·π·|x₁ - x₂|) for uniform phase distribution
            - Smaller β → wider kernel → more generalization
            - Larger β → narrower kernel → more discrimination

            **Typical Values:**
            - β = 0.01: Wide kernel, high generalization (classification)
            - β = 1.0: Medium kernel (default)
            - β = 10.0: Narrow kernel, low generalization (regression)

        seed : int or None, optional
            Random seed for generating base phasor (for reproducibility).
            Different seeds produce different random frequency vectors θ.
        phase_dist : str, optional
            Distribution for sampling frequency vector θ (default: 'uniform').

            **Available Distributions:**
            - 'uniform': θⱼ ~ Uniform[-π, π] → sinc kernel (default)
            - 'gaussian': θⱼ ~ N(0, 1) → Gaussian kernel approximation
            - 'laplace': θⱼ ~ Laplace(0, 1) → Exponential kernel, heavy tails
            - 'cauchy': θⱼ ~ Cauchy(0, 1) → Very heavy tails, long-range
            - 'student': θⱼ ~ Student-t(df=3) → Moderate tails, robust

            Different distributions induce different similarity kernels,
            affecting generalization properties.

        mixture_bandwidths : List[float] or None, optional
            List of K bandwidth values [β₁, β₂, ..., βₖ] for mixture encoding.

            **Mixture Encoding:**
            Instead of single bandwidth β, use weighted combination:
                z_mix(x) = Σₖ αₖ · φ^(βₖ·x)

            where αₖ are mixture_weights. This creates multi-scale representation
            combining coarse (small β) and fine (large β) kernels.

            **Example:**
            mixture_bandwidths = [0.01, 0.1, 1.0, 10.0]  # 4 scales
            Creates encoding with both local and global similarity.

        mixture_weights : List[float] or None, optional
            Weights αₖ for each bandwidth in mixture (must sum to 1).

            If None and mixture_bandwidths is provided, uses uniform weights:
                αₖ = 1/K for all k

            Weights can be:
            1. Hand-crafted (domain knowledge)
            2. Learned via `learn_mixture_weights()` (ridge regression)
            3. Uniform (default)

        Raises
        ------
        ValueError
            If phase_dist not in valid set, or if mixture_weights/mixture_bandwidths
            have mismatched lengths.

        Notes
        -----
        **Mathematical Foundation:**

        Fractional Power Encoding maps scalar x to hypervector via:
            z(x) = φ^(β·x_normalized)

        where:
        - φ = [e^(iθ₁), e^(iθ₂), ..., e^(iθₐ)] is base phasor (D dimensions)
        - θⱼ are random frequencies sampled from phase_dist
        - x_normalized ∈ [0, 1] is x mapped to unit interval
        - β is bandwidth parameter

        **Inner Product Kernel:**

        For uniform phase distribution θⱼ ~ Uniform[-π, π]:
            ⟨z(x₁), z(x₂)⟩ / D ≈ sinc(β·π·|x₁ - x₂|)

        This sinc kernel has important properties:
        - Smooth interpolation between similar values
        - Exact at x₁ = x₂ (similarity = 1)
        - Decreases monotonically with distance
        - Zero-crossings at integer multiples of 1/β

        **Comparison to Random Fourier Features:**

        FPE is equivalent to Random Fourier Features (Rahimi & Recht, 2007)
        for kernel approximation:
            k(x₁, x₂) ≈ φ(x₁)ᵀφ(x₂) / D

        where φ(x) = [cos(θ₁x), sin(θ₁x), ..., cos(θₐx), sin(θₐx)]

        For complex hypervectors, FPE uses complex exponentials instead:
            φ(x) = [e^(iθ₁x), e^(iθ₂x), ..., e^(iθₐx)]

        which provides more compact representation and supports exact
        fractional power operations in frequency domain.

        References
        ----------
        - Frady et al. (2021): "Computing on Functions Using Randomized
          Vector Representations" - Original FPE paper
        - Rahimi & Recht (2007): "Random Features for Large-Scale Kernel Machines"
        - Sutherland & Schneider (2015): "On the Error of Random Fourier Features"
        - Verges et al. (2025): "Learning Encoding Phasors with Fractional Power Encoding"

        Examples
        --------
        >>> # Basic FPE for temperature encoding
        >>> model = VSA.create('FHRR', dim=10000)
        >>> encoder = FractionalPowerEncoder(model, min_val=0, max_val=100)
        >>> temp_25 = encoder.encode(25.0)
        >>> temp_26 = encoder.encode(26.0)
        >>> similarity = model.similarity(temp_25, temp_26)  # ≈ 0.95

        >>> # Multi-scale mixture encoding
        >>> encoder_mix = FractionalPowerEncoder(
        ...     model, min_val=0, max_val=100,
        ...     mixture_bandwidths=[0.01, 0.1, 1.0, 10.0],
        ...     mixture_weights=[0.4, 0.3, 0.2, 0.1]  # Emphasize coarse scales
        ... )

        >>> # Alternative kernel via phase distribution
        >>> encoder_gauss = FractionalPowerEncoder(
        ...     model, min_val=0, max_val=100,
        ...     phase_dist='gaussian'  # Gaussian kernel instead of sinc
        ... )
        """
        super().__init__(model, min_val, max_val)

        self.bandwidth = bandwidth
        self.seed = seed

        # Distribution controls for frequencies (theta)
        self.phase_dist = (phase_dist or "uniform").lower()
        valid = {"uniform", "gaussian", "laplace", "cauchy", "student"}
        if self.phase_dist not in valid:
            raise ValueError(f"Unsupported phase_dist '{phase_dist}'. Choose from {sorted(valid)}.")

        # Mixture support (optional)
        self.mixture_bandwidths = mixture_bandwidths
        self.mixture_weights = mixture_weights
        if self.mixture_bandwidths is not None:
            if len(self.mixture_bandwidths) == 0:
                raise ValueError("mixture_bandwidths must be non-empty if provided")
            if self.mixture_weights is None:
                self.mixture_weights = [1.0 / len(self.mixture_bandwidths)] * len(self.mixture_bandwidths)
            if len(self.mixture_weights) != len(self.mixture_bandwidths):
                raise ValueError("mixture_weights must match mixture_bandwidths length")
            # Normalize weights
            s = sum(self.mixture_weights)
            if s <= 0:
                raise ValueError("mixture_weights must sum to positive value")
            self.mixture_weights = [w / s for w in self.mixture_weights]

        # Check complex vs real
        self.is_complex = self.model.space.space_name == "complex"

        # Base phases/frequencies θ_j
        # For uniform, we can derive from a random phasor; for others, sample numeric theta
        if self.phase_dist == "uniform":
            # Maintain backward compatibility using base phasor
            self.base_phasor = self._generate_base_phasor(seed)
            # Derive angles from the base phasor
            self.theta = self.backend.angle(self.base_phasor)
        else:
            # Numeric theta sampled in init; store as backend array
            self.theta = self._generate_theta_distribution(self.phase_dist, seed)
            # For complex path we do not need base_phasor; for real path, we’ll compute cos(theta * exponent)
            self.base_phasor = None

    def _generate_base_phasor(self, seed: int | None) -> Array:
        """
        Generate random phasor base vector with uniform phase distribution.

        For uniform phases φᵢ ~ Uniform[-π, π], this induces the sinc kernel:
        K(d) = sinc(πd)

        Args:
            seed: Random seed for reproducibility

        Returns:
            Base phasor vector φ = [e^(iφ₁), e^(iφ₂), ..., e^(iφₙ)]
        """
        # Generate random phasors using backend (fully backend-agnostic)
        if self.is_complex:
            # For complex models (FHRR), generate random phasors directly
            phasor = self.backend.random_phasor(
                shape=self.dimension,
                dtype='complex64',
                seed=seed
            )
        else:
            # For real models (HRR), generate phasors then project to real
            phasor_complex = self.backend.random_phasor(
                shape=self.dimension,
                dtype='complex64',
                seed=seed
            )
            # Project to real via inverse FFT
            phasor_real = self.backend.ifft(phasor_complex).real
            # Normalize to unit norm using backend
            phasor = self.backend.normalize(phasor_real)

        return phasor

    def _generate_theta_distribution(self, phase_dist: str, seed: int | None) -> Array:
        """
        Generate frequency vector θ according to specified distribution.

        Parameters
        ----------
        phase_dist : str
            Distribution name for sampling frequencies.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        Array
            Frequency vector θ of shape (D,) in backend format.

        Notes
        -----
        **Distribution Choices and Induced Kernels:**

        Different frequency distributions induce different similarity kernels
        via the Fourier transform relationship:

        1. **Uniform θⱼ ~ Uniform[-π, π]** (default):
           - Kernel: K(d) = sinc(π·d) = sin(π·d)/(π·d)
           - Properties: Smooth, monotonic decay, oscillatory
           - Best for: General-purpose continuous encoding
           - Zero-crossings at integer distances

        2. **Gaussian θⱼ ~ N(0, 1)**:
           - Kernel: K(d) ≈ exp(-d²/2) (Gaussian RBF kernel)
           - Properties: Smooth, no oscillations, fast decay
           - Best for: Local similarity, smooth interpolation
           - Widely used in kernel methods (SVMs, GPs)

        3. **Laplace θⱼ ~ Laplace(0, 1)**:
           - Kernel: K(d) ∝ exp(-|d|) (Exponential kernel)
           - Properties: Heavy tails, slower than Gaussian decay
           - Best for: Robust similarity, outlier tolerance
           - More forgiving to distant values

        4. **Cauchy θⱼ ~ Cauchy(0, 1)**:
           - Kernel: K(d) ∝ 1/(1 + d²) (Rational quadratic)
           - Properties: Very heavy tails, long-range interactions
           - Best for: Multi-scale similarity, hierarchical data
           - Cauchy kernel is limit of Student-t as df→∞

        5. **Student-t θⱼ ~ Student-t(df=3)**:
           - Kernel: K(d) ∝ (1 + d²/3)^(-2) (generalized Student)
           - Properties: Moderate heavy tails (df=3 chosen empirically)
           - Best for: Robust regression, noisy data
           - Interpolates between Gaussian (df→∞) and Cauchy (df→0)

        **Mathematical Background:**

        The relationship between frequency distribution p(θ) and
        similarity kernel K(d) follows from Bochner's theorem:

        A continuous kernel K(x₁, x₂) = K(x₁ - x₂) is positive definite
        if and only if K(d) is the Fourier transform of a non-negative
        measure (the frequency distribution p(θ)):

            K(d) = ∫ exp(i·θ·d) p(θ) dθ

        For FPE, the inner product is:
            ⟨z(x₁), z(x₂)⟩ / D ≈ 𝔼_θ[exp(i·θ·β·(x₁ - x₂))]
                                = ∫ exp(i·θ·β·d) p(θ) dθ
                                = K(β·d)

        where d = x₁ - x₂ is the distance between scalars.

        **Sampling Methods:**

        - **Uniform, Gaussian, Student-t**: Direct sampling from distribution
        - **Laplace**: Inverse CDF transform from uniform:
            θ = -sign(u) · log(1 - 2|u|)  where u ~ Uniform(-0.5, 0.5)
        - **Cauchy**: Inverse CDF transform:
            θ = tan(π·u)  where u ~ Uniform(-0.5, 0.5)

        **NumPy Usage Justification:**

        Uses local NumPy import because special distributions (Laplace, Cauchy)
        are not available in backend abstraction. Frequencies are converted
        to backend array immediately via `from_numpy()`.

        References
        ----------
        - Rahimi & Recht (2007): "Random Features for Large-Scale Kernel Machines"
          Section 3: Relationship between frequency distribution and kernel
        - Sutherland & Schneider (2015): "On the Error of Random Fourier Features"
          Analysis of approximation quality for different kernels
        - Bochner (1932): "Vorlesungen über Fouriersche Integrale"
          Original Bochner's theorem
        - Rasmussen & Williams (2006): "Gaussian Processes for Machine Learning"
          Chapter 4: Covariance functions and kernel design

        Examples
        --------
        >>> # Gaussian kernel for smooth similarity
        >>> model = VSA.create('FHRR', dim=10000)
        >>> enc = FractionalPowerEncoder(model, 0, 100, phase_dist='gaussian')

        >>> # Cauchy kernel for long-range similarity
        >>> enc_cauchy = FractionalPowerEncoder(model, 0, 100, phase_dist='cauchy')
        """
        import numpy as _np

        rng = _np.random.default_rng(seed)
        D = self.dimension
        if phase_dist == "gaussian":
            theta_np = rng.normal(0.0, 1.0, size=(D,)).astype(_np.float32)
        elif phase_dist == "laplace":
            # Laplace via inverse transform: scale=1
            u = rng.uniform(-0.5, 0.5, size=(D,)).astype(_np.float32)
            theta_np = (_np.sign(u) * _np.log1p(-2.0 * _np.abs(u))).astype(_np.float32) * -1.0
        elif phase_dist == "cauchy":
            u = rng.uniform(-0.5, 0.5, size=(D,)).astype(_np.float32)
            theta_np = _np.tan(_np.pi * u).astype(_np.float32)
        elif phase_dist == "student":
            theta_np = rng.standard_t(df=3.0, size=(D,)).astype(_np.float32)
        else:
            # Default to uniform angles; match base_phasor angle convention [-π, π]
            theta_np = rng.uniform(-_np.pi, _np.pi, size=(D,)).astype(_np.float32)

        return self.backend.from_numpy(theta_np)

    def encode(self, value: float) -> Array:
        """
        Encode scalar value to hypervector using fractional power.

        Parameters
        ----------
        value : float
            Scalar value to encode. Will be clipped to [min_val, max_val].

        Returns
        -------
        Array
            Encoded hypervector of shape (dimension,) in backend format.

        Notes
        -----
        **Single Bandwidth Encoding:**

        For single bandwidth β, implements:
            z(x) = φ^(β·x_normalized)

        where:
        - x_normalized = (value - min_val) / (max_val - min_val) ∈ [0, 1]
        - φ = [e^(iθ₁), ..., e^(iθₐ)] is base phasor with random frequencies θⱼ
        - Result is normalized according to model's space

        Element-wise computation:
            z_j(x) = e^(i·θⱼ·β·x_normalized)  (complex models)
            z_j(x) = cos(θⱼ·β·x_normalized)   (real models)

        **Mixture Encoding:**

        When mixture_bandwidths = [β₁, ..., βₖ] is provided, uses weighted sum:
            z_mix(x) = Σₖ αₖ · φ^(βₖ·x_normalized)

        where αₖ are mixture_weights (default: uniform αₖ = 1/K).

        **Advantages of Mixture Encoding:**

        1. **Multi-Scale Representation**: Combines coarse (small β) and
           fine (large β) similarity kernels in single hypervector

        2. **Improved Generalization**: Coarse scales provide robustness,
           fine scales provide discrimination

        3. **Learned Weights**: Weights αₖ can be learned via
           `learn_mixture_weights()` to optimize for specific task

        4. **Kernel Combination**: Mixture is equivalent to combining
           multiple kernels: K_mix(d) = Σₖ αₖ·K_βₖ(d)

        **Computational Complexity:**

        - Single bandwidth: O(D) operations (element-wise exponential)
        - Mixture with K bandwidths: O(K·D) operations
        - Backend operations (exp, multiply) are vectorized/GPU-accelerated

        **Normalization:**

        Output is normalized using model's normalization scheme:
        - FHRR/HRR: L2 normalization (unit norm)
        - MAP: Element-wise normalization
        - BSC/BSDC: No normalization (binary)

        This ensures hypervectors are in valid space for subsequent
        binding/bundling operations.

        Examples
        --------
        >>> # Basic encoding
        >>> model = VSA.create('FHRR', dim=10000)
        >>> encoder = FractionalPowerEncoder(model, min_val=0, max_val=100)
        >>> hv_25 = encoder.encode(25.0)  # Encode temperature 25°C
        >>> hv_26 = encoder.encode(26.0)
        >>> similarity = model.similarity(hv_25, hv_26)
        >>> print(f"Similarity: {similarity:.3f}")  # ≈ 0.950 (close values)

        >>> # Mixture encoding for multi-scale representation
        >>> encoder_mix = FractionalPowerEncoder(
        ...     model, min_val=0, max_val=100,
        ...     mixture_bandwidths=[0.01, 1.0, 100.0]
        ... )
        >>> hv_mix = encoder_mix.encode(25.0)  # Combines 3 scales

        >>> # Effect of bandwidth on similarity
        >>> enc_wide = FractionalPowerEncoder(model, 0, 100, bandwidth=0.1)
        >>> enc_narrow = FractionalPowerEncoder(model, 0, 100, bandwidth=10.0)
        >>> sim_wide = model.similarity(enc_wide.encode(25), enc_wide.encode(30))
        >>> sim_narrow = model.similarity(enc_narrow.encode(25), enc_narrow.encode(30))
        >>> # sim_wide > sim_narrow (wider kernel → more generalization)
        """
        # Normalize value to [0, 1]
        normalized = self.normalize(value)

        # Handle mixture: list of beta_k and weights alpha_k
        betas: list[float]
        alphas: list[float]
        if self.mixture_bandwidths is not None:
            betas = list(self.mixture_bandwidths)
            alphas = list(self.mixture_weights or [])
        else:
            betas = [self.bandwidth]
            alphas = [1.0]

        parts = []
        for alpha, beta in zip(alphas, betas, strict=True):
            exponent = beta * normalized
            if self.is_complex:
                # Complex: encode as exp(i * theta * exponent)
                theta = self.theta if self.theta is not None else self.backend.angle(self.base_phasor)
                phase = self.backend.multiply_scalar(theta, exponent)
                phasor = self.backend.exp(1j * phase)
                parts.append(self.backend.multiply_scalar(phasor, alpha))
            else:
                # Real: use cosine features directly: cos(theta * exponent)
                theta = self.theta if self.theta is not None else self.backend.angle(self.base_phasor)
                phase = self.backend.multiply_scalar(theta, exponent)
                # cos(phase) = Re(exp(i*phase))
                phasor = self.backend.real(self.backend.exp(1j * phase))
                parts.append(self.backend.multiply_scalar(phasor, alpha))

        if len(parts) == 1:
            encoded = parts[0]
        else:
            encoded = self.backend.sum(self.backend.stack(parts, axis=0), axis=0)

        # Normalize output according to space
        return self.model.normalize(encoded)

    def decode(
        self,
        hypervector: Array,
        resolution: int = 1000,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """
        Decode hypervector back to scalar value using two-stage optimization.

        Parameters
        ----------
        hypervector : Array
            Hypervector to decode (typically a noisy/bundled encoding).
        resolution : int, optional
            Number of grid points for coarse search (default: 1000).
            Higher resolution improves initial guess but increases cost.
        max_iterations : int, optional
            Maximum gradient descent iterations (default: 100).
            Typical convergence: 20-50 iterations.
        tolerance : float, optional
            Convergence tolerance for gradient descent (default: 1e-6).
            Stop when |Δx| < tolerance.

        Returns
        -------
        float
            Decoded scalar value in [min_val, max_val].

        Notes
        -----
        **Decoding Algorithm:**

        Uses two-stage optimization to find value x maximizing similarity:
            x* = argmax_x ⟨encode(x), hypervector⟩

        **Stage 1: Coarse Grid Search** (O(resolution · D))
        - Evaluate similarity at `resolution` uniformly-spaced points
        - Find x₀ with highest similarity
        - Provides good initialization for gradient descent

        **Stage 2: Gradient Descent** (O(max_iterations · D))
        - Starting from x₀, perform gradient ascent:
            x_{t+1} = x_t + η_t · ∇_x ⟨encode(x_t), hypervector⟩
        - Gradient computed via finite differences:
            ∇_x ≈ (sim(x + ε) - sim(x)) / ε
        - Step size η_t decays: η_t = η_0 · 0.95^t (prevents oscillation)
        - Clips updates to [0, 1] normalized range

        **Why This Works:**

        For FPE with sinc kernel K(x₁, x₂) = sinc(β·π·|x₁ - x₂|):
        - Similarity function is unimodal (single peak)
        - Peak occurs at x = x_true (encoded value)
        - Gradient descent converges to global maximum

        However, for noisy hypervectors (e.g., bundled encodings):
        - Multiple local maxima may exist
        - Coarse search reduces chance of local minimum trap
        - Wider kernels (small β) → smoother objective → easier optimization

        **Approximation Quality:**

        Decoding accuracy depends on several factors:

        1. **Dimension D**: Higher D → more accurate encoding → better decoding
           - D = 1000: Moderate accuracy (similarity ≈ 0.85)
           - D = 10000: High accuracy (similarity ≈ 0.99)

        2. **Signal-to-Noise Ratio**: Clean encoding vs bundled/noisy
           - Clean: Near-perfect recovery (error < 1%)
           - Bundled (10 items): Good recovery (error ≈ 5-10%)
           - Bundled (100 items): Degraded (error ≈ 20-30%)

        3. **Bandwidth β**: Wider kernels → smoother similarity landscape
           - β = 0.01: Very smooth, easy to optimize
           - β = 10.0: Narrow kernel, may have local maxima

        4. **Mixture Encoding**: Multiple bandwidths complicate landscape
           - May require finer grid search (higher resolution)
           - May need more gradient descent iterations

        **Computational Cost:**

        Total operations: O(resolution · D + max_iterations · D)

        Typical values:
        - resolution = 1000, max_iterations = 100, D = 10000
        - Total: ~1.1M evaluations
        - Runtime: ~0.1-1.0 seconds (CPU), ~0.01-0.1 seconds (GPU)

        For real-time applications, reduce resolution or max_iterations:
        - resolution = 100 (coarser search)
        - max_iterations = 20 (early stopping)

        **Comparison to Other Decoders:**

        - **Codebook Lookup** (LevelEncoder): O(K · D) for K levels
          Faster but discrete, no interpolation

        - **Resonator Network** (cleanup): O(iterations · M · D) for M items
          Better for structured/compositional decoding

        - **FPE Gradient Descent**: O(resolution · D + iterations · D)
          Best for continuous scalar recovery

        References
        ----------
        - Frady et al. (2021): "Computing on Functions Using Randomized
          Vector Representations" - Section on FPE decoding
        - Nocedal & Wright (2006): "Numerical Optimization" - Gradient descent
          methods and convergence analysis

        Examples
        --------
        >>> # Basic decoding
        >>> model = VSA.create('FHRR', dim=10000)
        >>> encoder = FractionalPowerEncoder(model, min_val=0, max_val=100)
        >>> hv = encoder.encode(25.0)
        >>> decoded = encoder.decode(hv)
        >>> print(f"Decoded: {decoded:.2f}")  # ≈ 25.00

        >>> # Decoding noisy hypervector (bundled encoding)
        >>> hv_bundle = model.bundle([encoder.encode(25.0), encoder.encode(26.0)])
        >>> decoded_bundle = encoder.decode(hv_bundle)
        >>> print(f"Decoded bundle: {decoded_bundle:.2f}")  # ≈ 25.5

        >>> # Fast decoding (lower resolution/iterations)
        >>> decoded_fast = encoder.decode(hv, resolution=100, max_iterations=20)
        """
        # Coarse search: evaluate on grid
        normalized_grid = self.backend.linspace(0, 1, resolution)

        best_similarity = -float('inf')
        best_normalized = 0.5  # Start in middle

        for norm_val_np in self.backend.to_numpy(normalized_grid):
            norm_val = float(norm_val_np)
            encoded = self.encode(self.denormalize(norm_val))
            similarity = float(
                self.backend.to_numpy(
                    self.model.similarity(encoded, hypervector)
                )
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_normalized = norm_val

        # Fine search: gradient descent around best coarse value
        # For simplicity, use finite differences for gradient
        current = best_normalized
        step_size = 0.01

        for _ in range(max_iterations):
            # Evaluate at current position
            encoded_curr = self.encode(self.denormalize(current))
            sim_curr = float(
                self.backend.to_numpy(
                    self.model.similarity(encoded_curr, hypervector)
                )
            )

            # Evaluate at current + epsilon
            epsilon = 1e-4
            encoded_plus = self.encode(self.denormalize(current + epsilon))
            sim_plus = float(
                self.backend.to_numpy(
                    self.model.similarity(encoded_plus, hypervector)
                )
            )

            # Compute gradient
            gradient = (sim_plus - sim_curr) / epsilon

            # Update (gradient ascent)
            new_current = current + step_size * gradient

            # Clip to [0, 1]
            new_current = max(0.0, min(1.0, new_current))

            # Check convergence
            if abs(new_current - current) < tolerance:
                break

            current = new_current
            step_size *= 0.95  # Decay step size

        # Denormalize and return
        return self.denormalize(current)

    @property
    def is_reversible(self) -> bool:
        """FPE supports approximate decoding."""
        return True

    @property
    def compatible_models(self) -> list[str]:
        """FPE works best with FHRR, also compatible with HRR."""
        return ["FHRR", "HRR"]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FractionalPowerEncoder("
            f"model={self.model.model_name}, "
            f"range=[{self.min_val}, {self.max_val}], "
            f"bandwidth={self.bandwidth}, "
            f"phase_dist={self.phase_dist}, "
            f"mixture={'yes' if self.mixture_bandwidths else 'no'}, "
            f"dimension={self.dimension})"
        )

    # ====== M2: Learned mixture weights (ridge-style closed form) ======
    def learn_mixture_weights(
        self,
        values: list[float],
        labels: list[int],
        reg: float = 1e-3,
    ) -> list[float]:
        """
        Learn mixture weights (alphas) for fixed mixture_bandwidths using a simple
        ridge-style objective that aligns encoded mixtures to per-class prototypes.

        Approach:
            - Build class prototypes p_c as the mean of current encodings (using current weights)
            - For each sample i, compute per-band encodings E_i = [e_{i1},...,e_{iK}] (shape d×K)
            - Solve (Σ E_i^T E_i + reg I) α = Σ E_i^T p_{y_i}
            - Project α onto simplex (nonnegative, sum=1)

        Args:
            values: list of scalar inputs
            labels: list of integer class labels (same length as values)
            reg: L2 regularization strength (default 1e-3)

        Returns:
            Learned mixture weights (list of floats summing to 1)

        Notes:
            - Requires mixture_bandwidths to be set (K>=2)
            - Uses numpy for solving normal equations; backend remains unchanged
        """
        import numpy as _np

        if self.mixture_bandwidths is None or len(self.mixture_bandwidths) < 2:
            raise ValueError("learn_mixture_weights requires mixture_bandwidths with K >= 2")

        # Prepare classes and group samples
        values = list(values)
        labels = list(labels)
        if len(values) != len(labels):
            raise ValueError("values and labels must have same length")

        classes = sorted(set(labels))
        K = len(self.mixture_bandwidths)
        # Build current encodings to compute class prototypes (using current mixture weights)
        encodings = [self.encode(v) for v in values]
        # Convert to numpy arrays for prototype computation
        enc_np = [_np.array(self.model.backend.to_numpy(e)) for e in encodings]
        # Class prototypes: mean of encodings per class (vector length d)
        prototypes = {}
        for c in classes:
            idxs = [i for i, y in enumerate(labels) if y == c]
            if not idxs:
                continue
            prototypes[c] = _np.mean(_np.stack([enc_np[i] for i in idxs], axis=0), axis=0)

        # Helper to compute per-band encodings matrix E_i (d×K) for a value
        def _per_band_matrix(val: float) -> _np.ndarray:
            assert self.mixture_bandwidths is not None
            norm = self.normalize(val)
            cols = []
            for beta in self.mixture_bandwidths:
                exponent = beta * norm
                theta = self.theta if self.theta is not None else self.model.backend.angle(self.base_phasor)
                phase = self.model.backend.multiply_scalar(theta, exponent)
                if self.is_complex:
                    ph = self.model.backend.exp(1j * phase)
                    col = self.model.backend.to_numpy(ph)
                else:
                    col = self.model.backend.to_numpy(self.model.backend.real(self.model.backend.exp(1j * phase)))
                cols.append(_np.array(col))
            # Stack columns to d×K
            return _np.stack(cols, axis=1)

        # Accumulate normal equations
        A = _np.zeros((K, K), dtype=_np.float64)
        b = _np.zeros((K,), dtype=_np.float64)
        for v, y in zip(values, labels, strict=True):
            E = _per_band_matrix(v)   # d×K
            p = prototypes[y]         # d
            # E^T E and E^T p
            A += E.T @ E
            b += E.T @ p

        # Regularization
        A += reg * _np.eye(K, dtype=_np.float64)
        # Solve
        try:
            alpha = _np.linalg.solve(A, b)
        except _np.linalg.LinAlgError:
            alpha = _np.linalg.lstsq(A, b, rcond=None)[0]

        # Project to simplex (≥0, sum=1)
        alpha = _np.maximum(alpha, 0.0)
        s = float(_np.sum(alpha))
        if s <= 0:
            alpha = _np.ones_like(alpha) / len(alpha)
        else:
            alpha = alpha / s

        # Update in encoder
        self.mixture_weights = [float(a) for a in alpha.tolist()]
        return self.mixture_weights


class ThermometerEncoder(ScalarEncoder):
    """
    Thermometer encoding for scalar values.

    Divides value range into N bins and encodes a value as the bundle
    of all bins it exceeds. Creates monotonic similarity profile.

    Simpler and more robust than FPE, but with coarser granularity.
    Works with all VSA models.

    References:
        Kanerva (2009): "Hyperdimensional Computing"
    """

    def __init__(
        self,
        model: VSAModel,
        min_val: float,
        max_val: float,
        n_bins: int = 100,
        seed: int | None = None
    ):
        """
        Initialize ThermometerEncoder.

        Args:
            model: VSA model (any)
            min_val: Minimum value of encoding range
            max_val: Maximum value of encoding range
            n_bins: Number of bins to divide range into (default 100)
            seed: Random seed for generating bin vectors

        Raises:
            ValueError: If n_bins < 2
        """
        super().__init__(model, min_val, max_val)

        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        self.n_bins = n_bins
        self.seed = seed

        # Generate random vectors for each bin
        self.bin_vectors = [
            model.random(seed=seed + i if seed is not None else None)
            for i in range(n_bins)
        ]

        # Compute bin edges
        self.bin_width = self.range / n_bins

    def encode(self, value: float) -> Array:
        """
        Encode scalar as bundle of all bins it exceeds.

        Args:
            value: Scalar value to encode

        Returns:
            Encoded hypervector (bundle of activated bins)
        """
        # Normalize value
        normalized = self.normalize(value)

        # Determine which bin the value falls into
        bin_index = int(normalized * self.n_bins)
        bin_index = min(bin_index, self.n_bins - 1)  # Handle edge case

        # Bundle all bins from 0 to bin_index (inclusive)
        if bin_index == 0:
            return self.bin_vectors[0]

        activated_bins = self.bin_vectors[:bin_index + 1]
        return self.model.bundle(activated_bins)

    def decode(self, hypervector: Array) -> float:
        """
        Decode is not implemented for ThermometerEncoder.

        Thermometer encoding is not easily reversible without
        storing additional information.

        Raises:
            NotImplementedError: Always raises
        """
        raise NotImplementedError(
            "ThermometerEncoder does not support decoding. "
            "Use FractionalPowerEncoder if decoding is required."
        )

    @property
    def is_reversible(self) -> bool:
        """Thermometer encoding is not reversible."""
        return False

    @property
    def compatible_models(self) -> list[str]:
        """Works with all VSA models."""
        return ["MAP", "FHRR", "HRR", "BSC", "GHRR", "VTB", "BSDC"]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ThermometerEncoder("
            f"model={self.model.model_name}, "
            f"range=[{self.min_val}, {self.max_val}], "
            f"n_bins={self.n_bins}, "
            f"dimension={self.dimension})"
        )


class LevelEncoder(ScalarEncoder):
    """
    Level (codebook) encoding for discrete scalar values.

    Maps discrete levels to random orthogonal vectors via lookup table.
    Fast (O(1) encode/decode) and exact for discrete values.

    Best used when you have a small number of discrete values rather
    than continuous range.

    Example:
        >>> # Encode weekdays (7 discrete values)
        >>> model = VSA.create('FHRR', dim=10000)
        >>> encoder = LevelEncoder(model, min_val=0, max_val=6, n_levels=7)
        >>> monday = encoder.encode(0)  # Exact encoding
        >>> friday = encoder.encode(4)
    """

    def __init__(
        self,
        model: VSAModel,
        min_val: float,
        max_val: float,
        n_levels: int,
        seed: int | None = None
    ):
        """
        Initialize LevelEncoder.

        Args:
            model: VSA model (any)
            min_val: Minimum value (corresponds to level 0)
            max_val: Maximum value (corresponds to level n_levels-1)
            n_levels: Number of discrete levels
            seed: Random seed for generating level vectors

        Raises:
            ValueError: If n_levels < 2
        """
        super().__init__(model, min_val, max_val)

        if n_levels < 2:
            raise ValueError(f"n_levels must be >= 2, got {n_levels}")

        self.n_levels = n_levels
        self.seed = seed

        # Generate random vector for each level
        self.level_vectors = [
            model.random(seed=seed + i if seed is not None else None)
            for i in range(n_levels)
        ]

        # Compute level width
        self.level_width = self.range / (n_levels - 1)

    def encode(self, value: float) -> Array:
        """
        Encode scalar to nearest level's hypervector.

        Args:
            value: Scalar value to encode

        Returns:
            Hypervector corresponding to nearest level
        """
        # Normalize to [0, 1]
        normalized = self.normalize(value)

        # Map to level index (round to nearest)
        level_index = int(round(normalized * (self.n_levels - 1)))
        level_index = max(0, min(level_index, self.n_levels - 1))

        return self.level_vectors[level_index]

    def decode(self, hypervector: Array) -> float:
        """
        Decode hypervector to nearest level value.

        Args:
            hypervector: Hypervector to decode

        Returns:
            Decoded scalar value (will be one of the discrete levels)
        """
        # Find most similar level vector
        best_similarity = -float('inf')
        best_level = 0

        for level_idx, level_vec in enumerate(self.level_vectors):
            similarity = float(
                self.backend.to_numpy(
                    self.model.similarity(hypervector, level_vec)
                )
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_level = level_idx

        # Convert level index back to value
        normalized = best_level / (self.n_levels - 1)
        return self.denormalize(normalized)

    @property
    def is_reversible(self) -> bool:
        """Level encoding is reversible (to nearest level)."""
        return True

    @property
    def compatible_models(self) -> list[str]:
        """Works with all VSA models."""
        return ["MAP", "FHRR", "HRR", "BSC", "GHRR", "VTB", "BSDC"]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LevelEncoder("
            f"model={self.model.model_name}, "
            f"range=[{self.min_val}, {self.max_val}], "
            f"n_levels={self.n_levels}, "
            f"dimension={self.dimension})"
        )
