from __future__ import annotations

from typing import Dict, Iterable, Optional

from holovec.encoders.base import Encoder
from holovec.models.base import VSAModel
from holovec.backends.base import Array


class VectorFPE(Encoder):
    """Multivariate Fractional Power Encoding (Random Fourier Features).

    Encodes x ∈ R^n as φ(x) = exp(i * W x * β), where W ∈ R^{D×n} is sampled
    from a chosen distribution (gaussian, laplace, cauchy, student, uniform).
    For complex spaces (FHRR/GHRR), returns unit phasors; for real spaces,
    returns cos(Wxβ) features.

    Parameters
    ----------
    model : VSAModel
        Destination model (determines space: complex vs real)
    input_dim : int
        Dimensionality of input vector x
    bandwidth : float | Array | numpy.ndarray
        Scalar β or per-dimension β vector (element-wise scaling of x)
    phase_dist : str
        One of {gaussian, laplace, cauchy, student, uniform}
    seed : int | None
        RNG seed
    """

    def __init__(
        self,
        model: VSAModel,
        input_dim: int,
        bandwidth: float | Array | "numpy.ndarray" = 1.0,
        phase_dist: str = "gaussian",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(model)
        self.input_dim = int(input_dim)
        self.bandwidth = bandwidth
        self.phase_dist = (phase_dist or "gaussian").lower()
        self.seed = seed
        valid = {"gaussian", "laplace", "cauchy", "student", "uniform"}
        if self.phase_dist not in valid:
            raise ValueError(f"Unsupported phase_dist '{phase_dist}'; choose from {sorted(valid)}")

        self.is_complex = self.model.space.space_name in ("complex",) or "matrix" in self.model.space.space_name
        # Sample W (D x n)
        self.W = self._sample_W(self.phase_dist, self.model.dimension, self.input_dim, seed)

    def _sample_W(self, phase_dist: str, D: int, n: int, seed: Optional[int]) -> Array:
        """Sample frequency matrix W from specified distribution.

        Parameters
        ----------
        phase_dist : str
            Distribution name: 'gaussian', 'laplace', 'cauchy', 'student', 'uniform'
        D : int
            Output dimension (hypervector dimension)
        n : int
            Input dimension (feature dimension)
        seed : int or None
            Random seed for reproducibility

        Returns
        -------
        Array
            Frequency matrix W of shape (D, n) in the model's backend format

        Notes
        -----
        **NumPy Usage Justification:**

        This method uses NumPy's random number generators because several
        required distributions are not available in the backend abstraction:

        1. **Laplace Distribution**:
           - Generated via inverse CDF: W = -sign(u) * log(1 - 2|u|)
           - Not available in PyTorch/JAX standard distributions

        2. **Cauchy Distribution**:
           - Generated via inverse CDF: W = tan(π * u)
           - PyTorch has `torch.distributions.Cauchy` but not JAX

        3. **Student-t Distribution**:
           - Available in NumPy as `standard_t(df=3)`
           - PyTorch/JAX have different APIs

        **Backend Compatibility:**

        The generated matrix W is immediately converted to the target backend
        via `backend.from_numpy()`, so this only affects initialization, not
        runtime encoding operations. All encoding computations (matrix multiply,
        exponentials) use the backend interface.

        **Distribution Choices:**

        Different distributions provide different frequency characteristics:
        - **Gaussian**: Standard RBF kernel approximation (default)
        - **Laplace**: Exponential kernel, heavier tails
        - **Cauchy**: Very heavy tails, long-range interactions
        - **Student-t**: Moderate tails (df=3), robust to outliers
        - **Uniform**: Bounded support, equal frequency coverage

        References
        ----------
        - Rahimi & Recht (2007): "Random Features for Large-Scale Kernel Machines"
        - Sutherland & Schneider (2015): "On the Error of Random Fourier Features"
        """
        # Use local NumPy import to avoid module-level backend dependency
        import numpy as _np
        import math

        rng = _np.random.default_rng(seed)

        if phase_dist == "gaussian":
            # Standard normal distribution N(0, 1)
            W = rng.normal(0.0, 1.0, size=(D, n)).astype(_np.float32)
        elif phase_dist == "laplace":
            # Laplace distribution via inverse CDF transform
            u = rng.uniform(-0.5, 0.5, size=(D, n)).astype(_np.float32)
            W = (_np.sign(u) * _np.log1p(-2.0 * _np.abs(u)) * -1.0).astype(_np.float32)
        elif phase_dist == "cauchy":
            # Cauchy distribution via inverse CDF: tan(π * u)
            u = rng.uniform(-0.5, 0.5, size=(D, n)).astype(_np.float32)
            W = _np.tan(math.pi * u).astype(_np.float32)
        elif phase_dist == "student":
            # Student-t distribution with 3 degrees of freedom
            W = rng.standard_t(df=3.0, size=(D, n)).astype(_np.float32)
        else:  # uniform
            # Uniform distribution over [-π, π]
            W = rng.uniform(-math.pi, math.pi, size=(D, n)).astype(_np.float32)

        # Convert to backend array for compatibility with PyTorch/JAX
        return self.model.backend.from_numpy(W)

    def _prepare_x(self, x: Array | Iterable[float]) -> Array:
        """Prepare input vector by converting to backend array and applying bandwidth.

        Parameters
        ----------
        x : Array or Iterable[float]
            Input vector (can be backend array, NumPy array, list, or tuple)

        Returns
        -------
        Array
            Input vector scaled by bandwidth in backend format

        Notes
        -----
        Accepts NumPy arrays as input for convenience (common format from
        scikit-learn, pandas, etc.) but converts immediately to backend array
        for all computations.
        """
        # Use local import to avoid module-level backend dependency
        import numpy as _np

        be = self.model.backend

        # Convert input to backend array if needed
        if isinstance(x, _np.ndarray):
            arr = be.from_numpy(x.astype(_np.float32))
        elif isinstance(x, list) or isinstance(x, tuple):
            arr = be.array(list(x), dtype="float32")
        else:
            # Already a backend array
            arr = x

        # Apply per-dimension or scalar bandwidth scaling
        if isinstance(self.bandwidth, (int, float)):
            # Scalar bandwidth: multiply all dimensions by same value
            return be.multiply_scalar(arr, float(self.bandwidth))
        else:
            # Per-dimension bandwidth: element-wise multiply
            bw = self.bandwidth
            if isinstance(bw, _np.ndarray):
                bw = be.from_numpy(bw.astype(_np.float32))
            return be.multiply(arr, bw)

    def encode(self, x: Array | Iterable[float]) -> Array:
        be = self.model.backend
        x_scaled = self._prepare_x(x)  # shape (n,)
        # angle = W @ x_scaled
        angle = be.matmul(self.W, x_scaled)  # shape (D,)
        if self.is_complex:
            return be.exp(1j * angle)
        else:
            return be.real(be.exp(1j * angle))

    def decode(self, hv: Array, candidates: Optional[Dict[str, Array]] = None, k: int = 1):
        """No closed-form inverse; use nearest in a codebook if provided."""
        if candidates is None:
            raise NotImplementedError("VectorFPE.decode requires a codebook for nearest lookup.")
        from holovec.utils.search import nearest_neighbors

        labels, sims = nearest_neighbors(hv, candidates, self.model, k=k, return_similarities=True)
        return list(zip(labels, sims or []))

    @property
    def is_reversible(self) -> bool:
        return False

    @property
    def compatible_models(self) -> list[str]:
        return ["FHRR", "HRR"]

    @property
    def input_type(self) -> str:
        return f"vector[float] length {self.input_dim}"
