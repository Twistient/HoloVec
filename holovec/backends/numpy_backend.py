"""NumPy backend implementation for holovec operations.

This is the default backend and requires only NumPy as a dependency.
"""

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np

from .base import Array, Backend


class NumPyBackend(Backend):
    """NumPy-based backend for VSA operations.

    This backend uses pure NumPy for all operations and serves as the
    default backend with minimal dependencies.
    """

    def __init__(self, seed: int | None = None):
        """Initialize NumPy backend with optional random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "numpy"

    def is_available(self) -> bool:
        return True  # NumPy is always available

    # ===== Capability Probes =====

    def supports_complex(self) -> bool:
        """NumPy fully supports complex number operations."""
        return True

    def supports_sparse(self) -> bool:
        """NumPy does not have native sparse array support.

        Note: scipy.sparse provides sparse arrays but is not part of core NumPy.
        """
        return False

    def supports_gpu(self) -> bool:
        """NumPy is CPU-only."""
        return False

    def supports_jit(self) -> bool:
        """NumPy does not have JIT compilation.

        Note: Numba can JIT-compile NumPy code but is a separate library.
        """
        return False

    def supports_device(self, device: str) -> bool:
        """NumPy only supports CPU."""
        return device.lower() in ('cpu', 'cpu:0')

    # ===== Array Creation =====

    def zeros(self, shape: int | tuple[int, ...], dtype: str = 'float32') -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: int | tuple[int, ...], dtype: str = 'float32') -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        return np.ones(shape, dtype=dtype)

    def random_normal(
        self,
        shape: int | tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: str = 'float32',
        seed: int | None = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return rng.normal(mean, std, shape).astype(dtype)

    def random_uniform(
        self,
        shape: int | tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        dtype: str = 'float32',
        seed: int | None = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return rng.uniform(low, high, shape).astype(dtype)

    def random_binary(
        self,
        shape: int | tuple[int, ...],
        p: float = 0.5,
        dtype: str = 'int32',
        seed: int | None = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return rng.binomial(1, p, shape).astype(dtype)

    def randint(
        self,
        shape: int | tuple[int, ...],
        low: int,
        high: int,
        dtype: str = 'int32',
        seed: int | None = None,
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return rng.integers(low, high, shape, dtype=np.dtype(dtype))

    def random_bipolar(
        self,
        shape: int | tuple[int, ...],
        p: float = 0.5,
        dtype: str = 'float32',
        seed: int | None = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        binary = rng.binomial(1, p, shape)
        return (2 * binary - 1).astype(dtype)

    def random_phasor(
        self,
        shape: int | tuple[int, ...],
        dtype: str = 'complex64',
        seed: int | None = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        # If last two dims are square, generate diagonal phasor matrices
        if isinstance(shape, tuple) and len(shape) >= 2 and shape[-1] == shape[-2]:
            batch = shape[:-2]
            m = shape[-1]
            # Angles for diagonal entries
            angles = rng.uniform(0, 2 * np.pi, batch + (m,))
            diag = np.exp(1j * angles).astype(dtype)
            out = np.zeros(batch + (m, m), dtype=dtype)
            # Fill diagonal efficiently by broadcasting
            idx = np.arange(m)
            out[..., idx, idx] = diag
            return out
        # Otherwise element-wise phasors
        angles = rng.uniform(0, 2 * np.pi, shape)
        return np.exp(1j * angles).astype(dtype)

    def array(self, data: object, dtype: str | None = None) -> Array:
        return np.array(data, dtype=dtype)

    # ===== Element-wise Operations =====

    def multiply(self, a: Array, b: Array) -> Array:
        return np.multiply(a, b)

    def add(self, a: Array, b: Array) -> Array:
        return np.add(a, b)

    def subtract(self, a: Array, b: Array) -> Array:
        return np.subtract(a, b)

    def divide(self, a: Array, b: Array) -> Array:
        return np.divide(a, b)

    def xor(self, a: Array, b: Array) -> Array:
        return np.bitwise_xor(a.astype(np.int32), b.astype(np.int32))

    def conjugate(self, a: Array) -> Array:
        return np.conjugate(a)

    # ===== Reductions =====

    def sum(self, a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def norm(self, a: Array, ord: int | str = 2, axis: int | None = None) -> Array:
        if isinstance(ord, str):
            if ord not in ("fro", "nuc"):
                raise ValueError(f"Unsupported norm order '{ord}' for NumPy backend")
            matrix_ord = cast(Literal["fro", "nuc"], ord)
            return np.linalg.norm(a, ord=matrix_ord)
        numeric_ord = float(ord)
        if axis is None:
            return np.linalg.norm(a, ord=numeric_ord)
        return np.linalg.norm(a, ord=numeric_ord, axis=axis)

    def dot(self, a: Array, b: Array) -> Array:
        return np.dot(a, b)

    def max(self, a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.max(a, axis=axis, keepdims=keepdims)

    def min(self, a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.min(a, axis=axis, keepdims=keepdims)

    def argmax(self, a: Array, axis: int | None = None) -> Array:
        return np.argmax(a, axis=axis)

    def argmin(self, a: Array, axis: int | None = None) -> Array:
        return np.argmin(a, axis=axis)

    # ===== Normalization =====

    def normalize(self, a: Array, ord: int | str = 2, axis: int | None = None, eps: float = 1e-12) -> Array:
        if isinstance(ord, str):
            if ord not in ("fro", "nuc"):
                raise ValueError(f"Unsupported norm order '{ord}' for NumPy backend")
            matrix_ord = cast(Literal["fro", "nuc"], ord)
            norm = np.linalg.norm(a, ord=matrix_ord)
        elif axis is None:
            norm = np.linalg.norm(a, ord=float(ord))
        else:
            norm = np.linalg.norm(a, ord=float(ord), axis=axis, keepdims=True)
        return a / (norm + eps)

    def softmax(self, a: Array, axis: int = -1) -> Array:
        """Softmax with numerical stability via max subtraction."""
        # Subtract max for numerical stability
        a_max = np.max(a, axis=axis, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=axis, keepdims=True)

    # ===== FFT Operations =====

    def fft(self, a: Array) -> Array:
        return np.fft.fft(a)

    def ifft(self, a: Array) -> Array:
        return np.fft.ifft(a)

    # ===== Circular Operations =====

    def circular_convolve(self, a: Array, b: Array) -> Array:
        """Circular convolution using real FFT for numerical stability."""
        n = a.shape[-1]
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        return np.fft.irfft(fa * fb, n=n)

    def circular_correlate(self, a: Array, b: Array) -> Array:
        """Circular correlation using real FFT for numerical stability."""
        n = a.shape[-1]
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        return np.fft.irfft(fa * np.conj(fb), n=n)

    # ===== Permutations =====

    def permute(self, a: Array, indices: Array) -> Array:
        return a[indices]

    def roll(self, a: Array, shift: int, axis: int | None = None) -> Array:
        return np.roll(a, shift, axis=axis)

    # ===== Similarity Measures =====

    def cosine_similarity(self, a: Array, b: Array) -> float:
        # Avoid a second full-array scan here; exact equality is still handled by
        # the normal cosine computation below.
        if a is b:
            return 1.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        prod = na * nb
        if prod == 0.0:
            return 0.0
        val = float(np.dot(a, b) / prod)
        # Clamp to [-1, 1] to avoid slight numeric overshoot
        if val > 1.0:
            return 1.0
        if val < -1.0:
            return -1.0
        return val

    def hamming_distance(self, a: Array, b: Array) -> float:
        """Hamming distance for binary/bipolar vectors."""
        return float(np.sum(a != b))

    def euclidean_distance(self, a: Array, b: Array) -> float:
        return float(np.linalg.norm(a - b))

    # ===== Utilities =====

    def shape(self, a: Array) -> tuple[int, ...]:
        return tuple(int(dim) for dim in a.shape)

    def dtype(self, a: Array) -> str:
        return str(a.dtype)

    def to_numpy(self, a: Array) -> np.ndarray:
        return np.asarray(a)

    def from_numpy(self, a: np.ndarray) -> Array:
        return a  # Already NumPy

    def clip(self, a: Array, min_val: float, max_val: float) -> Array:
        return np.clip(a, min_val, max_val)

    def abs(self, a: Array) -> Array:
        return np.abs(a)

    def sign(self, a: Array) -> Array:
        return np.sign(a)

    def astype(self, a: Array, dtype: str) -> Array:
        return a.astype(dtype)

    def threshold(self, a: Array, threshold: float, above: float = 1.0, below: float = 0.0) -> Array:
        return np.where(a >= threshold, above, below)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        return np.where(condition, x, y)

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return np.concatenate(arrays, axis=axis)

    # ===== Matrix Operations =====

    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication or batched matrix multiplication."""
        return np.matmul(a, b)

    def matrix_transpose(self, a: Array) -> Array:
        """Transpose last two dimensions."""
        # For 2D: simple transpose
        if len(a.shape) == 2:
            return np.transpose(a)
        # For 3D+: transpose last two dims
        axes = list(range(len(a.shape)))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        return np.transpose(a, axes=axes)

    def matrix_trace(self, a: Array) -> Array:
        """Compute trace of each matrix."""
        # For 2D: standard trace
        if len(a.shape) == 2:
            return np.trace(a)
        # For 3D+: trace along last two dims
        return np.trace(a, axis1=-2, axis2=-1)

    def svd(self, a: Array, full_matrices: bool = True) -> tuple[Array, Array, Array]:
        """Compute Singular Value Decomposition.

        For batched matrices (3D+), computes SVD for each matrix in the batch.
        """
        # NumPy's svd doesn't handle batched operations natively in older versions
        # Handle both 2D and batched cases
        if len(a.shape) == 2:
            # Simple 2D case
            return np.linalg.svd(a, full_matrices=full_matrices)
        else:
            # Batched case: process each matrix individually
            batch_shape = a.shape[:-2]
            m, n = a.shape[-2:]

            # Determine output shapes
            if full_matrices:
                u_shape = batch_shape + (m, m)
                vh_shape = batch_shape + (n, n)
            else:
                k = min(m, n)
                u_shape = batch_shape + (m, k)
                vh_shape = batch_shape + (k, n)
            s_shape = batch_shape + (min(m, n),)

            # Pre-allocate output arrays
            U_batch = np.zeros(u_shape, dtype=a.dtype)
            S_batch = np.zeros(s_shape, dtype=a.real.dtype)
            Vh_batch = np.zeros(vh_shape, dtype=a.dtype)

            # Flatten batch dimensions for iteration
            a_reshaped = a.reshape(-1, m, n)
            U_reshaped = U_batch.reshape(-1, *U_batch.shape[-2:])
            S_reshaped = S_batch.reshape(-1, S_batch.shape[-1])
            Vh_reshaped = Vh_batch.reshape(-1, *Vh_batch.shape[-2:])

            # Compute SVD for each matrix
            for i in range(a_reshaped.shape[0]):
                U, S, Vh = np.linalg.svd(a_reshaped[i], full_matrices=full_matrices)
                U_reshaped[i] = U
                S_reshaped[i] = S
                Vh_reshaped[i] = Vh

            return U_batch, S_batch, Vh_batch

    def reshape(self, a: Array, shape: tuple[int, ...]) -> Array:
        """Reshape array."""
        return np.reshape(a, shape)

    def eye(self, n: int, dtype: str = 'float32') -> Array:
        return np.eye(n, dtype=dtype)

    # ===== Additional Operations for Encoders =====

    def power(self, a: Array, exponent: float) -> Array:
        """Component-wise power: a^exponent."""
        return np.power(a, exponent)

    def angle(self, a: Array) -> Array:
        """Angle (phase) of complex numbers."""
        return np.angle(a)

    def linspace(self, start: float, stop: float, num: int) -> Array:
        """Create linearly spaced array."""
        return np.linspace(start, stop, num)

    def multiply_scalar(self, a: Array, scalar: float) -> Array:
        """Multiply array by scalar."""
        return a * scalar

    def exp(self, a: Array) -> Array:
        """Element-wise exponential."""
        return np.exp(a)

    def log(self, a: Array) -> Array:
        """Element-wise natural logarithm."""
        return np.log(a)

    def real(self, a: Array) -> Array:
        """Real part of complex array."""
        return np.real(a)

    def imag(self, a: Array) -> Array:
        """Imaginary part of complex array."""
        return np.imag(a)
