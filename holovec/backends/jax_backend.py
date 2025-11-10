"""JAX backend implementation for holovec operations.

This backend enables JIT compilation and automatic vectorization for fast
research code and automatic differentiation.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .base import Array, Backend, BackendNotAvailableError

try:
    import jax
    import jax.numpy as jnp
    from jax import random as jax_random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    jax_random = None


class JAXBackend(Backend):
    """JAX-based backend for VSA operations.

    This backend leverages JAX for JIT compilation, automatic differentiation,
    and functional programming patterns. Requires JAX to be installed.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize JAX backend.

        Args:
            seed: Random seed for reproducibility
        """
        if not self.is_available():
            raise BackendNotAvailableError("JAX is not installed. Install with: pip install jax jaxlib")

        self._seed = seed if seed is not None else 0
        self._key = jax_random.PRNGKey(self._seed)

    @property
    def name(self) -> str:
        return "jax"

    def is_available(self) -> bool:
        return JAX_AVAILABLE

    # ===== Capability Probes =====

    def supports_complex(self) -> bool:
        """JAX fully supports complex number operations."""
        return True

    def supports_sparse(self) -> bool:
        """JAX has experimental sparse array support (BCOO format).

        Note: jax.experimental.sparse exists but is not feature-complete.
        """
        return False  # Experimental, not production-ready

    def supports_gpu(self) -> bool:
        """JAX supports GPU acceleration via CUDA or Metal."""
        if not JAX_AVAILABLE:
            return False
        try:
            # Check if any GPU devices are available
            devices = jax.devices('gpu')
            return len(devices) > 0
        except RuntimeError:
            return False

    def supports_jit(self) -> bool:
        """JAX has powerful JIT compilation via XLA."""
        return True

    def supports_device(self, device: str) -> bool:
        """Check if JAX supports the specified device."""
        if not JAX_AVAILABLE:
            return False

        device_lower = device.lower()

        # CPU always supported
        if device_lower in ('cpu', 'cpu:0'):
            return True

        # Check GPU (CUDA or Metal)
        if device_lower in ('gpu', 'gpu:0') or device_lower.startswith('cuda'):
            try:
                devices = jax.devices('gpu')
                return len(devices) > 0
            except RuntimeError:
                return False

        # TPU support
        if device_lower.startswith('tpu'):
            try:
                devices = jax.devices('tpu')
                return len(devices) > 0
            except RuntimeError:
                return False

        return False

    def _get_key(self, seed: Optional[int] = None):
        """Get a PRNG key, optionally splitting the internal key."""
        if seed is not None:
            return jax_random.PRNGKey(seed)
        self._key, subkey = jax_random.split(self._key)
        return subkey

    # ===== Array Creation =====

    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype: str = 'float32') -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        jax_dtype = self._to_jax_dtype(dtype)
        return jnp.zeros(shape, dtype=jax_dtype)

    def ones(self, shape: Union[int, Tuple[int, ...]], dtype: str = 'float32') -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        jax_dtype = self._to_jax_dtype(dtype)
        return jnp.ones(shape, dtype=jax_dtype)

    def random_normal(
        self,
        shape: Union[int, Tuple[int, ...]],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        jax_dtype = self._to_jax_dtype(dtype)
        key = self._get_key(seed)
        return (jax_random.normal(key, shape, dtype=jax_dtype) * std + mean)

    def random_uniform(
        self,
        shape: Union[int, Tuple[int, ...]],
        low: float = 0.0,
        high: float = 1.0,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        jax_dtype = self._to_jax_dtype(dtype)
        key = self._get_key(seed)
        return jax_random.uniform(key, shape, dtype=jax_dtype, minval=low, maxval=high)

    def random_binary(
        self,
        shape: Union[int, Tuple[int, ...]],
        p: float = 0.5,
        dtype: str = 'int32',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        jax_dtype = self._to_jax_dtype(dtype)
        key = self._get_key(seed)
        return jax_random.bernoulli(key, p, shape).astype(jax_dtype)

    def random_bipolar(
        self,
        shape: Union[int, Tuple[int, ...]],
        p: float = 0.5,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        jax_dtype = self._to_jax_dtype(dtype)
        key = self._get_key(seed)
        binary = jax_random.bernoulli(key, p, shape)
        return (2 * binary - 1).astype(jax_dtype)

    def random_phasor(
        self,
        shape: Union[int, Tuple[int, ...]],
        dtype: str = 'complex64',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        jax_dtype = self._to_jax_dtype(dtype)
        key = self._get_key(seed)
        # If last two dims are square, generate diagonal phasor matrices
        if isinstance(shape, tuple) and len(shape) >= 2 and shape[-1] == shape[-2]:
            batch = shape[:-2]
            m = shape[-1]
            angles = jax_random.uniform(key, batch + (m,), minval=0.0, maxval=2 * np.pi)
            diag = jnp.exp(1j * angles).astype(jax_dtype)
            out = jnp.zeros(batch + (m, m), dtype=jax_dtype)
            idx = jnp.arange(m)
            out = out.at[..., idx, idx].set(diag)
            return out
        # Otherwise element-wise phasors
        angles = jax_random.uniform(key, shape, minval=0.0, maxval=2 * np.pi)
        return jnp.exp(1j * angles).astype(jax_dtype)

    def array(self, data, dtype: Optional[str] = None) -> Array:
        jax_dtype = self._to_jax_dtype(dtype) if dtype else None
        return jnp.array(data, dtype=jax_dtype)

    # ===== Element-wise Operations =====

    def multiply(self, a: Array, b: Array) -> Array:
        return jnp.multiply(a, b)

    def add(self, a: Array, b: Array) -> Array:
        return jnp.add(a, b)

    def subtract(self, a: Array, b: Array) -> Array:
        return jnp.subtract(a, b)

    def divide(self, a: Array, b: Array) -> Array:
        return jnp.divide(a, b)

    def xor(self, a: Array, b: Array) -> Array:
        return jnp.bitwise_xor(a.astype(jnp.int32), b.astype(jnp.int32))

    def conjugate(self, a: Array) -> Array:
        return jnp.conjugate(a)

    def exp(self, a: Array) -> Array:
        """Element-wise exponential."""
        return jnp.exp(a)

    def log(self, a: Array) -> Array:
        """Element-wise natural logarithm."""
        return jnp.log(a)

    # ===== Reductions =====

    def sum(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        return jnp.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        return jnp.mean(a, axis=axis, keepdims=keepdims)

    def norm(self, a: Array, ord: Union[int, str] = 2, axis: Optional[int] = None) -> Array:
        return jnp.linalg.norm(a, ord=ord, axis=axis)

    def dot(self, a: Array, b: Array) -> Array:
        return jnp.dot(a, b)

    def max(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        return jnp.max(a, axis=axis, keepdims=keepdims)

    def min(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        return jnp.min(a, axis=axis, keepdims=keepdims)

    def argmax(self, a: Array, axis: Optional[int] = None) -> Array:
        return jnp.argmax(a, axis=axis)

    def argmin(self, a: Array, axis: Optional[int] = None) -> Array:
        return jnp.argmin(a, axis=axis)

    # ===== Normalization =====

    def normalize(self, a: Array, ord: Union[int, str] = 2, axis: Optional[int] = None, eps: float = 1e-12) -> Array:
        norm = jnp.linalg.norm(a, ord=ord, axis=axis, keepdims=True)
        return a / (norm + eps)

    def softmax(self, a: Array, axis: int = -1) -> Array:
        """Softmax with numerical stability via max subtraction."""
        # Subtract max for numerical stability
        a_max = jnp.max(a, axis=axis, keepdims=True)
        exp_a = jnp.exp(a - a_max)
        return exp_a / jnp.sum(exp_a, axis=axis, keepdims=True)

    # ===== FFT Operations =====

    def fft(self, a: Array) -> Array:
        return jnp.fft.fft(a)

    def ifft(self, a: Array) -> Array:
        return jnp.fft.ifft(a)

    # ===== Circular Operations =====

    def circular_convolve(self, a: Array, b: Array) -> Array:
        """Circular convolution using FFT method."""
        result = jnp.fft.ifft(jnp.fft.fft(a) * jnp.fft.fft(b))
        return result.real if jnp.iscomplexobj(result) else result

    def circular_correlate(self, a: Array, b: Array) -> Array:
        """Circular correlation using FFT method."""
        result = jnp.fft.ifft(jnp.fft.fft(a) * jnp.conj(jnp.fft.fft(b)))
        return result.real if jnp.iscomplexobj(result) else result

    # ===== Permutations =====

    def permute(self, a: Array, indices: Array) -> Array:
        return a[indices]

    def roll(self, a: Array, shift: int, axis: Optional[int] = None) -> Array:
        return jnp.roll(a, shift, axis=axis)

    # ===== Similarity Measures =====

    def cosine_similarity(self, a: Array, b: Array) -> float:
        na = jnp.linalg.norm(a)
        nb = jnp.linalg.norm(b)
        prod = na * nb
        prod_np = float(np.array(prod))
        if prod_np == 0.0:
            return 0.0
        out = float(np.array(jnp.dot(a, b) / prod))
        if out > 1.0:
            return 1.0
        if out < -1.0:
            return -1.0
        return out

    def hamming_distance(self, a: Array, b: Array) -> float:
        """Hamming distance for binary/bipolar vectors."""
        return float(jnp.sum(a != b))

    def euclidean_distance(self, a: Array, b: Array) -> float:
        return float(jnp.linalg.norm(a - b))

    # ===== Utilities =====

    def shape(self, a: Array) -> Tuple[int, ...]:
        return tuple(a.shape)

    def dtype(self, a: Array) -> str:
        return str(a.dtype)

    def to_numpy(self, a: Array) -> np.ndarray:
        return np.array(a)

    def from_numpy(self, a: np.ndarray) -> Array:
        return jnp.array(a)

    def clip(self, a: Array, min_val: float, max_val: float) -> Array:
        return jnp.clip(a, min_val, max_val)

    def abs(self, a: Array) -> Array:
        return jnp.abs(a)

    def sign(self, a: Array) -> Array:
        return jnp.sign(a)

    def threshold(self, a: Array, threshold: float, above: float = 1.0, below: float = 0.0) -> Array:
        return jnp.where(a >= threshold, above, below)

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return jnp.stack(arrays, axis=axis)

    def concatenate(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return jnp.concatenate(arrays, axis=axis)

    # ===== Matrix Operations =====

    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication or batched matrix multiplication."""
        return jnp.matmul(a, b)

    def matrix_transpose(self, a: Array) -> Array:
        """Transpose last two dimensions."""
        # For 2D: simple transpose
        if len(a.shape) == 2:
            return jnp.transpose(a)
        # For 3D+: transpose last two dims
        axes = list(range(len(a.shape)))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        return jnp.transpose(a, axes=axes)

    def matrix_trace(self, a: Array) -> Array:
        """Compute trace of each matrix."""
        # For 2D: standard trace
        if len(a.shape) == 2:
            return jnp.trace(a)
        # For 3D+: trace along last two dims
        return jnp.trace(a, axis1=-2, axis2=-1)

    def svd(self, a: Array, full_matrices: bool = True) -> Tuple[Array, Array, Array]:
        """Compute Singular Value Decomposition.

        JAX's SVD natively supports batched operations.
        """
        # JAX returns (U, S, Vh) directly - exactly what we need!
        U, S, Vh = jnp.linalg.svd(a, full_matrices=full_matrices)
        return U, S, Vh

    def reshape(self, a: Array, shape: Tuple[int, ...]) -> Array:
        """Reshape array."""
        return jnp.reshape(a, shape)

    # ===== Additional Element-wise Utilities =====

    def power(self, a: Array, exponent: float) -> Array:
        return jnp.power(a, exponent)

    def angle(self, a: Array) -> Array:
        return jnp.angle(a)

    def real(self, a: Array) -> Array:
        return jnp.real(a)

    def imag(self, a: Array) -> Array:
        return jnp.imag(a)

    def multiply_scalar(self, a: Array, scalar: float) -> Array:
        return a * scalar

    def linspace(self, start: float, stop: float, num: int) -> Array:
        return jnp.linspace(start, stop, num)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        return jnp.where(condition, x, y)

    # ===== Helper Methods =====

    @staticmethod
    def _to_jax_dtype(dtype: str):
        """Convert string dtype to JAX dtype."""
        dtype_map = {
            'float16': jnp.float16,
            'float32': jnp.float32,
            'float64': jnp.float64,
            'int8': jnp.int8,
            'int16': jnp.int16,
            'int32': jnp.int32,
            'int64': jnp.int64,
            'uint8': jnp.uint8,
            'bool': jnp.bool_,
            'complex64': jnp.complex64,
            'complex128': jnp.complex128,
        }
        return dtype_map.get(dtype, jnp.float32)
