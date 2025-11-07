"""PyTorch backend implementation for holovec operations.

This backend enables GPU acceleration and integration with PyTorch models.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .base import Array, Backend, BackendNotAvailableError

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class TorchBackend(Backend):
    """PyTorch-based backend for VSA operations.

    This backend leverages PyTorch for GPU acceleration and neural network
    integration. Requires PyTorch to be installed.
    """

    def __init__(self, device: str = 'cpu', seed: Optional[int] = None):
        """Initialize PyTorch backend.

        Args:
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
            seed: Random seed for reproducibility
        """
        if not self.is_available():
            raise BackendNotAvailableError("PyTorch is not installed. Install with: pip install torch")

        self._device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)

    @property
    def name(self) -> str:
        return "torch"

    def is_available(self) -> bool:
        return TORCH_AVAILABLE

    # ===== Capability Probes =====

    def supports_complex(self) -> bool:
        """PyTorch fully supports complex number operations."""
        return True

    def supports_sparse(self) -> bool:
        """PyTorch has sparse tensor support (COO and CSR formats).

        Note: Sparse support is partial - not all operations work with sparse tensors.
        """
        return True

    def supports_gpu(self) -> bool:
        """PyTorch supports GPU acceleration via CUDA or Metal."""
        if not TORCH_AVAILABLE:
            return False
        return torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())

    def supports_jit(self) -> bool:
        """PyTorch has TorchScript JIT compilation but not as advanced as JAX.

        Note: TorchScript is mainly for deployment, not general computation like JAX.
        """
        return False  # We don't expose TorchScript as a primary feature

    def supports_device(self, device: str) -> bool:
        """Check if PyTorch supports the specified device."""
        if not TORCH_AVAILABLE:
            return False

        device_lower = device.lower()

        # CPU always supported
        if device_lower in ('cpu', 'cpu:0'):
            return True

        # Check CUDA
        if device_lower.startswith('cuda'):
            return torch.cuda.is_available()

        # Check Apple Metal (MPS)
        if device_lower.startswith('mps'):
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        return False

    @property
    def device(self) -> torch.device:
        """Return the current device."""
        return self._device

    # ===== Array Creation =====

    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype: str = 'float32') -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        torch_dtype = self._to_torch_dtype(dtype)
        return torch.zeros(shape, dtype=torch_dtype, device=self._device)

    def ones(self, shape: Union[int, Tuple[int, ...]], dtype: str = 'float32') -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        torch_dtype = self._to_torch_dtype(dtype)
        return torch.ones(shape, dtype=torch_dtype, device=self._device)

    def random_normal(
        self,
        shape: Union[int, Tuple[int, ...]],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        torch_dtype = self._to_torch_dtype(dtype)

        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None

        return torch.normal(mean, std, shape, generator=generator, dtype=torch_dtype, device=self._device)

    def random_uniform(
        self,
        shape: Union[int, Tuple[int, ...]],
        low: float = 0.0,
        high: float = 1.0,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        torch_dtype = self._to_torch_dtype(dtype)

        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None

        result = torch.rand(shape, generator=generator, dtype=torch_dtype, device=self._device)
        return result * (high - low) + low

    def random_binary(
        self,
        shape: Union[int, Tuple[int, ...]],
        p: float = 0.5,
        dtype: str = 'int32',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        torch_dtype = self._to_torch_dtype(dtype)

        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None

        result = torch.rand(shape, generator=generator, device=self._device)
        return (result < p).to(torch_dtype)

    def random_bipolar(
        self,
        shape: Union[int, Tuple[int, ...]],
        p: float = 0.5,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        torch_dtype = self._to_torch_dtype(dtype)

        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None

        result = torch.rand(shape, generator=generator, device=self._device)
        return torch.where(result < p, torch.tensor(1.0, dtype=torch_dtype, device=self._device),
                          torch.tensor(-1.0, dtype=torch_dtype, device=self._device))

    def random_phasor(
        self,
        shape: Union[int, Tuple[int, ...]],
        dtype: str = 'complex64',
        seed: Optional[int] = None
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else shape
        torch_dtype = self._to_torch_dtype(dtype)

        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None

        # If last two dims are square, generate diagonal phasor matrices
        if isinstance(shape, tuple) and len(shape) >= 2 and shape[-1] == shape[-2]:
            batch = shape[:-2]
            m = shape[-1]
            angles = torch.rand(batch + (m,), generator=generator, device=self._device) * 2 * np.pi
            diag = torch.exp(1j * angles).to(torch_dtype)
            out = torch.zeros(batch + (m, m), dtype=torch_dtype, device=self._device)
            idx = torch.arange(m, device=self._device)
            out[..., idx, idx] = diag
            return out

        # Otherwise element-wise phasors
        angles = torch.rand(shape, generator=generator, device=self._device) * 2 * np.pi
        return torch.exp(1j * angles).to(torch_dtype)

    def array(self, data, dtype: Optional[str] = None) -> Array:
        torch_dtype = self._to_torch_dtype(dtype) if dtype else None
        return torch.tensor(data, dtype=torch_dtype, device=self._device)

    # ===== Element-wise Operations =====

    def multiply(self, a: Array, b: Array) -> Array:
        return a * b

    def add(self, a: Array, b: Array) -> Array:
        return a + b

    def subtract(self, a: Array, b: Array) -> Array:
        return a - b

    def divide(self, a: Array, b: Array) -> Array:
        return a / b

    def xor(self, a: Array, b: Array) -> Array:
        return torch.bitwise_xor(a.to(torch.int32), b.to(torch.int32))

    def conjugate(self, a: Array) -> Array:
        return torch.conj(a)

    def exp(self, a: Array) -> Array:
        """Element-wise exponential."""
        return torch.exp(a)

    def log(self, a: Array) -> Array:
        """Element-wise natural logarithm."""
        return torch.log(a)

    # ===== Reductions =====

    def sum(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.sum(a)
        return torch.sum(a, dim=axis, keepdim=keepdims)

    def mean(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.mean(a)
        return torch.mean(a, dim=axis, keepdim=keepdims)

    def norm(self, a: Array, ord: Union[int, str] = 2, axis: Optional[int] = None) -> Array:
        if axis is None:
            return torch.linalg.norm(a, ord=ord)
        return torch.linalg.norm(a, ord=ord, dim=axis)

    def dot(self, a: Array, b: Array) -> Array:
        # Handle complex tensors explicitly: use sum(a * conj(b))
        if torch.is_complex(a) or torch.is_complex(b):
            return torch.sum(a.flatten() * torch.conj(b.flatten()))
        return torch.dot(a.flatten(), b.flatten())

    def max(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.max(a)
        return torch.max(a, dim=axis, keepdim=keepdims).values

    def min(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.min(a)
        return torch.min(a, dim=axis, keepdim=keepdims).values

    def argmax(self, a: Array, axis: Optional[int] = None) -> Array:
        if axis is None:
            return torch.argmax(a)
        return torch.argmax(a, dim=axis)

    def argmin(self, a: Array, axis: Optional[int] = None) -> Array:
        if axis is None:
            return torch.argmin(a)
        return torch.argmin(a, dim=axis)

    # ===== Normalization =====

    def normalize(self, a: Array, ord: Union[int, str] = 2, axis: Optional[int] = None, eps: float = 1e-12) -> Array:
        norm = torch.linalg.norm(a, ord=ord, dim=axis, keepdim=True)
        return a / (norm + eps)

    def softmax(self, a: Array, axis: int = -1) -> Array:
        """Softmax with numerical stability via max subtraction."""
        # Subtract max for numerical stability
        a_max = torch.max(a, dim=axis, keepdim=True).values
        exp_a = torch.exp(a - a_max)
        return exp_a / torch.sum(exp_a, dim=axis, keepdim=True)

    # ===== FFT Operations =====

    def fft(self, a: Array) -> Array:
        return torch.fft.fft(a)

    def ifft(self, a: Array) -> Array:
        return torch.fft.ifft(a)

    # ===== Circular Operations =====

    def circular_convolve(self, a: Array, b: Array) -> Array:
        """Circular convolution using FFT method."""
        result = torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b))
        return result.real if torch.is_complex(result) else result

    def circular_correlate(self, a: Array, b: Array) -> Array:
        """Circular correlation using FFT method."""
        result = torch.fft.ifft(torch.fft.fft(a) * torch.conj(torch.fft.fft(b)))
        return result.real if torch.is_complex(result) else result

    # ===== Permutations =====

    def permute(self, a: Array, indices: Array) -> Array:
        return a[indices]

    def roll(self, a: Array, shift: int, axis: Optional[int] = None) -> Array:
        if axis is None:
            dims = tuple(range(len(a.shape)))
        else:
            dims = (axis,)
        return torch.roll(a, shifts=shift, dims=dims)

    # ===== Similarity Measures =====

    def cosine_similarity(self, a: Array, b: Array) -> float:
        na = torch.linalg.norm(a)
        nb = torch.linalg.norm(b)
        prod = na * nb
        if float(prod.item()) == 0.0:
            return 0.0
        val = torch.dot(a.flatten(), b.flatten()) / prod
        out = float(val.item())
        if out > 1.0:
            return 1.0
        if out < -1.0:
            return -1.0
        return out

    def hamming_distance(self, a: Array, b: Array) -> float:
        """Hamming distance for binary/bipolar vectors."""
        return float(torch.sum(a != b).item())

    def euclidean_distance(self, a: Array, b: Array) -> float:
        return float(torch.linalg.norm(a - b).item())

    # ===== Utilities =====

    def shape(self, a: Array) -> Tuple[int, ...]:
        return tuple(a.shape)

    def dtype(self, a: Array) -> str:
        return str(a.dtype).replace('torch.', '')

    def to_numpy(self, a: Array) -> np.ndarray:
        return a.detach().cpu().numpy()

    def from_numpy(self, a: np.ndarray) -> Array:
        return torch.from_numpy(a).to(self._device)

    def clip(self, a: Array, min_val: float, max_val: float) -> Array:
        return torch.clamp(a, min_val, max_val)

    def abs(self, a: Array) -> Array:
        return torch.abs(a)

    def sign(self, a: Array) -> Array:
        return torch.sign(a)

    def threshold(self, a: Array, threshold: float, above: float = 1.0, below: float = 0.0) -> Array:
        return torch.where(a >= threshold, torch.tensor(above, device=self._device),
                          torch.tensor(below, device=self._device))

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        return torch.where(condition, x, y)

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return torch.stack(arrays, dim=axis)

    def concatenate(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return torch.cat(arrays, dim=axis)

    # ===== Matrix Operations =====

    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication or batched matrix multiplication."""
        return torch.matmul(a, b)

    def matrix_transpose(self, a: Array) -> Array:
        """Transpose last two dimensions."""
        # For 2D: simple transpose
        if len(a.shape) == 2:
            return a.T
        # For 3D+: transpose last two dims
        return a.transpose(-2, -1)

    def matrix_trace(self, a: Array) -> Array:
        """Compute trace of each matrix."""
        # For 2D: standard trace
        if len(a.shape) == 2:
            return torch.trace(a)
        # For 3D+: trace along last two dims
        # PyTorch doesn't have batched trace, so we do it manually
        return torch.diagonal(a, dim1=-2, dim2=-1).sum(-1)

    def reshape(self, a: Array, shape: Tuple[int, ...]) -> Array:
        """Reshape array."""
        return a.reshape(shape)

    # ===== Helper Methods =====

    @staticmethod
    def _to_torch_dtype(dtype: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'uint8': torch.uint8,
            'bool': torch.bool,
            'complex64': torch.complex64,
            'complex128': torch.complex128,
        }
        return dtype_map.get(dtype, torch.float32)

    # ===== Additional Element-wise Utilities =====

    def power(self, a: Array, exponent: float) -> Array:
        return torch.pow(a, exponent)

    def angle(self, a: Array) -> Array:
        return torch.angle(a)

    def real(self, a: Array) -> Array:
        return torch.real(a)

    def imag(self, a: Array) -> Array:
        return torch.imag(a)

    def multiply_scalar(self, a: Array, scalar: float) -> Array:
        return a * scalar

    def linspace(self, start: float, stop: float, num: int) -> Array:
        return torch.linspace(start, stop, steps=num, device=self._device)
