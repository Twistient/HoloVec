"""Abstract backend interface for holovec operations.

This module defines the abstract interface that all computational backends
must implement. Backends provide the underlying array operations for
different computation frameworks (NumPy, PyTorch, JAX, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

# Type alias for arrays (backend-specific)
Array = Any


class Backend(ABC):
    """Abstract base class for computational backends.

    All backends must implement these operations to support VSA computations
    across different frameworks (NumPy, PyTorch, JAX).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'numpy', 'torch', 'jax')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available in the current environment."""
        pass

    # ===== Capability Probes =====

    def supports_complex(self) -> bool:
        """Check if backend supports complex number operations.

        Complex operations are required for FHRR (Fourier HRR) and other
        frequency-domain VSA models.

        Returns:
            True if backend can handle complex dtypes (complex64, complex128)
        """
        return True  # Default: assume support (override in backend if needed)

    def supports_sparse(self) -> bool:
        """Check if backend supports sparse array operations.

        Sparse operations are beneficial for BSC (Binary Spatter Codes) and
        BSDC (Binary Sparse Distributed Codes) which have high sparsity.

        Returns:
            True if backend has native sparse array support
        """
        return False  # Default: no sparse support (override if available)

    def supports_gpu(self) -> bool:
        """Check if backend has GPU acceleration support.

        GPU support enables significant speedups for large-scale operations
        and is critical for production deployments.

        Returns:
            True if backend can utilize GPU hardware
        """
        return False  # Default: CPU only (override for PyTorch/JAX)

    def supports_jit(self) -> bool:
        """Check if backend supports Just-In-Time (JIT) compilation.

        JIT compilation can provide 10-100x speedups for certain operations
        by compiling Python code to optimized machine code.

        Returns:
            True if backend has JIT compilation (e.g., JAX, Numba)
        """
        return False  # Default: no JIT (override for JAX)

    def supports_device(self, device: str) -> bool:
        """Check if backend supports a specific device.

        Args:
            device: Device identifier (e.g., 'cpu', 'cuda', 'cuda:0', 'mps')

        Returns:
            True if the specified device is available

        Examples:
            >>> backend.supports_device('cpu')  # Always True
            >>> backend.supports_device('cuda')  # True if CUDA GPU available
            >>> backend.supports_device('mps')  # True if Apple Metal available
        """
        # Default: only CPU supported
        return device.lower() in ('cpu', 'cpu:0')

    # ===== Array Creation =====

    @abstractmethod
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype: str = 'float32') -> Array:
        """Create an array of zeros with the given shape and dtype."""
        pass

    @abstractmethod
    def ones(self, shape: Union[int, Tuple[int, ...]], dtype: str = 'float32') -> Array:
        """Create an array of ones with the given shape and dtype."""
        pass

    @abstractmethod
    def random_normal(
        self,
        shape: Union[int, Tuple[int, ...]],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        """Create an array of random values from a normal distribution."""
        pass

    @abstractmethod
    def random_uniform(
        self,
        shape: Union[int, Tuple[int, ...]],
        low: float = 0.0,
        high: float = 1.0,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        """Create an array of random values from a uniform distribution."""
        pass

    @abstractmethod
    def random_binary(
        self,
        shape: Union[int, Tuple[int, ...]],
        p: float = 0.5,
        dtype: str = 'int32',
        seed: Optional[int] = None
    ) -> Array:
        """Create a binary array with probability p of being 1."""
        pass

    @abstractmethod
    def random_bipolar(
        self,
        shape: Union[int, Tuple[int, ...]],
        p: float = 0.5,
        dtype: str = 'float32',
        seed: Optional[int] = None
    ) -> Array:
        """Create a bipolar array {-1, +1} with probability p of being +1."""
        pass

    @abstractmethod
    def random_phasor(
        self,
        shape: Union[int, Tuple[int, ...]],
        dtype: str = 'complex64',
        seed: Optional[int] = None
    ) -> Array:
        """Create an array of random unit phasors (complex numbers with magnitude 1)."""
        pass

    @abstractmethod
    def array(self, data: Any, dtype: Optional[str] = None) -> Array:
        """Create an array from Python data (list, tuple, etc.)."""
        pass

    # ===== Element-wise Operations =====

    @abstractmethod
    def multiply(self, a: Array, b: Array) -> Array:
        """Element-wise multiplication."""
        pass

    @abstractmethod
    def add(self, a: Array, b: Array) -> Array:
        """Element-wise addition."""
        pass

    @abstractmethod
    def subtract(self, a: Array, b: Array) -> Array:
        """Element-wise subtraction."""
        pass

    @abstractmethod
    def divide(self, a: Array, b: Array) -> Array:
        """Element-wise division."""
        pass

    @abstractmethod
    def xor(self, a: Array, b: Array) -> Array:
        """Element-wise XOR (for binary/bipolar)."""
        pass

    @abstractmethod
    def conjugate(self, a: Array) -> Array:
        """Complex conjugate (for complex arrays)."""
        pass

    @abstractmethod
    def exp(self, a: Array) -> Array:
        """Element-wise exponential: e^a.

        Args:
            a: Input array

        Returns:
            Array with exp applied element-wise
        """
        pass

    @abstractmethod
    def log(self, a: Array) -> Array:
        """Element-wise natural logarithm: ln(a).

        Args:
            a: Input array (must be positive)

        Returns:
            Array with log applied element-wise
        """
        pass

    # ===== Additional Element-wise Utilities =====

    @abstractmethod
    def power(self, a: Array, exponent: float) -> Array:
        """Element-wise power: a**exponent."""
        pass

    @abstractmethod
    def angle(self, a: Array) -> Array:
        """Element-wise phase/angle for complex arrays (radians)."""
        pass

    @abstractmethod
    def real(self, a: Array) -> Array:
        """Element-wise real part of (possibly complex) array."""
        pass

    @abstractmethod
    def imag(self, a: Array) -> Array:
        """Element-wise imaginary part of (possibly complex) array."""
        pass

    @abstractmethod
    def multiply_scalar(self, a: Array, scalar: float) -> Array:
        """Multiply array by a Python scalar."""
        pass

    @abstractmethod
    def linspace(self, start: float, stop: float, num: int) -> Array:
        """Create linearly spaced array of length num in [start, stop]."""
        pass

    # ===== Reductions =====

    @abstractmethod
    def sum(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        """Sum along an axis."""
        pass

    @abstractmethod
    def mean(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        """Mean along an axis."""
        pass

    @abstractmethod
    def norm(self, a: Array, ord: Union[int, str] = 2, axis: Optional[int] = None) -> Array:
        """Compute the norm of an array."""
        pass

    @abstractmethod
    def dot(self, a: Array, b: Array) -> Array:
        """Dot product of two vectors."""
        pass

    @abstractmethod
    def max(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        """Maximum value along an axis.

        Args:
            a: Input array
            axis: Axis along which to compute max (None for global max)
            keepdims: Whether to keep dimensions

        Returns:
            Maximum value(s)
        """
        pass

    @abstractmethod
    def min(self, a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        """Minimum value along an axis.

        Args:
            a: Input array
            axis: Axis along which to compute min (None for global min)
            keepdims: Whether to keep dimensions

        Returns:
            Minimum value(s)
        """
        pass

    @abstractmethod
    def argmax(self, a: Array, axis: Optional[int] = None) -> Array:
        """Index of maximum value along an axis.

        Args:
            a: Input array
            axis: Axis along which to find argmax (None for global argmax)

        Returns:
            Index/indices of maximum value(s)
        """
        pass

    @abstractmethod
    def argmin(self, a: Array, axis: Optional[int] = None) -> Array:
        """Index of minimum value along an axis.

        Args:
            a: Input array
            axis: Axis along which to find argmin (None for global argmin)

        Returns:
            Index/indices of minimum value(s)
        """
        pass

    # ===== Normalization =====

    @abstractmethod
    def normalize(self, a: Array, ord: Union[int, str] = 2, axis: Optional[int] = None, eps: float = 1e-12) -> Array:
        """Normalize an array to unit norm."""
        pass

    @abstractmethod
    def softmax(self, a: Array, axis: int = -1) -> Array:
        """Softmax function with numerical stability.

        Computes: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

        The max subtraction provides numerical stability by preventing overflow
        in the exponential function.

        Args:
            a: Input array
            axis: Axis along which to compute softmax

        Returns:
            Array with softmax applied along specified axis

        References:
            - Bricken & Pehlevan (2022): "Attention Approximates Sparse Distributed Memory"
            - Furlong & Eliasmith (2023): "Fractional binding in VSAs"
        """
        pass

    # ===== FFT Operations =====

    @abstractmethod
    def fft(self, a: Array) -> Array:
        """1D Fast Fourier Transform."""
        pass

    @abstractmethod
    def ifft(self, a: Array) -> Array:
        """1D Inverse Fast Fourier Transform."""
        pass

    # ===== Circular Operations =====

    @abstractmethod
    def circular_convolve(self, a: Array, b: Array) -> Array:
        """Circular convolution of two vectors."""
        pass

    @abstractmethod
    def circular_correlate(self, a: Array, b: Array) -> Array:
        """Circular correlation of two vectors."""
        pass

    # ===== Permutations =====

    @abstractmethod
    def permute(self, a: Array, indices: Array) -> Array:
        """Permute array elements according to indices."""
        pass

    @abstractmethod
    def roll(self, a: Array, shift: int, axis: Optional[int] = None) -> Array:
        """Roll array elements along an axis."""
        pass

    # ===== Similarity Measures =====

    @abstractmethod
    def cosine_similarity(self, a: Array, b: Array) -> float:
        """Compute cosine similarity between two vectors."""
        pass

    @abstractmethod
    def hamming_distance(self, a: Array, b: Array) -> float:
        """Compute Hamming distance between two binary/bipolar vectors."""
        pass

    @abstractmethod
    def euclidean_distance(self, a: Array, b: Array) -> float:
        """Compute Euclidean distance between two vectors."""
        pass

    # ===== Utilities =====

    @abstractmethod
    def shape(self, a: Array) -> Tuple[int, ...]:
        """Return the shape of an array."""
        pass

    @abstractmethod
    def dtype(self, a: Array) -> str:
        """Return the dtype of an array as a string."""
        pass

    @abstractmethod
    def to_numpy(self, a: Array) -> Any:
        """Convert array to NumPy array (for compatibility)."""
        pass

    @abstractmethod
    def from_numpy(self, a: Any) -> Array:
        """Create backend array from NumPy array."""
        pass

    @abstractmethod
    def clip(self, a: Array, min_val: float, max_val: float) -> Array:
        """Clip array values to [min_val, max_val]."""
        pass

    @abstractmethod
    def abs(self, a: Array) -> Array:
        """Element-wise absolute value."""
        pass

    @abstractmethod
    def sign(self, a: Array) -> Array:
        """Element-wise sign."""
        pass

    @abstractmethod
    def threshold(self, a: Array, threshold: float, above: float = 1.0, below: float = 0.0) -> Array:
        """Threshold array values."""
        pass

    @abstractmethod
    def where(self, condition: Array, x: Array, y: Array) -> Array:
        """Select elements from x or y depending on boolean condition."""
        pass

    @abstractmethod
    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        """Stack arrays along a new axis."""
        pass

    @abstractmethod
    def concatenate(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        """Concatenate arrays along an existing axis."""
        pass

    # ===== Matrix Operations (for GHRR, VTB) =====

    @abstractmethod
    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication (or batched matrix multiplication).

        Args:
            a: Matrix or batch of matrices
            b: Matrix or batch of matrices

        Returns:
            Matrix product
        """
        pass

    @abstractmethod
    def matrix_transpose(self, a: Array) -> Array:
        """Transpose last two dimensions of array.

        For 2D: standard transpose
        For 3D+: transpose last two dimensions (batch transpose)

        Args:
            a: Array with at least 2 dimensions

        Returns:
            Transposed array
        """
        pass

    @abstractmethod
    def matrix_trace(self, a: Array) -> Array:
        """Compute trace of matrix or batch of matrices.

        For 2D array: returns scalar
        For 3D+ array: returns trace of each matrix in batch

        Args:
            a: Matrix or batch of matrices (last 2 dims are matrix)

        Returns:
            Scalar or array of traces
        """
        pass

    @abstractmethod
    def reshape(self, a: Array, shape: Tuple[int, ...]) -> Array:
        """Reshape array to new shape.

        Args:
            a: Array to reshape
            shape: Target shape

        Returns:
            Reshaped array
        """
        pass


class BackendError(Exception):
    """Exception raised when a backend operation fails."""
    pass


class BackendNotAvailableError(BackendError):
    """Exception raised when a requested backend is not available."""
    pass
