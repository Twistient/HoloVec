"""Tests for backend implementations.

Ensures all backends provide consistent behavior and results.
"""

import pytest
import numpy as np

from holovec.backends import (
    NumPyBackend,
    get_backend,
    get_available_backends,
    is_backend_available,
)


# Check which backends are available for testing
AVAILABLE_BACKENDS = get_available_backends()


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend(request):
    """Parametrized fixture providing all available backends."""
    return get_backend(request.param)


class TestBackendAvailability:
    """Test backend detection and availability."""

    def test_numpy_always_available(self):
        """NumPy backend should always be available."""
        assert is_backend_available('numpy')
        assert 'numpy' in get_available_backends()

    def test_get_backend_numpy(self):
        """Should be able to create NumPy backend."""
        backend = get_backend('numpy')
        assert backend.name == 'numpy'
        assert isinstance(backend, NumPyBackend)

    def test_get_default_backend(self):
        """Should return a default backend when name is None."""
        backend = get_backend()
        assert backend is not None
        assert backend.name in AVAILABLE_BACKENDS


class TestBasicOperations:
    """Test basic array operations across all backends."""

    def test_zeros(self, backend):
        """Test zeros creation."""
        arr = backend.zeros(100, dtype='float32')
        assert backend.shape(arr) == (100,)
        assert np.allclose(backend.to_numpy(arr), 0.0)

    def test_ones(self, backend):
        """Test ones creation."""
        arr = backend.ones(50, dtype='float32')
        assert backend.shape(arr) == (50,)
        assert np.allclose(backend.to_numpy(arr), 1.0)

    def test_random_normal(self, backend):
        """Test normal random generation."""
        arr = backend.random_normal(1000, mean=0.0, std=1.0, seed=42)
        arr_np = backend.to_numpy(arr)

        # Check shape
        assert arr_np.shape == (1000,)

        # Check mean and std are approximately correct
        assert abs(np.mean(arr_np)) < 0.2  # Should be close to 0
        assert abs(np.std(arr_np) - 1.0) < 0.2  # Should be close to 1

    def test_random_uniform(self, backend):
        """Test uniform random generation."""
        arr = backend.random_uniform(1000, low=0.0, high=1.0, seed=42)
        arr_np = backend.to_numpy(arr)

        # Check range
        assert np.all(arr_np >= 0.0)
        assert np.all(arr_np <= 1.0)

        # Check mean approximately 0.5
        assert abs(np.mean(arr_np) - 0.5) < 0.1

    def test_random_binary(self, backend):
        """Test binary random generation."""
        arr = backend.random_binary(1000, p=0.5, seed=42)
        arr_np = backend.to_numpy(arr)

        # Check values are 0 or 1
        assert np.all((arr_np == 0) | (arr_np == 1))

        # Check approximately half are 1
        assert abs(np.mean(arr_np) - 0.5) < 0.1

    def test_random_bipolar(self, backend):
        """Test bipolar random generation."""
        arr = backend.random_bipolar(1000, p=0.5, seed=42)
        arr_np = backend.to_numpy(arr)

        # Check values are -1 or +1
        assert np.all((arr_np == -1) | (arr_np == 1))

        # Check approximately balanced
        assert abs(np.mean(arr_np)) < 0.2

    def test_random_phasor(self, backend):
        """Test phasor random generation."""
        arr = backend.random_phasor(100, seed=42)
        arr_np = backend.to_numpy(arr)

        # Check magnitudes are approximately 1
        magnitudes = np.abs(arr_np)
        assert np.allclose(magnitudes, 1.0, atol=1e-6)


class TestElementWiseOperations:
    """Test element-wise operations."""

    def test_multiply(self, backend):
        """Test element-wise multiplication."""
        a = backend.array([1.0, 2.0, 3.0])
        b = backend.array([2.0, 3.0, 4.0])
        result = backend.multiply(a, b)
        expected = np.array([2.0, 6.0, 12.0])
        assert np.allclose(backend.to_numpy(result), expected)

    def test_add(self, backend):
        """Test element-wise addition."""
        a = backend.array([1.0, 2.0, 3.0])
        b = backend.array([4.0, 5.0, 6.0])
        result = backend.add(a, b)
        expected = np.array([5.0, 7.0, 9.0])
        assert np.allclose(backend.to_numpy(result), expected)

    def test_subtract(self, backend):
        """Test element-wise subtraction."""
        a = backend.array([5.0, 7.0, 9.0])
        b = backend.array([1.0, 2.0, 3.0])
        result = backend.subtract(a, b)
        expected = np.array([4.0, 5.0, 6.0])
        assert np.allclose(backend.to_numpy(result), expected)

    def test_xor(self, backend):
        """Test XOR operation."""
        a = backend.array([0, 1, 0, 1], dtype='int32')
        b = backend.array([0, 0, 1, 1], dtype='int32')
        result = backend.xor(a, b)
        expected = np.array([0, 1, 1, 0])
        assert np.array_equal(backend.to_numpy(result), expected)


class TestReductions:
    """Test reduction operations."""

    def test_sum(self, backend):
        """Test sum reduction."""
        arr = backend.array([1.0, 2.0, 3.0, 4.0])
        result = backend.sum(arr)
        assert np.allclose(backend.to_numpy(result), 10.0)

    def test_mean(self, backend):
        """Test mean reduction."""
        arr = backend.array([1.0, 2.0, 3.0, 4.0])
        result = backend.mean(arr)
        assert np.allclose(backend.to_numpy(result), 2.5)

    def test_norm(self, backend):
        """Test norm computation."""
        arr = backend.array([3.0, 4.0])
        result = backend.norm(arr, ord=2)
        assert np.allclose(backend.to_numpy(result), 5.0)

    def test_dot(self, backend):
        """Test dot product."""
        a = backend.array([1.0, 2.0, 3.0])
        b = backend.array([4.0, 5.0, 6.0])
        result = backend.dot(a, b)
        expected = 1*4 + 2*5 + 3*6  # = 32
        assert np.allclose(backend.to_numpy(result), expected)


class TestNormalization:
    """Test normalization operations."""

    def test_normalize_l2(self, backend):
        """Test L2 normalization."""
        arr = backend.array([3.0, 4.0])
        result = backend.normalize(arr, ord=2)
        result_np = backend.to_numpy(result)

        # Check norm is 1
        assert np.allclose(np.linalg.norm(result_np), 1.0)

        # Check direction preserved
        expected = np.array([0.6, 0.8])
        assert np.allclose(result_np, expected)


class TestFFT:
    """Test FFT operations."""

    def test_fft_ifft_roundtrip(self, backend):
        """Test FFT and inverse FFT give back original."""
        arr = backend.random_normal(128, seed=42)
        fft_result = backend.fft(arr)
        roundtrip = backend.ifft(fft_result)

        # Should get back original (real part)
        arr_np = backend.to_numpy(arr)
        roundtrip_np = backend.to_numpy(roundtrip).real

        assert np.allclose(arr_np, roundtrip_np, atol=1e-5)


class TestCircularOperations:
    """Test circular convolution and correlation."""

    def test_circular_convolve(self, backend):
        """Test circular convolution."""
        a = backend.array([1.0, 2.0, 3.0, 4.0])
        b = backend.array([1.0, 0.0, 0.0, 0.0])
        result = backend.circular_convolve(a, b)

        # Convolving with [1,0,0,0] should give back original
        assert np.allclose(backend.to_numpy(result), backend.to_numpy(a), atol=1e-5)

    def test_circular_correlate(self, backend):
        """Test circular correlation."""
        a = backend.random_normal(64, seed=42)
        # Autocorrelation should have peak at 0
        result = backend.circular_correlate(a, a)
        result_np = backend.to_numpy(result)

        # Peak should be at index 0
        assert np.argmax(result_np) == 0


class TestSimilarityMeasures:
    """Test similarity and distance measures."""

    def test_cosine_similarity_identical(self, backend):
        """Cosine similarity of identical vectors should be 1."""
        a = backend.random_normal(100, seed=42)
        sim = backend.cosine_similarity(a, a)
        assert np.allclose(sim, 1.0, atol=1e-6)

    def test_cosine_similarity_orthogonal(self, backend):
        """Cosine similarity of orthogonal vectors should be ~0."""
        a = backend.array([1.0, 0.0, 0.0])
        b = backend.array([0.0, 1.0, 0.0])
        sim = backend.cosine_similarity(a, b)
        assert np.allclose(sim, 0.0, atol=1e-6)

    def test_hamming_distance_identical(self, backend):
        """Hamming distance of identical vectors should be 0."""
        a = backend.random_binary(100, seed=42)
        dist = backend.hamming_distance(a, a)
        assert dist == 0.0

    def test_euclidean_distance_identical(self, backend):
        """Euclidean distance of identical vectors should be 0."""
        a = backend.random_normal(100, seed=42)
        dist = backend.euclidean_distance(a, a)
        assert np.allclose(dist, 0.0, atol=1e-6)


class TestPermutations:
    """Test permutation operations."""

    def test_roll(self, backend):
        """Test circular roll."""
        arr = backend.array([1.0, 2.0, 3.0, 4.0])
        result = backend.roll(arr, shift=1)
        expected = np.array([4.0, 1.0, 2.0, 3.0])
        assert np.array_equal(backend.to_numpy(result), expected)

    def test_roll_negative(self, backend):
        """Test negative roll."""
        arr = backend.array([1.0, 2.0, 3.0, 4.0])
        result = backend.roll(arr, shift=-1)
        expected = np.array([2.0, 3.0, 4.0, 1.0])
        assert np.array_equal(backend.to_numpy(result), expected)


class TestBackendConsistency:
    """Test that different backends give consistent results."""

    @pytest.mark.parametrize("operation", [
        'random_normal',
        'random_bipolar',
        'random_binary',
    ])
    def test_random_consistency(self, operation):
        """Random operations with same seed should give same results."""
        if len(AVAILABLE_BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        # Create two different backends
        backend1 = get_backend(AVAILABLE_BACKENDS[0])
        backend2 = get_backend(AVAILABLE_BACKENDS[1] if len(AVAILABLE_BACKENDS) > 1 else AVAILABLE_BACKENDS[0])

        # Generate random vectors with same seed
        method1 = getattr(backend1, operation)
        method2 = getattr(backend2, operation)

        arr1 = method1(100, seed=42)
        arr2 = method2(100, seed=42)

        # Convert to numpy for comparison
        arr1_np = backend1.to_numpy(arr1)
        arr2_np = backend2.to_numpy(arr2)

        # Should be very close (allowing for minor numerical differences)
        assert np.allclose(arr1_np, arr2_np, atol=1e-5)


class TestNewMathOperations:
    """Test new mathematical operations (exp, log, softmax, max, min, argmax, argmin)."""

    def test_exp_basic(self, backend):
        """Test exponential function."""
        arr = backend.array([0.0, 1.0, 2.0])
        result = backend.exp(arr)
        expected = np.array([1.0, np.e, np.e**2])
        assert np.allclose(backend.to_numpy(result), expected, atol=1e-6)

    def test_log_basic(self, backend):
        """Test natural logarithm."""
        arr = backend.array([1.0, np.e, np.e**2])
        result = backend.log(arr)
        expected = np.array([0.0, 1.0, 2.0])
        assert np.allclose(backend.to_numpy(result), expected, atol=1e-6)

    def test_exp_log_roundtrip(self, backend):
        """Test exp(log(x)) == x."""
        arr = backend.random_uniform(100, low=0.1, high=10.0, seed=42)
        result = backend.exp(backend.log(arr))
        assert np.allclose(backend.to_numpy(result), backend.to_numpy(arr), atol=1e-5)

    def test_max_no_axis(self, backend):
        """Test max without axis (global max)."""
        arr = backend.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = backend.max(arr)
        assert np.allclose(backend.to_numpy(result), 5.0)

    def test_max_with_axis(self, backend):
        """Test max along specific axis."""
        arr = backend.array([[1.0, 5.0, 3.0],
                             [4.0, 2.0, 6.0]])
        result = backend.max(arr, axis=0)
        expected = np.array([4.0, 5.0, 6.0])
        assert np.allclose(backend.to_numpy(result), expected)

    def test_max_keepdims(self, backend):
        """Test max with keepdims."""
        arr = backend.array([[1.0, 5.0], [4.0, 2.0]])
        result = backend.max(arr, axis=1, keepdims=True)
        assert backend.shape(result) == (2, 1)
        expected = np.array([[5.0], [4.0]])
        assert np.allclose(backend.to_numpy(result), expected)

    def test_min_no_axis(self, backend):
        """Test min without axis (global min)."""
        arr = backend.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = backend.min(arr)
        assert np.allclose(backend.to_numpy(result), 1.0)

    def test_min_with_axis(self, backend):
        """Test min along specific axis."""
        arr = backend.array([[1.0, 5.0, 3.0],
                             [4.0, 2.0, 6.0]])
        result = backend.min(arr, axis=0)
        expected = np.array([1.0, 2.0, 3.0])
        assert np.allclose(backend.to_numpy(result), expected)

    def test_min_keepdims(self, backend):
        """Test min with keepdims."""
        arr = backend.array([[1.0, 5.0], [4.0, 2.0]])
        result = backend.min(arr, axis=1, keepdims=True)
        assert backend.shape(result) == (2, 1)
        expected = np.array([[1.0], [2.0]])
        assert np.allclose(backend.to_numpy(result), expected)

    def test_argmax_no_axis(self, backend):
        """Test argmax without axis (global argmax)."""
        arr = backend.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = backend.argmax(arr)
        assert int(backend.to_numpy(result)) == 1  # Index of max value (5.0)

    def test_argmax_with_axis(self, backend):
        """Test argmax along specific axis."""
        arr = backend.array([[1.0, 5.0, 3.0],
                             [4.0, 2.0, 6.0]])
        result = backend.argmax(arr, axis=0)
        expected = np.array([1, 0, 1])  # Indices of max along axis 0
        assert np.array_equal(backend.to_numpy(result), expected)

    def test_argmin_no_axis(self, backend):
        """Test argmin without axis (global argmin)."""
        arr = backend.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = backend.argmin(arr)
        assert int(backend.to_numpy(result)) == 0  # Index of min value (1.0)

    def test_argmin_with_axis(self, backend):
        """Test argmin along specific axis."""
        arr = backend.array([[1.0, 5.0, 3.0],
                             [4.0, 2.0, 6.0]])
        result = backend.argmin(arr, axis=0)
        expected = np.array([0, 1, 0])  # Indices of min along axis 0
        assert np.array_equal(backend.to_numpy(result), expected)

    def test_softmax_basic(self, backend):
        """Test softmax basic functionality."""
        arr = backend.array([1.0, 2.0, 3.0])
        result = backend.softmax(arr)

        # Softmax properties: sum to 1, all positive
        result_np = backend.to_numpy(result)
        assert np.allclose(np.sum(result_np), 1.0, atol=1e-6)
        assert np.all(result_np > 0)
        assert np.all(result_np < 1)

    def test_softmax_numerical_stability(self, backend):
        """Test softmax with large values (numerical stability)."""
        # Large values that would overflow without max subtraction
        arr = backend.array([1000.0, 1001.0, 1002.0])
        result = backend.softmax(arr)

        # Should still sum to 1 and be valid probabilities
        result_np = backend.to_numpy(result)
        assert np.allclose(np.sum(result_np), 1.0, atol=1e-6)
        assert np.all(result_np > 0)
        assert np.all(result_np < 1)
        assert not np.any(np.isnan(result_np))
        assert not np.any(np.isinf(result_np))

    def test_softmax_with_axis(self, backend):
        """Test softmax along specific axis."""
        arr = backend.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])
        result = backend.softmax(arr, axis=1)

        # Each row should sum to 1
        result_np = backend.to_numpy(result)
        row_sums = np.sum(result_np, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_softmax_zero_input(self, backend):
        """Test softmax with zero input."""
        arr = backend.array([0.0, 0.0, 0.0])
        result = backend.softmax(arr)

        # Should give uniform distribution
        result_np = backend.to_numpy(result)
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(result_np, expected, atol=1e-6)


class TestOperationsConsistency:
    """Test that new operations give consistent results across backends."""

    def test_exp_consistency(self):
        """Test exp gives same results across backends."""
        # Get truly available backends by trying to instantiate them
        available = []
        for name in AVAILABLE_BACKENDS:
            try:
                backend = get_backend(name)
                available.append((name, backend))
            except Exception:
                continue

        if len(available) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        arr_data = [0.0, 1.0, 2.0, -1.0]

        result1 = available[0][1].exp(available[0][1].array(arr_data))
        result2 = available[1][1].exp(available[1][1].array(arr_data))

        assert np.allclose(available[0][1].to_numpy(result1), available[1][1].to_numpy(result2), atol=1e-6)

    def test_softmax_consistency(self):
        """Test softmax gives same results across backends."""
        # Get truly available backends by trying to instantiate them
        available = []
        for name in AVAILABLE_BACKENDS:
            try:
                backend = get_backend(name)
                available.append((name, backend))
            except Exception:
                continue

        if len(available) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        arr_data = [1.0, 2.0, 3.0, 4.0, 5.0]

        result1 = available[0][1].softmax(available[0][1].array(arr_data))
        result2 = available[1][1].softmax(available[1][1].array(arr_data))

        assert np.allclose(available[0][1].to_numpy(result1), available[1][1].to_numpy(result2), atol=1e-6)


class TestCapabilityProbes:
    """Test backend capability probe system."""

    def test_supports_complex(self, backend):
        """Test supports_complex() probe."""
        result = backend.supports_complex()
        assert isinstance(result, bool)

        # All current backends support complex
        assert result is True

    def test_supports_sparse(self, backend):
        """Test supports_sparse() probe."""
        result = backend.supports_sparse()
        assert isinstance(result, bool)

        # NumPy and JAX don't support sparse (currently)
        # PyTorch does support sparse
        if backend.name == 'numpy':
            assert result is False
        elif backend.name == 'jax':
            assert result is False  # jax.experimental.sparse not production-ready
        elif backend.name == 'torch':
            assert result is True

    def test_supports_gpu(self, backend):
        """Test supports_gpu() probe."""
        result = backend.supports_gpu()
        assert isinstance(result, bool)

        # NumPy never supports GPU
        if backend.name == 'numpy':
            assert result is False
        # PyTorch and JAX may or may not have GPU depending on installation
        elif backend.name in ('torch', 'jax'):
            # Result is system-dependent, just check it returns a bool
            pass

    def test_supports_jit(self, backend):
        """Test supports_jit() probe."""
        result = backend.supports_jit()
        assert isinstance(result, bool)

        # Only JAX has JIT compilation
        if backend.name == 'jax':
            assert result is True
        else:
            assert result is False

    def test_supports_device_cpu(self, backend):
        """Test supports_device('cpu') - should always be True."""
        assert backend.supports_device('cpu') is True
        assert backend.supports_device('CPU') is True  # Case insensitive
        assert backend.supports_device('cpu:0') is True

    def test_supports_device_invalid(self, backend):
        """Test supports_device() with invalid device."""
        assert backend.supports_device('invalid_device') is False
        assert backend.supports_device('quantum') is False

    def test_capability_probes_consistent(self, backend):
        """Test that capability probes are consistent.

        If a backend claims to support a feature, it should actually work.
        """
        # If supports_complex, should be able to create complex arrays
        if backend.supports_complex():
            try:
                arr = backend.zeros(10, dtype='complex64')
                assert backend.dtype(arr) in ('complex64', 'complex128')
            except Exception as e:
                pytest.fail(f"Backend claims complex support but failed: {e}")

        # If supports_device('cpu'), should be able to use it
        if backend.supports_device('cpu'):
            try:
                arr = backend.zeros(10)
                # Should not raise
            except Exception as e:
                pytest.fail(f"Backend claims CPU support but failed: {e}")


class TestCapabilityProbeDetails:
    """Test specific capability probe behaviors for each backend."""

    def test_numpy_capabilities(self):
        """Test NumPy backend capabilities."""
        backend = get_backend('numpy')

        assert backend.supports_complex() is True
        assert backend.supports_sparse() is False
        assert backend.supports_gpu() is False
        assert backend.supports_jit() is False
        assert backend.supports_device('cpu') is True
        assert backend.supports_device('cuda') is False

    def test_torch_capabilities(self):
        """Test PyTorch backend capabilities (if available)."""
        if not is_backend_available('torch'):
            pytest.skip("PyTorch not available")

        backend = get_backend('torch')

        assert backend.supports_complex() is True
        assert backend.supports_sparse() is True  # PyTorch has sparse tensors
        # GPU support is system-dependent
        assert isinstance(backend.supports_gpu(), bool)
        assert backend.supports_jit() is False  # We don't expose TorchScript
        assert backend.supports_device('cpu') is True

    def test_jax_capabilities(self):
        """Test JAX backend capabilities (if available)."""
        if not is_backend_available('jax'):
            pytest.skip("JAX not available")

        backend = get_backend('jax')

        assert backend.supports_complex() is True
        assert backend.supports_sparse() is False  # Experimental, not production
        # GPU support is system-dependent
        assert isinstance(backend.supports_gpu(), bool)
        assert backend.supports_jit() is True  # JAX's key feature
        assert backend.supports_device('cpu') is True
