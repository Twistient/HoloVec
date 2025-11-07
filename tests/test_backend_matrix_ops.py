"""Tests for backend matrix operations.

Tests matrix operations needed for GHRR and VTB models.
"""

import pytest
import numpy as np

from holovec.backends import get_backend, get_available_backends


AVAILABLE_BACKENDS = get_available_backends()


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend(request):
    """Parametrized fixture providing all available backends."""
    return get_backend(request.param)


class TestMatrixMultiplication:
    """Test matrix multiplication operations."""

    def test_matmul_2d(self, backend):
        """Test 2D matrix multiplication."""
        a = backend.array([[1, 2], [3, 4]], dtype='float32')
        b = backend.array([[5, 6], [7, 8]], dtype='float32')
        result = backend.matmul(a, b)

        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        assert np.allclose(backend.to_numpy(result), expected)

    def test_matmul_batched(self, backend):
        """Test batched matrix multiplication."""
        # Batch of 3 matrices, each 2x2
        a = backend.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]]
        ], dtype='float32')

        b = backend.array([
            [[1, 0], [0, 1]],
            [[2, 0], [0, 2]],
            [[3, 0], [0, 3]]
        ], dtype='float32')

        result = backend.matmul(a, b)
        result_np = backend.to_numpy(result)

        # First should be identity multiply (unchanged)
        assert np.allclose(result_np[0], [[1, 2], [3, 4]])
        # Second should be 2x
        assert np.allclose(result_np[1], [[10, 12], [14, 16]])
        # Third should be 3x
        assert np.allclose(result_np[2], [[27, 30], [33, 36]])


class TestMatrixTranspose:
    """Test matrix transpose operations."""

    def test_transpose_2d(self, backend):
        """Test 2D matrix transpose."""
        a = backend.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
        result = backend.matrix_transpose(a)

        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
        assert np.allclose(backend.to_numpy(result), expected)

    def test_transpose_batched(self, backend):
        """Test batched matrix transpose."""
        # Batch of 2 matrices
        a = backend.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype='float32')

        result = backend.matrix_transpose(a)
        result_np = backend.to_numpy(result)

        assert np.allclose(result_np[0], [[1, 3], [2, 4]])
        assert np.allclose(result_np[1], [[5, 7], [6, 8]])

    def test_transpose_complex(self, backend):
        """Test transpose with complex matrices."""
        a = backend.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype='complex64')
        result = backend.matrix_transpose(a)

        # Note: This is transpose, not conjugate transpose
        expected = np.array([[1+2j, 5+6j], [3+4j, 7+8j]], dtype=np.complex64)
        assert np.allclose(backend.to_numpy(result), expected)


class TestMatrixTrace:
    """Test matrix trace operations."""

    def test_trace_2d(self, backend):
        """Test 2D matrix trace."""
        a = backend.array([[1, 2], [3, 4]], dtype='float32')
        result = backend.matrix_trace(a)

        assert np.allclose(backend.to_numpy(result), 5.0)  # 1 + 4

    def test_trace_batched(self, backend):
        """Test batched matrix trace."""
        # Batch of 3 matrices
        a = backend.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]]
        ], dtype='float32')

        result = backend.matrix_trace(a)
        result_np = backend.to_numpy(result)

        expected = np.array([5, 13, 21], dtype=np.float32)  # 1+4, 5+8, 9+12
        assert np.allclose(result_np, expected)

    def test_trace_complex(self, backend):
        """Test trace with complex matrices."""
        a = backend.array([[1+1j, 2], [3, 4+1j]], dtype='complex64')
        result = backend.matrix_trace(a)

        expected = (1+1j) + (4+1j)  # 5+2j
        assert np.allclose(backend.to_numpy(result), expected)


class TestReshape:
    """Test reshape operations."""

    def test_reshape_basic(self, backend):
        """Test basic reshape."""
        a = backend.array([1, 2, 3, 4, 5, 6], dtype='float32')
        result = backend.reshape(a, (2, 3))

        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        assert np.allclose(backend.to_numpy(result), expected)

    def test_reshape_to_3d(self, backend):
        """Test reshape to 3D."""
        a = backend.array(list(range(24)), dtype='float32')
        result = backend.reshape(a, (2, 3, 4))

        assert backend.shape(result) == (2, 3, 4)

    def test_reshape_flatten(self, backend):
        """Test reshape to flatten."""
        a = backend.array([[1, 2], [3, 4]], dtype='float32')
        result = backend.reshape(a, (4,))

        expected = np.array([1, 2, 3, 4], dtype=np.float32)
        assert np.allclose(backend.to_numpy(result), expected)


class TestGHRROperations:
    """Test operations as they would be used in GHRR."""

    def test_ghrr_binding_simulation(self, backend):
        """Simulate GHRR binding with batched matrices."""
        D = 10
        m = 3

        # Create two GHRR-style hypervectors (batches of m×m matrices)
        # Each hypervector is D matrices of size m×m
        h1 = backend.random_phasor((D, m, m), dtype='complex64')
        h2 = backend.random_phasor((D, m, m), dtype='complex64')

        # Binding in GHRR: element-wise matrix multiplication
        bound = backend.matmul(h1, h2)

        # Check shape is preserved
        assert backend.shape(bound) == (D, m, m)

        # Check all matrices are still approximately unitary
        # For unitary matrices: A^† A = I
        for i in range(3):  # Check first 3
            matrix = bound[i] if hasattr(bound, '__getitem__') else backend.to_numpy(bound)[i]
            if not isinstance(matrix, np.ndarray):
                matrix = backend.to_numpy(matrix)

            # Compute matrix^† @ matrix
            should_be_identity = np.conj(matrix.T) @ matrix
            identity = np.eye(m, dtype=np.complex64)

            # Should be close to identity
            assert np.allclose(should_be_identity, identity, atol=0.1)

    def test_ghrr_similarity_simulation(self, backend):
        """Simulate GHRR similarity computation."""
        D = 100
        m = 3

        # Two GHRR hypervectors
        h1 = backend.random_phasor((D, m, m), dtype='complex64')
        h2 = backend.random_phasor((D, m, m), dtype='complex64')

        # GHRR similarity: δ(H₁, H₂) = (1/mD) Re(tr(Σⱼ aⱼbⱼ†))

        # Compute h2^†
        h2_conj_t = backend.conjugate(backend.matrix_transpose(h2))

        # Element-wise matrix multiply
        products = backend.matmul(h1, h2_conj_t)

        # Compute traces
        traces = backend.matrix_trace(products)

        # Sum and normalize
        total = backend.sum(traces)
        similarity = backend.to_numpy(total).real / (m * D)

        # Similarity should be relatively small for random vectors
        assert abs(similarity) < 0.2

    def test_ghrr_unbinding_simulation(self, backend):
        """Simulate GHRR binding and unbinding."""
        D = 50
        m = 2

        # Create base vectors
        a = backend.random_phasor((D, m, m), dtype='complex64')
        b = backend.random_phasor((D, m, m), dtype='complex64')

        # Bind
        c = backend.matmul(a, b)

        # Unbind: c @ b^†
        b_conj_t = backend.conjugate(backend.matrix_transpose(b))
        a_recovered = backend.matmul(c, b_conj_t)

        # Compute similarity between a and a_recovered
        a_conj_t = backend.conjugate(backend.matrix_transpose(a))
        products = backend.matmul(a_recovered, a_conj_t)
        traces = backend.matrix_trace(products)
        total = backend.sum(traces)
        similarity = backend.to_numpy(total).real / (m * D)

        # Should have high similarity (unbinding should recover)
        assert similarity > 0.9


class TestCrossBackendConsistency:
    """Test that matrix operations give consistent results across backends."""

    @pytest.mark.skipif(len(AVAILABLE_BACKENDS) < 2,
                       reason="Need at least 2 backends for consistency test")
    def test_matmul_consistency(self):
        """Matrix multiplication should give same results across backends."""
        backend1 = get_backend(AVAILABLE_BACKENDS[0])
        backend2 = get_backend(AVAILABLE_BACKENDS[1] if len(AVAILABLE_BACKENDS) > 1 else AVAILABLE_BACKENDS[0])

        # Same random matrices
        np.random.seed(42)
        a_np = np.random.randn(5, 3, 3).astype(np.float32)
        b_np = np.random.randn(5, 3, 3).astype(np.float32)

        a1 = backend1.from_numpy(a_np)
        b1 = backend1.from_numpy(b_np)
        result1 = backend1.matmul(a1, b1)

        a2 = backend2.from_numpy(a_np)
        b2 = backend2.from_numpy(b_np)
        result2 = backend2.matmul(a2, b2)

        assert np.allclose(backend1.to_numpy(result1), backend2.to_numpy(result2), atol=1e-5)

    @pytest.mark.skipif(len(AVAILABLE_BACKENDS) < 2,
                       reason="Need at least 2 backends for consistency test")
    def test_trace_consistency(self):
        """Matrix trace should give same results across backends."""
        backend1 = get_backend(AVAILABLE_BACKENDS[0])
        backend2 = get_backend(AVAILABLE_BACKENDS[1] if len(AVAILABLE_BACKENDS) > 1 else AVAILABLE_BACKENDS[0])

        np.random.seed(42)
        a_np = np.random.randn(10, 4, 4).astype(np.float32)

        a1 = backend1.from_numpy(a_np)
        result1 = backend1.matrix_trace(a1)

        a2 = backend2.from_numpy(a_np)
        result2 = backend2.matrix_trace(a2)

        assert np.allclose(backend1.to_numpy(result1), backend2.to_numpy(result2), atol=1e-5)
