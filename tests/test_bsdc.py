"""Tests for BSDC (Binary Sparse Distributed Codes) model."""

import numpy as np
import pytest

from holovec import VSA
from holovec.models.bsdc import (
    BSDCModel,
    compare_sparse_vs_dense,
    expected_ones,
    optimal_sparsity,
)


class TestBSDCModel:
    """Test suite for BSDC model."""

    @pytest.fixture
    def model(self):
        """Create BSDC model for testing."""
        return VSA.create('BSDC', dim=1000, seed=42)

    @pytest.fixture
    def large_model(self):
        """Create larger BSDC model for capacity tests."""
        return VSA.create('BSDC', dim=10000, seed=42)

    def test_model_creation(self, model):
        """Test BSDC model can be created."""
        assert isinstance(model, BSDCModel)
        assert model.dimension == 1000
        assert model.model_name == "BSDC"

    def test_model_properties(self, model):
        """Test BSDC model properties."""
        assert model.is_self_inverse   # XOR is self-inverse
        assert model.is_commutative     # XOR is commutative
        assert model.is_exact_inverse   # XOR is exact inverse

    def test_optimal_sparsity(self, model):
        """Test that model uses optimal sparsity."""
        expected = 1.0 / np.sqrt(1000)
        assert np.isclose(model.sparsity, expected, rtol=1e-3)

    def test_random_generation_sparsity(self, model):
        """Test random vectors have correct sparsity."""
        a = model.random(seed=1)
        sparsity = model.measure_sparsity(a)

        # Should be close to optimal sparsity
        expected = model.sparsity
        # Allow some variation due to randomness
        assert 0.5 * expected < sparsity < 2.0 * expected

    def test_random_vectors_different(self, model):
        """Test random vectors are different."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        # Vectors should be different
        assert not np.allclose(a, b)

        # Hamming distance should be significant
        diff = model.backend.to_numpy(model.backend.xor(a, b))
        hamming = np.sum(diff)
        # For sparse codes, most differing bits come from non-overlapping 1s
        assert hamming > 0

    def test_bind(self, model):
        """Test binding operation."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        c = model.bind(a, b)

        # Result should be different from inputs
        assert not np.allclose(c, a)
        assert not np.allclose(c, b)

    def test_bind_self_inverse(self, model):
        """Test that binding is self-inverse: unbind(bind(a,b),b) = a."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        c = model.bind(a, b)
        a_recovered = model.unbind(c, b)

        # Should recover exactly (XOR is exact inverse)
        assert np.allclose(a, a_recovered)

    def test_bind_commutative(self, model):
        """Test that binding is commutative: a⊗b = b⊗a."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        ab = model.bind(a, b)
        ba = model.bind(b, a)

        # Results should be identical (XOR is commutative)
        assert np.allclose(ab, ba)

    def test_bundle_with_sparsity(self, model):
        """Test bundling maintains sparsity."""
        vectors = [model.random(seed=i) for i in range(5)]

        bundle = model.bundle(vectors, maintain_sparsity=True)

        # Sparsity should be close to target
        sparsity = model.measure_sparsity(bundle)
        expected = model.sparsity

        # Allow generous tolerance for small dimensions
        # (more variation expected with D=1000)
        assert 0.2 * expected < sparsity < 5.0 * expected

    def test_bundle_without_sparsity(self, model):
        """Test bundling with majority voting (no sparsity constraint)."""
        vectors = [model.random(seed=i) for i in range(5)]

        bundle = model.bundle(vectors, maintain_sparsity=False)

        # Sparsity may be different from target
        sparsity = model.measure_sparsity(bundle)
        # Just check it's a valid binary vector
        assert np.all(np.isin(model.backend.to_numpy(bundle), [0, 1]))

    def test_bundle_similarity(self, model):
        """Test bundled vector is similar to inputs."""
        vectors = [model.random(seed=i) for i in range(5)]
        bundle = model.bundle(vectors)

        # Bundle should have some similarity to each input
        # Note: For sparse codes, overlap similarity is used
        similarities = [model.similarity(bundle, v) for v in vectors]

        # At least some should have non-zero similarity
        assert any(s > 0.0 for s in similarities)

    def test_permute(self, model):
        """Test permutation operation."""
        a = model.random(seed=1)

        # Permute by different amounts
        a_p1 = model.permute(a, k=1)
        a_p2 = model.permute(a, k=2)

        # Permutations should be different
        assert not np.allclose(a_p1, a_p2)

        # Permutation preserves sparsity exactly
        orig_sparsity = model.measure_sparsity(a)
        perm_sparsity = model.measure_sparsity(a_p1)
        assert np.isclose(orig_sparsity, perm_sparsity)

        # Permuting by dimension returns to original
        a_full = model.permute(a, k=model.dimension)
        assert np.allclose(a, a_full)

    def test_rehash(self, model):
        """Test rehashing to restore sparsity."""
        # Create vector with wrong sparsity
        a = model.random(seed=1)

        # Artificially change sparsity by setting some bits
        a_np = model.backend.to_numpy(a).copy()
        # Set more bits to 1 (increase sparsity)
        zero_indices = np.where(a_np == 0)[0]
        a_np[zero_indices[:50]] = 1
        a_modified = model.backend.from_numpy(a_np.astype(np.int32))

        # Measure increased sparsity
        sparsity_before = model.measure_sparsity(a_modified)

        # Rehash to restore optimal sparsity
        a_rehashed = model.rehash(a_modified)

        # Should now have target sparsity
        sparsity_after = model.measure_sparsity(a_rehashed)
        expected = model.sparsity

        # Should be very close to target
        assert np.isclose(sparsity_after, expected, rtol=0.1)

    def test_encode_sequence_position(self, model):
        """Test sequence encoding with position binding."""
        items = [model.random(seed=i) for i in range(3)]

        seq = model.encode_sequence(items, use_ngrams=False)

        # Result should be valid binary vector
        assert np.all(np.isin(model.backend.to_numpy(seq), [0, 1]))

        # Should have reasonable sparsity
        sparsity = model.measure_sparsity(seq)
        assert 0 < sparsity < 0.5

    def test_encode_sequence_ngrams(self, model):
        """Test sequence encoding with n-grams."""
        items = [model.random(seed=i) for i in range(4)]

        # Bigrams
        seq = model.encode_sequence(items, use_ngrams=True, n=2)

        # Result should be valid binary vector
        assert np.all(np.isin(model.backend.to_numpy(seq), [0, 1]))

    def test_encode_short_sequence_ngrams(self, model):
        """Test n-gram encoding with sequence shorter than n."""
        items = [model.random(seed=i) for i in range(2)]

        # Try trigrams (n=3) with only 2 items
        seq = model.encode_sequence(items, use_ngrams=True, n=3)

        # Should fall back to simple bundle
        bundle = model.bundle(items)

        # Should be same or similar
        # (may not be identical due to sparsity maintenance)
        sim = model.similarity(seq, bundle)
        assert sim > 0.5

    def test_similarity_range(self, model):
        """Test similarity values are in valid range [0, 1]."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        # Self-similarity should be 1.0
        sim_aa = model.similarity(a, a)
        assert np.isclose(sim_aa, 1.0, rtol=1e-5)

        # Random vectors should have low similarity
        sim_ab = model.similarity(a, b)
        assert 0.0 <= sim_ab <= 1.0
        # For sparse codes, random overlap is unlikely
        assert sim_ab < 0.3

    def test_large_dimension_sparsity(self, large_model):
        """Test sparsity with larger dimension."""
        a = large_model.random(seed=1)

        sparsity = large_model.measure_sparsity(a)
        expected = 1.0 / np.sqrt(10000)  # 0.01

        # Should have about 100 ones in 10,000 dimensions
        ones = int(sparsity * 10000)
        expected_ones_val = expected_ones(10000)

        # Allow some variation
        assert 0.5 * expected_ones_val < ones < 2.0 * expected_ones_val

    def test_empty_bundle_raises(self, model):
        """Test that bundling empty sequence raises error."""
        with pytest.raises(ValueError, match="Cannot bundle empty sequence"):
            model.bundle([])

    def test_empty_sequence_raises(self, model):
        """Test that encoding empty sequence raises error."""
        with pytest.raises(ValueError, match="Cannot encode empty sequence"):
            model.encode_sequence([])


class TestBSDCHelpers:
    """Test helper functions for BSDC."""

    def test_optimal_sparsity_calculation(self):
        """Test optimal sparsity calculation."""
        # For D=10,000, optimal p = 1/100 = 0.01
        assert np.isclose(optimal_sparsity(10000), 0.01)

        # For D=100,000, optimal p = 1/316.23 ≈ 0.00316
        assert np.isclose(optimal_sparsity(100000), 0.00316, rtol=0.01)

        # For D=1,000, optimal p = 1/31.62 ≈ 0.0316
        assert np.isclose(optimal_sparsity(1000), 0.0316, rtol=0.01)

    def test_expected_ones_calculation(self):
        """Test expected ones calculation."""
        # For D=10,000 with optimal sparsity
        assert expected_ones(10000) == 100

        # For D=10,000 with custom sparsity
        assert expected_ones(10000, sparsity=0.05) == 500

        # For D=1,000 with optimal sparsity
        # 1/√1000 ≈ 0.0316, so ~31 ones
        ones = expected_ones(1000)
        assert 20 < ones < 50

    def test_compare_sparse_vs_dense(self):
        """Test comparison between sparse and dense codes."""
        result = compare_sparse_vs_dense(dimension=1000, trials=5)

        # Should return valid statistics
        assert 'memory_ratio' in result
        assert 'bsdc_recovery_mean' in result
        assert 'bsc_recovery_mean' in result
        assert 'bsdc_sparsity' in result

        # Memory ratio should show sparse is much more efficient
        # For D=1000, p=1/√1000≈0.0316, vs BSC p=0.5
        # Ratio should be ~0.0632
        assert result['memory_ratio'] < 0.15  # Sparse uses < 15% memory

        # Both should have perfect recovery (XOR is exact)
        assert result['bsdc_recovery_mean'] == 1.0
        assert result['bsc_recovery_mean'] == 1.0

        # Sparsity should be optimal
        expected = 1.0 / np.sqrt(1000)
        assert np.isclose(result['bsdc_sparsity'], expected, rtol=0.01)


class TestBSDCCapacity:
    """Test capacity and scaling properties of BSDC."""

    def test_multiple_bindings_preserve_recovery(self):
        """Test that multiple bindings still allow recovery."""
        model = VSA.create('BSDC', dim=5000, seed=42)

        # Create nested binding: ((a⊗b)⊗c)⊗d
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)
        d = model.random(seed=4)

        ab = model.bind(a, b)
        abc = model.bind(ab, c)
        abcd = model.bind(abc, d)

        # Unbind in reverse: recover a from abcd
        abc_recovered = model.unbind(abcd, d)
        ab_recovered = model.unbind(abc_recovered, c)
        a_recovered = model.unbind(ab_recovered, b)

        # Should recover exactly (XOR chain)
        sim = model.similarity(a, a_recovered)
        assert sim == 1.0  # Exact for XOR

    def test_large_bundle_capacity(self):
        """Test bundling many vectors."""
        model = VSA.create('BSDC', dim=5000, seed=42)

        # Create many random vectors
        n_vectors = 20
        vectors = [model.random(seed=i) for i in range(n_vectors)]

        # Bundle them
        bundle = model.bundle(vectors, maintain_sparsity=True)

        # Should still have reasonable sparsity
        sparsity = model.measure_sparsity(bundle)
        expected = model.sparsity

        # With sparsity maintenance, should be close to target
        assert 0.3 * expected < sparsity < 3.0 * expected

    def test_different_backends(self):
        """Test BSDC works with different backends."""
        backends = ['numpy']

        # Try PyTorch if available
        try:
            import torch
            backends.append('torch')
        except ImportError:
            pass

        # Try JAX if available
        try:
            import jax
            backends.append('jax')
        except ImportError:
            pass

        for backend_name in backends:
            model = VSA.create('BSDC', dim=1000, backend=backend_name, seed=42)

            a = model.random(seed=1)
            b = model.random(seed=2)

            # Test bind/unbind
            c = model.bind(a, b)
            a_recovered = model.unbind(c, b)

            # Should be exact
            assert np.allclose(
                model.backend.to_numpy(a),
                model.backend.to_numpy(a_recovered)
            ), f"Failed for backend: {backend_name}"


class TestBSDCMemoryEfficiency:
    """Test memory efficiency properties of BSDC."""

    def test_sparsity_reduces_memory(self):
        """Demonstrate memory savings from sparsity."""
        # Create two models: one with optimal sparsity, one dense
        sparse = VSA.create('BSDC', dim=10000, seed=42)
        dense = VSA.create('BSC', dim=10000, seed=42)

        a_sparse = sparse.random(seed=1)
        a_dense = dense.random(seed=1)

        # Count ones (proxy for memory in sparse representations)
        ones_sparse = float(sparse.backend.sum(a_sparse))
        ones_dense = float(dense.backend.sum(a_dense))

        # Sparse should have ~100 ones, dense should have ~5000
        assert ones_sparse < 200
        assert ones_dense > 4000

        # Ratio should be ~50x memory savings
        ratio = ones_sparse / ones_dense
        assert ratio < 0.05  # Sparse uses <5% memory


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
