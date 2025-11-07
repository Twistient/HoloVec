"""Tests for VSA model implementations.

Tests algebraic properties and correctness of VSA operations:
- Distributivity: a ⊗ (b + c) = (a ⊗ b) + (a ⊗ c)
- Invertibility: unbind(bind(a, b), b) ≈ a
- Commutativity: bind(a, b) = bind(b, a)
- Self-inverse: bind(a, b) = unbind(a, b)
"""

import pytest
import numpy as np

from holovec import VSA


# Test parameters
DIMENSIONS = [512, 1024]  # Test with different dimensions
MODELS = ['MAP', 'FHRR']  # Models to test


@pytest.fixture(params=MODELS)
def model_name(request):
    """Parametrized fixture for model names."""
    return request.param


@pytest.fixture
def model(model_name):
    """Create a VSA model for testing."""
    # Use smaller dimension for faster tests
    return VSA.create(model_name, dim=512, seed=42)


class TestModelCreation:
    """Test VSA model creation and configuration."""

    @pytest.mark.parametrize("model_type", MODELS)
    def test_create_model(self, model_type):
        """Test model creation."""
        model = VSA.create(model_type, dim=512)
        assert model is not None
        assert model.dimension == 512

    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_model_dimension(self, dim):
        """Test models work with different dimensions."""
        model = VSA.create('MAP', dim=dim)
        assert model.dimension == dim

        vec = model.random()
        assert model.backend.shape(vec) == (dim,)

    def test_model_properties(self, model, model_name):
        """Test model properties are correctly reported."""
        if model_name == 'MAP':
            assert model.is_self_inverse == True
            assert model.is_commutative == True
        elif model_name == 'FHRR':
            assert model.is_self_inverse == False
            assert model.is_commutative == True
            assert model.is_exact_inverse == True


class TestRandomVectorGeneration:
    """Test random vector generation."""

    def test_random_shape(self, model):
        """Random vectors should have correct shape."""
        vec = model.random()
        assert model.backend.shape(vec) == (model.dimension,)

    def test_random_different(self, model):
        """Different random vectors should be different."""
        vec1 = model.random()
        vec2 = model.random()

        # Should not be identical
        sim = model.similarity(vec1, vec2)
        # Note: For bipolar/binary spaces with Hamming similarity,
        # random vectors have expected similarity ~0.5, not 0.
        # For continuous spaces, expected similarity is ~0.
        assert abs(sim) < 0.9  # Should not be too similar

    def test_random_seed_reproducible(self, model):
        """Same seed should produce same vector."""
        vec1 = model.random(seed=42)
        vec2 = model.random(seed=42)

        sim = model.similarity(vec1, vec2)
        assert np.allclose(sim, 1.0, atol=1e-5)

    def test_random_sequence(self, model):
        """Should be able to generate multiple random vectors."""
        vecs = model.random_sequence(10, seed=42)
        assert len(vecs) == 10

        # All should be different from each other (not too similar)
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = model.similarity(vecs[i], vecs[j])
                # For bipolar/binary: expected ~0.5, for continuous: expected ~0
                assert abs(sim) < 0.9  # Should not be too similar


class TestBindingOperation:
    """Test binding operation."""

    def test_binding_dissimilar(self, model):
        """Binding should produce dissimilar result."""
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.bind(a, b)

        # c should be dissimilar to both a and b
        sim_a = model.similarity(c, a)
        sim_b = model.similarity(c, b)

        assert abs(sim_a) < 0.5
        assert abs(sim_b) < 0.5

    def test_binding_commutativity(self, model):
        """Test commutativity of binding."""
        if not model.is_commutative:
            pytest.skip("Model is not commutative")

        a = model.random(seed=1)
        b = model.random(seed=2)

        c1 = model.bind(a, b)
        c2 = model.bind(b, a)

        sim = model.similarity(c1, c2)
        assert np.allclose(sim, 1.0, atol=1e-5)

    def test_structured_similarity(self, model):
        """Binding should preserve structured similarity."""
        a1 = model.random(seed=1)
        a2 = model.random(seed=2)
        b = model.random(seed=3)

        # If a1 ≈ a2, then a1⊗b should be similar to a2⊗b
        # Create a2 as noisy version of a1
        a2 = model.bundle([a1, model.random(seed=4)])

        c1 = model.bind(a1, b)
        c2 = model.bind(a2, b)

        # c1 and c2 should have some similarity
        sim = model.similarity(c1, c2)
        # Not a strict requirement, but generally holds
        assert sim > 0.0


class TestUnbindingOperation:
    """Test unbinding operation."""

    def test_unbinding_recovery(self, model):
        """Unbinding should recover original vector."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        c = model.bind(a, b)
        a_recovered = model.unbind(c, b)

        sim = model.similarity(a, a_recovered)

        if model.is_exact_inverse:
            # For exact inverse models (FHRR), should be nearly perfect
            assert sim > 0.95
        else:
            # For approximate models, should still be good
            assert sim > 0.7

    def test_self_inverse_property(self, model):
        """For self-inverse models, bind = unbind."""
        if not model.is_self_inverse:
            pytest.skip("Model is not self-inverse")

        a = model.random(seed=1)
        b = model.random(seed=2)

        c1 = model.bind(a, b)
        c2 = model.unbind(a, b)

        sim = model.similarity(c1, c2)
        assert np.allclose(sim, 1.0, atol=1e-5)

    def test_unbinding_chain(self, model):
        """Test chain of binding and unbinding."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        # bind then unbind should give back a
        c = model.bind(a, b)
        a_recovered = model.unbind(c, b)

        # unbind again with a should give b
        b_recovered = model.unbind(c, a)

        sim_a = model.similarity(a, a_recovered)
        sim_b = model.similarity(b, b_recovered)

        if model.is_exact_inverse:
            assert sim_a > 0.95
            assert sim_b > 0.95
        else:
            assert sim_a > 0.7
            assert sim_b > 0.7


class TestBundlingOperation:
    """Test bundling (superposition) operation."""

    def test_bundling_similarity(self, model):
        """Bundled vector should be similar to inputs."""
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)

        bundled = model.bundle([a, b, c])

        # Should be similar to all inputs
        sim_a = model.similarity(bundled, a)
        sim_b = model.similarity(bundled, b)
        sim_c = model.similarity(bundled, c)

        assert sim_a > 0.3
        assert sim_b > 0.3
        assert sim_c > 0.3

    def test_bundling_single(self, model):
        """Bundling single vector should return similar vector."""
        a = model.random(seed=1)
        bundled = model.bundle([a])

        sim = model.similarity(a, bundled)
        assert np.allclose(sim, 1.0, atol=1e-5)

    def test_bundling_capacity(self, model):
        """Test bundling many vectors."""
        # Create 10 random vectors
        vectors = model.random_sequence(10, seed=42)

        # Bundle them
        bundled = model.bundle(vectors)

        # Check similarity to each
        similarities = [model.similarity(bundled, v) for v in vectors]

        # All should be positive (similar)
        assert all(s > 0.1 for s in similarities)

        # Mean similarity should be reasonable
        mean_sim = np.mean(similarities)
        assert mean_sim > 0.2

    def test_bundling_empty_raises(self, model):
        """Bundling empty sequence should raise error."""
        with pytest.raises(ValueError):
            model.bundle([])


class TestDistributivity:
    """Test distributivity: a ⊗ (b + c) = (a ⊗ b) + (a ⊗ c)."""

    def test_distributivity_over_addition(self, model):
        """Binding should distribute over bundling."""
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)

        # Left side: a ⊗ (b + c)
        bundled = model.bundle([b, c])
        left = model.bind(a, bundled)

        # Right side: (a ⊗ b) + (a ⊗ c)
        ab = model.bind(a, b)
        ac = model.bind(a, c)
        right = model.bundle([ab, ac])

        # Should be similar
        sim = model.similarity(left, right)
        assert sim > 0.7  # Not exact due to normalization


class TestPermutationOperation:
    """Test permutation operation."""

    def test_permutation_preserves_similarity(self, model):
        """Permutation should preserve similarity between vectors."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        # Similarity before permutation
        sim_before = model.similarity(a, b)

        # Permute both
        a_perm = model.permute(a, k=1)
        b_perm = model.permute(b, k=1)

        # Similarity after permutation
        sim_after = model.similarity(a_perm, b_perm)

        # Should be approximately the same
        assert np.allclose(sim_before, sim_after, atol=0.1)

    def test_permutation_invertible(self, model):
        """Permutation should be invertible."""
        a = model.random(seed=1)

        # Permute and unpermute
        a_perm = model.permute(a, k=5)
        a_back = model.unpermute(a_perm, k=5)

        # Should get back original
        sim = model.similarity(a, a_back)
        assert np.allclose(sim, 1.0, atol=1e-5)

    def test_permutation_for_sequences(self, model):
        """Test using permutation for sequence encoding."""
        # Encode sequence [a, b, c] using permutation
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)

        # Position encoding: s = a + ρ(b) + ρ²(c)
        sequence = model.bundle([
            a,
            model.permute(b, k=1),
            model.permute(c, k=2)
        ])

        # Should be able to query for position 0
        sim_a = model.similarity(sequence, a)
        assert sim_a > 0.2

        # Should be able to query for position 1
        b_at_0 = model.unpermute(sequence, k=1)
        sim_b = model.similarity(b_at_0, b)
        assert sim_b > 0.2


class TestMultipleBindings:
    """Test binding multiple vectors."""

    def test_bind_multiple(self, model):
        """Test binding sequence of vectors."""
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)

        # Bind all three
        result = model.bind_multiple([a, b, c])

        # Should be dissimilar to all inputs
        assert abs(model.similarity(result, a)) < 0.5
        assert abs(model.similarity(result, b)) < 0.5
        assert abs(model.similarity(result, c)) < 0.5

    def test_bind_multiple_recovery(self, model):
        """Test recovering from multiple bindings."""
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)

        # Bind: abc = a ⊗ b ⊗ c
        abc = model.bind_multiple([a, b, c])

        # Unbind c and b to get a
        bc = model.bind(b, c)
        a_recovered = model.unbind(abc, bc)

        sim = model.similarity(a, a_recovered)

        if model.is_exact_inverse:
            assert sim > 0.9
        else:
            assert sim > 0.5  # Approximate recovery degrades


class TestNormalization:
    """Test normalization operations."""

    def test_normalize_idempotent(self, model):
        """Normalizing twice should give same result."""
        vec = model.random(seed=1)
        norm1 = model.normalize(vec)
        norm2 = model.normalize(norm1)

        sim = model.similarity(norm1, norm2)
        assert np.allclose(sim, 1.0, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_bind_multiple_too_few(self, model):
        """Should raise error with fewer than 2 vectors."""
        a = model.random(seed=1)
        with pytest.raises(ValueError):
            model.bind_multiple([a])

    def test_dimension_mismatch_bind(self, model):
        """Binding vectors of different dimensions should fail."""
        # This test depends on backend checking dimensions
        # Some backends may broadcast, others may fail
        # Skip for now, but good to add backend dimension checking
        pytest.skip("Dimension checking not yet implemented")


# Model-specific tests

class TestMAPSpecific:
    """MAP-specific tests."""

    @pytest.fixture
    def map_model(self):
        return VSA.create('MAP', dim=512, seed=42)

    def test_bipolar_values(self, map_model):
        """MAP with bipolar space should have ±1 values."""
        if map_model.space.space_name != 'bipolar':
            pytest.skip("Not using bipolar space")

        vec = map_model.random()
        vec_np = map_model.backend.to_numpy(vec)

        # All values should be ±1
        assert np.all((vec_np == -1) | (vec_np == 1))


class TestFHRRSpecific:
    """FHRR-specific tests."""

    @pytest.fixture
    def fhrr_model(self):
        return VSA.create('FHRR', dim=512, seed=42)

    def test_unit_magnitude(self, fhrr_model):
        """FHRR vectors should have unit magnitude."""
        vec = fhrr_model.random()
        vec_np = fhrr_model.backend.to_numpy(vec)

        # All components should have magnitude 1
        magnitudes = np.abs(vec_np)
        assert np.allclose(magnitudes, 1.0, atol=1e-5)

    def test_exact_unbinding(self, fhrr_model):
        """FHRR should have exact unbinding."""
        a = fhrr_model.random(seed=1)
        b = fhrr_model.random(seed=2)

        c = fhrr_model.bind(a, b)
        a_recovered = fhrr_model.unbind(c, b)

        # Should be nearly exact
        sim = fhrr_model.similarity(a, a_recovered)
        assert sim > 0.99

    def test_fractional_power(self, fhrr_model):
        """Test fractional power encoding.

        Note: We use a small exponent (0.5) to avoid angle wrapping issues.
        For larger exponents, angles can wrap outside [-π, π], which breaks
        the reversibility of (z^α)^(1/α) = z due to complex logarithm branch cuts.
        """
        base = fhrr_model.random(seed=1)

        # Encode with small exponent to avoid angle wrapping
        exponent = 0.5
        encoded = fhrr_model.fractional_power(base, exponent)

        # Should still be unit magnitude
        encoded_np = fhrr_model.backend.to_numpy(encoded)
        magnitudes = np.abs(encoded_np)
        assert np.allclose(magnitudes, 1.0, atol=1e-5)

        # Inverse power should recover base (works for small exponents)
        decoded = fhrr_model.fractional_power(encoded, 1.0 / exponent)
        sim = fhrr_model.similarity(base, decoded)
        assert sim > 0.99
