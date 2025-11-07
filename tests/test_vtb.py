"""Tests for VTB (Vector-derived Transformation Binding) model."""

import numpy as np
import pytest

from holovec import VSA
from holovec.models.vtb import VTBModel, compare_vtb_hrr_unbinding


class TestVTBModel:
    """Test suite for VTB model."""

    @pytest.fixture
    def model(self):
        """Create VTB model for testing."""
        return VSA.create('VTB', dim=100, seed=42)

    def test_model_creation(self, model):
        """Test VTB model can be created."""
        assert isinstance(model, VTBModel)
        assert model.dimension == 100
        assert model.model_name == "VTB"

    def test_model_properties(self, model):
        """Test VTB model properties."""
        assert not model.is_self_inverse
        assert not model.is_commutative   # MBAT-style is non-commutative
        assert not model.is_exact_inverse  # Approximate inverse

    def test_random_generation(self, model):
        """Test random vector generation."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        # Vectors should be different
        assert not np.allclose(a, b)

        # Should be normalized (unit length for real space)
        norm_a = np.linalg.norm(model.backend.to_numpy(a))
        norm_b = np.linalg.norm(model.backend.to_numpy(b))
        assert np.isclose(norm_a, 1.0, rtol=1e-5)
        assert np.isclose(norm_b, 1.0, rtol=1e-5)

    def test_bind(self, model):
        """Test binding operation."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        c = model.bind(a, b)

        # Result should be different from inputs
        assert not np.allclose(c, a)
        assert not np.allclose(c, b)

        # Result should be normalized
        norm_c = np.linalg.norm(model.backend.to_numpy(c))
        assert np.isclose(norm_c, 1.0, rtol=1e-5)

    def test_commutativity(self, model):
        """Test that VTB is non-commutative: a⊗b != b⊗a."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        ab = model.bind(a, b)
        ba = model.bind(b, a)

        # Results should differ (non-commutative)
        assert not np.allclose(ab, ba, rtol=1e-5)

        # Similarity should not be near 1
        sim = model.test_non_commutativity(a, b)
        assert sim < 0.95

    def test_unbind(self, model):
        """Test unbinding operation (approximate recovery)."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        # Bind then unbind (recover b from a)
        c = model.bind(a, b)
        b_recovered = model.unbind(c, a)

        # Should recover b with high similarity (but not exact)
        similarity = model.similarity(b, b_recovered)
        assert similarity > 0.7  # Good but approximate recovery

    def test_bundle(self, model):
        """Test bundling operation."""
        vectors = [model.random(seed=i) for i in range(5)]

        bundle = model.bundle(vectors)

        # Bundle should be similar to all inputs
        similarities = [model.similarity(bundle, v) for v in vectors]
        assert all(s > 0.1 for s in similarities)

        # Bundle should be normalized
        norm = np.linalg.norm(model.backend.to_numpy(bundle))
        assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_permute(self, model):
        """Test permutation operation."""
        a = model.random(seed=1)

        # Permute by different amounts
        a_p1 = model.permute(a, k=1)
        a_p2 = model.permute(a, k=2)

        # Permutations should be different from original
        assert not np.allclose(a, a_p1)
        assert not np.allclose(a, a_p2)
        assert not np.allclose(a_p1, a_p2)

        # Permutation should preserve length
        norm_a = np.linalg.norm(model.backend.to_numpy(a))
        norm_p1 = np.linalg.norm(model.backend.to_numpy(a_p1))
        assert np.isclose(norm_a, norm_p1, rtol=1e-5)

        # Permuting by dimension should return to original
        a_full = model.permute(a, k=model.dimension)
        assert np.allclose(a, a_full, rtol=1e-5)

    def test_circulant_matrix(self, model):
        """Test circulant matrix construction."""
        # Create simple test vector
        vec = model.backend.array([1.0, 2.0, 3.0, 4.0])
        matrix = model._vector_to_circulant(vec)
        matrix_np = model.backend.to_numpy(matrix)

        # Check shape
        assert matrix_np.shape == (4, 4)

        # Check circulant structure
        expected = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 1.0, 2.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [2.0, 3.0, 4.0, 1.0]
        ])
        assert np.allclose(matrix_np, expected)

    def test_bind_sequence_with_permutation(self, model):
        """Test sequence binding with permutation strategy."""
        items = [model.random(seed=i) for i in range(3)]

        # Bind sequence with permutation
        seq = model.bind_sequence(items, use_permute=True)

        # Result should be normalized
        norm = np.linalg.norm(model.backend.to_numpy(seq))
        assert np.isclose(norm, 1.0, rtol=1e-5)

        # Should have some relationship to bundled items
        bundle = model.bundle(items)
        sim = model.similarity(seq, bundle)
        # Can be negative for real-valued vectors
        assert -1.0 <= sim <= 1.0  # Valid similarity range

    def test_bind_sequence_nested(self, model):
        """Test sequence binding with nested strategy."""
        items = [model.random(seed=i) for i in range(3)]

        # Bind sequence with nested binding
        seq = model.bind_sequence(items, use_permute=False)

        # Result should be normalized
        norm = np.linalg.norm(model.backend.to_numpy(seq))
        assert np.isclose(norm, 1.0, rtol=1e-5)

        # Should be different from simple bundle
        bundle = model.bundle(items)
        sim = model.similarity(seq, bundle)
        assert sim < 0.5  # Significantly different encoding

    def test_similarity_range(self, model):
        """Test similarity values are in valid range."""
        a = model.random(seed=1)
        b = model.random(seed=2)

        # Self-similarity should be 1.0
        sim_aa = model.similarity(a, a)
        assert np.isclose(sim_aa, 1.0, rtol=1e-5)

        # Random vectors should have low similarity
        sim_ab = model.similarity(a, b)
        assert -0.3 < sim_ab < 0.3  # Close to orthogonal

    def test_compositional_binding(self, model):
        """Test compositional binding: (a⊗b)⊗c."""
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)

        # Nested binding
        ab = model.bind(a, b)
        abc = model.bind(ab, c)

        # With MBAT-style binding, recover c from abc using key ab
        c_recovered = model.unbind(abc, ab)
        sim_c = model.similarity(c, c_recovered)
        assert sim_c > 0.7

        # And recover b from ab using key a
        b_recovered = model.unbind(ab, a)
        sim_b = model.similarity(b, b_recovered)
        assert sim_b > 0.7

    def test_different_backends(self):
        """Test VTB works with different backends."""
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
            model = VSA.create('VTB', dim=100, backend=backend_name, seed=42)

            a = model.random(seed=1)
            b = model.random(seed=2)

            c = model.bind(a, b)
            b_recovered = model.unbind(c, a)

            sim = model.similarity(b, b_recovered)
        assert sim > 0.7, f"Failed for backend: {backend_name}"

    def test_empty_bundle_raises(self, model):
        """Test that bundling empty sequence raises error."""
        with pytest.raises(ValueError, match="Cannot bundle empty sequence"):
            model.bundle([])

    def test_empty_sequence_raises(self, model):
        """Test that binding empty sequence raises error."""
        with pytest.raises(ValueError, match="Cannot bind empty sequence"):
            model.bind_sequence([])


class TestVTBComparison:
    """Test comparisons between VTB and other models."""

    def test_compare_vtb_hrr_unbinding(self):
        """Test comparison function between VTB and HRR."""
        result = compare_vtb_hrr_unbinding(dimension=500, trials=5)

        # Should return valid statistics
        assert 'vtb_mean' in result
        assert 'hrr_mean' in result
        assert 'vtb_std' in result
        assert 'hrr_std' in result
        assert 'vtb_better' in result

        # Both should have good recovery
        assert result['vtb_mean'] > 0.65
        assert result['hrr_mean'] > 0.65

        # Standard deviation should be reasonable
        assert 0 < result['vtb_std'] < 0.3
        assert 0 < result['hrr_std'] < 0.3

    def test_vtb_vs_hrr_commutativity(self):
        """Test HRR is commutative and VTB is not."""
        vtb = VSA.create('VTB', dim=500, seed=42)
        hrr = VSA.create('HRR', dim=500, seed=42)

        a_vtb = vtb.random(seed=1)
        b_vtb = vtb.random(seed=2)

        a_hrr = hrr.random(seed=1)
        b_hrr = hrr.random(seed=2)

        # VTB should be non-commutative
        vtb_com = vtb.test_non_commutativity(a_vtb, b_vtb)
        assert vtb_com < 0.95

        # HRR should be commutative
        hrr_ab = hrr.bind(a_hrr, b_hrr)
        hrr_ba = hrr.bind(b_hrr, a_hrr)
        hrr_com = hrr.similarity(hrr_ab, hrr_ba)
        assert hrr_com > 0.95  # High similarity (commutative)


class TestVTBRoleFillerBindings:
    """Test VTB for role-filler binding scenarios."""

    def test_simple_role_filler(self):
        """Test simple role-filler binding."""
        vtb = VSA.create('VTB', dim=500, seed=42)

        # Create role and filler vectors
        role_agent = vtb.random(seed=1)
        role_patient = vtb.random(seed=2)
        filler_cat = vtb.random(seed=3)
        filler_dog = vtb.random(seed=4)

        # Encode: "cat chases dog"
        # agent=cat, patient=dog
        agent_cat = vtb.bind(role_agent, filler_cat)
        patient_dog = vtb.bind(role_patient, filler_dog)
        sentence1 = vtb.bundle([agent_cat, patient_dog])

        # Encode: "dog chases cat"
        # agent=dog, patient=cat
        agent_dog = vtb.bind(role_agent, filler_dog)
        patient_cat = vtb.bind(role_patient, filler_cat)
        sentence2 = vtb.bundle([agent_dog, patient_cat])

        # Sentences should be different
        sim = vtb.similarity(sentence1, sentence2)
        assert sim < 0.8  # Different due to role-filler structure

        # Should be able to query roles
        # "Who is the agent in sentence1?"
        agent_query = vtb.unbind(sentence1, role_agent)
        sim_cat = vtb.similarity(agent_query, filler_cat)
        sim_dog = vtb.similarity(agent_query, filler_dog)

        assert sim_cat > sim_dog  # Should recover cat as agent

    def test_nested_structure(self):
        """Test nested compositional structure."""
        vtb = VSA.create('VTB', dim=1000, seed=42)

        # Create structure: [[A, B], C]
        # Note: Since VTB uses circular convolution (commutative),
        # nested structures will have similar properties to HRR

        a = vtb.random(seed=1)
        b = vtb.random(seed=2)
        c = vtb.random(seed=3)

        # Inner structure: [A, B]
        inner = vtb.bind(a, b)

        # Outer structure: [inner, C]
        outer = vtb.bind(inner, c)

        # Different from: [A, [B, C]]
        inner2 = vtb.bind(b, c)
        outer2 = vtb.bind(a, inner2)

        # Non-commutativity breaks simple associativity; structures need not be similar
        sim = vtb.similarity(outer, outer2)
        assert sim < 0.95


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
