"""Tests for BSDC Context-Dependent Thinning (CDT) binding mode.

CDT is an alternative binding operation for sparse binary codes that preserves
both structured and unstructured similarity, as described in Rachkovskij (2001).
"""

import numpy as np
import pytest

from holovec import VSA
from holovec.backends import get_available_backends, get_backend
from holovec.models.bsdc import BSDCModel

AVAILABLE_BACKENDS = get_available_backends()


class TestCDTBasics:
    """Basic tests for CDT binding mode."""

    def test_cdt_model_creation(self):
        """Test CDT model can be created."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        assert model.binding_mode == 'cdt'
        assert not model.is_self_inverse
        assert model.is_commutative

    def test_xor_model_creation(self):
        """Test XOR model (default) preserves existing behavior."""
        model = VSA.create('BSDC', dim=10000, seed=42)
        assert model.binding_mode == 'xor'
        assert model.is_self_inverse
        assert model.is_commutative

    def test_invalid_binding_mode_raises(self):
        """Test that invalid binding mode raises error."""
        with pytest.raises(ValueError, match="binding_mode must be"):
            BSDCModel(dimension=10000, binding_mode='invalid')

    def test_cdt_repr_includes_binding_mode(self):
        """Test that repr includes binding_mode."""
        model = VSA.create('BSDC', dim=1000, binding_mode='cdt')
        assert "binding_mode='cdt'" in repr(model)

        model_xor = VSA.create('BSDC', dim=1000)
        assert "binding_mode='xor'" in repr(model_xor)


class TestCDTUnstructuredSimilarity:
    """Tests for CDT preserving unstructured similarity.

    Unstructured similarity means the bound result remains similar to
    each of its components.
    """

    def test_cdt_result_similar_to_components(self):
        """Test that CDT result is similar to its components."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        a = model.random(seed=1)
        b = model.random(seed=2)
        bound = model.bind(a, b)

        sim_a = model.similarity(bound, a)
        sim_b = model.similarity(bound, b)

        # CDT should preserve similarity to components
        assert sim_a > 0.1, f"CDT result should be similar to component a, got {sim_a}"
        assert sim_b > 0.1, f"CDT result should be similar to component b, got {sim_b}"

    def test_cdt_three_components_similar(self):
        """Test that 3-component CDT result is similar to all components."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)

        bound = model.context_dependent_thinning([a, b, c])

        sim_a = model.similarity(bound, a)
        sim_b = model.similarity(bound, b)
        sim_c = model.similarity(bound, c)

        # All components should be similar to result
        assert sim_a > 0.1, f"3-CDT result should be similar to a, got {sim_a}"
        assert sim_b > 0.1, f"3-CDT result should be similar to b, got {sim_b}"
        assert sim_c > 0.1, f"3-CDT result should be similar to c, got {sim_c}"


class TestCDTStructuredSimilarity:
    """Tests for CDT preserving structured similarity.

    Structured similarity means similar inputs produce similar outputs.
    """

    def test_identical_inputs_identical_outputs(self):
        """Test that identical inputs produce identical CDT outputs."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        a = model.random(seed=1)
        b = model.random(seed=2)

        bound1 = model.bind(a, b)
        bound2 = model.bind(a, b)

        sim = model.similarity(bound1, bound2)
        assert sim > 0.99, f"Identical inputs should produce identical outputs, got {sim}"

    def test_similar_inputs_similar_outputs(self):
        """Test that similar inputs produce similar CDT outputs."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        a = model.random(seed=1)
        b = model.random(seed=2)

        # Create a' that is similar to a (shares most bits)
        a_np = model.backend.to_numpy(a)
        a_similar_np = a_np.copy()
        # Flip only 10% of the 1-bits
        ones_idx = np.where(a_np == 1)[0]
        flip_count = max(1, len(ones_idx) // 10)
        flip_idx = np.random.choice(ones_idx, size=flip_count, replace=False)
        a_similar_np[flip_idx] = 0
        a_similar = model.backend.from_numpy(a_similar_np)

        bound1 = model.bind(a, b)
        bound2 = model.bind(a_similar, b)

        sim = model.similarity(bound1, bound2)
        # Similar inputs should produce similar outputs
        assert sim > 0.5, f"Similar inputs should produce similar outputs, got {sim}"


class TestCDTSparsity:
    """Tests for CDT maintaining target sparsity."""

    def test_cdt_maintains_sparsity_2_components(self):
        """Test that 2-component CDT maintains target sparsity."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        a = model.random(seed=1)
        b = model.random(seed=2)

        bound = model.bind(a, b)
        density = float(np.sum(model.backend.to_numpy(bound))) / model.dimension

        # Should be close to target sparsity (within 2x)
        assert 0.005 < density < 0.03, f"CDT density {density} outside range [0.005, 0.03]"

    def test_cdt_maintains_sparsity_5_components(self):
        """Test that 5-component CDT maintains target sparsity."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        components = [model.random(seed=i) for i in range(5)]

        bound = model.context_dependent_thinning(components)
        density = float(np.sum(model.backend.to_numpy(bound))) / model.dimension

        # Should be close to target sparsity (within 2x)
        assert 0.005 < density < 0.03, f"5-CDT density {density} outside range"


class TestCDTvsXOR:
    """Compare CDT and XOR binding properties."""

    def test_xor_exact_recovery(self):
        """Test that XOR mode has exact unbinding."""
        model = VSA.create('BSDC', dim=10000, binding_mode='xor', seed=42)
        a = model.random(seed=1)
        b = model.random(seed=2)

        bound = model.bind(a, b)
        recovered = model.unbind(bound, b)

        sim = model.similarity(a, recovered)
        assert sim > 0.99, f"XOR should have exact recovery, got {sim}"

    def test_cdt_no_exact_recovery(self):
        """Test that CDT mode returns bound vector for unbind."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        a = model.random(seed=1)
        b = model.random(seed=2)

        bound = model.bind(a, b)
        result = model.unbind(bound, b)

        # CDT unbind returns the bound vector itself
        sim_to_bound = model.similarity(result, bound)
        assert sim_to_bound > 0.99, "CDT unbind should return the bound vector"

    def test_different_binding_modes_different_properties(self):
        """Test that XOR and CDT have different properties."""
        xor_model = VSA.create('BSDC', dim=10000, binding_mode='xor', seed=42)
        cdt_model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)

        assert xor_model.is_self_inverse
        assert not cdt_model.is_self_inverse

        assert xor_model.is_exact_inverse
        assert not cdt_model.is_exact_inverse


class TestCDTEdgeCases:
    """Edge case tests for CDT."""

    def test_cdt_single_component(self):
        """Test CDT with single component returns that component."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        a = model.random(seed=1)

        result = model.context_dependent_thinning([a])
        sim = model.similarity(result, a)
        assert sim > 0.99, "Single component CDT should return that component"

    def test_cdt_empty_raises(self):
        """Test CDT with empty list raises error."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)

        with pytest.raises(ValueError, match="Cannot bind empty"):
            model.context_dependent_thinning([])

    def test_cdt_permutations_deterministic(self):
        """Test that CDT permutations are deterministic with seed."""
        model1 = VSA.create('BSDC', dim=1000, binding_mode='cdt', seed=42)
        model2 = VSA.create('BSDC', dim=1000, binding_mode='cdt', seed=42)

        # Same seed should give same permutations
        assert np.array_equal(
            model1._cdt_permutations[0],
            model2._cdt_permutations[0]
        )

    def test_cdt_many_components(self):
        """Test CDT with many components."""
        model = VSA.create('BSDC', dim=10000, binding_mode='cdt', seed=42)
        components = [model.random(seed=i) for i in range(10)]

        bound = model.context_dependent_thinning(components)

        # Result should still be similar to components
        sims = [model.similarity(bound, c) for c in components]
        avg_sim = sum(sims) / len(sims)
        assert avg_sim > 0.05, f"10-component CDT avg similarity {avg_sim} too low"

        # And maintain some sparsity
        density = float(np.sum(model.backend.to_numpy(bound))) / model.dimension
        assert density < 0.1, f"10-component CDT too dense: {density}"


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_cdt_binding_runs_on_available_backends(backend_name):
    """CDT binding should remain backend-compatible on installed backends."""
    backend = get_backend(backend_name)
    model = BSDCModel(dimension=2048, backend=backend, seed=42, binding_mode='cdt')
    a = model.random(seed=1)
    b = model.random(seed=2)

    bound = model.bind(a, b)
    bound_np = backend.to_numpy(bound)

    assert bound_np.shape == (2048,)
    assert np.all(np.isin(bound_np, [0, 1]))
