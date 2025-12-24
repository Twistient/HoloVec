"""Tests for the analysis module.

This module tests capacity analysis tools including:
- theoretical_capacity()
- recommend_dimension()
- compare_models()
- empirical_capacity_test()
"""

import pytest

from holovec import VSA
from holovec.analysis import (
    compare_models,
    empirical_capacity_test,
    recommend_dimension,
    theoretical_capacity,
)


class TestTheoreticalCapacity:
    """Tests for theoretical_capacity function."""

    def test_fhrr_capacity(self):
        """Test FHRR capacity metrics."""
        capacity = theoretical_capacity('FHRR', dim=1000)

        # FHRR should have ~D/3 items per binding
        assert 300 < capacity['items_per_binding'] < 400
        assert capacity['recovery_accuracy'] == 1.0
        assert 'description' in capacity
        assert capacity['binding_depth'] >= 4

    def test_map_capacity(self):
        """Test MAP capacity metrics."""
        capacity = theoretical_capacity('MAP', dim=1000)

        # MAP has lower capacity than FHRR
        assert 50 < capacity['items_per_binding'] < 100
        assert capacity['recovery_accuracy'] == 1.0

    def test_hrr_capacity(self):
        """Test HRR capacity metrics."""
        capacity = theoretical_capacity('HRR', dim=1000)

        # HRR has approximate inverse
        assert capacity['recovery_accuracy'] < 1.0
        assert capacity['recovery_accuracy'] >= 0.7

    def test_bsc_capacity(self):
        """Test BSC capacity metrics."""
        capacity = theoretical_capacity('BSC', dim=1000)

        assert 'items_per_binding' in capacity
        assert capacity['recovery_accuracy'] == 1.0  # Self-inverse

    def test_bundle_capacity_scales_with_sqrt(self):
        """Test that bundle capacity scales with sqrt(dim)."""
        cap_1000 = theoretical_capacity('FHRR', dim=1000)
        cap_4000 = theoretical_capacity('FHRR', dim=4000)

        # Bundle capacity should roughly double when dim quadruples
        ratio = cap_4000['bundle_capacity'] / cap_1000['bundle_capacity']
        assert 1.8 < ratio < 2.2

    def test_unknown_model_raises(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            theoretical_capacity('UNKNOWN_MODEL', dim=1000)

    def test_case_insensitive(self):
        """Test that model name is case-insensitive."""
        cap_lower = theoretical_capacity('fhrr', dim=1000)
        cap_upper = theoretical_capacity('FHRR', dim=1000)
        cap_mixed = theoretical_capacity('Fhrr', dim=1000)

        assert cap_lower == cap_upper == cap_mixed

    def test_all_models_have_capacity(self):
        """Test that all known models return valid capacity."""
        models = ['MAP', 'FHRR', 'HRR', 'GHRR', 'VTB', 'BSC', 'BSDC', 'BSDC_SEG']
        for model in models:
            capacity = theoretical_capacity(model, dim=1000)
            assert 'items_per_binding' in capacity
            assert 'bundle_capacity' in capacity
            assert 'recovery_accuracy' in capacity
            assert 'binding_depth' in capacity
            assert 'description' in capacity


class TestRecommendDimension:
    """Tests for recommend_dimension function."""

    def test_basic_recommendation(self):
        """Test basic dimension recommendation."""
        dim = recommend_dimension('FHRR', n_items=100)

        # Should return a reasonable power of 2
        assert dim >= 256
        assert dim in [256, 512, 1024, 2048, 4096, 8192, 10000, 16384] or dim % 1000 == 0

    def test_more_items_needs_more_dim(self):
        """Test that more items requires higher dimension."""
        dim_small = recommend_dimension('MAP', n_items=10)
        dim_large = recommend_dimension('MAP', n_items=1000)

        assert dim_large > dim_small

    def test_more_bindings_needs_more_dim(self):
        """Test that more bindings requires higher dimension."""
        dim_1 = recommend_dimension('FHRR', n_items=50, n_bindings=1)
        dim_3 = recommend_dimension('FHRR', n_items=50, n_bindings=3)

        assert dim_3 >= dim_1

    def test_higher_accuracy_needs_more_dim(self):
        """Test that higher accuracy requires higher dimension."""
        dim_low = recommend_dimension('MAP', n_items=100, target_accuracy=0.90)
        dim_high = recommend_dimension('MAP', n_items=100, target_accuracy=0.99)

        assert dim_high >= dim_low

    def test_map_needs_more_than_fhrr(self):
        """Test that MAP requires more dimension than FHRR."""
        dim_fhrr = recommend_dimension('FHRR', n_items=100)
        dim_map = recommend_dimension('MAP', n_items=100)

        # MAP has lower capacity per dimension
        assert dim_map >= dim_fhrr

    def test_unknown_model_raises(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            recommend_dimension('UNKNOWN', n_items=100)


class TestCompareModels:
    """Tests for compare_models function."""

    def test_returns_all_models(self):
        """Test that comparison includes all models."""
        comparison = compare_models(dim=1000)

        assert 'FHRR' in comparison
        assert 'MAP' in comparison
        assert 'HRR' in comparison
        assert 'GHRR' in comparison

    def test_fhrr_has_highest_capacity(self):
        """Test that FHRR has highest items per binding."""
        comparison = compare_models(dim=1000)

        fhrr_capacity = comparison['FHRR']['items_per_binding']
        map_capacity = comparison['MAP']['items_per_binding']
        bsc_capacity = comparison['BSC']['items_per_binding']

        assert fhrr_capacity > map_capacity
        assert fhrr_capacity > bsc_capacity

    def test_all_have_required_keys(self):
        """Test that all models have required capacity keys."""
        comparison = compare_models(dim=1000)

        required_keys = ['items_per_binding', 'bundle_capacity', 'recovery_accuracy']
        for model, metrics in comparison.items():
            for key in required_keys:
                assert key in metrics, f"{model} missing {key}"


class TestEmpiricalCapacityTest:
    """Tests for empirical_capacity_test function."""

    @pytest.fixture
    def fhrr_model(self):
        """Create FHRR model for testing."""
        return VSA.create('FHRR', dim=512, seed=42)

    @pytest.fixture
    def map_model(self):
        """Create MAP model for testing."""
        return VSA.create('MAP', dim=512, seed=42)

    def test_returns_expected_keys(self, fhrr_model):
        """Test that result has expected structure."""
        result = empirical_capacity_test(
            fhrr_model,
            codebook_sizes=[10, 25],
            n_trials=5,
        )

        assert 'max_codebook_size' in result
        assert 'accuracies' in result
        assert 'dimension' in result
        assert 'model_name' in result

    def test_small_codebook_high_accuracy(self, fhrr_model):
        """Test that small codebooks have high accuracy."""
        result = empirical_capacity_test(
            fhrr_model,
            codebook_sizes=[10],
            n_trials=20,
        )

        # Small codebook should have very high accuracy
        assert result['accuracies'][10] > 0.9

    def test_accuracy_decreases_with_size(self, map_model):
        """Test that accuracy tends to decrease with codebook size."""
        result = empirical_capacity_test(
            map_model,
            codebook_sizes=[10, 100, 200],
            n_trials=20,
        )

        # Generally accuracy should decrease with size
        # (may not be strictly monotonic due to randomness)
        assert result['accuracies'][10] >= result['accuracies'][200] - 0.1

    def test_dimension_recorded(self, fhrr_model):
        """Test that model dimension is recorded."""
        result = empirical_capacity_test(
            fhrr_model,
            codebook_sizes=[10],
            n_trials=5,
        )

        assert result['dimension'] == 512

    def test_model_name_recorded(self, fhrr_model):
        """Test that model name is recorded."""
        result = empirical_capacity_test(
            fhrr_model,
            codebook_sizes=[10],
            n_trials=5,
        )

        assert result['model_name'] == 'FHRR'


class TestIntegration:
    """Integration tests for analysis module."""

    def test_recommend_then_verify(self):
        """Test that recommended dimension works empirically."""
        # Get recommendation
        dim = recommend_dimension('FHRR', n_items=50, n_bindings=1)

        # Create model with recommended dimension
        model = VSA.create('FHRR', dim=dim, seed=42)

        # Verify empirically
        result = empirical_capacity_test(
            model,
            codebook_sizes=[50],
            n_trials=20,
        )

        # Should achieve high accuracy with recommended dimension
        assert result['accuracies'][50] > 0.8

    def test_theoretical_vs_empirical(self):
        """Test that theoretical and empirical roughly agree."""
        dim = 1000
        model = VSA.create('FHRR', dim=dim, seed=42)

        # Theoretical capacity
        theory = theoretical_capacity('FHRR', dim=dim)

        # Empirical test
        empirical = empirical_capacity_test(
            model,
            codebook_sizes=[50, 100, 200],
            n_trials=20,
        )

        # At items_per_binding / 3 (conservative), should have high accuracy
        conservative_size = int(theory['items_per_binding'] / 3)
        if conservative_size in empirical['accuracies']:
            assert empirical['accuracies'][conservative_size] > 0.9
