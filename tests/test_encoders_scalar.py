"""
Comprehensive tests for scalar encoders.

Tests cover:
- Initialization and parameter validation
- Encoding and decoding functionality
- Property-based tests for mathematical properties
- Backend consistency (NumPy, PyTorch, JAX)
- Model compatibility
- Edge cases
"""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck

from holovec import VSA
from holovec.encoders import (
    FractionalPowerEncoder,
    ThermometerEncoder,
    LevelEncoder,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=["FHRR", "HRR"])
def fpe_compatible_model(request):
    """Create models compatible with FractionalPowerEncoder."""
    return VSA.create(request.param, dim=1000, backend="numpy")


@pytest.fixture
def any_model():
    """Create a model for testing encoders that work with any model."""
    return VSA.create("FHRR", dim=1000, backend="numpy")


# ============================================================================
# FractionalPowerEncoder Tests
# ============================================================================

class TestFractionalPowerEncoderInitialization:
    """Test FPE initialization and parameter validation."""

    def test_init_with_fhrr(self):
        """FPE should initialize successfully with FHRR."""
        model = VSA.create("FHRR", dim=1000)
        encoder = FractionalPowerEncoder(model, min_val=0.0, max_val=100.0)

        assert encoder.min_val == 0.0
        assert encoder.max_val == 100.0
        assert encoder.bandwidth == 1.0  # Default
        assert encoder.dimension == 1000

    def test_init_with_hrr(self):
        """FPE should initialize successfully with HRR."""
        model = VSA.create("HRR", dim=1000)
        encoder = FractionalPowerEncoder(model, min_val=-10.0, max_val=10.0)

        assert encoder.min_val == -10.0
        assert encoder.max_val == 10.0

    def test_init_with_incompatible_model(self):
        """FPE should reject incompatible models."""
        model = VSA.create("MAP", dim=1000)

        with pytest.raises(ValueError, match="not compatible"):
            FractionalPowerEncoder(model, min_val=0.0, max_val=100.0)

    def test_init_with_invalid_range(self):
        """FPE should reject invalid min/max values."""
        model = VSA.create("FHRR", dim=1000)

        # min_val >= max_val should fail
        with pytest.raises(ValueError, match="min_val must be less than max_val"):
            FractionalPowerEncoder(model, min_val=100.0, max_val=0.0)

        with pytest.raises(ValueError, match="min_val must be less than max_val"):
            FractionalPowerEncoder(model, min_val=50.0, max_val=50.0)

    def test_custom_bandwidth(self):
        """FPE should accept custom bandwidth parameter."""
        model = VSA.create("FHRR", dim=1000)
        encoder = FractionalPowerEncoder(
            model, min_val=0.0, max_val=100.0, bandwidth=0.01
        )

        assert encoder.bandwidth == 0.01

    def test_reproducibility_with_seed(self):
        """FPE with same seed should produce identical base phasors."""
        model = VSA.create("FHRR", dim=1000)

        encoder1 = FractionalPowerEncoder(
            model, min_val=0.0, max_val=100.0, seed=42
        )
        encoder2 = FractionalPowerEncoder(
            model, min_val=0.0, max_val=100.0, seed=42
        )

        # Base phasors should be identical
        diff = model.backend.to_numpy(
            model.backend.subtract(encoder1.base_phasor, encoder2.base_phasor)
        )
        assert np.allclose(diff, 0.0, atol=1e-10)


class TestFractionalPowerEncoderEncoding:
    """Test FPE encoding functionality."""

    def test_encode_returns_correct_shape(self, fpe_compatible_model):
        """Encoded vectors should have correct dimensionality."""
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        encoded = encoder.encode(50.0)
        assert encoded.shape == (1000,)

    def test_encode_within_range(self, fpe_compatible_model):
        """Encoding values within range should work."""
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        # Should not raise
        encoder.encode(0.0)
        encoder.encode(50.0)
        encoder.encode(100.0)

    def test_encode_clips_out_of_range(self, fpe_compatible_model):
        """Values outside range should be clipped."""
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        # These should clip to valid range (no error)
        encoded_below = encoder.encode(-10.0)
        encoded_min = encoder.encode(0.0)

        # Should be very similar (both clipped to min)
        similarity = fpe_compatible_model.similarity(encoded_below, encoded_min)
        assert float(fpe_compatible_model.backend.to_numpy(similarity)) > 0.99

    def test_encode_different_values_differ(self, fpe_compatible_model):
        """Different values should produce different encodings."""
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        encoded_25 = encoder.encode(25.0)
        encoded_75 = encoder.encode(75.0)

        similarity = fpe_compatible_model.similarity(encoded_25, encoded_75)
        # Should be somewhat different (not perfectly similar)
        assert float(fpe_compatible_model.backend.to_numpy(similarity)) < 0.9


class TestFractionalPowerEncoderDecoding:
    """Test FPE decoding functionality."""

    def test_decode_encode_roundtrip(self, fpe_compatible_model):
        """Decode(Encode(x)) should approximate x."""
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        original_values = [10.0, 25.0, 50.0, 75.0, 90.0]

        for original in original_values:
            encoded = encoder.encode(original)
            decoded = encoder.decode(encoded, resolution=500, max_iterations=50)

            # Should be close (within a few percent)
            error = abs(decoded - original)
            relative_error = error / 100.0  # Normalize by range
            assert relative_error < 0.05, f"Decode error too large: {error}"

    def test_is_reversible_property(self, fpe_compatible_model):
        """FPE should report as reversible."""
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        assert encoder.is_reversible is True


class TestFractionalPowerEncoderProperties:
    """Property-based tests for FPE mathematical properties."""

    @given(
        value1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        value2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_similarity_monotonicity(self, fpe_compatible_model, value1, value2):
        """
        Similarity should decrease with distance between values.

        Property: |x₁ - x₂| < |x₁ - x₃| → sim(x₁, x₂) > sim(x₁, x₃)
        """
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        # Skip if values too close
        if abs(value1 - value2) < 1.0:
            return

        # Encode three values: value1, value2, and midpoint
        midpoint = (value1 + value2) / 2.0

        encoded_1 = encoder.encode(value1)
        encoded_mid = encoder.encode(midpoint)
        encoded_2 = encoder.encode(value2)

        # Midpoint should be more similar to value1 than value2 is
        sim_to_mid = fpe_compatible_model.similarity(encoded_1, encoded_mid)
        sim_to_far = fpe_compatible_model.similarity(encoded_1, encoded_2)

        sim_to_mid_val = float(fpe_compatible_model.backend.to_numpy(sim_to_mid))
        sim_to_far_val = float(fpe_compatible_model.backend.to_numpy(sim_to_far))

        # This should generally hold (with some tolerance for noise)
        assert sim_to_mid_val >= sim_to_far_val - 0.1

    @given(value=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_self_similarity(self, fpe_compatible_model, value):
        """
        Encoding of same value should be perfectly similar to itself.

        Property: sim(encode(x), encode(x)) = 1.0
        """
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        encoded = encoder.encode(value)
        similarity = fpe_compatible_model.similarity(encoded, encoded)

        sim_val = float(fpe_compatible_model.backend.to_numpy(similarity))
        assert abs(sim_val - 1.0) < 1e-6

    @given(
        value1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        value2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_symmetry(self, fpe_compatible_model, value1, value2):
        """
        Similarity should be symmetric.

        Property: sim(encode(x₁), encode(x₂)) = sim(encode(x₂), encode(x₁))
        """
        encoder = FractionalPowerEncoder(
            fpe_compatible_model, min_val=0.0, max_val=100.0
        )

        encoded_1 = encoder.encode(value1)
        encoded_2 = encoder.encode(value2)

        sim_12 = fpe_compatible_model.similarity(encoded_1, encoded_2)
        sim_21 = fpe_compatible_model.similarity(encoded_2, encoded_1)

        sim_12_val = float(fpe_compatible_model.backend.to_numpy(sim_12))
        sim_21_val = float(fpe_compatible_model.backend.to_numpy(sim_21))

        assert abs(sim_12_val - sim_21_val) < 1e-6


# ============================================================================
# ThermometerEncoder Tests
# ============================================================================

class TestThermometerEncoder:
    """Test ThermometerEncoder functionality."""

    def test_init_with_any_model(self, any_model):
        """Thermometer encoder should work with any model."""
        encoder = ThermometerEncoder(any_model, min_val=0.0, max_val=100.0)

        assert encoder.n_bins == 100  # Default
        assert encoder.min_val == 0.0
        assert encoder.max_val == 100.0

    def test_init_with_custom_bins(self, any_model):
        """Should accept custom number of bins."""
        encoder = ThermometerEncoder(
            any_model, min_val=0.0, max_val=100.0, n_bins=50
        )

        assert encoder.n_bins == 50

    def test_init_invalid_bins(self, any_model):
        """Should reject invalid number of bins."""
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            ThermometerEncoder(any_model, min_val=0.0, max_val=100.0, n_bins=1)

    def test_encode_returns_correct_shape(self, any_model):
        """Encoded vectors should have correct shape."""
        encoder = ThermometerEncoder(any_model, min_val=0.0, max_val=100.0)

        encoded = encoder.encode(50.0)
        assert encoded.shape == (1000,)

    def test_encode_monotonic_similarity(self, any_model):
        """Higher values should activate more bins."""
        encoder = ThermometerEncoder(
            any_model, min_val=0.0, max_val=100.0, n_bins=10
        )

        # Encode values that span the range
        encoded_10 = encoder.encode(10.0)
        encoded_50 = encoder.encode(50.0)
        encoded_90 = encoder.encode(90.0)

        # 10 and 50 should be more similar than 10 and 90
        # (both include lower bins)
        sim_10_50 = any_model.similarity(encoded_10, encoded_50)
        sim_10_90 = any_model.similarity(encoded_10, encoded_90)

        sim_10_50_val = float(any_model.backend.to_numpy(sim_10_50))
        sim_10_90_val = float(any_model.backend.to_numpy(sim_10_90))

        assert sim_10_50_val > sim_10_90_val

    def test_decode_not_supported(self, any_model):
        """Thermometer encoder should not support decoding."""
        encoder = ThermometerEncoder(any_model, min_val=0.0, max_val=100.0)

        assert encoder.is_reversible is False

        encoded = encoder.encode(50.0)
        with pytest.raises(NotImplementedError):
            encoder.decode(encoded)

    def test_compatible_models(self, any_model):
        """Should report compatibility with all models."""
        encoder = ThermometerEncoder(any_model, min_val=0.0, max_val=100.0)

        all_models = ["MAP", "FHRR", "HRR", "BSC", "GHRR", "VTB", "BSDC"]
        for model_name in all_models:
            assert model_name in encoder.compatible_models


# ============================================================================
# LevelEncoder Tests
# ============================================================================

class TestLevelEncoder:
    """Test LevelEncoder functionality."""

    def test_init_with_discrete_levels(self, any_model):
        """Level encoder should initialize with discrete levels."""
        encoder = LevelEncoder(
            any_model, min_val=0.0, max_val=6.0, n_levels=7
        )

        assert encoder.n_levels == 7
        assert encoder.min_val == 0.0
        assert encoder.max_val == 6.0

    def test_init_invalid_levels(self, any_model):
        """Should reject invalid number of levels."""
        with pytest.raises(ValueError, match="n_levels must be >= 2"):
            LevelEncoder(any_model, min_val=0.0, max_val=10.0, n_levels=1)

    def test_encode_exact_levels(self, any_model):
        """Encoding exact level values should be deterministic."""
        encoder = LevelEncoder(
            any_model, min_val=0.0, max_val=10.0, n_levels=11, seed=42
        )

        # Encode same level multiple times
        encoded_5_first = encoder.encode(5.0)
        encoded_5_second = encoder.encode(5.0)

        # Should be identical
        diff = any_model.backend.to_numpy(
            any_model.backend.subtract(encoded_5_first, encoded_5_second)
        )
        assert np.allclose(diff, 0.0, atol=1e-10)

    def test_encode_rounds_to_nearest_level(self, any_model):
        """Values between levels should round to nearest."""
        encoder = LevelEncoder(
            any_model, min_val=0.0, max_val=10.0, n_levels=11, seed=42
        )

        # 4.6 should round to 5
        encoded_4_6 = encoder.encode(4.6)
        encoded_5 = encoder.encode(5.0)

        similarity = any_model.similarity(encoded_4_6, encoded_5)
        # Should be very similar (same level)
        assert float(any_model.backend.to_numpy(similarity)) > 0.99

    def test_decode_recovers_level(self, any_model):
        """Decoding should recover exact level."""
        encoder = LevelEncoder(
            any_model, min_val=0.0, max_val=10.0, n_levels=11, seed=42
        )

        levels = [0.0, 2.0, 5.0, 7.0, 10.0]

        for level in levels:
            encoded = encoder.encode(level)
            decoded = encoder.decode(encoded)

            # Should recover exact level
            assert abs(decoded - level) < 0.01

    def test_is_reversible(self, any_model):
        """Level encoder should be reversible."""
        encoder = LevelEncoder(
            any_model, min_val=0.0, max_val=10.0, n_levels=11
        )

        assert encoder.is_reversible is True

    def test_compatible_models(self, any_model):
        """Should work with all models."""
        encoder = LevelEncoder(
            any_model, min_val=0.0, max_val=10.0, n_levels=11
        )

        all_models = ["MAP", "FHRR", "HRR", "BSC", "GHRR", "VTB", "BSDC"]
        for model_name in all_models:
            assert model_name in encoder.compatible_models


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEncoderEdgeCases:
    """Test edge cases for all encoders."""

    def test_fpe_with_zero_bandwidth(self):
        """FPE with zero bandwidth should still work."""
        model = VSA.create("FHRR", dim=1000)
        encoder = FractionalPowerEncoder(
            model, min_val=0.0, max_val=100.0, bandwidth=0.0
        )

        # With bandwidth=0, all values should encode to same vector
        encoded_25 = encoder.encode(25.0)
        encoded_75 = encoder.encode(75.0)

        similarity = model.similarity(encoded_25, encoded_75)
        # Should be very similar (bandwidth=0 → constant encoding)
        assert float(model.backend.to_numpy(similarity)) > 0.95

    def test_thermometer_with_two_bins(self):
        """Thermometer with minimum bins should work."""
        model = VSA.create("FHRR", dim=1000)
        encoder = ThermometerEncoder(
            model, min_val=0.0, max_val=100.0, n_bins=2
        )

        # Should still encode without error
        encoded_low = encoder.encode(25.0)
        encoded_high = encoder.encode(75.0)

        # Should be different
        similarity = model.similarity(encoded_low, encoded_high)
        assert float(model.backend.to_numpy(similarity)) < 0.95

    def test_level_encoder_with_two_levels(self):
        """Level encoder with minimum levels (binary) should work."""
        model = VSA.create("FHRR", dim=1000)
        encoder = LevelEncoder(
            model, min_val=0.0, max_val=1.0, n_levels=2, seed=42
        )

        encoded_0 = encoder.encode(0.0)
        encoded_1 = encoder.encode(1.0)

        # Should be very different (quasi-orthogonal)
        similarity = model.similarity(encoded_0, encoded_1)
        assert abs(float(model.backend.to_numpy(similarity))) < 0.3


# ============================================================================
# Batch Encoding Tests
# ============================================================================

class TestBatchEncoding:
    """Test batch encoding functionality."""

    def test_fpe_batch_encode(self):
        """FPE should support batch encoding."""
        model = VSA.create("FHRR", dim=1000)
        encoder = FractionalPowerEncoder(model, min_val=0.0, max_val=100.0)

        values = [10.0, 25.0, 50.0, 75.0, 90.0]
        batch_encoded = encoder.encode_batch(values)

        assert len(batch_encoded) == 5
        for encoded in batch_encoded:
            assert encoded.shape == (1000,)

    def test_thermometer_batch_encode(self):
        """Thermometer should support batch encoding."""
        model = VSA.create("FHRR", dim=1000)
        encoder = ThermometerEncoder(model, min_val=0.0, max_val=100.0)

        values = [10.0, 50.0, 90.0]
        batch_encoded = encoder.encode_batch(values)

        assert len(batch_encoded) == 3

    def test_level_batch_encode(self):
        """Level encoder should support batch encoding."""
        model = VSA.create("FHRR", dim=1000)
        encoder = LevelEncoder(model, min_val=0.0, max_val=10.0, n_levels=11)

        values = [0.0, 5.0, 10.0]
        batch_encoded = encoder.encode_batch(values)

        assert len(batch_encoded) == 3


# ============================================================================
# Repr Tests
# ============================================================================

class TestEncoderRepresentations:
    """Test string representations."""

    def test_fpe_repr(self):
        """FPE should have informative repr."""
        model = VSA.create("FHRR", dim=1000)
        encoder = FractionalPowerEncoder(model, min_val=0.0, max_val=100.0)

        repr_str = repr(encoder)
        assert "FractionalPowerEncoder" in repr_str
        assert "FHRR" in repr_str
        assert "0.0" in repr_str
        assert "100.0" in repr_str

    def test_thermometer_repr(self):
        """Thermometer should have informative repr."""
        model = VSA.create("FHRR", dim=1000)
        encoder = ThermometerEncoder(model, min_val=0.0, max_val=100.0, n_bins=50)

        repr_str = repr(encoder)
        assert "ThermometerEncoder" in repr_str
        assert "n_bins=50" in repr_str

    def test_level_repr(self):
        """Level encoder should have informative repr."""
        model = VSA.create("FHRR", dim=1000)
        encoder = LevelEncoder(model, min_val=0.0, max_val=10.0, n_levels=11)

        repr_str = repr(encoder)
        assert "LevelEncoder" in repr_str
        assert "n_levels=11" in repr_str
