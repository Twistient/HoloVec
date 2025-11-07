"""
Tests for structured data encoders.

Tests the VectorEncoder and future structured encoders.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings

from holovec import VSA
from holovec.encoders import (
    VectorEncoder,
    FractionalPowerEncoder,
    ThermometerEncoder,
    LevelEncoder,
)


class TestVectorEncoderInitialization:
    """Test initialization of VectorEncoder."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)
        encoder = VectorEncoder(model, scalar_encoder=scalar_enc, n_dimensions=10)

        assert encoder.model == model
        assert encoder.backend == model.backend
        assert encoder.dimension == 1000
        assert encoder.scalar_encoder == scalar_enc
        assert encoder.n_dimensions == 10
        assert encoder.normalize_input is False
        assert len(encoder.dim_vectors) == 10

    def test_init_with_normalization(self):
        """Test initialization with input normalization enabled."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=-1, max_val=1, seed=42)
        encoder = VectorEncoder(
            model, scalar_encoder=scalar_enc, n_dimensions=5, normalize_input=True
        )

        assert encoder.normalize_input is True

    def test_init_with_seed(self):
        """Test initialization with seed for reproducibility."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)

        encoder1 = VectorEncoder(model, scalar_enc, n_dimensions=5, seed=123)
        encoder2 = VectorEncoder(model, scalar_enc, n_dimensions=5, seed=123)

        # Same seed should produce same dimension vectors
        for i in range(5):
            sim = float(model.similarity(encoder1.dim_vectors[i], encoder2.dim_vectors[i]))
            assert sim > 0.99  # Should be nearly identical

    def test_init_invalid_n_dimensions_fails(self):
        """Test that invalid n_dimensions raises error."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)

        with pytest.raises(ValueError, match="n_dimensions must be >= 1"):
            VectorEncoder(model, scalar_enc, n_dimensions=0)

        with pytest.raises(ValueError, match="n_dimensions must be >= 1"):
            VectorEncoder(model, scalar_enc, n_dimensions=-5)

    def test_init_wrong_scalar_encoder_type_fails(self):
        """Test that non-ScalarEncoder raises TypeError."""
        model = VSA.create('FHRR', dim=1000, seed=42)

        with pytest.raises(TypeError, match="must be a ScalarEncoder"):
            VectorEncoder(model, scalar_encoder="not_an_encoder", n_dimensions=5)

    def test_init_mismatched_model_fails(self):
        """Test that scalar_encoder with different model fails."""
        model1 = VSA.create('FHRR', dim=1000, seed=42)
        model2 = VSA.create('FHRR', dim=1000, seed=43)
        scalar_enc = FractionalPowerEncoder(model2, min_val=0, max_val=1, seed=42)

        with pytest.raises(ValueError, match="must use the same VSA model"):
            VectorEncoder(model1, scalar_encoder=scalar_enc, n_dimensions=5)


class TestVectorEncoderEncoding:
    """Test encoding functionality."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('FHRR', dim=2000, seed=42)

    @pytest.fixture
    def scalar_encoder(self, model):
        """Create test scalar encoder."""
        return FractionalPowerEncoder(model, min_val=0, max_val=10, seed=42)

    @pytest.fixture
    def encoder(self, model, scalar_encoder):
        """Create test vector encoder."""
        return VectorEncoder(model, scalar_encoder, n_dimensions=5, seed=42)

    def test_encode_simple_vector(self, encoder):
        """Test encoding a simple vector."""
        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hv = encoder.encode(vector)

        assert hv.shape == (encoder.dimension,)

    def test_encode_zero_vector(self, encoder):
        """Test encoding a zero vector."""
        vector = np.zeros(5)
        hv = encoder.encode(vector)

        assert hv.shape == (encoder.dimension,)

    def test_encode_wrong_shape_fails(self, encoder):
        """Test that wrong shape raises error."""
        vector_wrong = np.array([1.0, 2.0, 3.0])  # Should be 5D

        with pytest.raises(ValueError, match="Expected vector of shape"):
            encoder.encode(vector_wrong)

    def test_encode_with_normalization(self, model, scalar_encoder):
        """Test that normalization works correctly."""
        encoder_norm = VectorEncoder(
            model, scalar_encoder, n_dimensions=3, normalize_input=True
        )

        vector = np.array([3.0, 4.0, 0.0])  # Norm = 5.0
        hv = encoder_norm.encode(vector)

        # Should encode normalized version [0.6, 0.8, 0.0]
        assert hv.shape == (encoder_norm.dimension,)

    def test_different_vectors_differ(self, encoder):
        """Test that different vectors produce different encodings."""
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        hv1 = encoder.encode(v1)
        hv2 = encoder.encode(v2)

        similarity = float(encoder.model.similarity(hv1, hv2))
        assert similarity < 0.7  # Should be different

    def test_encode_accepts_list(self, encoder):
        """Test that encoding works with Python lists."""
        vector_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        hv = encoder.encode(vector_list)

        assert hv.shape == (encoder.dimension,)


class TestVectorEncoderSimilarity:
    """Test similarity properties of encodings."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('FHRR', dim=5000, seed=42)  # Higher dim for better properties

    @pytest.fixture
    def scalar_encoder(self, model):
        """Create test scalar encoder."""
        return FractionalPowerEncoder(model, min_val=0, max_val=10, seed=42)

    @pytest.fixture
    def encoder(self, model, scalar_encoder):
        """Create test vector encoder."""
        return VectorEncoder(model, scalar_encoder, n_dimensions=10, seed=42)

    def test_similar_vectors_high_similarity(self, encoder):
        """Test that similar vectors have high similarity."""
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        v2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.0])  # Small perturbation

        hv1 = encoder.encode(v1)
        hv2 = encoder.encode(v2)

        similarity = float(encoder.model.similarity(hv1, hv2))
        assert similarity > 0.7  # Should be similar

    def test_identical_vectors_perfect_similarity(self, encoder):
        """Test that identical vectors have similarity ~1.0."""
        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        hv1 = encoder.encode(vector)
        hv2 = encoder.encode(vector)

        similarity = float(encoder.model.similarity(hv1, hv2))
        assert similarity > 0.99  # Should be nearly perfect

    def test_partial_matching(self, encoder):
        """Test that partial matching works (some dimensions similar)."""
        # First 5 dimensions identical, last 5 different
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        v2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 9.0, 9.0, 9.0, 9.0])
        v3 = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])

        hv1 = encoder.encode(v1)
        hv2 = encoder.encode(v2)
        hv3 = encoder.encode(v3)

        sim_partial = float(encoder.model.similarity(hv1, hv2))
        sim_different = float(encoder.model.similarity(hv1, hv3))

        # Partial match should be more similar than completely different
        assert sim_partial > sim_different


class TestVectorEncoderDecoding:
    """Test decoding functionality."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('FHRR', dim=5000, seed=42)

    @pytest.fixture
    def scalar_encoder(self, model):
        """Create test scalar encoder."""
        return FractionalPowerEncoder(model, min_val=0, max_val=10, seed=42)

    @pytest.fixture
    def encoder(self, model, scalar_encoder):
        """Create test vector encoder."""
        return VectorEncoder(model, scalar_encoder, n_dimensions=5, seed=42)

    def test_decode_simple_vector(self, encoder):
        """Test decoding a simple vector."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)

        assert decoded.shape == original.shape
        # Should be approximately close
        assert np.allclose(original, decoded, atol=0.5)

    def test_encode_decode_roundtrip(self, encoder):
        """Test that encode-decode roundtrip preserves values approximately."""
        original = np.array([2.5, 5.0, 7.5, 1.0, 9.0])
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)

        # Allow some tolerance due to approximate decoding
        assert np.allclose(original, decoded, atol=1.0)

    def test_decode_with_non_reversible_encoder_fails(self, model):
        """Test that decoding with non-reversible scalar encoder fails."""
        thermometer_enc = ThermometerEncoder(model, min_val=0, max_val=10, n_bins=20)
        encoder = VectorEncoder(model, thermometer_enc, n_dimensions=3)

        vector = np.array([1.0, 2.0, 3.0])
        encoded = encoder.encode(vector)

        with pytest.raises(NotImplementedError, match="does not support decoding"):
            encoder.decode(encoded)


class TestVectorEncoderScalarEncoderIntegration:
    """Test integration with different scalar encoders."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('FHRR', dim=2000, seed=42)

    def test_works_with_fractional_power_encoder(self, model):
        """Test that VectorEncoder works with FractionalPowerEncoder."""
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=3)

        vector = np.array([0.1, 0.5, 0.9])
        hv = encoder.encode(vector)

        assert hv.shape == (2000,)
        assert encoder.is_reversible is True

    def test_works_with_thermometer_encoder(self, model):
        """Test that VectorEncoder works with ThermometerEncoder."""
        scalar_enc = ThermometerEncoder(model, min_val=0, max_val=10, n_bins=20)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=5)

        vector = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        hv = encoder.encode(vector)

        assert hv.shape == (2000,)
        assert encoder.is_reversible is False

    def test_works_with_level_encoder(self, model):
        """Test that VectorEncoder works with LevelEncoder."""
        scalar_enc = LevelEncoder(model, min_val=0, max_val=10, n_levels=11)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=4)

        vector = np.array([0.0, 3.0, 7.0, 10.0])
        hv = encoder.encode(vector)

        assert hv.shape == (2000,)
        assert encoder.is_reversible is True


class TestVectorEncoderDimensionality:
    """Test encoder with different dimensionalities."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('FHRR', dim=2000, seed=42)

    @pytest.fixture
    def scalar_encoder(self, model):
        """Create test scalar encoder."""
        return FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)

    def test_single_dimension(self, model, scalar_encoder):
        """Test encoder with single dimension."""
        encoder = VectorEncoder(model, scalar_encoder, n_dimensions=1)

        vector = np.array([0.5])
        hv = encoder.encode(vector)

        assert hv.shape == (2000,)

    def test_high_dimensionality(self, model, scalar_encoder):
        """Test encoder with high dimensionality (like MNIST)."""
        encoder = VectorEncoder(model, scalar_encoder, n_dimensions=784)  # 28x28

        vector = np.random.rand(784)
        hv = encoder.encode(vector)

        assert hv.shape == (2000,)
        assert len(encoder.dim_vectors) == 784

    def test_dimension_vectors_are_orthogonal(self, model, scalar_encoder):
        """Test that different dimension vectors are approximately orthogonal."""
        encoder = VectorEncoder(model, scalar_encoder, n_dimensions=10, seed=42)

        # Check orthogonality of dimension vectors
        for i in range(5):
            for j in range(i + 1, 5):
                sim = float(model.similarity(encoder.dim_vectors[i], encoder.dim_vectors[j]))
                # Random vectors should be approximately orthogonal
                assert abs(sim) < 0.3


class TestVectorEncoderProperties:
    """Test encoder properties."""

    def test_is_reversible_with_fpe(self):
        """Test is_reversible property with reversible scalar encoder."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=5)

        assert encoder.is_reversible is True

    def test_is_reversible_with_thermometer(self):
        """Test is_reversible property with non-reversible scalar encoder."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = ThermometerEncoder(model, min_val=0, max_val=1, n_bins=10)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=5)

        assert encoder.is_reversible is False

    def test_compatible_models_property(self):
        """Test compatible_models property."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=5)

        compatible = encoder.compatible_models
        assert 'FHRR' in compatible
        assert 'MAP' in compatible
        assert 'HRR' in compatible
        assert len(compatible) > 0

    def test_input_type_property(self):
        """Test input_type property."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=10)

        assert encoder.input_type == "10-dimensional vector"


class TestVectorEncoderRepresentation:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        model = VSA.create('FHRR', dim=1000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)
        encoder = VectorEncoder(
            model, scalar_enc, n_dimensions=128, normalize_input=True, seed=42
        )

        repr_str = repr(encoder)

        assert 'VectorEncoder' in repr_str
        assert 'FHRR' in repr_str
        assert 'FractionalPowerEncoder' in repr_str
        assert 'n_dimensions=128' in repr_str
        assert 'normalize_input=True' in repr_str


class TestVectorEncoderCompatibility:
    """Test compatibility with different VSA models."""

    @pytest.mark.parametrize('model_name', ['FHRR', 'HRR', 'MAP', 'BSC'])
    def test_works_with_multiple_models(self, model_name):
        """Test that encoder works with multiple VSA models."""
        model = VSA.create(model_name, dim=1000, seed=42)
        # Use LevelEncoder which works with all models
        scalar_enc = LevelEncoder(model, min_val=0, max_val=10, n_levels=11)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=5, seed=42)

        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hv = encoder.encode(vector)

        assert hv.shape == (1000,)


class TestVectorEncoderPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        n_dims=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_encode_produces_valid_shape(self, n_dims, seed):
        """Test that encoding always produces correct shape."""
        model = VSA.create('FHRR', dim=2000, seed=seed)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=10, seed=seed)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=n_dims, seed=seed)

        vector = np.random.rand(n_dims)
        hv = encoder.encode(vector)

        assert hv.shape == (2000,)

    @given(
        vector_values=st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=10
        )
    )
    def test_similar_vectors_similar_encoding(self, vector_values):
        """Test that similar vectors produce similar encodings."""
        model = VSA.create('FHRR', dim=3000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=10, seed=42)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=len(vector_values), seed=42)

        v1 = np.array(vector_values)
        # Create similar vector with small perturbation
        v2 = v1 + 0.01 * np.random.randn(len(vector_values))
        v2 = np.clip(v2, 0.0, 10.0)  # Keep in valid range

        hv1 = encoder.encode(v1)
        hv2 = encoder.encode(v2)

        similarity = float(model.similarity(hv1, hv2))
        # Small perturbation should give high similarity
        assert similarity > 0.6

    @given(
        n_dims=st.integers(min_value=2, max_value=10)
    )
    @settings(deadline=None)  # Decoding can be slow
    def test_encode_decode_roundtrip_shape(self, n_dims):
        """Test that decode returns correct shape."""
        model = VSA.create('FHRR', dim=3000, seed=42)
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=10, seed=42)
        encoder = VectorEncoder(model, scalar_enc, n_dimensions=n_dims, seed=42)

        vector = np.random.rand(n_dims) * 10
        encoded = encoder.encode(vector)
        decoded = encoder.decode(encoded)

        assert decoded.shape == vector.shape
        assert len(decoded) == n_dims
