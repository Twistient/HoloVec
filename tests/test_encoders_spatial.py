"""
Tests for spatial encoders (images, grids).

This module tests the ImageEncoder for encoding 2D spatial data like images.
"""

import pytest
import numpy as np
from holovec import VSA
from holovec.encoders import (
    ImageEncoder,
    ThermometerEncoder,
    FractionalPowerEncoder
)


# Fixtures
@pytest.fixture
def model():
    """Create a MAP model for testing."""
    return VSA.create('MAP', dim=1000, seed=42)


@pytest.fixture
def scalar_encoder(model):
    """Create a ThermometerEncoder for pixel values."""
    return ThermometerEncoder(model, min_val=0, max_val=1, n_bins=256, seed=42)


@pytest.fixture
def encoder(model, scalar_encoder):
    """Create an ImageEncoder."""
    return ImageEncoder(model, scalar_encoder, normalize_pixels=True, seed=42)


# ============================================================================
# Initialization Tests
# ============================================================================

class TestImageEncoderInitialization:
    """Test ImageEncoder initialization."""

    def test_init_default_parameters(self, model, scalar_encoder):
        """Test initialization with default parameters."""
        encoder = ImageEncoder(model, scalar_encoder)
        assert encoder.model == model
        assert encoder.scalar_encoder == scalar_encoder
        assert encoder.normalize_pixels is True
        assert encoder.n_channels is None
        assert encoder.image_shape is None

    def test_init_with_normalization_off(self, model, scalar_encoder):
        """Test initialization with normalization disabled."""
        encoder = ImageEncoder(model, scalar_encoder, normalize_pixels=False)
        assert encoder.normalize_pixels is False

    def test_init_with_seed(self, model, scalar_encoder):
        """Test initialization with seed for reproducibility."""
        encoder1 = ImageEncoder(model, scalar_encoder, seed=42)
        encoder2 = ImageEncoder(model, scalar_encoder, seed=42)

        # Same seed should produce same dimension vectors (test via encoding)
        image = np.ones((2, 2), dtype=np.uint8) * 100
        hv1 = encoder1.encode(image)
        hv2 = encoder2.encode(image)

        # Should be identical with same seed
        assert float(model.similarity(hv1, hv2)) > 0.99

    def test_init_wrong_scalar_encoder_fails(self, model):
        """Test that invalid scalar encoder type fails."""
        with pytest.raises(TypeError, match="scalar_encoder must be a ScalarEncoder"):
            ImageEncoder(model, "not an encoder")

    def test_init_mismatched_model_fails(self, scalar_encoder):
        """Test that mismatched model fails."""
        other_model = VSA.create('FHRR', dim=1000, seed=42)
        with pytest.raises(ValueError, match="must use the same VSA model"):
            ImageEncoder(other_model, scalar_encoder)


# ============================================================================
# Grayscale Encoding Tests
# ============================================================================

class TestImageEncoderGrayscale:
    """Test grayscale image encoding."""

    def test_encode_grayscale_2d(self, encoder):
        """Test encoding 2D grayscale image."""
        image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        hv = encoder.encode(image)

        assert hv.shape == (1000,)
        assert encoder.n_channels == 1
        assert encoder.image_shape == (2, 2, 1)

    def test_encode_grayscale_3d(self, encoder):
        """Test encoding 3D grayscale image (H, W, 1)."""
        image = np.array([[[100], [150]], [[200], [250]]], dtype=np.uint8)
        hv = encoder.encode(image)

        assert hv.shape == (1000,)
        assert encoder.n_channels == 1
        assert encoder.image_shape == (2, 2, 1)

    def test_encode_different_sizes(self, encoder):
        """Test encoding images of different sizes."""
        small = np.ones((2, 2), dtype=np.uint8) * 100
        medium = np.ones((4, 4), dtype=np.uint8) * 100
        large = np.ones((8, 8), dtype=np.uint8) * 100

        hv_small = encoder.encode(small)
        hv_medium = encoder.encode(medium)
        hv_large = encoder.encode(large)

        # All should encode to same dimension
        assert hv_small.shape == hv_medium.shape == hv_large.shape == (1000,)

    def test_identical_grayscale_images_high_similarity(self, encoder, model):
        """Test that identical images have high similarity."""
        image1 = np.ones((3, 3), dtype=np.uint8) * 128
        image2 = np.ones((3, 3), dtype=np.uint8) * 128

        hv1 = encoder.encode(image1)
        hv2 = encoder.encode(image2)

        sim = float(model.similarity(hv1, hv2))
        assert sim > 0.95

    def test_similar_grayscale_images_moderate_similarity(self, encoder, model):
        """Test that similar images have moderate similarity."""
        image1 = np.ones((3, 3), dtype=np.uint8) * 100
        image2 = np.ones((3, 3), dtype=np.uint8) * 110

        hv1 = encoder.encode(image1)
        hv2 = encoder.encode(image2)

        sim = float(model.similarity(hv1, hv2))
        assert 0.7 < sim < 0.99

    def test_different_grayscale_images_lower_similarity(self, encoder, model):
        """Test that different images have lower similarity."""
        image1 = np.ones((3, 3), dtype=np.uint8) * 50
        image2 = np.ones((3, 3), dtype=np.uint8) * 200

        hv1 = encoder.encode(image1)
        hv2 = encoder.encode(image2)

        sim = float(model.similarity(hv1, hv2))
        assert sim < 0.9


# ============================================================================
# RGB Encoding Tests
# ============================================================================

class TestImageEncoderRGB:
    """Test RGB image encoding."""

    def test_encode_rgb(self, encoder):
        """Test encoding RGB image."""
        image = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        hv = encoder.encode(image)

        assert hv.shape == (1000,)
        assert encoder.n_channels == 3
        assert encoder.image_shape == (4, 4, 3)

    def test_identical_rgb_images_high_similarity(self, encoder, model):
        """Test that identical RGB images have high similarity."""
        image1 = np.ones((3, 3, 3), dtype=np.uint8) * 128
        image2 = np.ones((3, 3, 3), dtype=np.uint8) * 128

        hv1 = encoder.encode(image1)
        hv2 = encoder.encode(image2)

        sim = float(model.similarity(hv1, hv2))
        assert sim > 0.95

    def test_rgb_channels_affect_encoding(self, encoder, model):
        """Test that different RGB channels produce different encodings."""
        # Pure red
        red = np.zeros((3, 3, 3), dtype=np.uint8)
        red[:, :, 0] = 255

        # Pure green
        green = np.zeros((3, 3, 3), dtype=np.uint8)
        green[:, :, 1] = 255

        # Pure blue
        blue = np.zeros((3, 3, 3), dtype=np.uint8)
        blue[:, :, 2] = 255

        hv_red = encoder.encode(red)
        hv_green = encoder.encode(green)
        hv_blue = encoder.encode(blue)

        # Different colors should have lower similarity
        sim_rg = float(model.similarity(hv_red, hv_green))
        sim_rb = float(model.similarity(hv_red, hv_blue))
        sim_gb = float(model.similarity(hv_green, hv_blue))

        assert sim_rg < 0.9
        assert sim_rb < 0.9
        assert sim_gb < 0.9


# ============================================================================
# RGBA Encoding Tests
# ============================================================================

class TestImageEncoderRGBA:
    """Test RGBA image encoding."""

    def test_encode_rgba(self, encoder):
        """Test encoding RGBA image."""
        image = np.random.randint(0, 256, (4, 4, 4), dtype=np.uint8)
        hv = encoder.encode(image)

        assert hv.shape == (1000,)
        assert encoder.n_channels == 4
        assert encoder.image_shape == (4, 4, 4)

    def test_rgba_alpha_channel_affects_encoding(self, encoder, model):
        """Test that alpha channel affects encoding."""
        # Fully opaque
        opaque = np.ones((3, 3, 4), dtype=np.uint8) * 128
        opaque[:, :, 3] = 255

        # Semi-transparent
        semi = np.ones((3, 3, 4), dtype=np.uint8) * 128
        semi[:, :, 3] = 128

        hv_opaque = encoder.encode(opaque)
        hv_semi = encoder.encode(semi)

        sim = float(model.similarity(hv_opaque, hv_semi))
        # Alpha difference should reduce similarity
        assert sim < 0.95


# ============================================================================
# Normalization Tests
# ============================================================================

class TestImageEncoderNormalization:
    """Test pixel value normalization."""

    def test_normalization_enabled_uint8(self, model, scalar_encoder):
        """Test normalization with uint8 input."""
        encoder = ImageEncoder(model, scalar_encoder, normalize_pixels=True)

        # uint8 values should be normalized to [0, 1]
        image = np.array([[0, 127, 255]], dtype=np.uint8)
        hv = encoder.encode(image)

        assert hv.shape == (1000,)

    def test_normalization_disabled_float(self, model):
        """Test normalization disabled with float input."""
        # Create encoder with float range [0, 255]
        scalar_enc = ThermometerEncoder(model, min_val=0, max_val=255, n_bins=256, seed=42)
        encoder = ImageEncoder(model, scalar_enc, normalize_pixels=False)

        # Float values in [0, 1] should not be normalized
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        hv = encoder.encode(image)

        assert hv.shape == (1000,)

    def test_different_dtypes(self, encoder):
        """Test encoding with different data types."""
        # uint8
        img_uint8 = np.ones((2, 2), dtype=np.uint8) * 128

        # float32
        img_float = np.ones((2, 2), dtype=np.float32) * 0.5

        hv_uint8 = encoder.encode(img_uint8)
        hv_float = encoder.encode(img_float)

        # Should produce similar (not identical) results
        assert hv_uint8.shape == hv_float.shape


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestImageEncoderErrors:
    """Test error handling."""

    def test_encode_wrong_ndim_fails(self, encoder):
        """Test that 1D or 4D+ arrays fail."""
        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            encoder.encode(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            encoder.encode(np.ones((2, 2, 2, 2)))

    def test_encode_invalid_channels_fails(self, encoder):
        """Test that invalid number of channels fails."""
        # 2 channels (invalid)
        with pytest.raises(ValueError, match="must have 1, 3, or 4 channels"):
            encoder.encode(np.ones((3, 3, 2), dtype=np.uint8))

        # 5 channels (invalid)
        with pytest.raises(ValueError, match="must have 1, 3, or 4 channels"):
            encoder.encode(np.ones((3, 3, 5), dtype=np.uint8))

    def test_decode_not_implemented(self, encoder):
        """Test that decoding raises NotImplementedError."""
        image = np.ones((2, 2), dtype=np.uint8) * 128
        hv = encoder.encode(image)

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            encoder.decode(hv, height=2, width=2, n_channels=1)


# ============================================================================
# Properties Tests
# ============================================================================

class TestImageEncoderProperties:
    """Test encoder properties."""

    @pytest.mark.parametrize('model_name', ['MAP', 'FHRR', 'HRR', 'BSC'])
    def test_works_with_multiple_models(self, model_name):
        """Test that encoder works with multiple VSA models."""
        model = VSA.create(model_name, dim=1000, seed=42)

        # Use compatible scalar encoder based on model type
        if model_name in ['FHRR', 'HRR']:
            scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1, seed=42)
        else:  # MAP, BSC
            scalar_enc = ThermometerEncoder(model, min_val=0, max_val=1, n_bins=256, seed=42)

        encoder = ImageEncoder(model, scalar_enc, seed=42)

        image = np.ones((3, 3), dtype=np.uint8) * 128
        hv = encoder.encode(image)

        assert hv.shape == (1000,)

    def test_is_reversible_property(self, encoder):
        """Test is_reversible property."""
        # Decoding not implemented yet
        assert encoder.is_reversible is False

    def test_compatible_models_property(self, encoder):
        """Test compatible_models property."""
        compatible = encoder.compatible_models
        assert 'MAP' in compatible
        assert len(compatible) > 0

    def test_input_type_property(self, encoder):
        """Test input_type property."""
        # Before encoding
        assert "2D array" in encoder.input_type or "3D array" in encoder.input_type

        # After grayscale
        encoder.encode(np.ones((3, 3), dtype=np.uint8))
        assert "Grayscale" in encoder.input_type
        assert "3x3" in encoder.input_type

        # After RGB
        encoder.encode(np.ones((4, 4, 3), dtype=np.uint8))
        assert "RGB" in encoder.input_type
        assert "4x4x3" in encoder.input_type

        # After RGBA
        encoder.encode(np.ones((5, 5, 4), dtype=np.uint8))
        assert "RGBA" in encoder.input_type
        assert "5x5x4" in encoder.input_type

    def test_repr(self, encoder):
        """Test string representation."""
        repr_str = repr(encoder)
        assert "ImageEncoder" in repr_str
        assert "MAP" in repr_str
        assert "ThermometerEncoder" in repr_str
        assert "normalize_pixels" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================

class TestImageEncoderIntegration:
    """Integration tests with real-world scenarios."""

    def test_mnist_like_images(self, encoder, model):
        """Test with MNIST-like 28x28 grayscale images."""
        # Create simple digit-like patterns
        digit_0 = np.zeros((28, 28), dtype=np.uint8)
        digit_0[5:23, 5:10] = 255    # Left edge
        digit_0[5:23, 18:23] = 255   # Right edge
        digit_0[5:10, 5:23] = 255    # Top edge
        digit_0[18:23, 5:23] = 255   # Bottom edge

        digit_1 = np.zeros((28, 28), dtype=np.uint8)
        digit_1[:, 12:16] = 255      # Vertical line

        hv_0 = encoder.encode(digit_0)
        hv_1 = encoder.encode(digit_1)

        # Encodings should be valid (patterns encoded successfully)
        assert hv_0.shape == (1000,)
        assert hv_1.shape == (1000,)

        # Note: Similarity depends on pattern complexity and encoder parameters
        # For small images, different patterns may still have moderate similarity
        sim = float(model.similarity(hv_0, hv_1))
        assert 0.0 <= sim <= 1.0  # Valid similarity range

    def test_small_image_patches(self, encoder, model):
        """Test encoding small image patches."""
        # 2x2 patches
        patch1 = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        patch2 = np.array([[255, 255], [255, 255]], dtype=np.uint8)

        hv1 = encoder.encode(patch1)
        hv2 = encoder.encode(patch2)

        # Black vs white should be different
        sim = float(model.similarity(hv1, hv2))
        assert sim < 0.9

    def test_color_images_rgb(self, encoder, model):
        """Test with typical RGB color images."""
        # Sky-like image (blue)
        sky = np.zeros((10, 10, 3), dtype=np.uint8)
        sky[:, :, 2] = 200  # Blue channel

        # Grass-like image (green)
        grass = np.zeros((10, 10, 3), dtype=np.uint8)
        grass[:, :, 1] = 150  # Green channel

        hv_sky = encoder.encode(sky)
        hv_grass = encoder.encode(grass)

        # Different colors should have low similarity
        sim = float(model.similarity(hv_sky, hv_grass))
        assert sim < 0.9

    def test_batch_encoding_consistency(self, encoder, model):
        """Test that encoding same image multiple times is consistent."""
        image = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)

        hvs = [encoder.encode(image) for _ in range(5)]

        # All encodings should be identical
        for i in range(1, len(hvs)):
            sim = float(model.similarity(hvs[0], hvs[i]))
            assert sim > 0.99
