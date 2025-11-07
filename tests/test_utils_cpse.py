"""Tests for CPSE/CPSD utilities.

Tests the utility functions for Context-Preserving SDR Encoding (CPSE)
and Context-Preserving SDR Decoding (CPSD).
"""

import pytest
import tempfile
import os
import json

from holovec.utils.cpse import (
    CPSEMetadata,
    generate_permutation_patterns,
    validate_cpse_convergence,
)
from holovec import VSA


# ============================================================================
# CPSEMetadata Tests
# ============================================================================

class TestCPSEMetadataInitialization:
    """Test CPSEMetadata initialization."""

    def test_init_valid_minimal(self):
        """Test initialization with minimal valid parameters."""
        metadata = CPSEMetadata(
            n_components=2,
            permutation_seeds=[42, 43]
        )
        assert metadata.n_components == 2
        assert metadata.permutation_seeds == [42, 43]
        assert metadata.base_seed == 42  # Default

    def test_init_valid_with_base_seed(self):
        """Test initialization with custom base seed."""
        metadata = CPSEMetadata(
            n_components=3,
            permutation_seeds=[100, 101, 102],
            base_seed=100
        )
        assert metadata.n_components == 3
        assert metadata.permutation_seeds == [100, 101, 102]
        assert metadata.base_seed == 100

    def test_init_fails_n_components_too_small(self):
        """Test that n_components < 2 fails."""
        with pytest.raises(ValueError, match="must be >= 2"):
            CPSEMetadata(n_components=1, permutation_seeds=[42])

    def test_init_fails_n_components_zero(self):
        """Test that n_components == 0 fails."""
        with pytest.raises(ValueError, match="must be >= 2"):
            CPSEMetadata(n_components=0, permutation_seeds=[])

    def test_init_fails_mismatched_seeds_length(self):
        """Test that mismatched seed length fails."""
        with pytest.raises(ValueError, match="Expected 3 permutation seeds, got 2"):
            CPSEMetadata(
                n_components=3,
                permutation_seeds=[42, 43]  # Only 2 seeds for 3 components
            )

    def test_init_fails_wrong_type_n_components(self):
        """Test that wrong type for n_components fails."""
        with pytest.raises(TypeError, match="n_components must be int"):
            CPSEMetadata(
                n_components="3",  # Wrong type
                permutation_seeds=[42, 43, 44]
            )

    def test_init_fails_wrong_type_permutation_seeds(self):
        """Test that wrong type for permutation_seeds fails."""
        with pytest.raises(TypeError, match="permutation_seeds must be list"):
            CPSEMetadata(
                n_components=3,
                permutation_seeds=(42, 43, 44)  # Tuple instead of list
            )

    def test_init_fails_wrong_type_base_seed(self):
        """Test that wrong type for base_seed fails."""
        with pytest.raises(TypeError, match="base_seed must be int"):
            CPSEMetadata(
                n_components=2,
                permutation_seeds=[42, 43],
                base_seed="42"  # Wrong type
            )


class TestCPSEMetadataSerialization:
    """Test CPSEMetadata serialization/deserialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = CPSEMetadata(3, [42, 43, 44], base_seed=42)
        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data['n_components'] == 3
        assert data['permutation_seeds'] == [42, 43, 44]
        assert data['base_seed'] == 42

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'n_components': 3,
            'permutation_seeds': [42, 43, 44],
            'base_seed': 42
        }
        metadata = CPSEMetadata.from_dict(data)

        assert metadata.n_components == 3
        assert metadata.permutation_seeds == [42, 43, 44]
        assert metadata.base_seed == 42

    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict/from_dict roundtrip works."""
        original = CPSEMetadata(5, [10, 11, 12, 13, 14], base_seed=10)
        data = original.to_dict()
        restored = CPSEMetadata.from_dict(data)

        assert restored.n_components == original.n_components
        assert restored.permutation_seeds == original.permutation_seeds
        assert restored.base_seed == original.base_seed

    def test_to_json_from_json_roundtrip(self):
        """Test that to_json/from_json roundtrip works."""
        original = CPSEMetadata(4, [20, 21, 22, 23], base_seed=20)

        # Use temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save and load
            original.to_json(temp_path)
            restored = CPSEMetadata.from_json(temp_path)

            # Verify
            assert restored.n_components == original.n_components
            assert restored.permutation_seeds == original.permutation_seeds
            assert restored.base_seed == original.base_seed
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_from_json_fails_missing_file(self):
        """Test that loading from non-existent file fails."""
        with pytest.raises(FileNotFoundError):
            CPSEMetadata.from_json('nonexistent_file.json')

    def test_from_dict_fails_missing_key(self):
        """Test that missing key in dict fails."""
        data = {
            'n_components': 3,
            # Missing 'permutation_seeds'
            'base_seed': 42
        }
        with pytest.raises(KeyError):
            CPSEMetadata.from_dict(data)


class TestCPSEMetadataProperties:
    """Test CPSEMetadata properties and methods."""

    def test_repr(self):
        """Test string representation."""
        metadata = CPSEMetadata(2, [42, 43])
        repr_str = repr(metadata)

        assert 'CPSEMetadata' in repr_str
        assert 'n_components=2' in repr_str
        assert 'permutation_seeds=[42, 43]' in repr_str
        assert 'base_seed=42' in repr_str

    def test_eq_identical(self):
        """Test equality for identical metadata."""
        m1 = CPSEMetadata(3, [42, 43, 44])
        m2 = CPSEMetadata(3, [42, 43, 44])

        assert m1 == m2

    def test_eq_different_n_components(self):
        """Test inequality for different n_components."""
        m1 = CPSEMetadata(3, [42, 43, 44])
        m2 = CPSEMetadata(2, [42, 43])

        assert m1 != m2

    def test_eq_different_permutation_seeds(self):
        """Test inequality for different permutation_seeds."""
        m1 = CPSEMetadata(3, [42, 43, 44])
        m2 = CPSEMetadata(3, [50, 51, 52])

        assert m1 != m2

    def test_eq_different_base_seed(self):
        """Test inequality for different base_seed."""
        m1 = CPSEMetadata(3, [42, 43, 44], base_seed=42)
        m2 = CPSEMetadata(3, [42, 43, 44], base_seed=100)

        assert m1 != m2

    def test_eq_with_non_metadata_object(self):
        """Test inequality with non-CPSEMetadata object."""
        metadata = CPSEMetadata(2, [42, 43])

        assert metadata != "not a metadata object"
        assert metadata != 42
        assert metadata != [2, [42, 43], 42]


# ============================================================================
# generate_permutation_patterns Tests
# ============================================================================

class TestGeneratePermutationPatterns:
    """Test generate_permutation_patterns function."""

    def test_generate_basic(self):
        """Test basic permutation pattern generation."""
        seeds = generate_permutation_patterns(n_patterns=5)

        assert isinstance(seeds, list)
        assert len(seeds) == 5
        assert seeds == [42, 43, 44, 45, 46]  # Default base_seed=42

    def test_generate_with_custom_base_seed(self):
        """Test generation with custom base seed."""
        seeds = generate_permutation_patterns(n_patterns=3, base_seed=100)

        assert len(seeds) == 3
        assert seeds == [100, 101, 102]

    def test_generate_single_pattern(self):
        """Test generation of single pattern."""
        seeds = generate_permutation_patterns(n_patterns=1)

        assert len(seeds) == 1
        assert seeds == [42]

    def test_generate_many_patterns(self):
        """Test generation of many patterns."""
        seeds = generate_permutation_patterns(n_patterns=100)

        assert len(seeds) == 100
        assert seeds[0] == 42
        assert seeds[-1] == 141
        # Check sequential
        for i in range(len(seeds) - 1):
            assert seeds[i+1] == seeds[i] + 1

    def test_generate_deterministic(self):
        """Test that generation is deterministic."""
        seeds1 = generate_permutation_patterns(n_patterns=10, base_seed=42)
        seeds2 = generate_permutation_patterns(n_patterns=10, base_seed=42)

        assert seeds1 == seeds2

    def test_generate_fails_zero_patterns(self):
        """Test that n_patterns == 0 fails."""
        with pytest.raises(ValueError, match="must be >= 1"):
            generate_permutation_patterns(n_patterns=0)

    def test_generate_fails_negative_patterns(self):
        """Test that negative n_patterns fails."""
        with pytest.raises(ValueError, match="must be >= 1"):
            generate_permutation_patterns(n_patterns=-5)

    def test_generate_fails_wrong_type_n_patterns(self):
        """Test that wrong type for n_patterns fails."""
        with pytest.raises(TypeError, match="n_patterns must be int"):
            generate_permutation_patterns(n_patterns="5")

    def test_generate_fails_wrong_type_base_seed(self):
        """Test that wrong type for base_seed fails."""
        with pytest.raises(TypeError, match="base_seed must be int"):
            generate_permutation_patterns(n_patterns=5, base_seed="42")


# ============================================================================
# validate_cpse_convergence Tests
# ============================================================================

class TestValidateCPSEConvergence:
    """Test validate_cpse_convergence function."""

    def test_convergence_identical_components(self):
        """Test convergence with identical components."""
        model = VSA.create('MAP', dim=1000, seed=42)

        # Create identical components
        original = [model.random(seed=i) for i in range(3)]
        decoded = original  # Same references

        converged, sims = validate_cpse_convergence(
            original, decoded, model, threshold=0.95
        )

        assert converged is True
        assert len(sims) == 3
        # All similarities should be 1.0 (identical)
        assert all(sim >= 0.99 for sim in sims)

    def test_convergence_high_similarity(self):
        """Test convergence with high similarity components."""
        model = VSA.create('MAP', dim=10000, seed=42)

        # Create original components
        original = [model.random(seed=i) for i in range(3)]

        # Create decoded with slight noise (still very similar)
        # Just use bundle without weights for a simpler test
        decoded = []
        for i, hv in enumerate(original):
            # Bundle original with small amount of noise
            noise = model.random(seed=100+i)
            # Multiple bundles to make original dominate
            noisy = model.bundle([hv, hv, hv, hv, noise])  # 4:1 ratio
            decoded.append(noisy)

        converged, sims = validate_cpse_convergence(
            original, decoded, model, threshold=0.85  # Lower threshold due to noise
        )

        assert converged is True
        assert len(sims) == 3
        # Similarities should be high
        assert all(sim >= 0.85 for sim in sims)

    def test_convergence_fails_low_similarity(self):
        """Test that convergence fails with low similarity."""
        model = VSA.create('MAP', dim=1000, seed=42)

        # Create original and completely different decoded
        original = [model.random(seed=i) for i in range(3)]
        decoded = [model.random(seed=i+100) for i in range(3)]  # Different seeds

        converged, sims = validate_cpse_convergence(
            original, decoded, model, threshold=0.95  # High threshold
        )

        assert converged is False
        assert len(sims) == 3
        # Random hypervectors have moderate similarity, but below 0.95 threshold
        assert all(sim < 0.95 for sim in sims)

    def test_convergence_custom_threshold(self):
        """Test convergence with custom threshold."""
        model = VSA.create('MAP', dim=1000, seed=42)

        original = [model.random(seed=i) for i in range(2)]
        decoded = original

        # Very strict threshold (but valid)
        converged, sims = validate_cpse_convergence(
            original, decoded, model, threshold=0.99
        )
        assert converged is True

        # Very lenient threshold
        converged, sims = validate_cpse_convergence(
            original, decoded, model, threshold=0.01
        )
        assert converged is True

    def test_convergence_fails_mismatched_lengths(self):
        """Test that mismatched component lengths fail."""
        model = VSA.create('MAP', dim=1000, seed=42)

        original = [model.random() for _ in range(3)]
        decoded = [model.random() for _ in range(2)]  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            validate_cpse_convergence(original, decoded, model)

    def test_convergence_fails_wrong_type_original(self):
        """Test that wrong type for original_components fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        decoded = [model.random() for _ in range(2)]

        with pytest.raises(TypeError, match="original_components must be list"):
            validate_cpse_convergence("not a list", decoded, model)

    def test_convergence_fails_wrong_type_decoded(self):
        """Test that wrong type for decoded_components fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        original = [model.random() for _ in range(2)]

        with pytest.raises(TypeError, match="decoded_components must be list"):
            validate_cpse_convergence(original, "not a list", model)

    def test_convergence_fails_wrong_type_model(self):
        """Test that wrong type for model fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        original = [model.random() for _ in range(2)]
        decoded = [model.random() for _ in range(2)]

        with pytest.raises(TypeError, match="model must be VSAModel"):
            validate_cpse_convergence(original, decoded, "not a model")

    def test_convergence_fails_threshold_out_of_range(self):
        """Test that threshold out of [0, 1] range fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        original = [model.random() for _ in range(2)]
        decoded = original

        with pytest.raises(ValueError, match="threshold must be in range"):
            validate_cpse_convergence(original, decoded, model, threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be in range"):
            validate_cpse_convergence(original, decoded, model, threshold=-0.1)

    def test_convergence_empty_lists(self):
        """Test convergence with empty lists."""
        model = VSA.create('MAP', dim=1000, seed=42)

        # Empty lists should technically converge (vacuously true)
        converged, sims = validate_cpse_convergence([], [], model)

        assert converged is True
        assert sims == []


# ============================================================================
# Integration Tests
# ============================================================================

class TestCPSEUtilitiesIntegration:
    """Integration tests combining multiple utilities."""

    def test_full_workflow_metadata_and_patterns(self):
        """Test complete workflow: generate patterns, create metadata, serialize."""
        # Generate patterns
        seeds = generate_permutation_patterns(n_patterns=5, base_seed=100)

        # Create metadata
        metadata = CPSEMetadata(n_components=5, permutation_seeds=seeds, base_seed=100)

        # Serialize to dict
        data = metadata.to_dict()

        # Restore from dict
        restored = CPSEMetadata.from_dict(data)

        # Verify everything matches
        assert restored.n_components == 5
        assert restored.permutation_seeds == seeds
        assert restored.base_seed == 100

    def test_metadata_with_convergence_validation(self):
        """Test using metadata with convergence validation."""
        # Setup
        model = VSA.create('MAP', dim=10000, seed=42)
        n_components = 3

        # Generate patterns and metadata
        seeds = generate_permutation_patterns(n_components, base_seed=42)
        metadata = CPSEMetadata(n_components, seeds, base_seed=42)

        # Simulate encoding/decoding
        original = [model.random() for _ in range(n_components)]

        # Perfect decoding (same vectors)
        decoded = original

        # Validate convergence
        converged, sims = validate_cpse_convergence(
            original, decoded, model, threshold=0.95
        )

        # Should converge since they're identical
        assert converged
        assert len(sims) == metadata.n_components
