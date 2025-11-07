"""Tests for general utility operations.

Tests the general-purpose operations for hypervector manipulation,
including top-k selection, noise injection, and similarity matrices.
"""

import pytest
import numpy as np

from holovec.utils.operations import (
    select_top_k,
    add_noise,
    similarity_matrix,
)
from holovec import VSA


# ============================================================================
# select_top_k Tests
# ============================================================================

class TestSelectTopKBasic:
    """Test select_top_k basic functionality."""

    def test_select_top_k_basic(self):
        """Test basic top-k selection."""
        items = {'a': 0.95, 'b': 0.87, 'c': 0.92, 'd': 0.75, 'e': 0.88}
        top = select_top_k(items, k=3)

        assert len(top) == 3
        assert top[0] == ('a', 0.95)  # Highest
        assert top[1] == ('c', 0.92)  # Second
        assert top[2] == ('e', 0.88)  # Third

    def test_select_top_k_sorted_descending(self):
        """Test that results are sorted by score (descending)."""
        items = {'x': 0.5, 'y': 0.9, 'z': 0.7}
        top = select_top_k(items, k=3)

        scores = [score for _, score in top]
        assert scores == sorted(scores, reverse=True)

    def test_select_top_k_single_item(self):
        """Test selecting single top item."""
        items = {'a': 0.8, 'b': 0.9, 'c': 0.7}
        top = select_top_k(items, k=1)

        assert len(top) == 1
        assert top[0] == ('b', 0.9)

    def test_select_top_k_all_items(self):
        """Test selecting all items."""
        items = {'a': 0.8, 'b': 0.9, 'c': 0.7}
        top = select_top_k(items, k=3)

        assert len(top) == 3
        # Should still be sorted
        assert top[0][1] >= top[1][1] >= top[2][1]


class TestSelectTopKErrors:
    """Test select_top_k error handling."""

    def test_select_top_k_fails_empty_items(self):
        """Test that empty items fails."""
        with pytest.raises(ValueError, match="must not be empty"):
            select_top_k({}, k=1)

    def test_select_top_k_fails_k_too_large(self):
        """Test that k > items size fails."""
        items = {'a': 0.5, 'b': 0.8}
        with pytest.raises(ValueError, match="cannot exceed items size"):
            select_top_k(items, k=5)

    def test_select_top_k_fails_k_zero(self):
        """Test that k=0 fails."""
        items = {'a': 0.5}
        with pytest.raises(ValueError, match="k must be >= 1"):
            select_top_k(items, k=0)

    def test_select_top_k_fails_wrong_type(self):
        """Test that wrong types fail."""
        with pytest.raises(TypeError, match="items must be dict"):
            select_top_k([], k=1)

        with pytest.raises(TypeError, match="k must be int"):
            select_top_k({'a': 0.5}, k="1")


# ============================================================================
# add_noise Tests
# ============================================================================

class TestAddNoiseBasic:
    """Test add_noise basic functionality."""

    def test_add_noise_reduces_similarity(self):
        """Test that adding noise reduces similarity."""
        # Use HRR which has continuous representations
        # MAP's discrete majority voting doesn't support fine-grained noise
        model = VSA.create('HRR', dim=10000, seed=42)
        original = model.random(seed=1)

        noisy = add_noise(original, model, noise_level=0.2)
        similarity = model.similarity(original, noisy)

        # Should be less than perfect but still high
        assert 0.6 < similarity < 1.0

    def test_add_noise_zero_level(self):
        """Test that noise_level=0.0 returns original."""
        model = VSA.create('MAP', dim=1000, seed=42)
        original = model.random(seed=1)

        noisy = add_noise(original, model, noise_level=0.0)
        similarity = model.similarity(original, noisy)

        assert similarity >= 0.99  # Should be nearly identical

    def test_add_noise_high_level(self):
        """Test that high noise level reduces similarity significantly."""
        # Use HRR for continuous noise injection
        model = VSA.create('HRR', dim=10000, seed=42)
        original = model.random(seed=1)

        noisy = add_noise(original, model, noise_level=0.5)
        similarity = model.similarity(original, noisy)

        # High noise should reduce similarity
        assert similarity < 0.9

    def test_add_noise_reproducible(self):
        """Test that noise with seed is reproducible."""
        model = VSA.create('MAP', dim=1000, seed=42)
        original = model.random(seed=1)

        noisy1 = add_noise(original, model, noise_level=0.2, seed=100)
        noisy2 = add_noise(original, model, noise_level=0.2, seed=100)

        similarity = model.similarity(noisy1, noisy2)
        assert similarity >= 0.99  # Should be nearly identical

    def test_add_noise_different_levels(self):
        """Test different noise levels produce different results."""
        # Use HRR for continuous noise injection
        model = VSA.create('HRR', dim=10000, seed=42)
        original = model.random(seed=1)

        low_noise = add_noise(original, model, noise_level=0.1)
        high_noise = add_noise(original, model, noise_level=0.4)

        sim_low = model.similarity(original, low_noise)
        sim_high = model.similarity(original, high_noise)

        # Low noise should be more similar than high noise
        assert sim_low > sim_high


class TestAddNoiseErrors:
    """Test add_noise error handling."""

    def test_add_noise_fails_invalid_noise_level(self):
        """Test that invalid noise_level fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vector = model.random()

        with pytest.raises(ValueError, match="noise_level must be in"):
            add_noise(vector, model, noise_level=1.5)

        with pytest.raises(ValueError, match="noise_level must be in"):
            add_noise(vector, model, noise_level=-0.1)

    def test_add_noise_fails_wrong_type(self):
        """Test that wrong types fail."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vector = model.random()

        with pytest.raises(TypeError, match="model must be VSAModel"):
            add_noise(vector, "not a model", noise_level=0.1)

        with pytest.raises(TypeError, match="noise_level must be numeric"):
            add_noise(vector, model, noise_level="0.1")

        with pytest.raises(TypeError, match="seed must be int or None"):
            add_noise(vector, model, noise_level=0.1, seed="42")


# ============================================================================
# similarity_matrix Tests
# ============================================================================

class TestSimilarityMatrixBasic:
    """Test similarity_matrix basic functionality."""

    def test_similarity_matrix_shape(self):
        """Test similarity matrix has correct shape."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vectors = [model.random(seed=i) for i in range(5)]

        matrix = similarity_matrix(vectors, model)

        assert matrix.shape == (5, 5)
        assert isinstance(matrix, np.ndarray)

    def test_similarity_matrix_diagonal(self):
        """Test that diagonal entries are near 1.0 (self-similarity)."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vectors = [model.random(seed=i) for i in range(3)]

        matrix = similarity_matrix(vectors, model)

        # Diagonal should be self-similarity (near 1.0)
        for i in range(3):
            assert matrix[i, i] >= 0.99

    def test_similarity_matrix_symmetric(self):
        """Test that similarity matrix is symmetric."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vectors = [model.random(seed=i) for i in range(3)]

        matrix = similarity_matrix(vectors, model)

        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T, rtol=1e-5)

    def test_similarity_matrix_with_labels(self):
        """Test similarity matrix with labels (doesn't affect computation)."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vectors = [model.random(seed=i) for i in range(3)]
        labels = ['a', 'b', 'c']

        matrix = similarity_matrix(vectors, model, labels=labels)

        assert matrix.shape == (3, 3)
        # Labels don't affect computation, just for user reference

    def test_similarity_matrix_single_vector(self):
        """Test similarity matrix with single vector."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vectors = [model.random(seed=1)]

        matrix = similarity_matrix(vectors, model)

        assert matrix.shape == (1, 1)
        assert matrix[0, 0] >= 0.99  # Self-similarity


class TestSimilarityMatrixErrors:
    """Test similarity_matrix error handling."""

    def test_similarity_matrix_fails_empty_vectors(self):
        """Test that empty vectors list fails."""
        model = VSA.create('MAP', dim=1000, seed=42)

        with pytest.raises(ValueError, match="must not be empty"):
            similarity_matrix([], model)

    def test_similarity_matrix_fails_wrong_label_length(self):
        """Test that mismatched labels length fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vectors = [model.random(seed=i) for i in range(3)]
        labels = ['a', 'b']  # Wrong length

        with pytest.raises(ValueError, match="labels length .* must match vectors length"):
            similarity_matrix(vectors, model, labels=labels)

    def test_similarity_matrix_fails_wrong_type(self):
        """Test that wrong types fail."""
        model = VSA.create('MAP', dim=1000, seed=42)
        vectors = [model.random()]

        with pytest.raises(TypeError, match="vectors must be list"):
            similarity_matrix("not a list", model)

        with pytest.raises(TypeError, match="model must be VSAModel"):
            similarity_matrix(vectors, "not a model")

        with pytest.raises(TypeError, match="labels must be list or None"):
            similarity_matrix(vectors, model, labels="not a list")
