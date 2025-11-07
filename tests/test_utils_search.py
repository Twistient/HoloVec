"""Tests for search utilities.

Tests the search functions for VSA codebook operations, including
k-nearest neighbors, threshold search, and batch similarity.
"""

import pytest

from holovec.utils.search import (
    nearest_neighbors,
    threshold_search,
    batch_similarity,
)
from holovec import VSA


# ============================================================================
# nearest_neighbors Tests
# ============================================================================

class TestNearestNeighborsBasic:
    """Test nearest_neighbors basic functionality."""

    def test_knn_finds_exact_match(self):
        """Test K-NN finds exact match as top result."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {
            'cat': model.random(seed=1),
            'dog': model.random(seed=2),
            'bird': model.random(seed=3),
            'fish': model.random(seed=4),
        }

        query = codebook['dog']
        labels, sims = nearest_neighbors(query, codebook, model, k=3)

        assert labels[0] == 'dog'  # Best match should be exact
        assert sims[0] >= 0.99  # Near-perfect similarity
        assert len(labels) == 3
        assert len(sims) == 3

    def test_knn_respects_k(self):
        """Test K-NN returns exactly k results."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {f'item_{i}': model.random(seed=i) for i in range(10)}
        query = codebook['item_5']

        for k in [1, 3, 5, 10]:
            labels, sims = nearest_neighbors(query, codebook, model, k=k)
            assert len(labels) == k
            assert len(sims) == k

    def test_knn_sorted_by_similarity(self):
        """Test K-NN results are sorted by similarity (descending)."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {f'item_{i}': model.random(seed=i) for i in range(10)}
        query = model.random(seed=100)

        labels, sims = nearest_neighbors(query, codebook, model, k=5)

        # Similarities should be in descending order
        for i in range(len(sims) - 1):
            assert sims[i] >= sims[i + 1]

    def test_knn_without_similarities(self):
        """Test K-NN with return_similarities=False."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {f'item_{i}': model.random(seed=i) for i in range(5)}
        query = codebook['item_2']

        labels, sims = nearest_neighbors(
            query, codebook, model, k=3, return_similarities=False
        )

        assert len(labels) == 3
        assert sims is None  # Should not return similarities

    def test_knn_single_neighbor(self):
        """Test K-NN with k=1."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {
            'a': model.random(seed=1),
            'b': model.random(seed=2),
            'c': model.random(seed=3),
        }

        query = codebook['b']
        labels, sims = nearest_neighbors(query, codebook, model, k=1)

        assert len(labels) == 1
        assert len(sims) == 1
        assert labels[0] == 'b'


class TestNearestNeighborsErrors:
    """Test nearest_neighbors error handling."""

    def test_knn_fails_empty_codebook(self):
        """Test that empty codebook fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        query = model.random()

        with pytest.raises(ValueError, match="must not be empty"):
            nearest_neighbors(query, {}, model, k=1)

    def test_knn_fails_k_too_large(self):
        """Test that k > codebook size fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        codebook = {'a': model.random(), 'b': model.random()}
        query = model.random()

        with pytest.raises(ValueError, match="cannot exceed codebook size"):
            nearest_neighbors(query, codebook, model, k=5)

    def test_knn_fails_k_zero(self):
        """Test that k=0 fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        codebook = {'a': model.random()}
        query = model.random()

        with pytest.raises(ValueError, match="k must be >= 1"):
            nearest_neighbors(query, codebook, model, k=0)

    def test_knn_fails_wrong_type_codebook(self):
        """Test that wrong codebook type fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        query = model.random()

        with pytest.raises(TypeError, match="codebook must be dict"):
            nearest_neighbors(query, [], model, k=1)

    def test_knn_fails_wrong_type_k(self):
        """Test that wrong k type fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        codebook = {'a': model.random()}
        query = model.random()

        with pytest.raises(TypeError, match="k must be int"):
            nearest_neighbors(query, codebook, model, k="3")


# ============================================================================
# threshold_search Tests
# ============================================================================

class TestThresholdSearchBasic:
    """Test threshold_search basic functionality."""

    def test_threshold_search_finds_high_similarity(self):
        """Test threshold search finds high-similarity matches."""
        model = VSA.create('MAP', dim=10000, seed=42)

        # Create codebook with one exact match
        exact = model.random(seed=1)
        codebook = {
            'exact': exact,
            'other1': model.random(seed=2),
            'other2': model.random(seed=3),
        }

        query = exact
        labels, sims = threshold_search(query, codebook, model, threshold=0.95)

        assert 'exact' in labels
        assert all(sim >= 0.95 for sim in sims)

    def test_threshold_search_filters_by_threshold(self):
        """Test threshold search filters correctly."""
        model = VSA.create('MAP', dim=10000, seed=42)

        codebook = {f'item_{i}': model.random(seed=i) for i in range(10)}
        query = codebook['item_5']

        # High threshold - should get fewer matches
        labels_high, _ = threshold_search(query, codebook, model, threshold=0.95)

        # Low threshold - should get more matches
        labels_low, _ = threshold_search(query, codebook, model, threshold=0.5)

        assert len(labels_high) <= len(labels_low)

    def test_threshold_search_sorted_by_similarity(self):
        """Test threshold search results are sorted."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {f'item_{i}': model.random(seed=i) for i in range(10)}
        query = model.random(seed=100)

        labels, sims = threshold_search(query, codebook, model, threshold=0.0)

        # Should be sorted descending
        for i in range(len(sims) - 1):
            assert sims[i] >= sims[i + 1]

    def test_threshold_search_empty_results(self):
        """Test threshold search with no matches above threshold."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {f'item_{i}': model.random(seed=i) for i in range(5)}
        query = model.random(seed=100)

        # Very high threshold - likely no matches
        labels, sims = threshold_search(query, codebook, model, threshold=0.99)

        assert isinstance(labels, list)
        assert isinstance(sims, list)
        # May be empty or have very few matches

    def test_threshold_search_without_similarities(self):
        """Test threshold search with return_similarities=False."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {f'item_{i}': model.random(seed=i) for i in range(5)}
        query = codebook['item_2']

        labels, sims = threshold_search(
            query, codebook, model, threshold=0.8, return_similarities=False
        )

        assert isinstance(labels, list)
        assert sims is None


class TestThresholdSearchErrors:
    """Test threshold_search error handling."""

    def test_threshold_search_fails_empty_codebook(self):
        """Test that empty codebook fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        query = model.random()

        with pytest.raises(ValueError, match="must not be empty"):
            threshold_search(query, {}, model)

    def test_threshold_search_fails_invalid_threshold(self):
        """Test that invalid threshold fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        codebook = {'a': model.random()}
        query = model.random()

        with pytest.raises(ValueError, match="threshold must be in"):
            threshold_search(query, codebook, model, threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be in"):
            threshold_search(query, codebook, model, threshold=-0.1)

    def test_threshold_search_fails_wrong_type(self):
        """Test that wrong types fail."""
        model = VSA.create('MAP', dim=1000, seed=42)
        codebook = {'a': model.random()}
        query = model.random()

        with pytest.raises(TypeError, match="codebook must be dict"):
            threshold_search(query, [], model)

        with pytest.raises(TypeError, match="threshold must be numeric"):
            threshold_search(query, codebook, model, threshold="0.8")


# ============================================================================
# batch_similarity Tests
# ============================================================================

class TestBatchSimilarityBasic:
    """Test batch_similarity basic functionality."""

    def test_batch_similarity_multiple_queries(self):
        """Test batch similarity with multiple queries."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {
            'a': model.random(seed=1),
            'b': model.random(seed=2),
            'c': model.random(seed=3),
        }

        queries = [
            codebook['a'],
            codebook['b'],
            model.random(seed=100),
        ]

        results = batch_similarity(queries, codebook, model)

        assert len(results) == 3  # One result per query
        for query_sims in results:
            assert isinstance(query_sims, dict)
            assert len(query_sims) == 3  # One similarity per codebook entry
            assert set(query_sims.keys()) == {'a', 'b', 'c'}

    def test_batch_similarity_correct_similarities(self):
        """Test batch similarity computes correct similarities."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {
            'x': model.random(seed=1),
            'y': model.random(seed=2),
        }

        # Query matching 'x'
        queries = [codebook['x']]
        results = batch_similarity(queries, codebook, model)

        # Should have highest similarity with 'x'
        assert results[0]['x'] > results[0]['y']
        assert results[0]['x'] >= 0.99

    def test_batch_similarity_single_query(self):
        """Test batch similarity with single query."""
        model = VSA.create('MAP', dim=1000, seed=42)

        codebook = {'a': model.random(seed=1), 'b': model.random(seed=2)}
        queries = [model.random(seed=100)]

        results = batch_similarity(queries, codebook, model)

        assert len(results) == 1
        assert isinstance(results[0], dict)


class TestBatchSimilarityErrors:
    """Test batch_similarity error handling."""

    def test_batch_similarity_fails_empty_queries(self):
        """Test that empty queries list fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        codebook = {'a': model.random()}

        with pytest.raises(ValueError, match="queries must not be empty"):
            batch_similarity([], codebook, model)

    def test_batch_similarity_fails_empty_codebook(self):
        """Test that empty codebook fails."""
        model = VSA.create('MAP', dim=1000, seed=42)
        queries = [model.random()]

        with pytest.raises(ValueError, match="codebook must not be empty"):
            batch_similarity(queries, {}, model)

    def test_batch_similarity_fails_wrong_type(self):
        """Test that wrong types fail."""
        model = VSA.create('MAP', dim=1000, seed=42)
        codebook = {'a': model.random()}
        query = model.random()

        with pytest.raises(TypeError, match="queries must be list"):
            batch_similarity("not a list", codebook, model)

        with pytest.raises(TypeError, match="codebook must be dict"):
            batch_similarity([query], [], model)
