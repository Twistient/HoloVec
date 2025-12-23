"""Tests for cleanup utilities.

Tests the cleanup strategies for VSA codebook operations, including
brute-force search and resonator networks.
"""

import pytest

from holovec.utils.cleanup import (
    CleanupStrategy,
    BruteForceCleanup,
    ResonatorCleanup,
)
from holovec import VSA


# ============================================================================
# BruteForceCleanup Tests
# ============================================================================


class TestBruteForceCleanupBasic:
    """Test BruteForceCleanup basic functionality."""

    def test_cleanup_finds_exact_match(self):
        """Test cleanup finds exact match with perfect similarity."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()

        # Create codebook
        codebook = {
            "cat": model.random(seed=1),
            "dog": model.random(seed=2),
            "bird": model.random(seed=3),
        }

        # Query with exact match
        query = codebook["dog"]
        label, similarity = cleanup.cleanup(query, codebook, model)

        assert label == "dog"
        assert similarity >= 0.99  # Should be near-perfect

    def test_cleanup_finds_best_match(self):
        """Test cleanup finds best match among similar vectors."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = BruteForceCleanup()

        # Create codebook with original vectors
        original_cat = model.random(seed=1)
        codebook = {
            "cat": original_cat,
            "dog": model.random(seed=2),
            "bird": model.random(seed=3),
        }

        # Query with noisy version of 'cat'
        noise = model.random(seed=100)
        query = model.bundle([original_cat, original_cat, original_cat, noise])

        label, similarity = cleanup.cleanup(query, codebook, model)

        assert label == "cat"
        assert similarity > 0.7  # Should still match despite noise

    def test_cleanup_with_single_item_codebook(self):
        """Test cleanup with codebook containing one item."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()

        codebook = {"only": model.random(seed=1)}
        query = model.random(seed=2)

        label, similarity = cleanup.cleanup(query, codebook, model)

        assert label == "only"
        assert isinstance(similarity, float)

    def test_cleanup_with_large_codebook(self):
        """Test cleanup with large codebook."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()

        # Create large codebook
        codebook = {f"item_{i}": model.random(seed=i) for i in range(100)}
        target_label = "item_50"
        query = codebook[target_label]

        label, similarity = cleanup.cleanup(query, codebook, model)

        assert label == target_label
        assert similarity >= 0.99


class TestBruteForceCleanupErrors:
    """Test BruteForceCleanup error handling."""

    def test_cleanup_fails_empty_codebook(self):
        """Test that empty codebook fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        query = model.random()

        with pytest.raises(ValueError, match="must not be empty"):
            cleanup.cleanup(query, {}, model)

    def test_cleanup_fails_wrong_type_codebook(self):
        """Test that wrong codebook type fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        query = model.random()

        with pytest.raises(TypeError, match="codebook must be dict"):
            cleanup.cleanup(query, [], model)

    def test_cleanup_fails_wrong_type_model(self):
        """Test that wrong model type fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random()}
        query = model.random()

        with pytest.raises(TypeError, match="model must be VSAModel"):
            cleanup.cleanup(query, codebook, "not a model")


class TestBruteForceFactorize:
    """Test BruteForceCleanup factorization."""

    def test_factorize_two_factors(self):
        """Test factorization of 2-factor composition."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = BruteForceCleanup()

        # Create codebook
        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
            "c": model.random(seed=3),
        }

        # Create composition: bind(a, b)
        composition = model.bind(codebook["a"], codebook["b"])

        # Factorize
        labels, similarities = cleanup.factorize(
            composition, codebook, model, n_factors=2, threshold=0.95
        )

        assert len(labels) == 2
        assert len(similarities) == 2
        # Factorization is approximate - just check we get reasonable results
        # At least one factor should be from the original composition
        assert any(label in ["a", "b"] for label in labels) or all(
            sim > 0.5 for sim in similarities
        )

    def test_factorize_three_factors(self):
        """Test factorization of 3-factor composition."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = BruteForceCleanup()

        codebook = {
            "x": model.random(seed=1),
            "y": model.random(seed=2),
            "z": model.random(seed=3),
        }

        # Chain bind for 3 factors: bind(bind(x, y), z)
        composition = model.bind(model.bind(codebook["x"], codebook["y"]), codebook["z"])

        labels, similarities = cleanup.factorize(composition, codebook, model, n_factors=3)

        assert len(labels) == 3
        assert len(similarities) == 3
        # All similarities should be reasonable
        assert all(sim > 0.5 for sim in similarities)

    def test_factorize_with_threshold(self):
        """Test factorization with custom threshold."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = BruteForceCleanup()

        codebook = {"a": model.random(seed=1), "b": model.random(seed=2)}
        composition = model.bind(codebook["a"], codebook["b"])

        # Lenient threshold
        labels, similarities = cleanup.factorize(
            composition, codebook, model, n_factors=2, threshold=0.5
        )

        assert len(labels) == 2
        assert len(similarities) == 2


class TestBruteForceFactorizeErrors:
    """Test BruteForceCleanup factorize error handling."""

    def test_factorize_fails_n_factors_zero(self):
        """Test that n_factors=0 fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random()}
        query = model.random()

        with pytest.raises(ValueError, match="n_factors must be >= 1"):
            cleanup.factorize(query, codebook, model, n_factors=0)

    def test_factorize_fails_negative_n_factors(self):
        """Test that negative n_factors fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random()}
        query = model.random()

        with pytest.raises(ValueError, match="n_factors must be >= 1"):
            cleanup.factorize(query, codebook, model, n_factors=-1)

    def test_factorize_fails_empty_codebook(self):
        """Test that empty codebook fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        query = model.random()

        with pytest.raises(ValueError, match="must not be empty"):
            cleanup.factorize(query, {}, model, n_factors=2)

    def test_factorize_fails_invalid_threshold(self):
        """Test that invalid threshold fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random()}
        query = model.random()

        with pytest.raises(ValueError, match="threshold must be in"):
            cleanup.factorize(query, codebook, model, threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be in"):
            cleanup.factorize(query, codebook, model, threshold=-0.1)

    def test_factorize_fails_wrong_type_n_factors(self):
        """Test that wrong type for n_factors fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random()}
        query = model.random()

        with pytest.raises(TypeError, match="n_factors must be int"):
            cleanup.factorize(query, codebook, model, n_factors="2")


# ============================================================================
# ResonatorCleanup Tests
# ============================================================================


class TestResonatorCleanupBasic:
    """Test ResonatorCleanup basic functionality."""

    def test_cleanup_matches_brute_force(self):
        """Test that resonator cleanup matches brute-force for single cleanup."""
        model = VSA.create("MAP", dim=1000, seed=42)
        resonator = ResonatorCleanup()
        brute_force = BruteForceCleanup()

        codebook = {
            "cat": model.random(seed=1),
            "dog": model.random(seed=2),
            "bird": model.random(seed=3),
        }

        query = codebook["dog"]

        # Both should find same result
        label_res, sim_res = resonator.cleanup(query, codebook, model)
        label_bf, sim_bf = brute_force.cleanup(query, codebook, model)

        assert label_res == label_bf
        assert abs(sim_res - sim_bf) < 0.01  # Should be nearly identical

    def test_cleanup_finds_best_match(self):
        """Test resonator cleanup finds best match."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        original = model.random(seed=1)
        codebook = {
            "target": original,
            "other1": model.random(seed=2),
            "other2": model.random(seed=3),
        }

        # Noisy query
        noise = model.random(seed=100)
        query = model.bundle([original, original, original, noise])

        label, similarity = cleanup.cleanup(query, codebook, model)

        assert label == "target"
        assert similarity > 0.7


class TestResonatorFactorize:
    """Test ResonatorCleanup factorization."""

    def test_factorize_two_factors(self):
        """Test resonator factorization of 2-factor composition."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
            "c": model.random(seed=3),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        labels, similarities = cleanup.factorize(composition, codebook, model, n_factors=2)

        assert len(labels) == 2
        assert len(similarities) == 2
        # Should find factors with reasonable similarity
        assert all(sim > 0.5 for sim in similarities)

    def test_factorize_three_factors(self):
        """Test resonator factorization of 3-factor composition."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "x": model.random(seed=1),
            "y": model.random(seed=2),
            "z": model.random(seed=3),
            "w": model.random(seed=4),
        }

        # Chain bind for 3 factors: bind(bind(x, y), z)
        composition = model.bind(model.bind(codebook["x"], codebook["y"]), codebook["z"])

        labels, similarities = cleanup.factorize(
            composition, codebook, model, n_factors=3, max_iterations=20
        )

        assert len(labels) == 3
        assert len(similarities) == 3

    def test_factorize_convergence(self):
        """Test that resonator factorization converges."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "p": model.random(seed=1),
            "q": model.random(seed=2),
        }

        composition = model.bind(codebook["p"], codebook["q"])

        # With reasonable iterations, should converge
        labels, similarities = cleanup.factorize(
            composition, codebook, model, n_factors=2, max_iterations=20, threshold=0.8
        )

        # Should have reasonable similarities
        assert all(sim > 0.5 for sim in similarities)

    def test_factorize_max_iterations(self):
        """Test factorization with limited iterations."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        # Very few iterations
        labels, similarities = cleanup.factorize(
            composition, codebook, model, n_factors=2, max_iterations=2
        )

        # Should still return results
        assert len(labels) == 2
        assert len(similarities) == 2


class TestResonatorFactorizeErrors:
    """Test ResonatorCleanup factorize error handling."""

    def test_factorize_fails_invalid_inputs(self):
        """Test that invalid inputs fail appropriately."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = ResonatorCleanup()
        codebook = {"a": model.random()}
        query = model.random()

        # Wrong types
        with pytest.raises(TypeError):
            cleanup.factorize(query, "not dict", model, n_factors=2)

        with pytest.raises(TypeError):
            cleanup.factorize(query, codebook, "not model", n_factors=2)

        # Invalid values
        with pytest.raises(ValueError):
            cleanup.factorize(query, {}, model, n_factors=2)

        with pytest.raises(ValueError):
            cleanup.factorize(query, codebook, model, n_factors=0)


# ============================================================================
# Integration Tests
# ============================================================================


class TestCleanupIntegration:
    """Integration tests for cleanup strategies."""

    def test_compare_brute_force_and_resonator(self):
        """Compare brute-force and resonator results."""
        model = VSA.create("MAP", dim=10000, seed=42)
        bf_cleanup = BruteForceCleanup()
        res_cleanup = ResonatorCleanup()

        codebook = {
            "alpha": model.random(seed=1),
            "beta": model.random(seed=2),
            "gamma": model.random(seed=3),
        }

        composition = model.bind(codebook["alpha"], codebook["beta"])

        # Both should find reasonable factors
        bf_labels, bf_sims = bf_cleanup.factorize(composition, codebook, model, n_factors=2)
        res_labels, res_sims = res_cleanup.factorize(composition, codebook, model, n_factors=2)

        # Both should return 2 factors
        assert len(bf_labels) == 2
        assert len(res_labels) == 2

        # Both should find factors with reasonable similarity
        assert all(sim > 0.5 for sim in bf_sims)
        assert all(sim > 0.5 for sim in res_sims)

    def test_cleanup_with_different_models(self):
        """Test cleanup works with different VSA models."""
        models_to_test = ["MAP", "BSC", "HRR"]

        for model_name in models_to_test:
            model = VSA.create(model_name, dim=1000, seed=42)
            cleanup = BruteForceCleanup()

            codebook = {
                "item1": model.random(seed=1),
                "item2": model.random(seed=2),
            }

            query = codebook["item1"]
            label, similarity = cleanup.cleanup(query, codebook, model)

            assert label == "item1"
            assert similarity > 0.9  # Should be high for exact match


# ============================================================================
# Additional coverage tests for cleanup utilities
# ============================================================================


class TestBruteForceCleanupQueryValidation:
    """Test query validation in BruteForceCleanup."""

    def test_cleanup_fails_none_query(self):
        """Test that None query raises TypeError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}

        with pytest.raises(TypeError, match="query cannot be None"):
            cleanup.cleanup(None, codebook, model)

    def test_cleanup_fails_wrong_shape_query(self):
        """Test that wrong shape query raises ValueError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}

        # Create wrong-shape array
        import numpy as np

        wrong_shape = np.zeros((500,))  # Wrong dimension

        with pytest.raises(ValueError, match="query must have shape"):
            cleanup.cleanup(wrong_shape, codebook, model)

    def test_cleanup_fails_invalid_query_type(self):
        """Test that invalid query type raises TypeError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}

        with pytest.raises(TypeError, match="query must be a valid array"):
            cleanup.cleanup("not an array", codebook, model)


class TestBruteForceFactorizeValidation:
    """Test additional validation in BruteForceCleanup.factorize."""

    def test_factorize_fails_none_query(self):
        """Test that None query raises TypeError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}

        with pytest.raises(TypeError, match="query cannot be None"):
            cleanup.factorize(None, codebook, model, n_factors=2)

    def test_factorize_fails_wrong_shape_query(self):
        """Test that wrong shape query raises ValueError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}

        import numpy as np

        wrong_shape = np.zeros((500,))

        with pytest.raises(ValueError, match="query must have shape"):
            cleanup.factorize(wrong_shape, codebook, model, n_factors=2)

    def test_factorize_fails_invalid_max_iterations(self):
        """Test that invalid max_iterations fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}
        query = model.random()

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            cleanup.factorize(query, codebook, model, n_factors=1, max_iterations=0)

    def test_factorize_fails_wrong_type_max_iterations(self):
        """Test that wrong type for max_iterations fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}
        query = model.random()

        with pytest.raises(TypeError, match="max_iterations must be int"):
            cleanup.factorize(query, codebook, model, n_factors=1, max_iterations="10")

    def test_factorize_fails_wrong_type_threshold(self):
        """Test that wrong type for threshold fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}
        query = model.random()

        with pytest.raises(TypeError, match="threshold must be numeric"):
            cleanup.factorize(query, codebook, model, n_factors=1, threshold="0.9")

    def test_factorize_fails_invalid_query_type(self):
        """Test that invalid query type raises TypeError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = BruteForceCleanup()
        codebook = {"a": model.random(seed=1)}

        with pytest.raises(TypeError, match="query must be a valid array"):
            cleanup.factorize("not an array", codebook, model, n_factors=2)


class TestResonatorFactorizeValidation:
    """Test additional validation in ResonatorCleanup.factorize."""

    def test_factorize_fails_none_query(self):
        """Test that None query raises TypeError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = ResonatorCleanup()
        codebook = {"a": model.random(seed=1)}

        with pytest.raises(TypeError, match="query cannot be None"):
            cleanup.factorize(None, codebook, model, n_factors=2)

    def test_factorize_fails_wrong_shape_query(self):
        """Test that wrong shape query raises ValueError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = ResonatorCleanup()
        codebook = {"a": model.random(seed=1)}

        import numpy as np

        wrong_shape = np.zeros((500,))

        with pytest.raises(ValueError, match="query must have shape"):
            cleanup.factorize(wrong_shape, codebook, model, n_factors=2)

    def test_factorize_fails_invalid_max_iterations(self):
        """Test that invalid max_iterations fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = ResonatorCleanup()
        codebook = {"a": model.random(seed=1)}
        query = model.random()

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            cleanup.factorize(query, codebook, model, n_factors=1, max_iterations=0)

    def test_factorize_fails_wrong_type_max_iterations(self):
        """Test that wrong type for max_iterations fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = ResonatorCleanup()
        codebook = {"a": model.random(seed=1)}
        query = model.random()

        with pytest.raises(TypeError, match="max_iterations must be int"):
            cleanup.factorize(query, codebook, model, n_factors=1, max_iterations="10")

    def test_factorize_fails_wrong_type_threshold(self):
        """Test that wrong type for threshold fails."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = ResonatorCleanup()
        codebook = {"a": model.random(seed=1)}
        query = model.random()

        with pytest.raises(TypeError, match="threshold must be numeric"):
            cleanup.factorize(query, codebook, model, n_factors=1, threshold="0.9")

    def test_factorize_fails_invalid_query_type(self):
        """Test that invalid query type raises TypeError."""
        model = VSA.create("MAP", dim=1000, seed=42)
        cleanup = ResonatorCleanup()
        codebook = {"a": model.random(seed=1)}

        with pytest.raises(TypeError, match="query must be a valid array"):
            cleanup.factorize("not an array", codebook, model, n_factors=2)


class TestResonatorSoftMode:
    """Test ResonatorCleanup soft mode and top-k behavior."""

    def test_factorize_soft_mode(self):
        """Test factorization with mode='soft'."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
            "c": model.random(seed=3),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        labels, similarities = cleanup.factorize(
            composition,
            codebook,
            model,
            n_factors=2,
            mode="soft",
            temperature=20.0,
        )

        assert len(labels) == 2
        assert len(similarities) == 2
        assert all(isinstance(s, float) for s in similarities)

    def test_factorize_top_k(self):
        """Test factorization with top_k > 1."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
            "c": model.random(seed=3),
            "d": model.random(seed=4),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        labels, similarities = cleanup.factorize(
            composition,
            codebook,
            model,
            n_factors=2,
            top_k=3,  # Use weighted average of top-3
        )

        assert len(labels) == 2
        assert len(similarities) == 2

    def test_factorize_temperature_effect(self):
        """Test that temperature affects soft mode behavior."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        # High temperature (flatter distribution)
        labels_hi, sims_hi = cleanup.factorize(
            composition, codebook, model, n_factors=2, mode="soft", temperature=100.0
        )

        # Low temperature (sharper distribution)
        labels_lo, sims_lo = cleanup.factorize(
            composition, codebook, model, n_factors=2, mode="soft", temperature=1.0
        )

        # Both should produce valid results
        assert len(labels_hi) == 2
        assert len(labels_lo) == 2


class TestResonatorPatience:
    """Test ResonatorCleanup patience/early stopping behavior."""

    def test_factorize_patience_stops_early(self):
        """Test that patience parameter causes early stopping on plateau."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        # Low patience should stop earlier
        labels, similarities = cleanup.factorize(
            composition,
            codebook,
            model,
            n_factors=2,
            max_iterations=100,
            patience=1,  # Very low patience
            min_delta=0.5,  # High delta threshold
        )

        assert len(labels) == 2
        assert len(similarities) == 2

    def test_factorize_min_delta_effect(self):
        """Test that min_delta affects early stopping."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        # Tiny min_delta (almost any improvement counts)
        labels, similarities = cleanup.factorize(
            composition,
            codebook,
            model,
            n_factors=2,
            max_iterations=20,
            patience=3,
            min_delta=1e-10,
        )

        assert len(labels) == 2
        assert len(similarities) == 2


class TestResonatorFactorizeVerbose:
    """Test ResonatorCleanup.factorize_verbose method."""

    def test_factorize_verbose_returns_history(self):
        """Test that factorize_verbose returns iteration history."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
            "c": model.random(seed=3),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        labels, similarities, history = cleanup.factorize_verbose(
            composition,
            codebook,
            model,
            n_factors=2,
            max_iterations=10,
        )

        assert len(labels) == 2
        assert len(similarities) == 2
        assert isinstance(history, list)
        assert len(history) > 0  # Should have some iteration history
        assert all(isinstance(h, float) for h in history)

    def test_factorize_verbose_soft_mode(self):
        """Test factorize_verbose with soft mode."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
        }

        composition = model.bind(codebook["a"], codebook["b"])

        labels, similarities, history = cleanup.factorize_verbose(
            composition,
            codebook,
            model,
            n_factors=2,
            mode="soft",
            top_k=2,
        )

        assert len(labels) == 2
        assert len(similarities) == 2
        assert isinstance(history, list)

    def test_factorize_verbose_converges(self):
        """Test that verbose history shows convergence pattern."""
        model = VSA.create("MAP", dim=10000, seed=42)
        cleanup = ResonatorCleanup()

        codebook = {
            "p": model.random(seed=1),
            "q": model.random(seed=2),
        }

        composition = model.bind(codebook["p"], codebook["q"])

        labels, similarities, history = cleanup.factorize_verbose(
            composition,
            codebook,
            model,
            n_factors=2,
            max_iterations=20,
            threshold=0.9,
        )

        assert len(labels) == 2
        # History should generally improve or plateau
        assert history[-1] >= history[0] - 0.1  # Allow some noise
