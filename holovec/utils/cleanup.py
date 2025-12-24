"""Cleanup and factorization strategies for VSA codebook operations.

This module provides cleanup strategies for vector symbolic architectures,
including brute-force codebook search and resonator networks for iterative
multi-factor unbinding.

Key Features:
    - Abstract CleanupStrategy interface for extensibility
    - BruteForceCleanup: Exhaustive codebook search (baseline)
    - ResonatorCleanup: Iterative factorization (10-100x speedup)
    - AttentionResonatorCleanup: Modern Hopfield attention-based cleanup
    - Support for single and multi-factor unbinding

Based on:
    Kymn et al. (2024) "Attention Mechanisms in Vector Symbolic Architectures"
    - Resonator Networks for multi-factor unbinding
    - Typical convergence: 5-15 iterations with 0.99 threshold
    - Performance: 10-100x speedup over brute-force

    Yeung et al. (2024) "Self-Attention Based Semantic Decomposition in VSAs"
    - Attention-based resonator update rule
    - Exponential memory capacity via modern Hopfield networks
    - Works with FHRR continuous vectors (traditional fails)
    - 10-100x lower complexity than traditional resonator

References:
    Paper: Kymn et al. (2024) - Attention and Resonator specifications
    Paper: Yeung et al. (2024) - Self-attention resonator networks
    Paper: Ramsauer et al. (2021) - "Hopfield Networks is All You Need"
    Related: Kanerva (2009) - Hyperdimensional computing principles

Mathematical Foundation:
    - Cleanup: Find argmax_i sim(query, codebook[i])
    - Factorization: Iteratively unbind factors until convergence
    - Convergence: similarity >= threshold or max_iterations reached
    - Attention update: x_hat = X @ softmax(β * X^H @ (s * o_hat) / D)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..backends.base import Array
from ..models.base import VSAModel

if TYPE_CHECKING:
    from collections.abc import Sequence


class CleanupStrategy(ABC):
    """Abstract base class for cleanup strategies.

    Cleanup strategies search a codebook to find the closest match(es) to
    a query hypervector. Different strategies offer trade-offs between
    speed, accuracy, and support for multi-factor unbinding.

    Implementing classes must define:
        - cleanup(): Single-factor codebook search
        - factorize(): Multi-factor iterative unbinding

    Examples:
        >>> # Create a cleanup strategy
        >>> strategy = BruteForceCleanup()
        >>>
        >>> # Single-factor cleanup
        >>> label, similarity = strategy.cleanup(query, codebook, model)
        >>> print(f"Best match: {label} (similarity: {similarity:.3f})")
        >>>
        >>> # Multi-factor factorization
        >>> labels, similarities = strategy.factorize(
        ...     query, codebook, model, n_factors=3
        ... )
        >>> print(f"Factors: {labels}")

    Attributes:
        None (abstract class)

    References:
        Kanerva (2009): Hyperdimensional Computing
        Kymn et al. (2024): Attention Mechanisms in VSAs
    """

    @abstractmethod
    def cleanup(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
    ) -> tuple[str, float]:
        """Find the best matching codebook entry for a query.

        Args:
            query: Query hypervector to clean up
            codebook: Dictionary mapping labels to hypervectors
            model: VSA model for similarity computation

        Returns:
            Tuple of (label, similarity) for the best match

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If codebook is empty

        Examples:
            >>> label, sim = strategy.cleanup(query, codebook, model)
            >>> print(f"Best: {label} with similarity {sim:.3f}")
        """
        pass

    @abstractmethod
    def factorize(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
        n_factors: int = 2,
        max_iterations: int = 20,
        threshold: float = 0.99,
    ) -> tuple[list[str], list[float]]:
        """Factorize a composition into constituent factors.

        Iteratively unbinds factors from a composite hypervector by
        finding the best match, unbinding it, and repeating.

        Args:
            query: Composite hypervector to factorize
            codebook: Dictionary mapping labels to hypervectors
            model: VSA model for bind/unbind/similarity operations
            n_factors: Number of factors to extract (default: 2)
            max_iterations: Maximum iterations per factor (default: 20)
            threshold: Convergence threshold for similarity (default: 0.99)

        Returns:
            Tuple of:
                - labels: List of factor labels in extraction order
                - similarities: List of similarities for each factor

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If n_factors < 1 or codebook is empty

        Examples:
            >>> # Factorize a 3-factor composition
            >>> labels, sims = strategy.factorize(
            ...     query, codebook, model, n_factors=3
            ... )
            >>> print(f"Factors: {labels}")
            >>> print(f"Similarities: {[f'{s:.3f}' for s in sims]}")
        """
        pass


class BruteForceCleanup(CleanupStrategy):
    """Brute-force cleanup via exhaustive codebook search.

    This is the baseline cleanup strategy that computes similarity between
    the query and every codebook entry, returning the best match. Simple
    and effective, but slow for large codebooks.

    Performance:
        - Time complexity: O(n × d) for n items, d dimensions
        - Space complexity: O(1)
        - Best for: Small codebooks (< 1000 items)

    Examples:
        >>> # Create strategy
        >>> cleanup = BruteForceCleanup()
        >>>
        >>> # Single cleanup
        >>> label, sim = cleanup.cleanup(query, codebook, model)
        >>> print(f"Found: {label}")
        >>>
        >>> # Multi-factor factorization
        >>> labels, sims = cleanup.factorize(query, codebook, model, n_factors=3)
        >>> print(f"Factors: {labels}")

    References:
        Kanerva (2009): Classic cleanup operation
    """

    def cleanup(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
    ) -> tuple[str, float]:
        """Find best match via exhaustive search.

        Computes similarity between query and every codebook entry,
        returning the label with highest similarity.

        Args:
            query: Query hypervector to clean up
            codebook: Dictionary mapping labels to hypervectors
            model: VSA model for similarity computation

        Returns:
            Tuple of (label, similarity) for the best match

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If codebook is empty

        Examples:
            >>> label, sim = cleanup.cleanup(query, codebook, model)
            >>> print(f"Best match: {label} (sim: {sim:.3f})")
        """
        # Type validation
        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")

        # Value validation
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")

        # Array shape validation (ensure query is 1-D vector matching model dimension)
        try:
            query_shape = model.backend.shape(query)
            expected_shape = (model.dimension,)
            if query_shape != expected_shape:
                raise ValueError(
                    f"query must have shape {expected_shape}, got {query_shape}. "
                    f"Ensure query is a 1-D hypervector matching model dimension."
                )
        except (AttributeError, TypeError) as e:
            raise TypeError(
                f"query must be a valid array compatible with model backend, got {type(query)}. "
                f"Backend error: {e}"
            )

        # Compute similarities for all entries
        best_label = None
        best_similarity = float('-inf')

        for label, vector in codebook.items():
            similarity = model.similarity(query, vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label

        return best_label, float(best_similarity)

    def factorize(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
        n_factors: int = 2,
        max_iterations: int = 20,
        threshold: float = 0.99,
    ) -> tuple[list[str], list[float]]:
        """Factorize via iterative cleanup and unbinding.

        Repeatedly finds the best match, unbinds it from the query,
        and continues until n_factors are extracted or convergence.

        Args:
            query: Composite hypervector to factorize
            codebook: Dictionary mapping labels to hypervectors
            model: VSA model for bind/unbind/similarity operations
            n_factors: Number of factors to extract (default: 2)
            max_iterations: Maximum iterations per factor (default: 20)
            threshold: Convergence threshold for similarity (default: 0.99)

        Returns:
            Tuple of:
                - labels: List of factor labels in extraction order
                - similarities: List of similarities for each factor

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If n_factors < 1 or codebook is empty

        Examples:
            >>> labels, sims = cleanup.factorize(
            ...     query, codebook, model, n_factors=3, threshold=0.95
            ... )
            >>> print(f"Extracted {len(labels)} factors")
        """
        # Type validation
        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")
        if not isinstance(n_factors, int):
            raise TypeError(f"n_factors must be int, got {type(n_factors)}")
        if not isinstance(max_iterations, int):
            raise TypeError(f"max_iterations must be int, got {type(max_iterations)}")
        if not isinstance(threshold, int | float):
            raise TypeError(f"threshold must be numeric, got {type(threshold)}")

        # Value validation
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        # Array shape validation (ensure query is 1-D vector matching model dimension)
        try:
            query_shape = model.backend.shape(query)
            expected_shape = (model.dimension,)
            if query_shape != expected_shape:
                raise ValueError(
                    f"query must have shape {expected_shape}, got {query_shape}. "
                    f"Ensure query is a 1-D hypervector matching model dimension."
                )
        except (AttributeError, TypeError) as e:
            raise TypeError(
                f"query must be a valid array compatible with model backend, got {type(query)}. "
                f"Backend error: {e}"
            )

        # Extract factors iteratively
        labels = []
        similarities = []
        current = query

        for _ in range(n_factors):
            # Find best match
            label, similarity = self.cleanup(current, codebook, model)
            labels.append(label)
            similarities.append(similarity)

            # Check convergence
            if similarity >= threshold:
                # High similarity - factor found
                pass

            # Unbind the found factor and continue
            factor_vector = codebook[label]
            current = model.unbind(current, factor_vector)

        return labels, similarities


class ResonatorCleanup(CleanupStrategy):
    """Resonator network cleanup via iterative refinement.

    Implements the resonator network algorithm from Kymn et al. (2024),
    which uses iterative attention mechanisms to refine factor estimates.
    Achieves 10-100x speedup over brute-force for multi-factor unbinding.

    Algorithm:
        1. Initialize estimates for all factors
        2. For each iteration:
            a. Unbind other factors to isolate target
            b. Cleanup against codebook
            c. Update estimate
        3. Repeat until convergence or max_iterations

    Performance:
        - Convergence: Typically 5-15 iterations
        - Speedup: 10-100x over brute-force
        - Best for: Multi-factor compositions (3+ factors)

    Examples:
        >>> # Create resonator cleanup
        >>> cleanup = ResonatorCleanup()
        >>>
        >>> # Single cleanup (same as brute-force)
        >>> label, sim = cleanup.cleanup(query, codebook, model)
        >>>
        >>> # Multi-factor with resonator (much faster)
        >>> labels, sims = cleanup.factorize(
        ...     query, codebook, model, n_factors=5, threshold=0.99
        ... )
        >>> print(f"Converged with {len(labels)} factors")

    Attributes:
        None (stateless)

    References:
        Kymn et al. (2024): Attention Mechanisms in VSAs
            - Section 3: Resonator Networks
            - Algorithm 1: Iterative factorization
    """

    def cleanup(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
    ) -> tuple[str, float]:
        """Find best match via exhaustive search.

        For single-factor cleanup, resonator networks reduce to brute-force
        search. Use factorize() for multi-factor speedup.

        Args:
            query: Query hypervector to clean up
            codebook: Dictionary mapping labels to hypervectors
            model: VSA model for similarity computation

        Returns:
            Tuple of (label, similarity) for the best match

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If codebook is empty

        Examples:
            >>> label, sim = cleanup.cleanup(query, codebook, model)
        """
        # For single cleanup, resonator = brute-force
        # Use the brute-force implementation
        brute_force = BruteForceCleanup()
        return brute_force.cleanup(query, codebook, model)

    def factorize(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
        n_factors: int = 2,
        max_iterations: int = 20,
        threshold: float = 0.99,
        # Refinements
        temperature: float = 20.0,
        top_k: int = 1,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = 'hard',
    ) -> tuple[list[str], list[float]]:
        """Factorize via resonator network iteration.

        Uses iterative attention to refine factor estimates simultaneously,
        achieving much faster convergence than sequential unbinding.

        Algorithm (from Kymn et al. 2024):
            1. Initialize: estimates = [random from codebook] × n_factors
            2. Repeat for max_iterations:
                a. For each factor i:
                    - Unbind all other estimates from query
                    - Cleanup result against codebook
                    - Update estimate[i]
                b. Check convergence (all similarities >= threshold)
            3. Return final estimates and similarities

        Args:
            query: Composite hypervector to factorize
            codebook: Dictionary mapping labels to hypervectors
            model: VSA model for bind/unbind/similarity operations
            n_factors: Number of factors to extract (default: 2)
            max_iterations: Maximum iterations (default: 20)
            threshold: Convergence threshold for similarity (default: 0.99)

        Returns:
            Tuple of:
                - labels: List of factor labels
                - similarities: List of similarities for each factor

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If n_factors < 1 or codebook is empty

        Examples:
            >>> # Fast multi-factor unbinding
            >>> labels, sims = cleanup.factorize(
            ...     query, codebook, model, n_factors=5
            ... )
            >>> print(f"Factors: {labels}")
            >>> print(f"Avg similarity: {sum(sims)/len(sims):.3f}")
        """
        # Type validation
        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")
        if not isinstance(n_factors, int):
            raise TypeError(f"n_factors must be int, got {type(n_factors)}")
        if not isinstance(max_iterations, int):
            raise TypeError(f"max_iterations must be int, got {type(max_iterations)}")
        if not isinstance(threshold, int | float):
            raise TypeError(f"threshold must be numeric, got {type(threshold)}")

        # Value validation
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        # Array shape validation (ensure query is 1-D vector matching model dimension)
        try:
            query_shape = model.backend.shape(query)
            expected_shape = (model.dimension,)
            if query_shape != expected_shape:
                raise ValueError(
                    f"query must have shape {expected_shape}, got {query_shape}. "
                    f"Ensure query is a 1-D hypervector matching model dimension."
                )
        except (AttributeError, TypeError) as e:
            raise TypeError(
                f"query must be a valid array compatible with model backend, got {type(query)}. "
                f"Backend error: {e}"
            )

        # Initialize estimates with deterministic codebook entries (cycle)
        codebook_labels = list(codebook.keys())
        estimates = []
        estimate_labels = []

        for i in range(n_factors):
            # Use modulo to cycle through codebook if n_factors > codebook size
            label = codebook_labels[i % len(codebook_labels)]
            estimates.append(codebook[label])
            estimate_labels.append(label)

        # Iterative refinement with optional early stopping
        best_avg = -1.0
        no_improve = 0
        for iteration in range(max_iterations):
            converged = True

            for i in range(n_factors):
                # Unbind all OTHER estimates from query to isolate factor i
                isolated = query
                for j in range(n_factors):
                    if j != i:
                        isolated = model.unbind(isolated, estimates[j])

                # Compute similarities to entire codebook
                sims: list[tuple[str, float]] = []
                for lbl, vec in codebook.items():
                    sims.append((lbl, float(model.similarity(isolated, vec))))
                # Sort by similarity desc
                sims.sort(key=lambda t: t[1], reverse=True)

                # Hard vs soft update
                use_soft = (mode == 'soft') or (top_k > 1)
                if not use_soft:
                    label, similarity = sims[0]
                    estimates[i] = codebook[label]
                    estimate_labels[i] = label
                else:
                    # Take top-K and softmax-weight them
                    k = min(max(2, top_k), len(sims))
                    top = sims[:k]
                    import numpy as _np
                    vals = _np.array([s for _, s in top], dtype=_np.float64)
                    # temperature > 0; larger → flatter
                    logits = vals * float(temperature)
                    logits = logits - logits.max()
                    w = _np.exp(logits)
                    w = w / (w.sum() + 1e-12)
                    # Bundle weighted
                    parts = []
                    for (lbl, _s), wt in zip(top, w.tolist()):
                        parts.append(model.backend.multiply_scalar(codebook[lbl], float(wt)))
                    estimates[i] = model.backend.sum(model.backend.stack(parts, axis=0), axis=0)
                    # Label: top-1 for reporting
                    estimate_labels[i] = top[0][0]
                    similarity = float(top[0][1])

                # Check convergence for this factor
                if similarity < threshold:
                    converged = False

            # Global early stopping on plateau
            # Compute avg isolated similarity across factors
            curr_sims = []
            for i in range(n_factors):
                isolated = query
                for j in range(n_factors):
                    if j != i:
                        isolated = model.unbind(isolated, estimates[j])
                curr_sims.append(float(model.similarity(isolated, estimates[i])))
            avg_sim = sum(curr_sims) / max(1, len(curr_sims))

            if avg_sim > best_avg + min_delta:
                best_avg = avg_sim
                no_improve = 0
            else:
                no_improve += 1

            if converged or no_improve >= patience:
                break

        # Compute final similarities (as in original API)
        similarities: list[float] = []
        for i in range(n_factors):
            isolated = query
            for j in range(n_factors):
                if j != i:
                    isolated = model.unbind(isolated, estimates[j])
            similarity = model.similarity(isolated, estimates[i])
            similarities.append(float(similarity))

        return estimate_labels, similarities

    def factorize_verbose(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
        n_factors: int = 2,
        max_iterations: int = 20,
        threshold: float = 0.99,
        temperature: float = 20.0,
        top_k: int = 1,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = 'hard',
    ) -> tuple[list[str], list[float], list[float]]:
        """Like factorize(), but also returns avg-similarity history per iteration."""
        # Lightweight wrapper: capture avg similarity after each iteration
        # Re-implement loop to record history.
        # Initialize estimates
        codebook_labels = list(codebook.keys())
        estimates = []
        estimate_labels = []
        for i in range(n_factors):
            label = codebook_labels[i % len(codebook_labels)]
            estimates.append(codebook[label])
            estimate_labels.append(label)

        history: list[float] = []
        best_avg = -1.0
        no_improve = 0
        for _iter in range(max_iterations):
            converged = True
            for i in range(n_factors):
                isolated = query
                for j in range(n_factors):
                    if j != i:
                        isolated = model.unbind(isolated, estimates[j])
                sims = [(lbl, float(model.similarity(isolated, vec))) for lbl, vec in codebook.items()]
                sims.sort(key=lambda t: t[1], reverse=True)
                use_soft = (mode == 'soft') or (top_k > 1)
                if not use_soft:
                    label, similarity = sims[0]
                    estimates[i] = codebook[label]
                    estimate_labels[i] = label
                else:
                    k = min(max(2, top_k), len(sims))
                    top = sims[:k]
                    import numpy as _np
                    vals = _np.array([s for _, s in top], dtype=_np.float64)
                    logits = vals * float(temperature)
                    logits = logits - logits.max()
                    w = _np.exp(logits)
                    w = w / (w.sum() + 1e-12)
                    parts = []
                    for (lbl, _s), wt in zip(top, w.tolist()):
                        parts.append(model.backend.multiply_scalar(codebook[lbl], float(wt)))
                    estimates[i] = model.backend.sum(model.backend.stack(parts, axis=0), axis=0)
                    estimate_labels[i] = top[0][0]
                    similarity = float(top[0][1])
                if similarity < threshold:
                    converged = False

            # record avg similarity across factors
            curr_sims = []
            for i in range(n_factors):
                isolated = query
                for j in range(n_factors):
                    if j != i:
                        isolated = model.unbind(isolated, estimates[j])
                curr_sims.append(float(model.similarity(isolated, estimates[i])))
            avg_sim = sum(curr_sims) / max(1, len(curr_sims))
            history.append(avg_sim)

            if avg_sim > best_avg + min_delta:
                best_avg = avg_sim
                no_improve = 0
            else:
                no_improve += 1
            if converged or no_improve >= patience:
                break

        # Final similarities
        final_sims = []
        for i in range(n_factors):
            isolated = query
            for j in range(n_factors):
                if j != i:
                    isolated = model.unbind(isolated, estimates[j])
            final_sims.append(float(model.similarity(isolated, estimates[i])))
        return estimate_labels, final_sims, history


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / (np.sum(exp_x) + 1e-12)


class AttentionResonatorCleanup(CleanupStrategy):
    """Attention-based resonator network using modern Hopfield update rule.

    This is the state-of-the-art resonator implementation based on
    Yeung et al. (2024) "Self-Attention Based Semantic Decomposition in VSAs".
    It uses the modern Hopfield network update rule with exponential memory
    capacity, providing dramatic improvements over traditional resonators.

    Key Advantages over Traditional Resonator:
        - Works with FHRR continuous vectors (traditional fails ~0% accuracy)
        - 10-100x lower complexity (iterations / success rate)
        - Exponential memory capacity vs linear
        - Much better accuracy for F > 2 factors
        - Higher robustness to cross-correlation noise

    Algorithm (from Yeung et al. 2024):
        For bipolar vectors:
            x_hat = X @ softmax(β * X^T @ (s * o_hat) / D)

        For FHRR (complex) vectors:
            x_hat = X @ softmax(β * Re[X^H @ (s * o_hat^{-1})] / D)

        Where:
            - X: codebook matrix (n × D)
            - s: composite vector to factorize
            - o_hat: product of other factor estimates (using inverses)
            - β: inverse temperature (higher = sharper, more accurate)
            - D: vector dimension

    Performance (from paper, F factors, search space M = n^F):
        | F | Traditional Accuracy | Attention Accuracy | Speedup |
        |---|---------------------|-------------------|---------|
        | 2 | ~100%               | ~100%             | 1.5x    |
        | 3 | ~30%                | ~70%              | 2x      |
        | 4 | ~20%                | ~50%              | 3x      |
        | 5 | ~15%                | ~60%              | 5x      |

    Examples:
        >>> # Create attention-based resonator
        >>> cleanup = AttentionResonatorCleanup(beta=250.0)
        >>>
        >>> # Multi-factor factorization (works with FHRR!)
        >>> labels, sims = cleanup.factorize(
        ...     query, codebook, fhrr_model, n_factors=4
        ... )
        >>> print(f"Factors: {labels}")
        >>> print(f"Avg similarity: {sum(sims)/len(sims):.3f}")

    Attributes:
        beta: Inverse temperature (default 250.0). Higher = more accurate but
              slower convergence. Recommended range: 50-500.
        max_iterations: Maximum iterations before stopping (default 100).
        convergence_threshold: Similarity threshold for convergence (default 0.99).
        patience: Early stopping patience for no improvement (default 5).

    References:
        Yeung et al. (2024): "Self-Attention Based Semantic Decomposition in VSAs"
            arXiv:2403.13218
        Ramsauer et al. (2021): "Hopfield Networks is All You Need"
            - Shows equivalence between modern Hopfield and self-attention
            - Proves exponential memory capacity
    """

    def __init__(
        self,
        beta: float = 250.0,
        max_iterations: int = 100,
        convergence_threshold: float = 0.99,
        patience: int = 5,
    ):
        """Initialize attention-based resonator.

        Args:
            beta: Inverse temperature for softmax. Higher values give sharper
                  attention (closer to argmax). Recommended: 100-500.
                  - β=50: Fast but less accurate
                  - β=250: Good balance (paper default)
                  - β=500: Most accurate but slower
            max_iterations: Maximum iterations before giving up.
            convergence_threshold: Similarity threshold for convergence.
            patience: Stop early if no improvement for this many iterations.
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if not (0.0 < convergence_threshold <= 1.0):
            raise ValueError(
                f"convergence_threshold must be in (0, 1], got {convergence_threshold}"
            )
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.beta = beta
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.patience = patience

    def cleanup(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
    ) -> tuple[str, float]:
        """Find best match via attention-weighted cleanup.

        For single-factor cleanup, uses softmax attention over codebook
        and returns the highest-weighted entry.

        Args:
            query: Query hypervector to clean up
            codebook: Dictionary mapping labels to hypervectors
            model: VSA model for similarity computation

        Returns:
            Tuple of (label, similarity) for the best match

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If codebook is empty
        """
        # Validation
        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")

        # For single cleanup, compute attention weights and return best
        labels = list(codebook.keys())
        vectors = [codebook[lbl] for lbl in labels]

        # Compute similarities
        similarities = np.array([
            float(model.similarity(query, vec)) for vec in vectors
        ])

        # Find best match
        best_idx = int(np.argmax(similarities))
        return labels[best_idx], float(similarities[best_idx])

    def factorize(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
        n_factors: int = 2,
        max_iterations: int | None = None,
        threshold: float | None = None,
    ) -> tuple[list[str], list[float]]:
        """Factorize via attention-based resonator network.

        Uses the modern Hopfield network update rule with softmax attention,
        providing exponential memory capacity and support for continuous
        (FHRR) vectors.

        The key insight from Yeung et al. (2024) is to replace the sign
        nonlinearity in traditional resonators with softmax attention:

            Traditional: x_hat = sgn(X @ X^T @ (s * o_hat))
            Attention:   x_hat = X @ softmax(β * X^T @ (s * o_hat) / D)

        For FHRR vectors, we use conjugate for unbinding and take the real
        part of the similarity:

            x_hat = X @ softmax(β * Re[X^H @ (s * o_hat^{-1})] / D)

        Args:
            query: Composite hypervector to factorize
            codebook: Dictionary mapping labels to hypervectors
            model: VSA model for bind/unbind/similarity operations
            n_factors: Number of factors to extract (default: 2)
            max_iterations: Maximum iterations (default: self.max_iterations)
            threshold: Convergence threshold (default: self.convergence_threshold)

        Returns:
            Tuple of:
                - labels: List of factor labels
                - similarities: List of similarities for each factor

        Raises:
            TypeError: If arguments are not correct types
            ValueError: If n_factors < 1 or codebook is empty
        """
        # Use instance defaults if not provided
        if max_iterations is None:
            max_iterations = self.max_iterations
        if threshold is None:
            threshold = self.convergence_threshold

        # Type validation
        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")
        if not isinstance(n_factors, int):
            raise TypeError(f"n_factors must be int, got {type(n_factors)}")

        # Value validation
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")

        # Extract codebook as matrix and labels
        labels = list(codebook.keys())
        n_items = len(labels)
        dim = model.dimension

        # Stack codebook vectors into matrix (n_items × dim)
        codebook_vectors = [codebook[lbl] for lbl in labels]
        codebook_matrix = model.backend.stack(codebook_vectors, axis=0)

        # Detect if complex (FHRR) model
        is_complex = hasattr(model.space, 'is_complex') and model.space.is_complex

        # Initialize estimates as mean of codebook (as recommended in paper)
        # This gives a neutral starting point that lets the algorithm explore
        codebook_stacked = model.backend.stack(codebook_vectors, axis=0)
        codebook_mean = model.backend.mean(codebook_stacked, axis=0)

        # For FHRR, normalize the mean to unit magnitude
        if is_complex and hasattr(model, 'normalize'):
            codebook_mean = model.normalize(codebook_mean)

        estimates = [codebook_mean for _ in range(n_factors)]

        # Track convergence
        best_avg_sim = -1.0
        no_improve_count = 0

        for iteration in range(max_iterations):
            converged = True

            for j in range(n_factors):
                # Compute product of OTHER estimates
                # For non-self-inverse models, we need to use proper inverses
                other_product = self._compute_other_product(
                    estimates, j, model, is_complex
                )

                # Unbind to get noisy estimate of factor j
                noisy_estimate = model.unbind(query, other_product)

                # Compute similarities with all codebook entries
                # For FHRR: Re[X^H @ noisy] / D
                # For bipolar: X^T @ noisy / D
                if is_complex:
                    # Complex: use conjugate transpose and take real part
                    # similarities[i] = Re[codebook[i]^* · noisy] / D
                    similarities = np.array([
                        float(np.real(
                            model.backend.sum(
                                model.backend.multiply(
                                    model.backend.conjugate(codebook_vectors[i]),
                                    noisy_estimate
                                )
                            )
                        )) / dim
                        for i in range(n_items)
                    ])
                else:
                    # Real: standard dot product
                    similarities = np.array([
                        float(model.similarity(noisy_estimate, codebook_vectors[i]))
                        for i in range(n_items)
                    ])

                # Apply softmax attention with temperature β
                # weights = softmax(β * similarities)
                attention_weights = _softmax(self.beta * similarities)

                # Update estimate as weighted combination
                # x_hat = X @ weights = Σ_i weights[i] * codebook[i]
                new_estimate = self._weighted_combination(
                    codebook_vectors, attention_weights, model
                )

                # Normalize if needed (for FHRR, maintain unit magnitude)
                if hasattr(model, 'normalize'):
                    new_estimate = model.normalize(new_estimate)

                estimates[j] = new_estimate

                # Check convergence for this factor
                best_sim = float(np.max(similarities))
                if best_sim < threshold:
                    converged = False

            # Compute average similarity across all factors for early stopping
            avg_sim = self._compute_avg_similarity(
                query, estimates, codebook_vectors, model, is_complex, dim
            )

            if avg_sim > best_avg_sim + 1e-4:
                best_avg_sim = avg_sim
                no_improve_count = 0
            else:
                no_improve_count += 1

            if converged or no_improve_count >= self.patience:
                break

        # Extract final labels and similarities
        final_labels = []
        final_similarities = []

        for j in range(n_factors):
            # Find closest codebook entry to each estimate
            if is_complex:
                similarities = np.array([
                    float(np.real(
                        model.backend.sum(
                            model.backend.multiply(
                                model.backend.conjugate(codebook_vectors[i]),
                                estimates[j]
                            )
                        )
                    )) / dim
                    for i in range(n_items)
                ])
            else:
                similarities = np.array([
                    float(model.similarity(estimates[j], codebook_vectors[i]))
                    for i in range(n_items)
                ])

            best_idx = int(np.argmax(similarities))
            final_labels.append(labels[best_idx])
            final_similarities.append(float(similarities[best_idx]))

        return final_labels, final_similarities

    def _compute_other_product(
        self,
        estimates: list[Array],
        exclude_idx: int,
        model: VSAModel,
        is_complex: bool,
    ) -> Array:
        """Compute product of all estimates except one.

        This computes the product of OTHER factor estimates (not their inverses).
        The inverse will be applied by model.unbind() when we unbind this
        product from the composite to isolate the target factor.

        Args:
            estimates: List of current factor estimates
            exclude_idx: Index to exclude from product
            model: VSA model
            is_complex: Whether model uses complex vectors (unused but kept for API)

        Returns:
            Product of all other estimates (NOT inverted - unbind handles that)
        """
        n_factors = len(estimates)

        if n_factors == 1:
            # Single factor - return identity (ones for multiplicative)
            if is_complex:
                return model.backend.ones(model.dimension, dtype=np.complex128)
            return model.backend.ones(model.dimension, dtype=np.float64)

        # Collect vectors to multiply (excluding target)
        # Note: We do NOT compute inverses here - unbind() will handle that
        others = []
        for i in range(n_factors):
            if i != exclude_idx:
                others.append(estimates[i])

        # Bind all together
        result = others[0]
        for vec in others[1:]:
            result = model.bind(result, vec)

        return result

    def _weighted_combination(
        self,
        vectors: list[Array],
        weights: np.ndarray,
        model: VSAModel,
    ) -> Array:
        """Compute weighted combination of vectors.

        Args:
            vectors: List of codebook vectors
            weights: Softmax attention weights
            model: VSA model

        Returns:
            Weighted sum: Σ_i weights[i] * vectors[i]
        """
        # Scale each vector by its weight and sum
        weighted = []
        for vec, w in zip(vectors, weights):
            scaled = model.backend.multiply_scalar(vec, float(w))
            weighted.append(scaled)

        # Sum all weighted vectors
        stacked = model.backend.stack(weighted, axis=0)
        return model.backend.sum(stacked, axis=0)

    def _compute_avg_similarity(
        self,
        query: Array,
        estimates: list[Array],
        codebook_vectors: list[Array],
        model: VSAModel,
        is_complex: bool,
        dim: int,
    ) -> float:
        """Compute average max similarity across factors.

        Used for early stopping - tracks how well we're doing overall.
        """
        n_factors = len(estimates)
        n_items = len(codebook_vectors)
        total_sim = 0.0

        for j in range(n_factors):
            if is_complex:
                similarities = [
                    float(np.real(
                        model.backend.sum(
                            model.backend.multiply(
                                model.backend.conjugate(codebook_vectors[i]),
                                estimates[j]
                            )
                        )
                    )) / dim
                    for i in range(n_items)
                ]
            else:
                similarities = [
                    float(model.similarity(estimates[j], codebook_vectors[i]))
                    for i in range(n_items)
                ]
            total_sim += max(similarities)

        return total_sim / n_factors

    def factorize_verbose(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
        n_factors: int = 2,
        max_iterations: int | None = None,
        threshold: float | None = None,
    ) -> tuple[list[str], list[float], list[float]]:
        """Like factorize(), but also returns avg-similarity history per iteration.

        Useful for debugging and analyzing convergence behavior.

        Returns:
            Tuple of:
                - labels: List of factor labels
                - similarities: List of similarities for each factor
                - history: List of avg similarities per iteration
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
        if threshold is None:
            threshold = self.convergence_threshold

        # Extract codebook
        labels_list = list(codebook.keys())
        n_items = len(labels_list)
        dim = model.dimension
        codebook_vectors = [codebook[lbl] for lbl in labels_list]

        is_complex = hasattr(model.space, 'is_complex') and model.space.is_complex

        # Initialize estimates as mean of codebook (as recommended in paper)
        codebook_stacked = model.backend.stack(codebook_vectors, axis=0)
        codebook_mean = model.backend.mean(codebook_stacked, axis=0)

        if is_complex and hasattr(model, 'normalize'):
            codebook_mean = model.normalize(codebook_mean)

        estimates = [codebook_mean for _ in range(n_factors)]

        history: list[float] = []
        best_avg_sim = -1.0
        no_improve_count = 0

        for iteration in range(max_iterations):
            converged = True

            for j in range(n_factors):
                other_product = self._compute_other_product(
                    estimates, j, model, is_complex
                )
                noisy_estimate = model.unbind(query, other_product)

                if is_complex:
                    similarities = np.array([
                        float(np.real(
                            model.backend.sum(
                                model.backend.multiply(
                                    model.backend.conjugate(codebook_vectors[i]),
                                    noisy_estimate
                                )
                            )
                        )) / dim
                        for i in range(n_items)
                    ])
                else:
                    similarities = np.array([
                        float(model.similarity(noisy_estimate, codebook_vectors[i]))
                        for i in range(n_items)
                    ])

                attention_weights = _softmax(self.beta * similarities)
                new_estimate = self._weighted_combination(
                    codebook_vectors, attention_weights, model
                )

                if hasattr(model, 'normalize'):
                    new_estimate = model.normalize(new_estimate)

                estimates[j] = new_estimate

                if float(np.max(similarities)) < threshold:
                    converged = False

            # Record history
            avg_sim = self._compute_avg_similarity(
                query, estimates, codebook_vectors, model, is_complex, dim
            )
            history.append(avg_sim)

            if avg_sim > best_avg_sim + 1e-4:
                best_avg_sim = avg_sim
                no_improve_count = 0
            else:
                no_improve_count += 1

            if converged or no_improve_count >= self.patience:
                break

        # Extract final results
        final_labels = []
        final_similarities = []

        for j in range(n_factors):
            if is_complex:
                similarities = np.array([
                    float(np.real(
                        model.backend.sum(
                            model.backend.multiply(
                                model.backend.conjugate(codebook_vectors[i]),
                                estimates[j]
                            )
                        )
                    )) / dim
                    for i in range(n_items)
                ])
            else:
                similarities = np.array([
                    float(model.similarity(estimates[j], codebook_vectors[i]))
                    for i in range(n_items)
                ])

            best_idx = int(np.argmax(similarities))
            final_labels.append(labels_list[best_idx])
            final_similarities.append(float(similarities[best_idx]))

        return final_labels, final_similarities, history
