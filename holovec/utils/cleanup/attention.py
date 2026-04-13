"""Attention-based cleanup implementations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...backends.base import Array
from ...models.base import VSAModel
from .base import CleanupStrategy


def _softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return np.asarray(exp_x / (np.sum(exp_x) + 1e-12), dtype=np.float64)


def _uses_complex_similarity(model: VSAModel) -> bool:
    """Return whether a model uses complex- or matrix-valued similarity."""
    space_name = model.space.space_name
    return space_name == "complex" or space_name.startswith("matrix_")


class AttentionResonatorCleanup(CleanupStrategy):
    """Attention-based resonator network using the modern Hopfield update rule."""

    def __init__(
        self,
        beta: float = 250.0,
        max_iterations: int = 100,
        convergence_threshold: float = 0.99,
        patience: int = 5,
    ) -> None:
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
        """Find best match via attention-weighted cleanup."""
        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")

        labels = list(codebook.keys())
        vectors = [codebook[lbl] for lbl in labels]
        similarities = np.array([float(model.similarity(query, vec)) for vec in vectors])
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
        temperature: float = 20.0,
        top_k: int = 1,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = "hard",
        **kwargs: object,
    ) -> tuple[list[str], list[float]]:
        """Factorize via attention-based resonator network."""
        if max_iterations is None:
            max_iterations = self.max_iterations
        if threshold is None:
            threshold = self.convergence_threshold

        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")
        if not isinstance(n_factors, int):
            raise TypeError(f"n_factors must be int, got {type(n_factors)}")
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")

        labels = list(codebook.keys())
        n_items = len(labels)
        dim = model.dimension
        codebook_vectors = [codebook[lbl] for lbl in labels]
        is_complex = _uses_complex_similarity(model)

        codebook_stacked = model.backend.stack(codebook_vectors, axis=0)
        codebook_mean = model.backend.mean(codebook_stacked, axis=0)
        if is_complex and hasattr(model, "normalize"):
            codebook_mean = model.normalize(codebook_mean)

        estimates = [codebook_mean for _ in range(n_factors)]
        best_avg_sim = -1.0
        no_improve_count = 0

        for _iteration in range(max_iterations):
            converged = True

            for j in range(n_factors):
                other_product = self._compute_other_product(estimates, j, model, is_complex)
                noisy_estimate = model.unbind(query, other_product)

                if is_complex:
                    similarities = np.array(
                        [
                            float(
                                np.real(
                                    model.backend.sum(
                                        model.backend.multiply(
                                            model.backend.conjugate(codebook_vectors[i]),
                                            noisy_estimate,
                                        )
                                    )
                                )
                            )
                            / dim
                            for i in range(n_items)
                        ]
                    )
                else:
                    similarities = np.array(
                        [float(model.similarity(noisy_estimate, codebook_vectors[i])) for i in range(n_items)]
                    )

                attention_weights = _softmax(self.beta * similarities)
                new_estimate = self._weighted_combination(codebook_vectors, attention_weights, model)

                if hasattr(model, "normalize"):
                    new_estimate = model.normalize(new_estimate)

                estimates[j] = new_estimate

                best_sim = float(np.max(similarities))
                if best_sim < threshold:
                    converged = False

            avg_sim = self._compute_avg_similarity(
                query, estimates, codebook_vectors, model, is_complex, dim
            )

            if avg_sim > best_avg_sim + min_delta:
                best_avg_sim = avg_sim
                no_improve_count = 0
            else:
                no_improve_count += 1

            if converged or no_improve_count >= self.patience:
                break

        final_labels: list[str] = []
        final_similarities: list[float] = []
        for j in range(n_factors):
            if is_complex:
                similarities = np.array(
                    [
                        float(
                            np.real(
                                model.backend.sum(
                                    model.backend.multiply(
                                        model.backend.conjugate(codebook_vectors[i]),
                                        estimates[j],
                                    )
                                )
                            )
                        )
                        / dim
                        for i in range(n_items)
                    ]
                )
            else:
                similarities = np.array(
                    [float(model.similarity(estimates[j], codebook_vectors[i])) for i in range(n_items)]
                )

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
        """Compute the product of all estimates except one."""
        n_factors = len(estimates)
        if n_factors == 1:
            if is_complex:
                return model.backend.ones(model.dimension, dtype="complex128")
            return model.backend.ones(model.dimension, dtype="float64")

        others: list[Array] = []
        for i in range(n_factors):
            if i != exclude_idx:
                others.append(estimates[i])

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
        """Compute a weighted combination of vectors."""
        weighted: list[Array] = []
        for vec, w in zip(vectors, weights, strict=True):
            weighted.append(model.backend.multiply_scalar(vec, float(w)))

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
        """Compute average max similarity across factors for early stopping."""
        n_factors = len(estimates)
        n_items = len(codebook_vectors)
        total_sim = 0.0

        for j in range(n_factors):
            if is_complex:
                similarities = [
                    float(
                        np.real(
                            model.backend.sum(
                                model.backend.multiply(
                                    model.backend.conjugate(codebook_vectors[i]),
                                    estimates[j],
                                )
                            )
                        )
                    )
                    / dim
                    for i in range(n_items)
                ]
            else:
                similarities = [
                    float(model.similarity(estimates[j], codebook_vectors[i])) for i in range(n_items)
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
        """Like factorize(), but also returns average-similarity history."""
        if max_iterations is None:
            max_iterations = self.max_iterations
        if threshold is None:
            threshold = self.convergence_threshold

        labels_list = list(codebook.keys())
        n_items = len(labels_list)
        dim = model.dimension
        codebook_vectors = [codebook[lbl] for lbl in labels_list]
        is_complex = _uses_complex_similarity(model)

        codebook_stacked = model.backend.stack(codebook_vectors, axis=0)
        codebook_mean = model.backend.mean(codebook_stacked, axis=0)
        if is_complex and hasattr(model, "normalize"):
            codebook_mean = model.normalize(codebook_mean)

        estimates = [codebook_mean for _ in range(n_factors)]
        history: list[float] = []
        best_avg_sim = -1.0
        no_improve_count = 0

        for _iteration in range(max_iterations):
            converged = True

            for j in range(n_factors):
                other_product = self._compute_other_product(estimates, j, model, is_complex)
                noisy_estimate = model.unbind(query, other_product)

                if is_complex:
                    similarities = np.array(
                        [
                            float(
                                np.real(
                                    model.backend.sum(
                                        model.backend.multiply(
                                            model.backend.conjugate(codebook_vectors[i]),
                                            noisy_estimate,
                                        )
                                    )
                                )
                            )
                            / dim
                            for i in range(n_items)
                        ]
                    )
                else:
                    similarities = np.array(
                        [float(model.similarity(noisy_estimate, codebook_vectors[i])) for i in range(n_items)]
                    )

                attention_weights = _softmax(self.beta * similarities)
                new_estimate = self._weighted_combination(codebook_vectors, attention_weights, model)

                if hasattr(model, "normalize"):
                    new_estimate = model.normalize(new_estimate)

                estimates[j] = new_estimate

                if float(np.max(similarities)) < threshold:
                    converged = False

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

        final_labels: list[str] = []
        final_similarities: list[float] = []
        for j in range(n_factors):
            if is_complex:
                similarities = np.array(
                    [
                        float(
                            np.real(
                                model.backend.sum(
                                    model.backend.multiply(
                                        model.backend.conjugate(codebook_vectors[i]),
                                        estimates[j],
                                    )
                                )
                            )
                        )
                        / dim
                        for i in range(n_items)
                    ]
                )
            else:
                similarities = np.array(
                    [float(model.similarity(estimates[j], codebook_vectors[i])) for i in range(n_items)]
                )

            best_idx = int(np.argmax(similarities))
            final_labels.append(labels_list[best_idx])
            final_similarities.append(float(similarities[best_idx]))

        return final_labels, final_similarities, history
