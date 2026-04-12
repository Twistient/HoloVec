"""Brute-force cleanup strategy implementations."""

from __future__ import annotations

from ...backends.base import Array
from ...models.base import VSAModel
from .base import CleanupStrategy


class BruteForceCleanup(CleanupStrategy):
    """Brute-force cleanup via exhaustive codebook search."""

    def cleanup(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
    ) -> tuple[str, float]:
        """Find best match via exhaustive search."""
        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")

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
            ) from e

        codebook_iter = iter(codebook.items())
        best_label, best_vector = next(codebook_iter)
        best_similarity = float(model.similarity(query, best_vector))

        for label, vector in codebook_iter:
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
        max_iterations: int | None = None,
        threshold: float | None = None,
        temperature: float = 20.0,
        top_k: int = 1,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = "hard",
        **kwargs: object,
    ) -> tuple[list[str], list[float]]:
        """Factorize via iterative cleanup and unbinding."""
        if query is None:
            raise TypeError("query cannot be None")
        if not isinstance(codebook, dict):
            raise TypeError(f"codebook must be dict, got {type(codebook)}")
        if not isinstance(model, VSAModel):
            raise TypeError(f"model must be VSAModel, got {type(model)}")
        if not isinstance(n_factors, int):
            raise TypeError(f"n_factors must be int, got {type(n_factors)}")
        if max_iterations is None:
            max_iterations = 20
        if threshold is None:
            threshold = 0.99
        if not isinstance(max_iterations, int):
            raise TypeError(f"max_iterations must be int, got {type(max_iterations)}")
        if not isinstance(threshold, int | float):
            raise TypeError(f"threshold must be numeric, got {type(threshold)}")

        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if len(codebook) == 0:
            raise ValueError("codebook must not be empty")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

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
            ) from e

        labels: list[str] = []
        similarities: list[float] = []
        current = query

        for _ in range(n_factors):
            label, similarity = self.cleanup(current, codebook, model)
            labels.append(label)
            similarities.append(similarity)
            factor_vector = codebook[label]
            current = model.unbind(current, factor_vector)

        return labels, similarities
