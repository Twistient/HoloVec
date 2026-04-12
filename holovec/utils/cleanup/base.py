"""Shared interfaces for cleanup and factorization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...backends.base import Array
from ...models.base import VSAModel


class CleanupStrategy(ABC):
    """Abstract base class for cleanup strategies."""

    @abstractmethod
    def cleanup(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
    ) -> tuple[str, float]:
        """Find the best matching codebook entry for a query."""

    @abstractmethod
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
        """Factorize a composition into constituent factors."""
