"""Traditional resonator cleanup implementations."""

from __future__ import annotations

from ...backends.base import Array
from ...models.base import VSAModel
from .base import CleanupStrategy
from .bruteforce import BruteForceCleanup


class ResonatorCleanup(CleanupStrategy):
    """Resonator network cleanup via iterative refinement."""

    def cleanup(
        self,
        query: Array,
        codebook: dict[str, Array],
        model: VSAModel,
    ) -> tuple[str, float]:
        """Single-factor cleanup reduces to brute force."""
        return BruteForceCleanup().cleanup(query, codebook, model)

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
        """Factorize via resonator network iteration."""
        if max_iterations is None:
            max_iterations = 20
        if threshold is None:
            threshold = 0.99

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

        codebook_labels = list(codebook.keys())
        estimates: list[Array] = []
        estimate_labels: list[str] = []
        for i in range(n_factors):
            label = codebook_labels[i % len(codebook_labels)]
            estimates.append(codebook[label])
            estimate_labels.append(label)

        best_avg = -1.0
        no_improve = 0
        for _iteration in range(max_iterations):
            converged = True

            for i in range(n_factors):
                isolated = query
                for j in range(n_factors):
                    if j != i:
                        isolated = model.unbind(isolated, estimates[j])

                sims: list[tuple[str, float]] = []
                for lbl, vec in codebook.items():
                    sims.append((lbl, float(model.similarity(isolated, vec))))
                sims.sort(key=lambda t: t[1], reverse=True)

                use_soft = (mode == "soft") or (top_k > 1)
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
                    for (lbl, _score), wt in zip(top, w.tolist(), strict=True):
                        parts.append(model.backend.multiply_scalar(codebook[lbl], float(wt)))
                    estimates[i] = model.backend.sum(model.backend.stack(parts, axis=0), axis=0)
                    estimate_labels[i] = top[0][0]
                    similarity = float(top[0][1])

                if similarity < threshold:
                    converged = False

            curr_sims: list[float] = []
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

        similarities: list[float] = []
        for i in range(n_factors):
            isolated = query
            for j in range(n_factors):
                if j != i:
                    isolated = model.unbind(isolated, estimates[j])
            similarities.append(float(model.similarity(isolated, estimates[i])))

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
        mode: str = "hard",
    ) -> tuple[list[str], list[float], list[float]]:
        """Like factorize(), but also returns average-similarity history."""
        codebook_labels = list(codebook.keys())
        estimates: list[Array] = []
        estimate_labels: list[str] = []
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
                use_soft = (mode == "soft") or (top_k > 1)
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
                    for (lbl, _score), wt in zip(top, w.tolist(), strict=True):
                        parts.append(model.backend.multiply_scalar(codebook[lbl], float(wt)))
                    estimates[i] = model.backend.sum(model.backend.stack(parts, axis=0), axis=0)
                    estimate_labels[i] = top[0][0]
                    similarity = float(top[0][1])
                if similarity < threshold:
                    converged = False

            curr_sims: list[float] = []
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

        final_sims: list[float] = []
        for i in range(n_factors):
            isolated = query
            for j in range(n_factors):
                if j != i:
                    isolated = model.unbind(isolated, estimates[j])
            final_sims.append(float(model.similarity(isolated, estimates[i])))
        return estimate_labels, final_sims, history
