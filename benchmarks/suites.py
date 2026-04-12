"""Benchmark suites for HoloVec.

The benchmark philosophy is model-aware:

- do not compare unlike models with a single universal score
- report quality and speed together
- keep suite outputs flat so they can be written to JSON or CSV
"""

from __future__ import annotations

import random
import time
from statistics import fmean

from holovec import VSA
from holovec.retrieval import Codebook, ItemStore
from holovec.utils.cleanup import BruteForceCleanup, ResonatorCleanup
from holovec.utils.search import find_by_segment_pattern, segment_pattern

RowValue = str | int | float | bool
BenchmarkRow = dict[str, RowValue]

SUITE_MODELS: dict[str, tuple[str, ...]] = {
    "primitives": ("FHRR", "MAP", "HRR", "BSC", "BSDC", "BSDC-SEG", "GHRR", "VTB"),
    "bundle-capacity": ("FHRR", "MAP", "HRR", "BSC", "BSDC"),
    "approximate-unbinding": ("HRR", "VTB"),
    "cleanup-factorization": ("MAP",),
    "order-sensitivity": ("GHRR", "VTB"),
    "sparse-retrieval": ("BSDC", "BSDC-SEG"),
}


def suite_names() -> tuple[str, ...]:
    """Return the known suite names."""
    return tuple(SUITE_MODELS)


def default_dimension(model_name: str, smoke: bool) -> int:
    """Return a model-specific default dimension."""
    if model_name == "BSDC":
        return 5000 if smoke else 20000
    if model_name == "BSDC-SEG":
        return 240 if smoke else 400
    if model_name == "GHRR":
        return 48 if smoke else 96
    if model_name == "VTB":
        return 128 if smoke else 256
    return 2048 if smoke else 4096


def default_model_kwargs(model_name: str, smoke: bool) -> dict[str, object]:
    """Return model-specific factory kwargs."""
    if model_name == "BSDC":
        return {"sparsity": 0.02}
    if model_name == "BSDC-SEG":
        return {"segments": 12 if smoke else 20}
    if model_name == "GHRR":
        return {"matrix_size": 3, "diagonality": 0.4}
    if model_name == "VTB":
        return {"n_bases": 4, "temperature": 50.0}
    return {}


def create_model(
    model_name: str,
    backend: str,
    smoke: bool,
    dimension: int | None = None,
    **overrides: object,
):
    """Create a model for a benchmark suite."""
    dim = default_dimension(model_name, smoke) if dimension is None else dimension
    kwargs = default_model_kwargs(model_name, smoke)
    kwargs.update(overrides)
    return VSA.create(model_name, dim=dim, backend=backend, seed=7, **kwargs)


def make_row(
    *,
    suite: str,
    case: str,
    model: str,
    backend: str,
    dimension: int,
    metric: str,
    value: float,
    unit: str,
    notes: str = "",
) -> BenchmarkRow:
    """Create a flat benchmark row."""
    return {
        "suite": suite,
        "case": case,
        "model": model,
        "backend": backend,
        "dimension": dimension,
        "metric": metric,
        "value": value,
        "unit": unit,
        "notes": notes,
    }


def time_seconds(fn, iterations: int) -> float:
    """Measure mean wall-clock seconds for a callable."""
    for _ in range(3):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations


def primitives_suite(model_name: str, backend: str, smoke: bool, dimension: int | None) -> list[BenchmarkRow]:
    """Measure primitive ops with basic quality guardrails."""
    model = create_model(model_name, backend, smoke, dimension)
    iterations = 10 if smoke else 100

    a = model.random(seed=1)
    b = model.random(seed=2)
    c = model.random(seed=3)

    rows: list[BenchmarkRow] = []
    operations = {
        "random": lambda: model.random(seed=999),
        "bind": lambda: model.bind(a, b),
        "unbind": lambda: model.unbind(model.bind(a, b), b),
        "bundle": lambda: model.bundle([a, b, c]),
        "permute": lambda: model.permute(a, k=1),
        "similarity": lambda: model.similarity(a, b),
    }
    for name, fn in operations.items():
        rows.append(
            make_row(
                suite="primitives",
                case=name,
                model=model_name,
                backend=backend,
                dimension=model.dimension,
                metric="seconds",
                value=time_seconds(fn, iterations),
                unit="s",
                notes=f"mean over {iterations} iterations",
            )
        )

    recovery = float(model.similarity(a, model.unbind(model.bind(a, b), b)))
    bundle_member = fmean(float(model.similarity(model.bundle([a, b, c]), vec)) for vec in (a, b, c))
    commutativity = float(model.similarity(model.bind(a, b), model.bind(b, a)))

    rows.extend(
        [
            make_row(
                suite="primitives",
                case="bind-unbind",
                model=model_name,
                backend=backend,
                dimension=model.dimension,
                metric="recovery_similarity",
                value=recovery,
                unit="ratio",
                notes="single-pair bind then unbind",
            ),
            make_row(
                suite="primitives",
                case="bundle-3",
                model=model_name,
                backend=backend,
                dimension=model.dimension,
                metric="member_similarity",
                value=bundle_member,
                unit="ratio",
                notes="mean similarity from a 3-item bundle to its members",
            ),
            make_row(
                suite="primitives",
                case="bind-order",
                model=model_name,
                backend=backend,
                dimension=model.dimension,
                metric="commutativity_similarity",
                value=commutativity,
                unit="ratio",
                notes="sim(bind(a,b), bind(b,a))",
            ),
        ]
    )
    return rows


def bundle_capacity_suite(
    model_name: str,
    backend: str,
    smoke: bool,
    dimension: int | None,
) -> list[BenchmarkRow]:
    """Measure bundled-item retrieval accuracy in the Schlegel-style setup."""
    model = create_model(model_name, backend, smoke, dimension)
    item_memory_size = 64 if smoke else 256
    trials = 3 if smoke else 8
    k_values = [1, 4, 8] if smoke else [1, 4, 8, 16, 32]

    rows: list[BenchmarkRow] = []
    labels = [f"item_{index}" for index in range(item_memory_size)]
    for k in k_values:
        accuracies: list[float] = []
        timings: list[float] = []
        for trial in range(trials):
            trial_model = create_model(model_name, backend, smoke, dimension)
            items = {
                label: trial_model.random(seed=trial * 1000 + index)
                for index, label in enumerate(labels)
            }
            chosen = random.Random(trial).sample(labels, k=k)
            bundle = trial_model.bundle([items[label] for label in chosen])
            store = ItemStore(trial_model).fit(Codebook(items, backend=trial_model.backend))

            start = time.perf_counter()
            hits = store.query(bundle, k=k)
            timings.append(time.perf_counter() - start)

            retrieved = {label for label, _score in hits}
            accuracies.append(len(retrieved & set(chosen)) / k)

        case = f"k={k}"
        rows.extend(
            [
                make_row(
                    suite="bundle-capacity",
                    case=case,
                    model=model_name,
                    backend=backend,
                    dimension=model.dimension,
                    metric="retrieval_accuracy",
                    value=fmean(accuracies),
                    unit="ratio",
                    notes="fraction of bundled items returned in top-k cleanup",
                ),
                make_row(
                    suite="bundle-capacity",
                    case=case,
                    model=model_name,
                    backend=backend,
                    dimension=model.dimension,
                    metric="seconds",
                    value=fmean(timings),
                    unit="s",
                    notes="mean top-k retrieval time",
                ),
            ]
        )
    return rows


def approximate_unbinding_suite(
    model_name: str,
    backend: str,
    smoke: bool,
    dimension: int | None,
) -> list[BenchmarkRow]:
    """Measure final recovery after sequential bind/unbind chains."""
    model = create_model(model_name, backend, smoke, dimension)
    depths = [2, 4] if smoke else [2, 4, 8]
    trials = 3 if smoke else 8

    rows: list[BenchmarkRow] = []
    for depth in depths:
        similarities: list[float] = []
        timings: list[float] = []
        for trial in range(trials):
            trial_model = create_model(model_name, backend, smoke, dimension)
            original = trial_model.random(seed=trial * 1000)
            binders = [trial_model.random(seed=trial * 1000 + index + 1) for index in range(depth)]

            start = time.perf_counter()
            result = original
            for binder in binders:
                result = trial_model.bind(result, binder)
            for binder in reversed(binders):
                result = trial_model.unbind(result, binder)
            timings.append(time.perf_counter() - start)
            similarities.append(float(trial_model.similarity(original, result)))

        case = f"depth={depth}"
        rows.extend(
            [
                make_row(
                    suite="approximate-unbinding",
                    case=case,
                    model=model_name,
                    backend=backend,
                    dimension=model.dimension,
                    metric="final_recovery_similarity",
                    value=fmean(similarities),
                    unit="ratio",
                    notes="similarity to original after full bind/unbind chain",
                ),
                make_row(
                    suite="approximate-unbinding",
                    case=case,
                    model=model_name,
                    backend=backend,
                    dimension=model.dimension,
                    metric="seconds",
                    value=fmean(timings),
                    unit="s",
                    notes="end-to-end chain time",
                ),
            ]
        )
    return rows


def cleanup_factorization_suite(
    model_name: str,
    backend: str,
    smoke: bool,
    dimension: int | None,
) -> list[BenchmarkRow]:
    """Compare brute-force sequential factorization with resonator cleanup."""
    model = create_model(model_name, backend, smoke, dimension)
    codebook_size = 32 if smoke else 128
    factor_counts = [2, 3] if smoke else [2, 3, 4]
    trials = 3 if smoke else 6

    rows: list[BenchmarkRow] = []
    labels = [f"item_{index}" for index in range(codebook_size)]
    for n_factors in factor_counts:
        brute_acc: list[float] = []
        brute_time: list[float] = []
        resonator_acc: list[float] = []
        resonator_time: list[float] = []

        for trial in range(trials):
            trial_model = create_model(model_name, backend, smoke, dimension)
            codebook = {
                label: trial_model.random(seed=trial * 1000 + index)
                for index, label in enumerate(labels)
            }
            chosen = random.Random(trial).sample(labels, k=n_factors)
            composite = trial_model.bind_multiple([codebook[label] for label in chosen])

            brute_force = BruteForceCleanup()
            resonator = ResonatorCleanup()

            start = time.perf_counter()
            brute_labels, _ = brute_force.factorize(
                composite,
                codebook,
                trial_model,
                n_factors=n_factors,
                threshold=0.6,
            )
            brute_time.append(time.perf_counter() - start)
            brute_acc.append(len(set(brute_labels[:n_factors]) & set(chosen)) / n_factors)

            start = time.perf_counter()
            resonator_labels, _ = resonator.factorize(
                composite,
                codebook,
                trial_model,
                n_factors=n_factors,
                threshold=0.6,
            )
            resonator_time.append(time.perf_counter() - start)
            resonator_acc.append(len(set(resonator_labels[:n_factors]) & set(chosen)) / n_factors)

        case = f"n_factors={n_factors}"
        rows.extend(
            [
                make_row(
                    suite="cleanup-factorization",
                    case=case,
                    model=model_name,
                    backend=backend,
                    dimension=model.dimension,
                    metric="brute_force_accuracy",
                    value=fmean(brute_acc),
                    unit="ratio",
                    notes="set-overlap accuracy for iterative brute-force factorization",
                ),
                make_row(
                    suite="cleanup-factorization",
                    case=case,
                    model=model_name,
                    backend=backend,
                    dimension=model.dimension,
                    metric="brute_force_seconds",
                    value=fmean(brute_time),
                    unit="s",
                    notes="mean factorization time",
                ),
                make_row(
                    suite="cleanup-factorization",
                    case=case,
                    model=model_name,
                    backend=backend,
                    dimension=model.dimension,
                    metric="resonator_accuracy",
                    value=fmean(resonator_acc),
                    unit="ratio",
                    notes="set-overlap accuracy for resonator factorization",
                ),
                make_row(
                    suite="cleanup-factorization",
                    case=case,
                    model=model_name,
                    backend=backend,
                    dimension=model.dimension,
                    metric="resonator_seconds",
                    value=fmean(resonator_time),
                    unit="s",
                    notes="mean factorization time",
                ),
            ]
        )
    return rows


def order_sensitivity_suite(
    model_name: str,
    backend: str,
    smoke: bool,
    dimension: int | None,
) -> list[BenchmarkRow]:
    """Measure non-commutativity for order-sensitive models."""
    if model_name == "GHRR":
        dim = default_dimension(model_name, smoke) if dimension is None else dimension
        alphas = [0.0, 0.4, 1.0]
        trials = 4 if smoke else 8
        rows: list[BenchmarkRow] = []
        for alpha in alphas:
            scores: list[float] = []
            recoveries: list[float] = []
            for trial in range(trials):
                model = create_model(
                    model_name,
                    backend,
                    smoke,
                    dim,
                    matrix_size=3,
                    diagonality=alpha,
                )
                a = model.random(seed=100 + 2 * trial)
                b = model.random(seed=101 + 2 * trial)
                scores.append(1.0 - float(model.similarity(model.bind(a, b), model.bind(b, a))))
                recoveries.append(float(model.similarity(a, model.unbind(model.bind(a, b), b))))
            case = f"diagonality={alpha:.2f}"
            rows.extend(
                [
                    make_row(
                        suite="order-sensitivity",
                        case=case,
                        model=model_name,
                        backend=backend,
                        dimension=dim,
                        metric="noncommutativity",
                        value=fmean(scores),
                        unit="ratio",
                        notes="1 - sim(bind(a,b), bind(b,a))",
                    ),
                    make_row(
                        suite="order-sensitivity",
                        case=case,
                        model=model_name,
                        backend=backend,
                        dimension=dim,
                        metric="recovery_similarity",
                        value=fmean(recoveries),
                        unit="ratio",
                        notes="single bind/unbind recovery quality",
                    ),
                ]
            )
        return rows

    model = create_model(model_name, backend, smoke, dimension)
    trials = 4 if smoke else 8
    scores: list[float] = []
    recoveries: list[float] = []
    for trial in range(trials):
        a = model.random(seed=100 + 2 * trial)
        b = model.random(seed=101 + 2 * trial)
        scores.append(1.0 - float(model.similarity(model.bind(a, b), model.bind(b, a))))
        recoveries.append(float(model.similarity(a, model.unbind(model.bind(a, b), b))))
    return [
        make_row(
            suite="order-sensitivity",
            case="default",
            model=model_name,
            backend=backend,
            dimension=model.dimension,
            metric="noncommutativity",
            value=fmean(scores),
            unit="ratio",
            notes="1 - sim(bind(a,b), bind(b,a))",
        ),
        make_row(
            suite="order-sensitivity",
            case="default",
            model=model_name,
            backend=backend,
            dimension=model.dimension,
            metric="recovery_similarity",
            value=fmean(recoveries),
            unit="ratio",
            notes="single bind/unbind recovery quality",
        ),
    ]


def sparse_retrieval_suite(
    model_name: str,
    backend: str,
    smoke: bool,
    dimension: int | None,
) -> list[BenchmarkRow]:
    """Measure sparse-model retrieval with model-appropriate tasks."""
    if model_name == "BSDC":
        model = create_model(model_name, backend, smoke, dimension)
        item_count = 64 if smoke else 256
        trials = 3 if smoke else 8
        labels = [f"item_{index}" for index in range(item_count)]
        accuracies: list[float] = []
        for trial in range(trials):
            trial_model = create_model(model_name, backend, smoke, dimension)
            items = {
                label: trial_model.random(seed=trial * 1000 + index)
                for index, label in enumerate(labels)
            }
            store = ItemStore(trial_model).fit(Codebook(items, backend=trial_model.backend))
            target_label = labels[trial % len(labels)]
            noisy_query = trial_model.bundle(
                [
                    items[target_label],
                    items[target_label],
                    items[target_label],
                    trial_model.random(seed=999 + trial),
                ]
            )
            top_label = store.query(noisy_query, k=1)[0][0]
            accuracies.append(float(top_label == target_label))

        return [
            make_row(
                suite="sparse-retrieval",
                case="noisy-top1",
                model=model_name,
                backend=backend,
                dimension=model.dimension,
                metric="top1_accuracy",
                value=fmean(accuracies),
                unit="ratio",
                notes="top-1 recovery from sparse noisy queries",
            )
        ]

    model = create_model(model_name, backend, smoke, dimension)
    trials = 3 if smoke else 8
    exact_hits: list[float] = []
    partial_hits: list[float] = []
    recoveries: list[float] = []

    for trial in range(trials):
        trial_model = create_model(model_name, backend, smoke, dimension)
        space = trial_model.space
        codebook = {f"item_{index}": trial_model.random(seed=trial * 1000 + index) for index in range(16)}
        target_label = "item_1"
        target = codebook[target_label]
        pattern = segment_pattern(target, space)
        query_pattern = pattern[:3] + [None] * (space.segments - 3)

        exact = find_by_segment_pattern(codebook, space, query_pattern, match_mode="exact")
        partial = find_by_segment_pattern(
            codebook,
            space,
            query_pattern,
            match_mode="fraction",
            min_fraction=0.5,
        )
        exact_hits.append(float(any(label == target_label for label, _score in exact)))
        partial_hits.append(float(any(label == target_label for label, _score in partial)))

        role = codebook["item_2"]
        recovered = trial_model.unbind(trial_model.bind(target, role), role)
        recoveries.append(float(trial_model.similarity(target, recovered)))

    return [
        make_row(
            suite="sparse-retrieval",
            case="segment-exact",
            model=model_name,
            backend=backend,
            dimension=model.dimension,
            metric="hit_rate",
            value=fmean(exact_hits),
            unit="ratio",
            notes="target appears in exact segment-pattern matches",
        ),
        make_row(
            suite="sparse-retrieval",
            case="segment-fraction",
            model=model_name,
            backend=backend,
            dimension=model.dimension,
            metric="hit_rate",
            value=fmean(partial_hits),
            unit="ratio",
            notes="target appears in fractional segment-pattern matches",
        ),
        make_row(
            suite="sparse-retrieval",
            case="bind-unbind",
            model=model_name,
            backend=backend,
            dimension=model.dimension,
            metric="recovery_similarity",
            value=fmean(recoveries),
            unit="ratio",
            notes="self-inverse recovery on segment-sparse bindings",
        ),
    ]


SUITE_FUNCS = {
    "primitives": primitives_suite,
    "bundle-capacity": bundle_capacity_suite,
    "approximate-unbinding": approximate_unbinding_suite,
    "cleanup-factorization": cleanup_factorization_suite,
    "order-sensitivity": order_sensitivity_suite,
    "sparse-retrieval": sparse_retrieval_suite,
}


def run_suite(
    suite_name: str,
    model_name: str,
    backend: str,
    smoke: bool,
    dimension: int | None = None,
) -> list[BenchmarkRow]:
    """Run a single benchmark suite for a model."""
    if suite_name not in SUITE_FUNCS:
        raise ValueError(f"Unknown suite {suite_name!r}")
    return SUITE_FUNCS[suite_name](model_name, backend, smoke, dimension)
