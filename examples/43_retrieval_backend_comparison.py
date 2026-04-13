"""Compare ItemStore search backends on a real retrieval workload.

Run:
    python examples/43_retrieval_backend_comparison.py --model all --build-rust
    python examples/43_retrieval_backend_comparison.py --model BSDC --items 5000 --queries 200
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import numpy as np

from holovec import VSA
from holovec.retrieval import Codebook, ItemStore
from holovec.retrieval.rust_search import (
    build_rust_search_library,
    rust_search_library_path,
)

SUPPORTED_MODELS = ("MAP", "BSC", "BSDC", "BSDC-SEG")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*SUPPORTED_MODELS, "all"], default="all")
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--bundle-size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a smaller fast comparison.")
    parser.add_argument(
        "--build-rust",
        action="store_true",
        help="Build the Rust retrieval library before benchmarking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args(argv)


def default_dimension(model_name: str, smoke: bool) -> int:
    if model_name == "BSDC":
        return 5000 if smoke else 20000
    if model_name == "BSDC-SEG":
        return 240 if smoke else 400
    return 2048 if smoke else 4096


def default_model_kwargs(model_name: str, smoke: bool) -> dict[str, object]:
    if model_name == "BSDC":
        return {"sparsity": 0.02}
    if model_name == "BSDC-SEG":
        return {"segments": 12 if smoke else 20}
    return {}


def resolve_models(requested_model: str) -> list[str]:
    if requested_model == "all":
        return list(SUPPORTED_MODELS)
    return [requested_model]


def make_workload(
    model_name: str,
    backend: str,
    smoke: bool,
    *,
    dimension: int | None,
    items: int,
    queries: int,
    bundle_size: int,
) -> tuple[object, dict[str, object], list[object]]:
    dim = default_dimension(model_name, smoke) if dimension is None else dimension
    model = VSA.create(
        model_name,
        dim=dim,
        backend=backend,
        seed=7,
        **default_model_kwargs(model_name, smoke),
    )
    labels = [f"item_{index}" for index in range(items)]
    codebook = {label: model.random(seed=1_000_000 + index) for index, label in enumerate(labels)}

    rng = random.Random(17)
    query_vectors = []
    for _ in range(queries):
        chosen = rng.sample(labels, k=bundle_size)
        query_vectors.append(model.bundle([codebook[label] for label in chosen]))
    return model, codebook, query_vectors


def benchmark_store(
    model: object,
    items: dict[str, object],
    queries: list[object],
    *,
    search_backend: str,
    k: int,
) -> tuple[dict[str, float], list[list[tuple[str, float]]]]:
    fit_start = time.perf_counter()
    store = ItemStore(model, search_backend=search_backend).fit(
        Codebook(items, backend=model.backend)
    )
    fit_seconds = time.perf_counter() - fit_start

    first_start = time.perf_counter()
    first_result = store.query(queries[0], k=k, fast=True)
    first_query_seconds = time.perf_counter() - first_start

    latencies: list[float] = []
    results = [first_result]
    for query in queries[1:]:
        start = time.perf_counter()
        results.append(store.query(query, k=k, fast=True))
        latencies.append(time.perf_counter() - start)

    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1)) if ordered else 0
    steady_mean_ms = fmean(latencies) * 1000.0 if latencies else 0.0
    steady_p95_ms = ordered[p95_index] * 1000.0 if ordered else 0.0
    return (
        {
            "fit_ms": fit_seconds * 1000.0,
            "first_query_ms": first_query_seconds * 1000.0,
            "steady_query_ms": steady_mean_ms,
            "steady_p95_ms": steady_p95_ms,
        },
        results,
    )


def compare_results(
    reference: list[list[tuple[str, float]]],
    candidate: list[list[tuple[str, float]]],
) -> dict[str, float]:
    exact_order = 0
    exact_set = 0
    overlap_total = 0.0
    max_abs_score_diff = 0.0

    for reference_hits, candidate_hits in zip(reference, candidate, strict=True):
        reference_labels = [label for label, _ in reference_hits]
        candidate_labels = [label for label, _ in candidate_hits]
        if reference_labels == candidate_labels:
            exact_order += 1
        if set(reference_labels) == set(candidate_labels):
            exact_set += 1
        overlap_total += len(set(reference_labels) & set(candidate_labels)) / len(reference_labels)
        max_abs_score_diff = max(
            max_abs_score_diff,
            float(
                np.max(
                    np.abs(
                        np.asarray([score for _, score in reference_hits], dtype=np.float64)
                        - np.asarray([score for _, score in candidate_hits], dtype=np.float64)
                    )
                )
            ),
        )

    return {
        "exact_topk_order_match_rate": exact_order / len(reference),
        "exact_topk_set_match_rate": exact_set / len(reference),
        "mean_topk_overlap_rate": overlap_total / len(reference),
        "max_abs_score_diff": max_abs_score_diff,
    }


def run_model(
    model_name: str,
    backend: str,
    smoke: bool,
    *,
    dimension: int | None,
    items: int,
    queries: int,
    bundle_size: int,
    k: int,
) -> dict[str, object]:
    model, codebook, query_vectors = make_workload(
        model_name,
        backend,
        smoke,
        dimension=dimension,
        items=items,
        queries=queries,
        bundle_size=bundle_size,
    )
    numpy_metrics, numpy_results = benchmark_store(
        model,
        codebook,
        query_vectors,
        search_backend="numpy",
        k=k,
    )
    rust_metrics, rust_results = benchmark_store(
        model,
        codebook,
        query_vectors,
        search_backend="rust",
        k=k,
    )

    return {
        "model": model_name,
        "backend": backend,
        "dimension": int(model.dimension),
        "items": items,
        "queries": queries,
        "bundle_size": bundle_size,
        "k": k,
        "numpy": numpy_metrics,
        "rust": {
            **rust_metrics,
            "steady_speedup_vs_numpy": (
                numpy_metrics["steady_query_ms"] / rust_metrics["steady_query_ms"]
                if rust_metrics["steady_query_ms"] > 0.0
                else 0.0
            ),
            **compare_results(numpy_results, rust_results),
        },
    }


def print_summary(runs: list[dict[str, object]]) -> None:
    print("ItemStore backend comparison")
    print("============================")
    for run in runs:
        numpy_metrics = run["numpy"]
        rust_metrics = run["rust"]
        print(
            f"{run['model']}: dim={run['dimension']}, items={run['items']}, "
            f"queries={run['queries']}, k={run['k']}"
        )
        print(
            "  numpy:"
            f" fit={numpy_metrics['fit_ms']:.3f} ms,"
            f" first_query={numpy_metrics['first_query_ms']:.3f} ms,"
            f" steady={numpy_metrics['steady_query_ms']:.3f} ms/query"
        )
        print(
            "  rust :"
            f" fit={rust_metrics['fit_ms']:.3f} ms,"
            f" first_query={rust_metrics['first_query_ms']:.3f} ms,"
            f" steady={rust_metrics['steady_query_ms']:.3f} ms/query,"
            f" speedup={rust_metrics['steady_speedup_vs_numpy']:.2f}x"
        )
        print(
            "  parity:"
            f" order={rust_metrics['exact_topk_order_match_rate']:.3f},"
            f" set={rust_metrics['exact_topk_set_match_rate']:.3f},"
            f" overlap={rust_metrics['mean_topk_overlap_rate']:.3f},"
            f" max_score_diff={rust_metrics['max_abs_score_diff']:.3e}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.build_rust:
        build_rust_search_library(release=True)

    rust_library = rust_search_library_path()
    if not rust_library.exists():
        raise SystemExit(
            "Rust retrieval library not found. "
            "Run with --build-rust or build it manually via "
            "'cargo build --release --manifest-path prototypes/rust_search/Cargo.toml'."
        )

    items = args.items if args.items is not None else (256 if args.smoke else 5000)
    queries = args.queries if args.queries is not None else (16 if args.smoke else 200)
    bundle_size = args.bundle_size if args.bundle_size is not None else (2 if args.smoke else 4)
    k = args.k if args.k is not None else (4 if args.smoke else 8)

    runs = [
        run_model(
            model_name,
            args.backend,
            args.smoke,
            dimension=args.dim,
            items=items,
            queries=queries,
            bundle_size=bundle_size,
            k=k,
        )
        for model_name in resolve_models(args.model)
    ]

    payload = {
        "metadata": {
            "example": "43_retrieval_backend_comparison.py",
            "backend": args.backend,
            "smoke": args.smoke,
            "rust_library": str(rust_library),
        },
        "runs": runs,
    }

    print_summary(runs)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote JSON summary to {args.output}")

    if args.smoke:
        print("SMOKE OK: 43_retrieval_backend_comparison")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
