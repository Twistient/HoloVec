"""Benchmark the Rust retrieval prototype against exact Python baselines."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import TypeVar

from holovec.backends.base import Array
from holovec.models.base import VSAModel
from holovec.utils.search import (
    PreparedSearchIndex,
    nearest_neighbors,
    prepare_search_index,
    query_prepared_index,
)

from .rust_search import (
    RustPreparedIndex,
    build_rust_search_library,
    prepare_rust_search_from_index,
)
from .suites import create_model

SUPPORTED_MODELS = ("MAP", "BSC", "HRR", "BSDC", "BSDC-SEG")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*SUPPORTED_MODELS, "all"], default="all")
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--bundle-size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--build-rust", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def default_items(smoke: bool) -> int:
    return 512 if smoke else 5000


def default_queries(smoke: bool) -> int:
    return 32 if smoke else 200


def default_bundle_size(smoke: bool) -> int:
    return 2 if smoke else 4


def default_k(smoke: bool) -> int:
    return 4 if smoke else 8


def model_overrides(model_name: str, smoke: bool) -> dict[str, object]:
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
) -> tuple[VSAModel, dict[str, Array], list[Array]]:
    model = create_model(
        model_name,
        backend,
        smoke,
        dimension=dimension,
        **model_overrides(model_name, smoke),
    )
    labels = [f"item_{index}" for index in range(items)]
    codebook = {label: model.random(seed=1_000_000 + index) for index, label in enumerate(labels)}
    rng = random.Random(17)
    query_vectors = []
    for _ in range(queries):
        chosen = rng.sample(labels, k=bundle_size)
        query_vectors.append(model.bundle([codebook[label] for label in chosen]))
    return model, codebook, query_vectors


Runner = TypeVar("Runner")


def benchmark_runner(
    build: Callable[[], Runner],
    run_query: Callable[[Runner, Array], tuple[list[str], list[float]]],
    queries: list[Array],
) -> tuple[dict[str, float], list[tuple[list[str], list[float]]]]:
    build_start = time.perf_counter()
    runner = build()
    build_seconds = time.perf_counter() - build_start

    for query in queries[: min(3, len(queries))]:
        run_query(runner, query)

    latencies: list[float] = []
    results: list[tuple[list[str], list[float]]] = []
    for query in queries:
        start = time.perf_counter()
        results.append(run_query(runner, query))
        latencies.append(time.perf_counter() - start)

    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return (
        {
            "build_seconds": build_seconds,
            "mean_query_ms": fmean(latencies) * 1000.0,
            "p95_query_ms": ordered[p95_index] * 1000.0,
            "total_query_seconds": sum(latencies),
        },
        results,
    )


def compare_results(
    reference: list[tuple[list[str], list[float]]],
    candidate: list[tuple[list[str], list[float]]],
) -> dict[str, float]:
    exact_topk_order = 0
    exact_topk_set = 0
    overlap_total = 0.0
    max_abs_score_diff = 0.0
    for (ref_labels, ref_scores), (cand_labels, cand_scores) in zip(
        reference, candidate, strict=True
    ):
        if ref_labels == cand_labels:
            exact_topk_order += 1
        if set(ref_labels) == set(cand_labels):
            exact_topk_set += 1
        overlap_total += len(set(ref_labels) & set(cand_labels)) / len(ref_labels)
        if ref_scores and cand_scores:
            max_abs_score_diff = max(
                max_abs_score_diff,
                max(abs(left - right) for left, right in zip(ref_scores, cand_scores, strict=True)),
            )
    return {
        "exact_topk_order_match_rate": exact_topk_order / len(reference),
        "exact_topk_set_match_rate": exact_topk_set / len(reference),
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
    prepared_index = prepare_search_index(codebook, model)

    def scalar_runner(_unused: None, query: Array) -> tuple[list[str], list[float]]:
        labels, scores = nearest_neighbors(query, codebook, model, k=k, return_similarities=True)
        if scores is None:
            raise RuntimeError("scalar benchmark expected similarity scores")
        return labels, scores

    def prepared_runner(index: PreparedSearchIndex, query: Array) -> tuple[list[str], list[float]]:
        labels, scores = query_prepared_index(
            query,
            index,
            model,
            k=k,
            return_similarities=True,
        )
        if scores is None:
            raise RuntimeError("prepared benchmark expected similarity scores")
        return labels, scores

    def rust_runner(index: RustPreparedIndex, query: Array) -> tuple[list[str], list[float]]:
        return index.query(query, k=k)

    scalar_metrics, scalar_results = benchmark_runner(lambda: None, scalar_runner, query_vectors)
    prepared_metrics, prepared_results = benchmark_runner(
        lambda: prepare_search_index(codebook, model),
        prepared_runner,
        query_vectors,
    )
    rust_metrics, rust_results = benchmark_runner(
        lambda: prepare_rust_search_from_index(prepare_search_index(codebook, model), model),
        rust_runner,
        query_vectors,
    )

    return {
        "model": model_name,
        "backend": backend,
        "dimension": int(model.dimension),
        "prepared_mode": prepared_index.mode,
        "items": items,
        "queries": queries,
        "bundle_size": bundle_size,
        "k": k,
        "engines": {
            "scalar": {
                **scalar_metrics,
                "speedup_vs_scalar": 1.0,
            },
            "prepared_numpy": {
                **prepared_metrics,
                "speedup_vs_scalar": scalar_metrics["mean_query_ms"]
                / prepared_metrics["mean_query_ms"],
                **compare_results(scalar_results, prepared_results),
            },
            "rust": {
                **rust_metrics,
                "speedup_vs_scalar": scalar_metrics["mean_query_ms"]
                / rust_metrics["mean_query_ms"],
                "speedup_vs_prepared_numpy": prepared_metrics["mean_query_ms"]
                / rust_metrics["mean_query_ms"],
                **compare_results(scalar_results, rust_results),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    items = args.items if args.items is not None else default_items(args.smoke)
    queries = args.queries if args.queries is not None else default_queries(args.smoke)
    bundle_size = (
        args.bundle_size if args.bundle_size is not None else default_bundle_size(args.smoke)
    )
    k = args.k if args.k is not None else default_k(args.smoke)

    if args.build_rust:
        build_rust_search_library(release=True)

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
            "suite": "rust-retrieval-prototype",
            "backend": args.backend,
            "smoke": args.smoke,
        },
        "runs": runs,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote prototype benchmark results to {args.output}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
