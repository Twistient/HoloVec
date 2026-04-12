"""CLI runner for literature-informed HoloVec benchmark suites."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from .suites import SUITE_MODELS, BenchmarkRow, run_suite, suite_names

JSONFormat = dict[str, object]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=[*suite_names(), "all"],
        required=True,
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Model to benchmark, or 'all' for the suite default model set.",
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        help="Backend name passed to VSA.create(). Default: numpy.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Override the suite's default dimension for the selected model.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small fast configuration intended for CI smoke tests.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format. Default: json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path for JSON or CSV results.",
    )
    return parser.parse_args(argv)


def resolve_models(suite: str, requested_model: str) -> list[str]:
    """Resolve which models a suite should run."""
    if suite == "all":
        if requested_model != "all":
            raise ValueError("--model must be 'all' when --suite=all")
        all_models: list[str] = []
        for suite_models in SUITE_MODELS.values():
            for model in suite_models:
                if model not in all_models:
                    all_models.append(model)
        return all_models

    suite_models = list(SUITE_MODELS[suite])
    if requested_model == "all":
        return suite_models
    if requested_model not in suite_models:
        raise ValueError(
            f"Model {requested_model!r} is not supported by suite {suite!r}. "
            f"Expected one of: {suite_models}"
        )
    return [requested_model]


def write_csv(output: Path, rows: list[BenchmarkRow]) -> None:
    """Write benchmark rows as CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite",
        "case",
        "model",
        "backend",
        "dimension",
        "metric",
        "value",
        "unit",
        "notes",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    output: Path,
    *,
    suite: str,
    model: str,
    backend: str,
    smoke: bool,
    rows: list[BenchmarkRow],
) -> None:
    """Write benchmark rows as JSON."""
    payload: JSONFormat = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "suite": suite,
            "model": model,
            "backend": backend,
            "smoke": smoke,
            "row_count": len(rows),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    suite_list = list(suite_names()) if args.suite == "all" else [args.suite]
    rows: list[BenchmarkRow] = []
    for suite in suite_list:
        models = resolve_models(suite, args.model)
        for model in models:
            rows.extend(run_suite(suite, model, args.backend, args.smoke, args.dim))

    if args.format == "csv":
        write_csv(args.output, rows)
    else:
        write_json(
            args.output,
            suite=args.suite,
            model=args.model,
            backend=args.backend,
            smoke=cast(bool, args.smoke),
            rows=rows,
        )

    print(f"Wrote {len(rows)} benchmark rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
