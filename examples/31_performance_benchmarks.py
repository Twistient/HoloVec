"""Wrapper around the benchmark CLI for exploratory local runs.

Run:
    python examples/31_performance_benchmarks.py
    python examples/31_performance_benchmarks.py --smoke
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.run import main as run_benchmarks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="primitives", help="Benchmark suite to run.")
    parser.add_argument("--model", default="FHRR", help="Model to benchmark.")
    parser.add_argument("--backend", default="numpy", help="Backend to benchmark.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny fast benchmark.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "benchmark-output.json"
        benchmark_argv = [
            "--suite",
            args.suite,
            "--model",
            args.model,
            "--backend",
            args.backend,
            "--output",
            str(output),
        ]
        if args.smoke:
            benchmark_argv.append("--smoke")

        run_benchmarks(benchmark_argv)
        print(output.read_text(encoding="utf-8"))
        if args.smoke:
            print("SMOKE OK: 31_performance_benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
