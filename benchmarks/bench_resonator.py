"""Compatibility wrapper for the cleanup-factorization benchmark suite."""

from __future__ import annotations

from pathlib import Path

from benchmarks.run import main as run_main


def main() -> int:
    return run_main(
        [
            "--suite",
            "cleanup-factorization",
            "--model",
            "MAP",
            "--backend",
            "numpy",
            "--output",
            str(Path("artifacts") / "cleanup-factorization.json"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
