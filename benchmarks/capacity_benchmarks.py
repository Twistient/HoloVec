"""Compatibility wrapper for the bundle-capacity benchmark suite."""

from __future__ import annotations

from pathlib import Path

from benchmarks.run import main as run_main


def main() -> int:
    return run_main(
        [
            "--suite",
            "bundle-capacity",
            "--model",
            "all",
            "--backend",
            "numpy",
            "--output",
            str(Path("artifacts") / "bundle-capacity.json"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
