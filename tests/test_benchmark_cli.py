from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "primitives.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.run",
            "--suite",
            "primitives",
            "--model",
            "FHRR",
            "--backend",
            "numpy",
            "--smoke",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    rows = data["rows"]
    assert data["metadata"]["suite"] == "primitives"
    assert any(row["metric"] == "seconds" for row in rows)
    assert any(row["metric"] == "recovery_similarity" for row in rows)


def test_benchmark_cli_writes_csv(tmp_path: Path) -> None:
    output = tmp_path / "order.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.run",
            "--suite",
            "order-sensitivity",
            "--model",
            "GHRR",
            "--backend",
            "numpy",
            "--smoke",
            "--format",
            "csv",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert any(row["metric"] == "noncommutativity" for row in rows)
