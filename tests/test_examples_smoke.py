from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    "00_quickstart.py",
    "02_models_comparison.py",
    "10_encoders_scalar.py",
    "13_encoders_position_binding.py",
    "26_retrieval_basics.py",
    "27_cleanup_strategies.py",
    "41_model_ghrr_diagonality.py",
    "42_model_bsdc_seg.py",
]


@pytest.mark.parametrize("filename", EXAMPLES)
def test_canonical_example_smoke(filename: str) -> None:
    path = ROOT / "examples" / filename
    result = subprocess.run(
        [sys.executable, str(path), "--smoke"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"{filename} failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"SMOKE OK: {filename.removesuffix('.py')}" in result.stdout
