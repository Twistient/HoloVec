"""Explore the effect of GHRR diagonality on non-commutativity.

Run:
    python examples/41_model_ghrr_diagonality.py
    python examples/41_model_ghrr_diagonality.py --smoke
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from statistics import fmean

from holovec import VSA


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a smaller fast configuration intended for automated smoke tests.",
    )
    return parser.parse_args(argv)


def noncommutativity_score(model_name: str, dim: int, trials: int, **kwargs: object) -> float:
    model = VSA.create(model_name, dim=dim, seed=7, **kwargs)
    scores: list[float] = []
    for trial in range(trials):
        a = model.random(seed=100 + 2 * trial)
        b = model.random(seed=101 + 2 * trial)
        ab = model.bind(a, b)
        ba = model.bind(b, a)
        scores.append(1.0 - float(model.similarity(ab, ba)))
    return fmean(scores)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dim = 48 if args.smoke else 96
    trials = 4 if args.smoke else 8
    alphas = [0.0, 0.5, 1.0] if args.smoke else [0.0, 0.25, 0.5, 0.75, 1.0]

    sweep: dict[float, float] = {}
    for alpha in alphas:
        sweep[alpha] = noncommutativity_score(
            "GHRR",
            dim=dim,
            trials=trials,
            matrix_size=3,
            diagonality=alpha,
        )

    exact_model = VSA.create("GHRR", dim=dim, matrix_size=3, diagonality=0.4, seed=7)
    a = exact_model.random(seed=1)
    b = exact_model.random(seed=2)
    recovery = float(exact_model.similarity(a, exact_model.unbind(exact_model.bind(a, b), b)))

    print("GHRR diagonality")
    print("================")
    for alpha in alphas:
        print(f"diagonality={alpha:>4.2f} -> non-commutativity={sweep[alpha]:.3f}")
    print(f"exact inverse recovery check: {recovery:.3f}")

    if args.smoke:
        assert sweep[0.0] > sweep[1.0]
        assert recovery > 0.95
        print("SMOKE OK: 41_model_ghrr_diagonality")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
