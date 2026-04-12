"""Compare the major HoloVec model families.

Run:
    python examples/02_models_comparison.py
    python examples/02_models_comparison.py --smoke
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dense_dim = 2048 if args.smoke else 4096
    sparse_dim = 5000 if args.smoke else 20000
    segment_dim = 240 if args.smoke else 400
    matrix_dim = 48 if args.smoke else 96
    vtb_dim = 128 if args.smoke else 256

    configs: list[tuple[str, int, dict[str, object]]] = [
        ("FHRR", dense_dim, {}),
        ("MAP", dense_dim, {}),
        ("HRR", dense_dim, {}),
        ("BSC", dense_dim, {}),
        ("BSDC", sparse_dim, {"sparsity": 0.02}),
        ("BSDC-SEG", segment_dim, {"segments": 12 if args.smoke else 20}),
        ("GHRR", matrix_dim, {"matrix_size": 3, "diagonality": 0.4}),
        ("VTB", vtb_dim, {"n_bases": 4, "temperature": 50.0}),
    ]

    rows: dict[str, dict[str, float | bool | str]] = {}
    for name, dim, kwargs in configs:
        model = VSA.create(name, dim=dim, seed=7, **kwargs)
        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.random(seed=3)

        recovered = model.unbind(model.bind(a, b), b)
        commute = float(model.similarity(model.bind(a, b), model.bind(b, a)))
        bundle = model.bundle([a, b, c])
        bundle_member = fmean(float(model.similarity(bundle, vec)) for vec in (a, b, c))

        rows[name] = {
            "space": model.space.space_name,
            "exact_inverse": model.is_exact_inverse,
            "self_inverse": model.is_self_inverse,
            "recovery": float(model.similarity(a, recovered)),
            "commutativity": commute,
            "bundle_member_similarity": bundle_member,
        }

    print("Model family comparison")
    print("=======================")
    print(
        f"{'Model':<9} {'Space':<16} {'Exact':<5} {'Self':<5} "
        f"{'Recover':>8} {'Commute':>8} {'Bundle':>8}"
    )
    for name, *_rest in configs:
        row = rows[name]
        print(
            f"{name:<9} {str(row['space']):<16} "
            f"{str(row['exact_inverse']):<5} {str(row['self_inverse']):<5} "
            f"{float(row['recovery']):>8.3f} {float(row['commutativity']):>8.3f} "
            f"{float(row['bundle_member_similarity']):>8.3f}"
        )

    print()
    print("Default guidance:")
    print("- FHRR is the general-purpose default.")
    print("- MAP and BSC are attractive when self-inverse algebra matters.")
    print("- GHRR and VTB are the order-sensitive options.")
    print("- BSDC and BSDC-SEG are the sparse families.")

    if args.smoke:
        assert rows["FHRR"]["recovery"] > 0.95
        assert rows["GHRR"]["commutativity"] < rows["FHRR"]["commutativity"]
        print("SMOKE OK: 02_models_comparison")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
