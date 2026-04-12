"""Cleanup and factorization comparison.

Run:
    python examples/27_cleanup_strategies.py
    python examples/27_cleanup_strategies.py --smoke
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from holovec import VSA
from holovec.utils.cleanup import BruteForceCleanup, ResonatorCleanup


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
    dim = 2048 if args.smoke else 4096

    model = VSA.create("MAP", dim=dim, seed=7)
    codebook = {f"item_{i}": model.random(seed=100 + i) for i in range(6 if args.smoke else 10)}
    expected = {"item_0", "item_1", "item_2"}

    brute_force = BruteForceCleanup()
    resonator = ResonatorCleanup()

    single_label, single_similarity = brute_force.cleanup(codebook["item_1"], codebook, model)

    composite = model.bind_multiple(
        [
            codebook["item_0"],
            codebook["item_1"],
            codebook["item_2"],
        ]
    )

    brute_labels, brute_sims = brute_force.factorize(
        composite,
        codebook,
        model,
        n_factors=3,
        threshold=0.6,
    )
    resonator_labels, resonator_sims, history = resonator.factorize_verbose(
        composite,
        codebook,
        model,
        n_factors=3,
        max_iterations=10 if args.smoke else 20,
        threshold=0.6,
    )

    print("Cleanup strategies")
    print("==================")
    print(f"single-factor cleanup: {single_label} ({single_similarity:.3f})")
    print(f"brute-force iterative factorization: {list(zip(brute_labels, brute_sims, strict=True))}")
    print(f"resonator factorization: {list(zip(resonator_labels, resonator_sims, strict=True))}")
    print(f"resonator average-similarity history: {[round(value, 3) for value in history]}")
    print("note: brute-force cleanup is reliable for top-1 lookup, while resonator cleanup is the")
    print("maintained multi-factor recovery path.")

    if args.smoke:
        assert single_label == "item_1"
        assert set(resonator_labels[:3]) == expected
        assert len(history) >= 1
        print("SMOKE OK: 27_cleanup_strategies")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
