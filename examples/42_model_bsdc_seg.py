"""Release-facing BSDC-SEG example.

Run:
    python examples/42_model_bsdc_seg.py
    python examples/42_model_bsdc_seg.py --smoke
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from holovec import VSA
from holovec.utils.search import find_by_segment_pattern, segment_pattern


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
    dim = 240 if args.smoke else 400
    segments = 12 if args.smoke else 20

    model = VSA.create("BSDC-SEG", dim=dim, segments=segments, seed=7)
    space = model.space
    codebook = {f"item_{i}": model.random(seed=30 + i) for i in range(8)}

    target = codebook["item_1"]
    pattern = segment_pattern(target, space)
    query_pattern = pattern[:3] + [None] * (space.segments - 3)

    exact_matches = find_by_segment_pattern(codebook, space, query_pattern, match_mode="exact")
    partial_matches = find_by_segment_pattern(
        codebook,
        space,
        query_pattern,
        match_mode="fraction",
        min_fraction=0.5,
    )

    role = codebook["item_4"]
    recovered = model.unbind(model.bind(target, role), role)
    recovery = float(model.similarity(target, recovered))

    print("BSDC-SEG")
    print("========")
    print(f"segments: {space.segments}")
    print(f"target pattern prefix: {pattern[:5]}")
    print(f"exact matches on first 3 segments: {exact_matches}")
    print(f"fractional matches on first 3 segments: {partial_matches}")
    print(f"self-inverse recovery: {recovery:.3f}")

    if args.smoke:
        assert recovery > 0.95
        assert any(label == "item_1" for label, _score in exact_matches)
        assert any(label == "item_1" for label, _score in partial_matches)
        print("SMOKE OK: 42_model_bsdc_seg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
