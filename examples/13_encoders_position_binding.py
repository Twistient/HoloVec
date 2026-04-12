"""Sequence encoding with PositionBindingEncoder.

Run:
    python examples/13_encoders_position_binding.py
    python examples/13_encoders_position_binding.py --smoke
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from holovec import VSA
from holovec.encoders import PositionBindingEncoder


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
    encoder = PositionBindingEncoder(model, seed=42)

    reference = ["alice", "likes", "tea"]
    prefix_extension = ["alice", "likes", "tea", "daily"]
    reordered = ["tea", "likes", "alice"]

    hv_reference = encoder.encode(reference)
    hv_prefix_extension = encoder.encode(prefix_extension)
    hv_reordered = encoder.encode(reordered)
    decoded = encoder.decode(hv_reference, max_positions=3, threshold=0.2)

    sim_prefix = float(model.similarity(hv_reference, hv_prefix_extension))
    sim_reordered = float(model.similarity(hv_reference, hv_reordered))

    print("Position binding")
    print("================")
    print(f"reference: {reference}")
    print(f"decoded:   {decoded}")
    print(f"shared-prefix similarity: {sim_prefix:.3f}")
    print(f"reordered similarity:     {sim_reordered:.3f}")

    if args.smoke:
        assert decoded == reference
        assert sim_prefix > sim_reordered
        print("SMOKE OK: 13_encoders_position_binding")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
