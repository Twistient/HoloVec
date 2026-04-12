"""Scalar encoder walkthrough for HoloVec.

Run:
    python examples/10_encoders_scalar.py
    python examples/10_encoders_scalar.py --smoke
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from holovec import VSA
from holovec.encoders import FractionalPowerEncoder, LevelEncoder, ThermometerEncoder


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

    fhrr_model = VSA.create("FHRR", dim=dim, seed=7)
    map_model = VSA.create("MAP", dim=dim, seed=7)

    fpe = FractionalPowerEncoder(
        fhrr_model,
        min_val=0.0,
        max_val=100.0,
        bandwidth=1.5,
        seed=1,
    )
    thermometer = ThermometerEncoder(
        map_model,
        min_val=0.0,
        max_val=100.0,
        n_bins=12 if args.smoke else 20,
        seed=2,
    )
    level = LevelEncoder(
        map_model,
        min_val=0.0,
        max_val=6.0,
        n_levels=7,
        seed=3,
    )

    fpe_close = float(fhrr_model.similarity(fpe.encode(24.0), fpe.encode(25.0)))
    fpe_far = float(fhrr_model.similarity(fpe.encode(24.0), fpe.encode(70.0)))

    thermo_close = float(map_model.similarity(thermometer.encode(20.0), thermometer.encode(25.0)))
    thermo_far = float(map_model.similarity(thermometer.encode(20.0), thermometer.encode(85.0)))

    decoded_level = level.decode(level.encode(3.0))

    print("Scalar encoders")
    print("===============")
    print(f"FPE close vs far similarity: {fpe_close:.3f} vs {fpe_far:.3f}")
    print(f"Thermometer close vs far similarity: {thermo_close:.3f} vs {thermo_far:.3f}")
    print(f"Level decode(3): {decoded_level:.1f}")
    print(f"Level is reversible: {level.is_reversible}")

    if args.smoke:
        assert fpe_close > fpe_far
        assert thermo_close > thermo_far
        assert int(round(decoded_level)) == 3
        print("SMOKE OK: 10_encoders_scalar")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
