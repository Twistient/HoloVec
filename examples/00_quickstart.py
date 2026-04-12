"""Release-facing quickstart for HoloVec.

Run:
    python examples/00_quickstart.py
    python examples/00_quickstart.py --smoke
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from holovec import VSA
from holovec.encoders import FractionalPowerEncoder
from holovec.retrieval import Codebook, ItemStore


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

    model = VSA.create("FHRR", dim=dim, seed=7)
    temperature = FractionalPowerEncoder(
        model,
        min_val=0.0,
        max_val=100.0,
        bandwidth=1.5,
        seed=3,
    )

    role_color = model.random(seed=1)
    role_temp = model.random(seed=2)
    red = model.random(seed=10)
    blue = model.random(seed=11)
    apple = model.random(seed=12)

    record = model.bundle(
        [
            model.bind(role_color, red),
            model.bind(role_temp, temperature.encode(24.0)),
        ]
    )

    recovered_color = model.unbind(record, role_color)
    recovered_temp = model.unbind(record, role_temp)

    store = ItemStore(model).fit(
        Codebook(
            {
                "red": red,
                "blue": blue,
                "apple": apple,
            },
            backend=model.backend,
        )
    )

    top_hit = store.query(recovered_color, k=1)[0]
    close_temp = float(model.similarity(recovered_temp, temperature.encode(24.0)))
    far_temp = float(model.similarity(recovered_temp, temperature.encode(75.0)))

    print("Quickstart")
    print("==========")
    print(f"model: {model.model_name} dim={model.dimension} backend={model.backend.name}")
    print(f"bind/unbind similarity: {float(model.similarity(red, recovered_color)):.3f}")
    print(f"top retrieved color: {top_hit[0]} ({top_hit[1]:.3f})")
    print(f"temperature similarity to 24C: {close_temp:.3f}")
    print(f"temperature similarity to 75C: {far_temp:.3f}")

    if args.smoke:
        assert top_hit[0] == "red"
        assert close_temp > far_temp
        print("SMOKE OK: 00_quickstart")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
