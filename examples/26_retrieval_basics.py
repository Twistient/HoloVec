"""Release-facing retrieval walkthrough.

Run:
    python examples/26_retrieval_basics.py
    python examples/26_retrieval_basics.py --smoke
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from holovec import VSA
from holovec.retrieval import Codebook, ItemStore
from holovec.utils.search import threshold_search


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
    items = {
        "apple": model.random(seed=10),
        "banana": model.random(seed=11),
        "pear": model.random(seed=12),
        "cherry": model.random(seed=13),
    }

    codebook = Codebook(items, backend=model.backend)
    store = ItemStore(model).fit(codebook)

    noisy_apple = model.bundle([items["apple"], items["apple"], items["apple"], model.random(seed=99)])
    top_hits = store.query(noisy_apple, k=2)
    threshold_labels, threshold_sims = threshold_search(
        noisy_apple,
        dict(codebook.items()),
        model,
        threshold=0.2,
    )

    pair = model.bind(items["banana"], items["pear"])
    factor_labels, factor_sims = store.factorize(pair, n_factors=2)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "fruit-codebook.npz"
        codebook.save(str(path))
        restored = Codebook.load(str(path), backend=model.backend)
        restored_store = ItemStore(model).fit(restored)
        restored_top = restored_store.query(items["pear"], k=1)[0]

    print("Retrieval basics")
    print("================")
    print(f"top hits for noisy apple: {top_hits}")
    print(f"threshold hits: {list(zip(threshold_labels, threshold_sims or [], strict=True))}")
    print(f"factorized labels: {factor_labels}")
    print(f"factorized similarities: {[f'{sim:.3f}' for sim in factor_sims]}")
    print(f"restored codebook top hit for pear: {restored_top}")

    if args.smoke:
        assert top_hits[0][0] == "apple"
        assert set(factor_labels[:2]) == {"banana", "pear"}
        assert restored_top[0] == "pear"
        print("SMOKE OK: 26_retrieval_basics")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
