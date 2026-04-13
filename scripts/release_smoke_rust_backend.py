from __future__ import annotations

import os
from pathlib import Path

from holovec import VSA
from holovec.retrieval import Codebook, ItemStore
from holovec.retrieval.rust_search import rust_search_library_path


def main() -> None:
    library_path = rust_search_library_path()
    if not library_path.exists():
        raise FileNotFoundError(f"Rust search library not found: {library_path}")

    expect_packaged = os.getenv("HOLOVEC_EXPECT_PACKAGED_RUST") == "1"
    if expect_packaged and "_native" not in library_path.parts:
        raise RuntimeError(f"Expected packaged Rust library, got: {library_path}")

    model = VSA.create("MAP", dim=256, backend="numpy", seed=7)
    items = {f"item_{index}": model.random(seed=1_000 + index) for index in range(8)}
    store = ItemStore(model, search_backend="rust").fit(Codebook(items, backend=model.backend))
    results = store.query(items["item_3"], k=3, fast=True)

    if results[0][0] != "item_3":
        raise RuntimeError(f"Unexpected top result: {results}")

    print(f"SMOKE OK: release_smoke_rust_backend ({Path(library_path).name})")


if __name__ == "__main__":
    main()
