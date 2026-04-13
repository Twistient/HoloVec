# Rust Retrieval Prototype

This prototype isolates the prepared top-k retrieval hotspot behind a small Rust
`cdylib` loaded from Python via `ctypes`.

It currently covers the exact retrieval modes that map cleanly to a narrow FFI:

- continuous cosine search (`HRR`)
- discrete exact-match search (`MAP`, `BSC`)
- sparse overlap search (`BSDC`)
- sparse segment-pattern search (`BSDC-SEG`)

It deliberately does not yet cover `FHRR` or `GHRR`, because the first question
to answer is whether a Rust rewrite beats the fixed exact NumPy path on the
non-complex retrieval kernels.

The library now exposes this as an opt-in runtime backend for retrieval stores:

```python
from holovec.retrieval import ItemStore
from holovec.retrieval.rust_search import build_rust_search_library

build_rust_search_library(release=True)
store = ItemStore(model, search_backend="rust").fit(items)
hits = store.query(query, k=5, fast=True)
```

Production integration currently opts into Rust only for:

- discrete exact-match retrieval (`MAP`, `BSC`)
- sparse overlap retrieval (`BSDC`)
- sparse segment-pattern retrieval (`BSDC-SEG`)

Dense cosine retrieval remains on the prepared NumPy path by default because the
prototype did not beat NumPy there.

## Build

```bash
cargo build --release --manifest-path prototypes/rust_search/Cargo.toml
```

## Benchmark

```bash
uv run python -m benchmarks.prototype_retrieval \
  --model all \
  --build-rust \
  --output artifacts/rust-prototype.json
```
