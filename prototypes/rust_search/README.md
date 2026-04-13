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
