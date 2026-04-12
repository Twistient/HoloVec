# HoloVec Benchmarks

The benchmark runner is:

```bash
python -m benchmarks.run --suite <suite> --model <model|all> --output <path>
```

The benchmark philosophy is model-aware:

- do not compare unlike models with a single universal score
- report quality and speed together
- prefer literature-motivated workloads over isolated microbenchmarks
- keep CI on smoke-sized workloads only

## Suites

| Suite | What it measures | Default models |
|-------|------------------|----------------|
| `primitives` | timing for `random`, `bind`, `unbind`, `bundle`, `permute`, `similarity`, plus recovery/commutativity guardrails | all models |
| `bundle-capacity` | bundled-item cleanup accuracy with top-k retrieval | `FHRR`, `MAP`, `HRR`, `BSC`, `BSDC` |
| `approximate-unbinding` | sequential bind/unbind recovery for approximate-inverse models | `HRR`, `VTB` |
| `cleanup-factorization` | brute-force vs resonator factorization quality and time | `MAP` |
| `order-sensitivity` | non-commutativity and recovery for order-sensitive models | `GHRR`, `VTB` |
| `sparse-retrieval` | sparse noisy retrieval or segment-pattern hit rate | `BSDC`, `BSDC-SEG` |

## Examples

```bash
python -m benchmarks.run \
  --suite primitives \
  --model FHRR \
  --backend numpy \
  --smoke \
  --output artifacts/primitives-fhrr.json

python -m benchmarks.run \
  --suite order-sensitivity \
  --model GHRR \
  --format csv \
  --output artifacts/ghrr-order.csv
```

## Notes

- `--smoke` is for CI and quick local checks, not publishable measurements.
- JSON is best for tooling and release artifacts.
- CSV is convenient for spreadsheets and docs tables.
- Use the methodology page in `docs/guides/benchmarks.md` before drawing cross-model conclusions.
