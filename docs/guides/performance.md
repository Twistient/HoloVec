Guidance for measuring and improving HoloVec workloads without hard-coding benchmark claims that
may not match your dimension, backend, or retrieval task.

## Do Not Optimize Against a Single Toy Metric

For VSA systems, "performance" is a combination of:

- latency and throughput
- memory footprint
- retrieval quality
- cleanup or factorization quality
- robustness under bundling load, noise, or distractors

That means `bind()` timing alone is not enough. A slower model can still be the better system if it
reduces cleanup error, supports the right algebra, or holds more structure at the same dimension.

## Benchmark the Workload You Actually Care About

Measure separate workloads instead of collapsing everything into one score:

- primitive ops: `random`, `bind`, `unbind`, `bundle`, `permute`, `similarity`
- retrieval: nearest neighbor, threshold retrieval, codebook sweeps
- structure recovery: role-filler unbinding, factorization, sequence cleanup
- task quality: classification accuracy, reconstruction accuracy, factor recovery, noise tolerance

This matters because model families behave differently:

- exact-inverse models should be judged on exact recovery and compositional fidelity
- approximate-inverse models should be judged on cleanup quality and graceful degradation
- sparse models need overlap or segment-aware retrieval tests
- non-commutative models need order-sensitive workloads, not just symmetric binding tests

## Backend Selection

| Situation | Recommended Backend | Why |
|-----------|---------------------|-----|
| development and release gating | NumPy | simplest environment, lowest setup cost |
| GPU-backed batched workloads | PyTorch | accelerator support and familiar tensor ecosystem |
| repeated compiled workloads | JAX | JIT can help when the same computation shape repeats |

NumPy remains the release-blocking backend. Treat PyTorch and JAX as environment-dependent support
paths unless you are explicitly testing them in your deployment environment.

## Dimension and Model Tradeoffs

Use dimension as a budget, not a vanity metric.

- lower dimensions are useful for prototyping and smoke tests
- mid-range dense dimensions are usually the practical starting point for production
- large dimensions help only if the retrieval task or bundling load justifies them
- sparse models trade arithmetic simplicity for memory efficiency and task-specific retrieval

The right dimension depends on:

- number of bundled items
- amount of structured composition
- cleanup strategy
- distractor count
- desired false-positive and false-negative rates

## Avoid Invalid Cross-Model Comparisons

Do not treat all models as if they should win the same benchmark:

- `GHRR` and `VTB` should be tested on order-sensitive or nested structure tasks
- `BSDC` and `BSDC-SEG` should be tested with sparse retrieval or segment-pattern workloads
- `HRR` should not be judged by exact-inverse expectations
- `MAP`, `BSC`, and `BSDC-SEG` are especially relevant for self-inverse cleanup or hardware-flavored
  use cases

If a benchmark ignores the model's intended algebra, the result is mostly noise.

## Batch and Vectorized Paths

Prefer codebook-level or backend-native operations whenever they exist.

```python
from holovec.retrieval import Codebook, ItemStore

store = ItemStore(model).fit(Codebook(items, backend=model.backend))
top_hits = store.query(query, k=10)
```

This is usually the right baseline before building a custom batched pipeline.

## Optional Rust Retrieval Backend

For discrete and sparse retrieval-heavy workloads, HoloVec now exposes an opt-in
Rust-backed prepared search path on `ItemStore` and `AssocStore`.

```python
from holovec.retrieval import Codebook, ItemStore
from holovec.retrieval.rust_search import build_rust_search_library

build_rust_search_library(release=True)
store = ItemStore(model, search_backend="rust").fit(Codebook(items, backend=model.backend))
top_hits = store.query(query, k=10, fast=True)
```

This is intentionally opt-in. Exact prepared NumPy remains the default backend,
and unsupported or unavailable native paths fall back to NumPy.

If you want a reviewer-facing comparison on the public API surface, run:

```bash
uv run python examples/43_retrieval_backend_comparison.py --model all --build-rust
```

That script compares `ItemStore(search_backend="numpy")` and
`ItemStore(search_backend="rust")` directly, including first-query index build
cost and steady-state query latency.

## Measure Locally

Use a tiny harness that warms up the path you care about:

```python
import time


def benchmark(fn, iterations=100):
    for _ in range(10):
        fn()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations


seconds = benchmark(lambda: model.bind(a, b))
print(f"bind(): {seconds * 1_000:.3f} ms")
```

For JAX, account for compile warmup separately from steady-state timing.

## Memory Guidance

- dense complex or real models scale roughly with dimension times element size
- sparse models can reduce memory significantly, but only when the task actually benefits from
  sparsity
- matrix-valued models can be the right choice for structure even when they cost more per binding

`VSA.create()` validates only the documented factory kwargs. If you want custom precision or other
backend-specific array behavior, handle that in application-specific backend code rather than
assuming a generic factory passthrough.

## Current Documentation Policy

This page intentionally avoids fixed timing tables. Release-facing benchmark numbers should come
from the benchmark suite and methodology docs, not from hand-maintained prose.

Use [Benchmark Methodology](./benchmarks.md) and `python -m benchmarks.run ...` for reproducible
measurements rather than copying an old table into design docs or release notes.

## Canonical Examples

- [examples/02_models_comparison.py](https://github.com/Twistient/HoloVec/blob/master/examples/02_models_comparison.py)
- [examples/27_cleanup_strategies.py](https://github.com/Twistient/HoloVec/blob/master/examples/27_cleanup_strategies.py)
- [examples/41_model_ghrr_diagonality.py](https://github.com/Twistient/HoloVec/blob/master/examples/41_model_ghrr_diagonality.py)
- [examples/42_model_bsdc_seg.py](https://github.com/Twistient/HoloVec/blob/master/examples/42_model_bsdc_seg.py)
- [examples/43_retrieval_backend_comparison.py](https://github.com/Twistient/HoloVec/blob/master/examples/43_retrieval_backend_comparison.py)

## See Also

- [Backends](../architecture/backends.md)
- [Model Comparison](../models/index.md)
- [Patterns](./patterns.md)
