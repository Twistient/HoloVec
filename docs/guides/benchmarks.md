Benchmark methodology for HoloVec.

This guide defines what we benchmark, why the suites are shaped the way they are, and which
literature each suite is meant to reflect.

## Benchmark Philosophy

HDC/VSA benchmarking has to be model-aware.

The comparison literature already highlights that useful evaluation covers at least:

- bundle capacity
- non-exact unbinding quality
- the interaction of binding and bundling in query answering
- application-level behavior rather than timing in isolation

That structure is explicit in Schlegel, Neubert, and Protzel's comparison study:

- Schlegel et al. evaluate "(1) the capacity of bundles, (2) the approximation quality of
  non-exact unbinding operations, (3) the influence of combining binding and bundling operations
  on the query answering performance, and (4) the performance on two example applications"
  ([Schlegel et al. 2022](https://arxiv.org/abs/2001.11797)).

The HDC/VSA survey literature also argues that different models and data transformations should be
understood in terms of their algebraic properties rather than treated as interchangeable vectors:

- [Kleyko et al. 2022, Part I](https://redwood.berkeley.edu/wp-content/uploads/2022/11/2022_CSUR_survey_HDCVSA_part_1.pdf)
- [Kleyko et al. 2023, Part II](https://redwood.berkeley.edu/wp-content/uploads/2022/11/2023_CSUR_survey_HDCVSA_part_2.pdf)

That means:

- exact-inverse, self-inverse, approximate-inverse, sparse, and non-commutative models should not
  be forced into one scoreboard
- quality metrics are as important as timing
- the benchmark suite must expose the workload assumptions directly

## Literature Mapping

| Suite | Why it exists | Primary literature anchor |
|-------|---------------|---------------------------|
| `primitives` | sanity-check core ops and record baseline timings | survey-level cross-model grounding from Kleyko et al. |
| `bundle-capacity` | bundled-item recovery under cleanup | [Schlegel et al. 2022](https://arxiv.org/abs/2001.11797) |
| `approximate-unbinding` | sequential bind/unbind degradation on approximate models | [Schlegel et al. 2022](https://arxiv.org/abs/2001.11797) |
| `cleanup-factorization` | multi-factor recovery with cleanup dynamics | [Frady et al. / Resonator Networks](https://pubmed.ncbi.nlm.nih.gov/33080162/) and follow-on factorization work |
| `order-sensitivity` | non-commutativity and exact recovery for directional models | [Yeung et al. 2024](https://arxiv.org/abs/2405.09689) and matrix-binding literature such as Gallant and Okaywe (2013) |
| `sparse-retrieval` | sparse overlap and segment-pattern retrieval | [Rachkovskij and Kussul 2001](https://redwood.berkeley.edu/wp-content/uploads/2021/08/RachkovskijKussul.pdf) |

## Model-Specific Expectations

### Exact-inverse models

Examples: `FHRR`, `GHRR`

Expect:

- very high bind/unbind recovery on clean pairs
- strong compositional recovery on structured queries
- sensitivity to the underlying structure, not just nearest-neighbor speed

### Self-inverse models

Examples: `MAP`, `BSC`, `BSDC-SEG`

Expect:

- strong cleanup and factorization behavior on clean compositions
- simple algebra that often makes them attractive for hardware or discrete pipelines

### Approximate-inverse models

Examples: `HRR`, `VTB`

Expect:

- graceful degradation instead of perfect recovery
- quality to depend more strongly on cleanup strategy and task formulation

### Sparse models

Examples: `BSDC`, `BSDC-SEG`

Expect:

- retrieval behavior to depend on overlap or segment structure, not just cosine-like scoring
- different capacity and error regimes from dense continuous models

### Non-commutative models

Examples: `GHRR`, `VTB`

Expect:

- order-sensitive workloads to reveal their value
- symmetric workloads to under-test them

## Runner

The benchmark runner is:

```bash
python -m benchmarks.run --suite <suite> --model <model|all> --output <path>
```

Useful examples:

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

Supported suites:

- `primitives`
- `bundle-capacity`
- `approximate-unbinding`
- `cleanup-factorization`
- `order-sensitivity`
- `sparse-retrieval`

## Output Policy

- JSON is the default and is the preferred archival format.
- CSV is supported for spreadsheet and docs workflows.
- Outputs are written to local artifacts, not committed benchmark result blobs.

## CI Policy

CI only smoke-tests the runner on tiny workloads. It does not enforce benchmark thresholds.

That is intentional. Before `v1.0`, we want reproducible methodology and correct model-aware task
selection first. Hard regression thresholds can come later once we have stable published baselines.
