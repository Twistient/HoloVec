HoloVec ships multiple VSA models because there is no single best algebra for every task. The
important distinctions are:

- exact inverse vs approximate inverse
- commutative vs order-sensitive binding
- dense vs sparse representations
- vector vs matrix-valued spaces

Use this page to narrow the model family first, then move to the individual model page.

## Comparison at a Glance

| Model | Space | Inverse Style | Order Sensitive | Main Strength | Main Tradeoff |
|-------|-------|---------------|-----------------|---------------|---------------|
| [FHRR](../models/fhrr.md) | Complex | Exact | No | Strong default for compositional retrieval and continuous encoders | Dense complex vectors |
| [GHRR](../models/ghrr.md) | Matrix | Exact | Yes | Non-commutative exact binding for nested or directional structures | Matrix operations are heavier |
| [MAP](../models/map.md) | Bipolar | Self-inverse | No | Simple fast algebra and strong cleanup behavior | Lower expressive structure than matrix models |
| [HRR](../models/hrr.md) | Real/Bipolar | Approximate | No | Classic baseline used in literature | Approximate inverse and cleanup dependence |
| [VTB](../models/vtb.md) | Matrix/Real | Approximate | Yes | Directional and asymmetric relations | Approximate recovery |
| [BSC](../models/bsc.md) | Binary | Self-inverse | No | Binary-friendly and exact XOR-style recovery | Discrete/binary representation only |
| [BSDC](../models/bsdc.md) | Sparse | Approximate | No | Sparse memory-efficient storage | Approximate inverse, retrieval quality depends on sparsity |
| [BSDC-SEG](../models/bsdc-seg.md) | Sparse segment | Self-inverse | No | Segment-sparse pattern matching and fast structured search | Model-specific segment assumptions |

## Recommended Defaults

| If you need... | Start with... | Why |
|----------------|---------------|-----|
| a general-purpose model | `FHRR` | exact inverse, strong cleanup, good encoder support |
| order-sensitive or directional structure | `GHRR` or `VTB` | non-commutative binding |
| simple exact discrete algebra | `MAP` or `BSC` | self-inverse recovery |
| sparse storage | `BSDC` or `BSDC-SEG` | memory efficiency and segment-aware retrieval |
| literature reproduction baseline | `HRR` | classic holographic reduced representations |

## Model Families

### Exact-Inverse Models

`FHRR` and `GHRR` recover bound factors exactly in the noiseless case.

```python
from holovec import VSA

model = VSA.create("FHRR", dim=4096)
a = model.random(seed=1)
b = model.random(seed=2)

recovered = model.unbind(model.bind(a, b), b)
print(float(model.similarity(a, recovered)))
```

Choose these when recovery quality matters more than raw simplicity.

### Self-Inverse Models

`MAP`, `BSC`, and `BSDC-SEG` are attractive when you want exact discrete-style recovery without a
separate inverse operation.

```python
model = VSA.create("MAP", dim=4096)
a = model.random(seed=1)
b = model.random(seed=2)

recovered = model.unbind(model.bind(a, b), b)
print(float(model.similarity(a, recovered)))
```

These are good cleanup and factorization workhorses.

### Approximate-Inverse Models

`HRR`, `VTB`, and `BSDC` usually rely more heavily on cleanup memories or task-specific thresholds.

```python
model = VSA.create("HRR", dim=4096)
a = model.random(seed=1)
b = model.random(seed=2)

recovered = model.unbind(model.bind(a, b), b)
print(float(model.similarity(a, recovered)))
```

Approximate recovery is not a defect on its own, but it changes how you should benchmark and
evaluate the model.

## Order Sensitivity

If `bind(a, b)` and `bind(b, a)` should mean different things, use a non-commutative model.

```python
ghrr = VSA.create("GHRR", dim=96, matrix_size=3, diagonality=0.4)
a = ghrr.random(seed=1)
b = ghrr.random(seed=2)

ab = ghrr.bind(a, b)
ba = ghrr.bind(b, a)
print(float(ghrr.similarity(ab, ba)))
```

Use `GHRR` when you also want exact inverse behavior. Use `VTB` when directional structure matters
but approximate recovery is acceptable.

## Sparse and Segment-Sparse Models

The sparse models are not just "small dense models." They have different retrieval behavior and
should be measured differently:

- `BSDC`: sparse overlap-based similarities and approximate inverse behavior
- `BSDC-SEG`: one active bit per segment, segment-pattern search, self-inverse recovery

Typical valid factory calls:

```python
VSA.create("BSDC", dim=20000, sparsity=0.01, binding_mode="cdt")
VSA.create("BSDC-SEG", dim=400, segments=20)
```

## Configuration Through the Factory

`VSA.create()` now validates supported kwargs instead of ignoring them. Model-specific examples:

```python
VSA.create("GHRR", dim=96, matrix_size=3, diagonality=0.4)
VSA.create("VTB", dim=512, n_bases=4, temperature=50.0)
VSA.create("BSDC", dim=20000, sparsity=0.01, binding_mode="cdt")
VSA.create("BSDC-SEG", dim=400, segments=20)
```

If you need unsupported backend precision or array options, handle them in application-specific
backend code rather than assuming they pass through the factory.

## Quantitative Comparisons

This page intentionally avoids hard-coded benchmark tables. Operation speed, cleanup quality, and
capacity all depend on:

- dimension
- cleanup strategy
- backend
- sparsity or matrix parameters
- the exact workload being measured

Use [Performance Guidance](../guides/performance.md) for measurement advice, and use the upcoming
benchmark suite for release-facing quantitative comparisons rather than copying a generic table.

## Canonical Example

See [examples/02_models_comparison.py](https://github.com/Twistient/HoloVec/blob/master/examples/02_models_comparison.py)
for the maintained runnable comparison script.

## See Also

- [Core Concepts](../getting-started/core-concepts.md)
- [Spaces](../architecture/spaces.md)
- [Performance Guidance](../guides/performance.md)
- [Migration Notes](../guides/migration.md)
