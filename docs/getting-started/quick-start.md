This guide gets you from install to structured retrieval with the smallest set of concepts that
matter in practice.

## 1. Create a Model

```python
from holovec import VSA

model = VSA.create("FHRR", dim=4096, seed=7)
```

`FHRR` is the default recommendation for new projects: exact inverse binding, good cleanup
behavior, and strong compatibility with continuous encoders.

## 2. Use the Core Algebra

```python
a = model.random(seed=1)
b = model.random(seed=2)

bound = model.bind(a, b)
recovered = model.unbind(bound, b)
bundle = model.bundle([a, b, bound])

print(float(model.similarity(a, recovered)))  # close to 1.0
print(float(model.similarity(bundle, a)))     # positive but lower
```

The three operations you use most are:

- `bind`: associate two vectors into a structured relation
- `bundle`: superpose multiple vectors into one memory
- `permute`: encode position or order without losing the vector type

## 3. Encode Real Values

```python
from holovec.encoders import FractionalPowerEncoder

temperature = FractionalPowerEncoder(
    model,
    min_val=0.0,
    max_val=100.0,
    bandwidth=1.5,
    seed=3,
)

v24 = temperature.encode(24.0)
v25 = temperature.encode(25.0)
v80 = temperature.encode(80.0)

print(float(model.similarity(v24, v25)))  # nearby values remain similar
print(float(model.similarity(v24, v80)))  # distant values separate
```

Use:

- `FractionalPowerEncoder` for continuous values
- `ThermometerEncoder` for ordinal bins
- `LevelEncoder` for discrete recoverable levels

## 4. Build a Simple Structured Memory

```python
from holovec.retrieval import Codebook, ItemStore

role_color = model.random(seed=10)
role_temp = model.random(seed=11)
red = model.random(seed=20)
blue = model.random(seed=21)

record = model.bundle(
    [
        model.bind(role_color, red),
        model.bind(role_temp, temperature.encode(24.0)),
    ]
)

store = ItemStore(model).fit(
    Codebook({"red": red, "blue": blue}, backend=model.backend)
)

recovered_color = model.unbind(record, role_color)
print(store.query(recovered_color, k=1))
```

This is the standard VSA pattern:

1. encode values
2. bind them to roles or positions
3. bundle them into a memory
4. unbind with the role and clean up against a codebook

## 5. Know the Supported Factory Surface

`VSA.create()` now rejects unsupported kwargs instead of ignoring them. Common valid calls:

```python
VSA.create("BSDC", dim=20000, sparsity=0.01, binding_mode="cdt")
VSA.create("BSDC-SEG", dim=400, segments=20)
VSA.create("GHRR", dim=96, matrix_size=3, diagonality=0.4)
VSA.create("VTB", dim=512, n_bases=4, temperature=50.0)
VSA.create("FHRR", dim=4096, backend="torch", device="cuda")
```

## Canonical Example

The runnable version of this walkthrough lives in
[examples/00_quickstart.py](https://github.com/Twistient/HoloVec/blob/master/examples/00_quickstart.py).
It supports `--smoke` and is executed in pytest as part of the release-facing example set.

## Next Steps

- [Core Concepts](../getting-started/core-concepts.md)
- [Model Comparison](../models/index.md)
- [Patterns](../guides/patterns.md)
- [Migration Notes](../guides/migration.md)
