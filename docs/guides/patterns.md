Common VSA patterns and design recipes.

These are the patterns worth stabilizing before `v1` because they map directly to the maintained
example set and public API:

- role-filler records
- sequence encoding
- prototype retrieval
- cleanup and factorization
- order-sensitive composition

## Role-Filler Records

Use role vectors to bind typed fields into one record.

```python
from holovec import VSA
from holovec.encoders import FractionalPowerEncoder
from holovec.retrieval import Codebook, ItemStore

model = VSA.create("FHRR", dim=4096, seed=7)
temperature = FractionalPowerEncoder(model, 0.0, 100.0, bandwidth=1.5, seed=3)

ROLE_NAME = model.random(seed=1)
ROLE_TEMP = model.random(seed=2)

alice = model.random(seed=10)
twenty_four = temperature.encode(24.0)

record = model.bundle(
    [
        model.bind(ROLE_NAME, alice),
        model.bind(ROLE_TEMP, twenty_four),
    ]
)

name_store = ItemStore(model).fit(Codebook({"alice": alice}, backend=model.backend))
print(name_store.query(model.unbind(record, ROLE_NAME), k=1))
```

Use this pattern for records, frames, entity state, and compact knowledge storage.

## Sequence Encoding

`PositionBindingEncoder` gives you an order-sensitive representation without inventing a separate
position bookkeeping scheme yourself.

```python
from holovec import VSA
from holovec.encoders import PositionBindingEncoder

model = VSA.create("MAP", dim=4096, seed=7)
encoder = PositionBindingEncoder(model, seed=42)

reference = ["alice", "likes", "tea"]
reordered = ["tea", "likes", "alice"]

hv_reference = encoder.encode(reference)
hv_reordered = encoder.encode(reordered)

print(float(model.similarity(hv_reference, hv_reference)))
print(float(model.similarity(hv_reference, hv_reordered)))
print(encoder.decode(hv_reference, max_positions=3, threshold=0.2))
```

Use this when order matters and you want approximate sequence matching or sequence cleanup.

## Prototype Retrieval

Bundles work well as class prototypes or memory slots.

```python
from holovec import VSA
from holovec.retrieval import Codebook, ItemStore

model = VSA.create("MAP", dim=4096, seed=7)

cat_prototype = model.bundle([model.random(seed=10), model.random(seed=11)])
dog_prototype = model.bundle([model.random(seed=20), model.random(seed=21)])

store = ItemStore(model).fit(
    Codebook(
        {
            "cat": cat_prototype,
            "dog": dog_prototype,
        },
        backend=model.backend,
    )
)

query = model.bundle([model.random(seed=10), model.random(seed=11)])
print(store.query(query, k=1))
```

Use this for one-shot or few-shot classification when you already have encoded exemplars.

## Cleanup and Factorization

Keep a codebook of known factors, then clean or factorize against it.

```python
from holovec import VSA
from holovec.utils.cleanup import BruteForceCleanup, ResonatorCleanup

model = VSA.create("MAP", dim=4096, seed=7)
codebook = {f"item_{i}": model.random(seed=100 + i) for i in range(6)}

composite = model.bind_multiple(
    [
        codebook["item_0"],
        codebook["item_1"],
        codebook["item_2"],
    ]
)

brute_force = BruteForceCleanup()
resonator = ResonatorCleanup()

print(brute_force.factorize(composite, codebook, model, n_factors=3, threshold=0.6))
print(resonator.factorize(composite, codebook, model, n_factors=3, threshold=0.6))
```

For exact or self-inverse models, this is the standard route to recovering structured factors from
bound compositions.

## Order-Sensitive Composition

If `bind(a, b)` should differ from `bind(b, a)`, use a non-commutative model.

```python
from holovec import VSA

model = VSA.create("GHRR", dim=96, matrix_size=3, diagonality=0.4, seed=7)
a = model.random(seed=1)
b = model.random(seed=2)

ab = model.bind(a, b)
ba = model.bind(b, a)
print(float(model.similarity(ab, ba)))
```

This is the right pattern for directional relations, nested symbolic structures, and role-sensitive
composition.

## Pattern Selection Checklist

- Use `FHRR` when you want a strong default with exact inverse behavior.
- Use `MAP` or `BSC` when self-inverse algebra simplifies cleanup or deployment.
- Use `GHRR` or `VTB` when order and asymmetry are first-class constraints.
- Use `BSDC` or `BSDC-SEG` when sparse retrieval or memory footprint dominates the design.
- Always benchmark the full workload, not just `bind()` in isolation.

## Canonical Examples

- [examples/00_quickstart.py](https://github.com/Twistient/HoloVec/blob/master/examples/00_quickstart.py)
- [examples/13_encoders_position_binding.py](https://github.com/Twistient/HoloVec/blob/master/examples/13_encoders_position_binding.py)
- [examples/26_retrieval_basics.py](https://github.com/Twistient/HoloVec/blob/master/examples/26_retrieval_basics.py)
- [examples/27_cleanup_strategies.py](https://github.com/Twistient/HoloVec/blob/master/examples/27_cleanup_strategies.py)
# Encode with repetition
def encode_robust(item, model, copies=3):
    parts = [model.permute(item, k=i) for i in range(copies)]
    return model.bundle(parts)

# Decode by querying each copy
def decode_robust(noisy, model, copies=3):
    votes = []
    for i in range(copies):
        recovered = model.unpermute(noisy, k=i)
        votes.append(recovered)
    return model.bundle(votes)  # Majority vote
```

**Use for**: Noisy channels, unreliable storage

---

## See Also

- [Quick Start](../getting-started/quick-start.md) — Basic operations
- [Tutorials](../guides/patterns.md) — Full examples
- [Core Concepts](../getting-started/core-concepts.md) — Operation definitions
