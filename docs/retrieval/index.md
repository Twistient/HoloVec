HoloVec provides lightweight retrieval primitives for storing labeled hypervectors, running nearest-neighbor lookup, and working with simple associative memories.

## Components

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **Codebook** | Label → vector mapping | Vocabulary, symbol tables, persistence |
| **ItemStore** | Top-k retrieval and factorization | Cleanup, nearest-neighbor lookup |
| **AssocStore** | Key-value associations | Heteroassociative memory |

## Codebook

A `Codebook` maps string labels to hypervectors while preserving insertion order.

### Basic Usage

```python
from holovec import VSA
from holovec.retrieval import Codebook

model = VSA.create("FHRR", dim=2048)

items = {
    "apple": model.random(seed=1),
    "banana": model.random(seed=2),
    "cherry": model.random(seed=3),
}
codebook = Codebook(items, backend=model.backend)

apple_vec = codebook["apple"]
print("apple" in codebook)  # True
print(len(codebook))        # 3
```

### Adding Items

```python
codebook.add("date", model.random(seed=4))

codebook.extend({
    "elderberry": model.random(seed=5),
    "fig": model.random(seed=6),
})
```

### Persistence

```python
codebook.save("fruit_codebook.npz")
loaded = Codebook.load("fruit_codebook.npz", backend=model.backend)
```

Legacy `.npz` files created by older HoloVec releases used pickle-backed label arrays. They now fail closed by default. To migrate one of those files, load it once with `allow_unsafe_legacy=True`, then re-save it:

```python
legacy = Codebook.load(
    "old_codebook.npz",
    backend=model.backend,
    allow_unsafe_legacy=True,
)
legacy.save("migrated_codebook.npz")
```

---

## ItemStore

`ItemStore` wraps a `Codebook` and a cleanup strategy.

### Basic Usage

```python
from holovec.retrieval import ItemStore

store = ItemStore(model).fit(codebook)

query = codebook["apple"]
results = store.query(query, k=3)

for label, similarity in results:
    print(f"{label}: {similarity:.3f}")
```

### Search Utilities

For thresholded or batched search, use the public search helpers:

```python
from holovec.search import batch_similarity, threshold_search

codebook_dict = dict(codebook.items())
labels, sims = threshold_search(query, codebook_dict, model, threshold=0.8)
all_sims = batch_similarity([query], codebook_dict, model)
```

### Factorization

```python
from holovec.utils.cleanup import ResonatorCleanup

roles = Codebook(
    {
        "shape": model.random(seed=10),
        "color": model.random(seed=11),
    },
    backend=model.backend,
)

composition = model.bind(roles["shape"], roles["color"])
store = ItemStore(model, cleanup=ResonatorCleanup()).fit(roles)
labels, similarities = store.factorize(composition, n_factors=2)
```

### Persistence

```python
store.save("items.npz")
restored = ItemStore.load(model, "items.npz")
```

If you need to migrate a legacy pickle-backed archive, pass `allow_unsafe_legacy=True` to `ItemStore.load(...)` once, then re-save the store.

---

## AssocStore

`AssocStore` keeps aligned key and value codebooks.

### Basic Usage

```python
from holovec.retrieval import AssocStore

store = AssocStore(model)
store.add("ball", model.random(seed=1), model.random(seed=101))
store.add("cube", model.random(seed=2), model.random(seed=102))

label, value_vec = store.query_value(store.keys["ball"])
print(label)  # ball
```

### Bulk Fit

```python
keys = {
    "red": model.random(seed=1),
    "blue": model.random(seed=2),
}
values = {
    "red": model.random(seed=101),
    "blue": model.random(seed=102),
}

store = AssocStore(model).fit(keys, values)
ranked = store.query_label(keys["blue"], k=2)
```

### Persistence

```python
store.save("keys.npz", "values.npz")
restored = AssocStore.load(model, "keys.npz", "values.npz")
```

As with `Codebook` and `ItemStore`, legacy pickle-backed archives can be migrated via `allow_unsafe_legacy=True`.

---

## Workflow Example

```python
from holovec import VSA
from holovec.retrieval import Codebook, ItemStore
from holovec.encoders import FractionalPowerEncoder

model = VSA.create("FHRR", dim=4096)

OBJECT = model.random(seed=1)
COLOR = model.random(seed=2)
SIZE = model.random(seed=3)

objects = Codebook(
    {
        "ball": model.random(seed=10),
        "cube": model.random(seed=11),
        "cone": model.random(seed=12),
    },
    backend=model.backend,
)
colors = Codebook(
    {
        "red": model.random(seed=20),
        "blue": model.random(seed=21),
        "green": model.random(seed=22),
    },
    backend=model.backend,
)

size_encoder = FractionalPowerEncoder(model, min_val=0, max_val=100)

big_red_ball = model.bundle(
    [
        model.bind(OBJECT, objects["ball"]),
        model.bind(COLOR, colors["red"]),
        model.bind(SIZE, size_encoder.encode(80)),
    ]
)

object_query = model.unbind(big_red_ball, OBJECT)
object_store = ItemStore(model).fit(objects)
print(object_store.query(object_query, k=1)[0][0])  # ball
```

---

## See Also

- [Cleanup Strategies](./cleanup.md) — cleanup and factorization methods
- [Patterns](../guides/patterns.md) — common usage patterns
- [Model-HRR](../models/hrr.md) — when cleanup is especially useful
