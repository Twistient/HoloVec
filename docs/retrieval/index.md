HoloVec provides retrieval mechanisms for storing, querying, and cleaning up hypervectors.

## Components

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **Codebook** | Label → vector mapping | Vocabulary, symbol tables |
| **ItemStore** | High-level retrieval | K-NN, batch queries |
| **AssocStore** | Key-value associations | Associative memory |

## Codebook

A `Codebook` maps string labels to hypervectors.

### Basic Usage

```python
from holovec import VSA
from holovec.retrieval import Codebook

model = VSA.create('FHRR', dim=2048)

# Create from dictionary
items = {
    "apple": model.random(seed=1),
    "banana": model.random(seed=2),
    "cherry": model.random(seed=3)
}
codebook = Codebook(items, backend=model.backend)

# Access vectors
apple_vec = codebook["apple"]

# Check membership
print("apple" in codebook)  # True
print(len(codebook))        # 3
```

### Adding Items

```python
# Add single item
codebook.add("date", model.random(seed=4))

# Add multiple items
codebook.update({
    "elderberry": model.random(seed=5),
    "fig": model.random(seed=6)
})
```

### Persistence

```python
# Save to JSON
codebook.save("fruit_codebook.json")

# Load from JSON
loaded = Codebook.load("fruit_codebook.json", backend=model.backend)
```

---

## ItemStore

`ItemStore` provides high-level retrieval with cleanup strategies.

### Basic Usage

```python
from holovec.retrieval import ItemStore

# Create and fit
store = ItemStore(model)
store.fit(codebook)

# Query for nearest neighbors
query = codebook["apple"] + model.random() * 0.1  # Noisy apple
results = store.query(query, k=3)

for label, similarity in results:
    print(f"{label}: {similarity:.3f}")
# apple: 0.95
# banana: 0.02
# cherry: 0.01
```

### Threshold Query

```python
# Find all items above threshold
results = store.query_threshold(query, threshold=0.5)
```

### Batch Queries

```python
# Query multiple vectors at once
queries = [noisy_apple, noisy_banana, noisy_cherry]
batch_results = store.batch_query(queries, k=1)
```

---

## AssocStore

`AssocStore` implements key-value associative memory using binding.

### Basic Usage

```python
from holovec.retrieval import AssocStore

store = AssocStore(model)

# Store associations
key1 = model.random(seed=1)
value1 = model.random(seed=2)
store.store(key1, value1)

# Retrieve by key
retrieved = store.retrieve(key1)
print(model.similarity(value1, retrieved))  # High
```

### Multiple Associations

```python
# Store multiple key-value pairs
keys = [model.random(seed=i) for i in range(10)]
values = [model.random(seed=i+100) for i in range(10)]

for k, v in zip(keys, values):
    store.store(k, v)

# Retrieve
for i, k in enumerate(keys[:3]):
    retrieved = store.retrieve(k)
    sim = model.similarity(values[i], retrieved)
    print(f"Key {i}: similarity = {sim:.3f}")
```

### Cleanup Integration

```python
# Use cleanup for better retrieval
from holovec.utils.cleanup import BruteForceCleanup

cleanup = BruteForceCleanup(model)
value_codebook = Codebook({f"v{i}": v for i, v in enumerate(values)}, backend=model.backend)

retrieved_raw = store.retrieve(keys[0])
cleaned = cleanup.cleanup(retrieved_raw, value_codebook)
```

---

## Workflow Example

Complete workflow for a semantic memory system:

```python
from holovec import VSA
from holovec.retrieval import Codebook, ItemStore
from holovec.encoders import FractionalPowerEncoder

# Setup
model = VSA.create('FHRR', dim=4096)

# Define roles
OBJECT = model.random(seed=1)
COLOR = model.random(seed=2)
SIZE = model.random(seed=3)

# Define fillers
objects = Codebook({
    "ball": model.random(seed=10),
    "cube": model.random(seed=11),
    "cone": model.random(seed=12)
}, backend=model.backend)

colors = Codebook({
    "red": model.random(seed=20),
    "blue": model.random(seed=21),
    "green": model.random(seed=22)
}, backend=model.backend)

size_encoder = FractionalPowerEncoder(model, min_val=0, max_val=100)

# Create structured representation: "big red ball"
big_red_ball = model.bundle([
    model.bind(OBJECT, objects["ball"]),
    model.bind(COLOR, colors["red"]),
    model.bind(SIZE, size_encoder.encode(80))
])

# Query: what object is this?
query_result = model.unbind(big_red_ball, OBJECT)
store = ItemStore(model).fit(objects)
results = store.query(query_result, k=1)
print(f"Object: {results[0][0]}")  # ball

# Query: what color?
query_result = model.unbind(big_red_ball, COLOR)
store = ItemStore(model).fit(colors)
results = store.query(query_result, k=1)
print(f"Color: {results[0][0]}")  # red
```

---

## Cleanup Strategies

For models with approximate inverse (HRR, VTB), cleanup improves retrieval:

```python
from holovec.utils.cleanup import BruteForceCleanup, ResonatorCleanup

# Brute force: check all items
bf_cleanup = BruteForceCleanup(model)
cleaned = bf_cleanup.cleanup(noisy_vector, codebook)

# Resonator: iterative refinement (faster for large codebooks)
res_cleanup = ResonatorCleanup(model, iterations=10)
cleaned = res_cleanup.cleanup(noisy_vector, codebook)
```

See [Cleanup Strategies](../retrieval/cleanup.md) for details.

---

## See Also

- [Cleanup Strategies](../retrieval/cleanup.md) — Cleanup methods
- [Patterns](../guides/patterns.md) — Common usage patterns
- [Model-HRR](../models/hrr.md) — When cleanup is essential
