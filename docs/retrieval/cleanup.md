Cleanup strategies project noisy or composite hypervectors back to known items in a codebook.

## Why Cleanup?

After operations like unbinding (especially with approximate models) or bundling, vectors become noisy:

```python
# HRR unbinding is approximate
model = VSA.create('HRR', dim=10000)
a, b = model.random(seed=1), model.random(seed=2)
c = model.bind(a, b)
a_recovered = model.unbind(c, b)

print(model.similarity(a, a_recovered))  # ~0.70 (not 1.0!)
```

Cleanup finds the closest known item:

```python
# With cleanup
cleaned = cleanup.cleanup(a_recovered, codebook)
print(model.similarity(a, cleaned))  # ~1.0 (if a is in codebook)
```

## Available Strategies

| Strategy | Method | Speed | Best For |
|----------|--------|-------|----------|
| **BruteForce** | Check all items | O(n×d) | Small codebooks (<1000) |
| **Resonator** | Iterative refinement | O(k×d) | Large codebooks, composites |

---

## BruteForceCleanup

Exhaustively compares with every item in the codebook.

### Usage

```python
from holovec import VSA
from holovec.retrieval import Codebook
from holovec.utils.cleanup import BruteForceCleanup

model = VSA.create('HRR', dim=10000)

# Create codebook
items = {f"item_{i}": model.random(seed=i) for i in range(100)}
codebook = Codebook(items, backend=model.backend)

# Create cleanup
cleanup = BruteForceCleanup(model)

# Clean up noisy vector
noisy = items["item_42"] + model.random() * 0.3
cleaned, label, similarity = cleanup.cleanup(noisy, codebook)

print(f"Cleaned to: {label} (sim={similarity:.3f})")  # item_42
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| threshold | None | Minimum similarity to accept |

### When to Use

- Small codebooks (< 1000 items)
- Need guaranteed best match
- One-time queries (not batched)

---

## ResonatorCleanup

Iterative algorithm based on resonator networks (Kymn et al. 2024).

### How It Works

1. Start with initial estimate x⁰
2. Iterate: x^(t+1) = normalize(similarity_matrix × x^t)
3. Converge to attractor (known item)

### Usage

```python
from holovec.utils.cleanup import ResonatorCleanup

# Create resonator
cleanup = ResonatorCleanup(
    model,
    iterations=10,
    soft=True
)

# Clean up
cleaned, label, similarity = cleanup.cleanup(noisy, codebook)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| iterations | 10 | Max iterations |
| soft | True | Soft vs hard update |
| convergence_threshold | 0.999 | Early stopping |

### Soft vs Hard Mode

**Soft resonator** (default):
- Smooth updates via weighted average
- Better for noisy input
- More stable convergence

**Hard resonator**:
- Winner-take-all updates
- Faster convergence
- Can oscillate with noise

```python
# Hard resonator
hard_cleanup = ResonatorCleanup(model, iterations=10, soft=False)
```

### Factorization

Resonators can decompose bound composites:

```python
# c = bind(a, b) where a, b are from codebook
codebook_a = Codebook(...)  # Possible values of a
codebook_b = Codebook(...)  # Possible values of b

# Factorize
a_cleaned, b_cleaned = cleanup.factorize(c, [codebook_a, codebook_b])
```

### When to Use

- Large codebooks (1000+ items)
- Composite vector factorization
- Repeated queries (can precompute)

---

## Performance Comparison

| Codebook Size | BruteForce | Resonator (10 iter) |
|---------------|------------|---------------------|
| 100 | 0.1 ms | 0.3 ms |
| 1,000 | 1.0 ms | 0.5 ms |
| 10,000 | 10 ms | 0.8 ms |
| 100,000 | 100 ms | 1.5 ms |

Resonator becomes faster for large codebooks due to sublinear convergence.

---

## Cleanup with Different Models

| Model | Cleanup Needed? | Recommended Strategy |
|-------|-----------------|---------------------|
| FHRR | Rarely (exact inverse) | BruteForce if needed |
| GHRR | Rarely (exact inverse) | BruteForce if needed |
| MAP | After bundling | BruteForce |
| HRR | Always | Resonator |
| VTB | Always | Resonator |
| BSC | After bundling | BruteForce |
| BSDC | After operations | BruteForce |

---

## Example: Complete Cleanup Pipeline

```python
from holovec import VSA
from holovec.retrieval import Codebook, ItemStore
from holovec.utils.cleanup import ResonatorCleanup

# Setup
model = VSA.create('HRR', dim=10000)

# Create word embeddings
words = ["apple", "banana", "cherry", "date", "elderberry"]
word_vectors = Codebook(
    {w: model.random(seed=hash(w)) for w in words},
    backend=model.backend
)

# Create cleanup
cleanup = ResonatorCleanup(model, iterations=20)

# Store and retrieve with cleanup
def semantic_query(query_word, context_word):
    """Find word most related to query given context."""
    q = word_vectors[query_word]
    ctx = word_vectors[context_word]

    # Bind to create association
    associated = model.bind(q, ctx)

    # Unbind with query
    result = model.unbind(associated, q)

    # Clean up to find closest word
    cleaned, label, sim = cleanup.cleanup(result, word_vectors)
    return label, sim

# Test
result, sim = semantic_query("apple", "cherry")
print(f"Result: {result} (sim={sim:.3f})")
```

---

## References

- Kymn, C., et al. (2024). Resonator Networks for Learning Compositional Representations

## See Also

- [Retrieval-Overview](../retrieval/index.md) — Retrieval components
- [Model-HRR](../models/hrr.md) — Approximate inverse model
- [Model-VTB](../models/vtb.md) — Another approximate model
