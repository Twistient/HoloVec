Vector spaces define the type of vectors used by each VSA model. They determine how random vectors are generated and how similarity is measured.

## Overview

| Space | Values | Similarity | Models |
|-------|--------|------------|--------|
| **Bipolar** | {-1, +1} | Cosine | MAP, HRR |
| **Complex** | Unit phasors (e^iθ) | Cosine | FHRR |
| **Binary** | {0, 1} | Hamming | BSC |
| **Sparse** | Sparse {0, 1} | Overlap | BSDC |
| **SparseSegment** | Segment-sparse {0, 1} | Overlap | BSDC-SEG |
| **Matrix** | Unitary matrices | Trace | GHRR, VTB |

## When Spaces Matter

Most users don't interact with spaces directly—`VSA.create()` automatically selects the correct space for each model. However, understanding spaces helps when:

- Choosing a model for specific hardware constraints
- Debugging similarity computations
- Implementing custom models

## Space Details

### Bipolar Space

**Values:** Each dimension is -1 or +1.

**Random Generation:** Uniform choice of {-1, +1} per dimension.

**Similarity:** Cosine similarity:
$$\text{sim}(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||}$$

**Used by:** [MAP](../models/map.md), [HRR](../models/hrr.md)

```python
from holovec import VSA

model = VSA.create('MAP', dim=1000)
v = model.random()
# v contains only -1 and +1 values
print(set(v.tolist()))  # {-1.0, 1.0}
```

### Complex Space

**Values:** Unit phasors e^iθ where θ ∈ [0, 2π).

**Random Generation:** Uniform random phase per dimension.

**Similarity:** Real part of complex cosine:
$$\text{sim}(a, b) = \text{Re}\left(\frac{a \cdot \bar{b}}{||a|| \cdot ||b||}\right)$$

**Used by:** [FHRR](../models/fhrr.md)

```python
model = VSA.create('FHRR', dim=1000)
v = model.random()
# v contains complex numbers on the unit circle
print(abs(v[0]))  # ~1.0 (unit magnitude)
```

!!! note
    Complex vectors enable exact inverse binding through conjugation, giving FHRR its superior capacity.

### Binary Space

**Values:** Each dimension is 0 or 1.

**Random Generation:** Bernoulli(0.5) per dimension.

**Similarity:** Normalized Hamming distance:
$$\text{sim}(a, b) = 1 - \frac{\text{hamming}(a, b)}{d}$$

**Used by:** [BSC](../models/bsc.md)

```python
model = VSA.create('BSC', dim=1000)
v = model.random()
print(set(v.tolist()))  # {0.0, 1.0}
```

### Sparse Space

**Values:** Sparse binary vectors with fixed number of 1s.

**Random Generation:** *k* random positions set to 1, rest are 0.

**Similarity:** Overlap coefficient:
$$\text{sim}(a, b) = \frac{|a \cap b|}{k}$$

**Used by:** [BSDC](../models/bsdc.md)

**Parameters:**
- `dim`: Total dimensions
- `sparsity`: Number of active bits (k)

```python
model = VSA.create('BSDC', dim=10000, sparsity=100)
v = model.random()
print(v.sum())  # ~100 active bits
```

### Sparse Segment Space

**Values:** Segment-structured sparse binary vectors.

**Random Generation:** Vector divided into segments, exactly one 1 per segment.

**Similarity:** Segment overlap.

**Used by:** [BSDC-SEG](../models/bsdc-seg.md)

**Parameters:**
- `dim`: Total dimensions
- `segments`: Number of segments

```python
model = VSA.create('BSDC_SEG', dim=10000, segments=100)
v = model.random()
# Each segment of 100 dimensions has exactly one 1
```

### Matrix Space

**Values:** Unitary matrices (U†U = I).

**Random Generation:** Random orthogonal/unitary matrix.

**Similarity:** Normalized trace:
$$\text{sim}(A, B) = \frac{\text{Re}(\text{tr}(A^\dagger B))}{d}$$

**Used by:** [GHRR](../models/ghrr.md), [VTB](../models/vtb.md)

!!! note
    Matrix-based models trade memory (O(d²) per vector) for non-commutative binding, useful when order matters.

## Similarity Properties

All spaces guarantee these properties:

| Property | Description |
|----------|-------------|
| Self-similarity = 1 | `sim(a, a) = 1.0` |
| Symmetry | `sim(a, b) = sim(b, a)` |
| Bounded | `-1 ≤ sim(a, b) ≤ 1` |
| Random orthogonality | `E[sim(random, random)] ≈ 0` |

## Choosing a Space (via Model)

You don't choose spaces directly—you choose a model, and the space follows:

| If you need... | Choose Model | Space |
|----------------|--------------|-------|
| Best capacity | FHRR | Complex |
| Hardware-friendly | MAP or BSC | Bipolar/Binary |
| Memory efficiency | BSDC | Sparse |
| Non-commutative binding | GHRR | Matrix |
| Classic baseline | HRR | Bipolar |

## See Also

- [Models](../models/index.md) — Model comparison
- [Architecture Overview](../architecture/index.md) — How spaces fit in
- [Core Concepts](../getting-started/core-concepts.md) — Mathematical foundations
