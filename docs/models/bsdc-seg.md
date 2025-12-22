**Segmented Binary Sparse Distributed Codes**

BSDC-SEG uses segment-structured sparse binary vectors where exactly one bit is active per segment. This provides deterministic sparsity and fast segment-wise operations.

## Properties

| Property | Value |
|----------|-------|
| Binding | XOR |
| Inverse | Self (exact) |
| Commutative | Yes |
| Self-inverse | Yes |
| Space | Segment-sparse binary |
| Structure | 1 active bit per segment |

## When to Use

- Deterministic sparsity required
- Fast segment-based retrieval
- When sparsity must be exactly maintained
- Structured sparse representations
- Fast set operations

## Theory

### Vector Space

BSDC-SEG vectors are divided into S segments of length L:

$$d = S \times L$$

Each segment has exactly one active bit:

$$\sum_{i \in \text{segment}_k} v_i = 1, \quad \forall k$$

### Example Structure

For d=12, S=3, L=4:

```
Segment 0    Segment 1    Segment 2
[0 1 0 0]    [0 0 1 0]    [1 0 0 0]
     ↑            ↑            ↑
  active       active       active
```

### Binding Operation

XOR binding, which maintains segment structure:

$$c = a \oplus b$$

After XOR, each segment may have 0 or 2 active bits. Normalization restores exactly 1 per segment.

### Bundling Operation

Segment-wise majority voting:

1. Count votes per position within each segment
2. Select position with maximum count
3. Ties broken deterministically (lowest index)

### Similarity Measure

Fraction of matching segments:

$$\text{sim}(a, b) = \frac{\text{matching segments}}{S}$$

## Capacity Analysis

Capacity depends on segment configuration:

$$n_{max} \approx 0.05 \cdot S$$

| Config | Segments | Max Items |
|--------|----------|-----------|
| d=1000, S=100 | 100 | ~5 |
| d=10000, S=100 | 100 | ~5 |
| d=10000, S=1000 | 1000 | ~50 |

More segments = higher capacity but more memory.

## Code Examples

### Basic Usage

```python
from holovec import VSA

# Create BSDC-SEG model: 10000 dims, 100 segments
model = VSA.create('BSDC_SEG', dim=10000, segments=100)

print(f"Segments: {model.segments}")         # 100
print(f"Segment length: {model.segment_length}")  # 100

# Generate hypervectors
a = model.random(seed=1)
b = model.random(seed=2)

# Verify structure: exactly 100 active bits
print(f"Active bits: {a.sum()}")  # 100
```

### Verifying Segment Structure

```python
import numpy as np

a = model.random(seed=1)
a_np = np.array(a).reshape(model.segments, model.segment_length)

# Each row (segment) has exactly one 1
for seg_idx in range(5):
    print(f"Segment {seg_idx}: {a_np[seg_idx].sum()} active bit(s)")
```

### Binding and Recovery

```python
# XOR binding
c = model.bind(a, b)

# Self-inverse: exact recovery
a_recovered = model.unbind(c, b)
print(model.similarity(a, a_recovered))  # 1.0
```

### Segment-Based Similarity

```python
# Similarity = fraction of matching segments
a = model.random(seed=1)
b = model.random(seed=2)

# Random vectors have ~1/L matching segments by chance
expected_sim = 1.0 / model.segment_length
actual_sim = model.similarity(a, b)
print(f"Expected: {expected_sim:.3f}, Actual: {actual_sim:.3f}")
```

## Comparison with BSDC

| Aspect | BSDC | BSDC-SEG |
|--------|------|----------|
| Sparsity | Probabilistic | Exact |
| Structure | Unstructured | Segmented |
| Binding | XOR (approx) | XOR (exact after norm) |
| Similarity | Overlap | Segment match |
| Use case | General sparse | Structured retrieval |

## Implementation Details

### Segment Normalization

After XOR, segments are normalized:

```python
def normalize_segment(segment):
    # Keep position with maximum value (vote)
    # Ties: choose lowest index
    winner = np.argmax(segment)
    result = np.zeros_like(segment)
    result[winner] = 1
    return result
```

### Fast Retrieval

Segment structure enables efficient retrieval:

```python
# Find matching segments in O(S) instead of O(d)
def fast_similarity(a, b, segments, seg_len):
    matches = 0
    for s in range(segments):
        start = s * seg_len
        end = start + seg_len
        # Compare which index is active
        a_active = np.argmax(a[start:end])
        b_active = np.argmax(b[start:end])
        if a_active == b_active:
            matches += 1
    return matches / segments
```

## Comparison with Similar Models

| vs Model | BSDC-SEG Advantage | BSDC-SEG Disadvantage |
|----------|-------------------|----------------------|
| BSDC | Deterministic sparsity | Less flexible |
| BSC | Structured retrieval | More complex |
| MAP | Exact self-inverse | Lower capacity |

## References

- Kleyko, D., et al. (2023). A Survey on Hyperdimensional Computing
- Rachkovskij, D. A. (2001). Representation and Processing of Structures

## See Also

- [Models Overview](../models/index.md) — Compare all models
- [Model-BSDC](../models/bsdc.md) — Unstructured sparse
- [Model-BSC](../models/bsc.md) — Dense binary
