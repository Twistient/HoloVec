**Binary Sparse Distributed Codes**

BSDC uses sparse binary vectors where only a small fraction of bits are set to 1. This provides memory efficiency and biological plausibility while maintaining hyperdimensional properties.

## Properties

| Property | Value |
|----------|-------|
| Binding | XOR |
| Inverse | Self (approximate) |
| Commutative | Yes |
| Self-inverse | Yes |
| Space | Sparse binary |
| Sparsity | p = 1/√d (optimal) |

## When to Use

- Memory-constrained systems
- Very high dimensions (d > 10,000)
- Biologically plausible models (cortical codes)
- Sparse data structures available
- When storage scales with active bits

## Theory

### Vector Space

BSDC vectors are sparse binary:

$$v_i \in \{0, 1\}, \quad ||v||_0 = k$$

where k is the number of active (1) bits.

### Optimal Sparsity

The optimal sparsity maximizes information capacity:

$$p^* = \frac{1}{\sqrt{d}}$$

| Dimension | Sparsity | Active Bits |
|-----------|----------|-------------|
| 1,000 | 3.2% | ~32 |
| 10,000 | 1.0% | ~100 |
| 100,000 | 0.32% | ~316 |

### Binding Operation

Binding uses XOR, which approximately preserves sparsity:

$$c = a \oplus b$$

Expected sparsity of result: $2p(1-p) \approx 2p$ for small p.

### Bundling Operation

Bundling uses majority voting with sparsity preservation:

1. Sum all vectors element-wise
2. Select top-k positions with highest counts
3. k chosen to maintain target sparsity

### Similarity Measure

Overlap coefficient:

$$\text{sim}(a, b) = \frac{|a \cap b|}{k}$$

where k is the expected number of active bits.

## Capacity Analysis

BSDC has lower bundle capacity than dense codes due to sparse overlap dynamics:

$$n_{max} \approx 0.01 \cdot d$$

| Dimension | Max Items |
|-----------|-----------|
| 2048 | ~20 |
| 4096 | ~40 |
| 10000 | ~100 |

!!! note
    While capacity per dimension is lower, BSDC uses far less memory, allowing much higher dimensions. At dim=100,000 with 1% sparsity, you get ~1000 item capacity with only ~2KB storage per vector.

## Memory Efficiency

| Dimension | Dense Storage | Sparse Storage | Ratio |
|-----------|---------------|----------------|-------|
| 10,000 | 10 KB | 200 B | 50× |
| 100,000 | 100 KB | 600 B | 160× |
| 1,000,000 | 1 MB | 2 KB | 500× |

Sparse storage: indices of active bits (e.g., 100 × 16-bit indices = 200 bytes).

## Code Examples

### Basic Usage

```python
from holovec import VSA

# Create BSDC model with default optimal sparsity
model = VSA.create('BSDC', dim=10000)

print(f"Sparsity: {model.sparsity:.4f}")  # ~0.01
print(f"Expected ones: {int(model.sparsity * 10000)}")  # ~100

# Generate sparse hypervectors
a = model.random(seed=1)
b = model.random(seed=2)

print(f"Actual ones in a: {a.sum()}")  # ~100
```

### Custom Sparsity

```python
# More sparse (less memory, lower capacity)
sparse_model = VSA.create('BSDC', dim=10000, sparsity=0.005)

# Less sparse (more memory, higher capacity)
dense_model = VSA.create('BSDC', dim=10000, sparsity=0.02)
```

### Measuring Sparsity

```python
# After binding, sparsity changes slightly
c = model.bind(a, b)
actual_sparsity = model.measure_sparsity(c)
print(f"Result sparsity: {actual_sparsity:.4f}")

# Rehash to restore target sparsity
c_rehashed = model.rehash(c)
print(f"After rehash: {model.measure_sparsity(c_rehashed):.4f}")
```

### Sparse Storage

```python
import numpy as np

# Get indices of active bits (sparse representation)
a = model.random(seed=1)
active_indices = np.where(a == 1)[0]
print(f"Storage: {len(active_indices) * 2} bytes")  # 2 bytes per index
```

## Biological Plausibility

BSDC mirrors cortical neural codes:

| Property | BSDC | Cortex |
|----------|------|--------|
| Sparsity | ~1% | ~1-5% |
| Encoding | Distributed | Distributed |
| Binding | XOR-like | Synaptic modulation |

## Comparison with Similar Models

| vs Model | BSDC Advantage | BSDC Disadvantage |
|----------|---------------|-------------------|
| BSC | Much less memory | Slightly lower capacity |
| MAP | Sparse storage | Approximate operations |
| BSDC-SEG | More flexible sparsity | No segment structure |

## References

- Kanerva, P. (1988). Sparse Distributed Memory
- Rachkovskij, D. A., & Kussul, E. M. (2001). Binding and normalization of binary sparse codes
- Kleyko, D., et al. (2023). A Survey on Hyperdimensional Computing

## See Also

- [Models Overview](../models/index.md) — Compare all models
- [Model-BSC](../models/bsc.md) — Dense binary
- [Model-BSDC-SEG](../models/bsdc-seg.md) — Segment-structured sparse
