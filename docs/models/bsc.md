**Binary Spatter Codes**

BSC uses XOR binding on binary vectors {0, 1}. It's mathematically equivalent to MAP with bipolar vectors and is ideal for FPGA and low-power implementations.

## Properties

| Property | Value |
|----------|-------|
| Binding | XOR |
| Inverse | Self (XOR = unbind) |
| Commutative | Yes |
| Self-inverse | Yes |
| Space | Binary {0, 1} |
| Complexity | O(d) bitwise ops |

## When to Use

- FPGA implementation
- Ultra-low-power devices
- Bitwise operations only (no multiply)
- Maximum hardware efficiency
- When storage is measured in bits

## Theory

### Vector Space

BSC vectors are binary:

$$v_i \in \{0, 1\}$$

Each dimension is independently sampled with P(v_i = 1) = 0.5.

### Relationship to MAP

BSC and MAP are equivalent via transformation:

$$\text{bipolar} \leftrightarrow \text{binary}: x \mapsto \frac{x+1}{2}$$

| BSC | MAP |
|-----|-----|
| 0 | -1 |
| 1 | +1 |

XOR on binary = element-wise multiply on bipolar.

### Binding Operation

Binding is XOR:

$$(a \oplus b)_i = a_i \oplus b_i$$

| a | b | a⊕b |
|---|---|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### Self-Inverse Property

XOR is self-inverse:

$$a \oplus b \oplus b = a$$

Since $b \oplus b = 0$ (identity).

### Bundling Operation

Bundling uses majority vote:

$$d_i = \begin{cases} 1 & \text{if } \sum_j v_{j,i} > n/2 \\ 0 & \text{otherwise} \end{cases}$$

### Similarity Measure

BSC uses normalized Hamming distance:

$$\text{sim}(a, b) = 1 - \frac{\text{hamming}(a, b)}{d}$$

## Capacity Analysis

BSC has similar capacity to MAP (they are mathematically equivalent):

$$n_{max} \approx 0.03 \cdot d$$

| Dimension | Max Items |
|-----------|-----------|
| 1024 | ~25 |
| 2048 | ~50 |
| 4096 | ~100 |
| 10000 | ~250 |

!!! note
    Like MAP, BSC has a ~0.5 noise floor (expected Hamming similarity for random binary vectors), which limits effective capacity despite high absolute similarity after bundling.

## Code Examples

### Basic Usage

```python
from holovec import VSA

# Create BSC model
model = VSA.create('BSC', dim=10000)

# Generate random binary hypervectors
a = model.random(seed=1)
b = model.random(seed=2)

print(set(a.tolist()))  # {0.0, 1.0}

# XOR binding
c = model.bind(a, b)

# Self-inverse recovery
a_recovered = model.bind(c, b)  # or model.unbind(c, b)
print(model.similarity(a, a_recovered))  # 1.0
```

### Conversion to/from Bipolar

```python
# Convert binary to bipolar (for MAP compatibility)
a_binary = model.random()
a_bipolar = model.to_bipolar(a_binary)  # {0,1} → {-1,+1}

# Convert back
a_binary_again = model.from_bipolar(a_bipolar)
```

### Hardware-Oriented Example

```python
import numpy as np

# Pure bitwise operations
def xor_bind(a, b):
    return np.bitwise_xor(a.astype(np.uint8), b.astype(np.uint8))

def majority_bundle(vectors, threshold=None):
    if threshold is None:
        threshold = len(vectors) / 2
    summed = np.sum(vectors, axis=0)
    return (summed > threshold).astype(np.uint8)

def hamming_similarity(a, b):
    return 1 - np.sum(a != b) / len(a)
```

## Hardware Implementation

### FPGA Resources (10,000 dim)

| Component | Resources |
|-----------|-----------|
| Vector storage | 10,000 bits |
| XOR binding | 10,000 LUT1s |
| Majority vote | ~30 adder bits |
| Similarity | Popcount + divide |

### Verilog Sketch

```verilog
// XOR binding - single cycle
assign c = a ^ b;

// Popcount for similarity (parallel tree)
always @(*) begin
    hamming = 0;
    for (i = 0; i < DIM; i = i + 1)
        hamming = hamming + (a[i] ^ b[i]);
end
```

## Comparison with Similar Models

| vs Model | BSC Advantage | BSC Disadvantage |
|----------|--------------|------------------|
| MAP | 1 bit vs 32 bits storage | Hamming vs cosine similarity |
| BSDC | Dense (50% ones) | More storage than sparse |
| HRR | Much simpler hardware | Lower capacity |

## References

- Kanerva, P. (1996). Binary Spatter-Coding of Ordered K-Tuples
- Kanerva, P. (2009). Hyperdimensional Computing

## See Also

- [Models Overview](../models/index.md) — Compare all models
- [Model-MAP](../models/map.md) — Bipolar equivalent
- [Model-BSDC](../models/bsdc.md) — Sparse binary alternative
