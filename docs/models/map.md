**Multiply-Add-Permute**

MAP uses element-wise multiplication for binding, which is self-inverse. It's one of the simplest VSA models, making it ideal for hardware deployment and neuromorphic computing.

## Properties

| Property | Value |
|----------|-------|
| Binding | Element-wise multiply |
| Inverse | Self (bind = unbind) |
| Commutative | Yes |
| Self-inverse | Yes |
| Space | Bipolar {-1, +1} |
| Complexity | O(d) per operation |

## When to Use

- Hardware deployment (FPGA, ASIC)
- Neuromorphic computing
- Real-time systems with strict latency requirements
- When self-inverse property simplifies logic
- Resource-constrained environments

## Theory

### Vector Space

MAP vectors use bipolar values:

$$v_i \in \{-1, +1\}$$

Each dimension is independently sampled with equal probability.

### Binding Operation

Binding is element-wise multiplication (Hadamard product):

$$c_i = a_i \cdot b_i$$

For bipolar vectors, this is equivalent to XOR:
- (+1)(+1) = +1
- (+1)(-1) = -1
- (-1)(+1) = -1
- (-1)(-1) = +1

### Self-Inverse Property

Binding is its own inverse:

$$\text{unbind}(c, b) = c \odot b = (a \odot b) \odot b = a \odot (b \odot b) = a \odot \mathbf{1} = a$$

Since $b_i \cdot b_i = 1$ for all bipolar values.

### Bundling Operation

Bundling sums vectors and applies majority vote:

$$d_i = \text{sign}\left(\sum_j v_{j,i}\right)$$

Ties are broken by the first vector's value.

## Capacity Analysis

MAP has moderate capacity due to its ~0.5 noise floor (random bipolar similarity):

$$n_{max} \approx 0.03 \cdot d$$

| Dimension | Max Items |
|-----------|-----------|
| 1024 | ~30 |
| 2048 | ~60 |
| 4096 | ~120 |
| 10000 | ~300 |

!!! note
    MAP similarity between random vectors is ~0.5 (not 0), so signal-to-noise ratio is lower than FHRR/HRR despite high absolute similarity after bundling.

## Code Examples

### Basic Usage

```python
from holovec import VSA

# Create MAP model
model = VSA.create('MAP', dim=10000)

# Generate random hypervectors
a = model.random(seed=1)
b = model.random(seed=2)

# Bind (same as unbind for MAP)
c = model.bind(a, b)

# Unbind using bind (self-inverse)
a_recovered = model.bind(c, b)  # or model.unbind(c, b)

print(model.similarity(a, a_recovered))  # 1.0 (exact)
```

### Self-Inverse Demonstration

```python
# Binding is self-inverse
c = model.bind(a, b)

# Both methods recover a:
a_via_bind = model.bind(c, b)
a_via_unbind = model.unbind(c, b)

# They're identical
print(model.similarity(a_via_bind, a_via_unbind))  # 1.0
```

### Hardware-Friendly Operations

```python
import numpy as np

# Convert bipolar to binary for hardware
a_bipolar = model.random()
a_binary = (a_bipolar + 1) / 2  # {-1,+1} → {0,1}

# XOR binding in binary
def xor_bind(a_bin, b_bin):
    return np.logical_xor(a_bin, b_bin).astype(float)
```

## Comparison with Similar Models

| vs Model | MAP Advantage | MAP Disadvantage |
|----------|--------------|------------------|
| FHRR | Simpler, faster | Lower capacity |
| BSC | Same ops, real-valued | Slightly more memory |
| HRR | Self-inverse, simpler | Less expressive |
| GHRR | Much simpler | No non-commutativity |

## Hardware Implementation

MAP is ideal for hardware because:

1. **Binary operations**: Bipolar can be stored as single bits
2. **XOR binding**: Single gate per dimension
3. **Majority bundling**: Popcount and threshold
4. **No multiply-accumulate**: Pure logic operations

### FPGA Resource Estimate (10,000 dim)

| Resource | Count |
|----------|-------|
| LUTs for binding | ~10,000 |
| FFs for storage | ~10,000 |
| Comparators | ~10,000 |

## References

- Kanerva, P. (2009). Hyperdimensional Computing: An Introduction
- Schlegel, K., et al. (2022). A comparison of vector symbolic architectures

## See Also

- [Models Overview](../models/index.md) — Compare all models
- [Model-BSC](../models/bsc.md) — Binary version
- [Performance](../guides/performance.md) — Benchmarks
