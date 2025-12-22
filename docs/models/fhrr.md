**Fourier Holographic Reduced Representations**

FHRR uses complex-valued vectors (unit phasors) with element-wise complex multiplication for binding. It achieves the best capacity among VSA models and has exact inverse through complex conjugation.

## Properties

| Property | Value |
|----------|-------|
| Binding | Element-wise complex multiply |
| Inverse | Exact (via conjugation) |
| Commutative | Yes |
| Self-inverse | No |
| Space | Complex (unit phasors) |
| Capacity | ~0.06 items/dim |

## When to Use

- **Default choice** when unsure which model to use
- Encoding continuous values (via fractional power)
- Maximum capacity requirements
- Applications where exact recovery matters
- Neural network integration (complex ops supported)

## Theory

### Vector Space

FHRR vectors live on the complex unit circle. Each dimension contains a unit phasor:

$$z_i = e^{i\theta_i}$$

where θ_i ∈ [0, 2π) is sampled uniformly at random.

### Binding Operation

Binding is element-wise complex multiplication:

$$c_i = a_i \cdot b_i = e^{i(\theta_a + \theta_b)}$$

This adds phase angles, creating a result dissimilar to both inputs.

### Unbinding Operation

Unbinding uses complex conjugation:

$$\text{unbind}(c, b)_i = c_i \cdot \bar{b}_i = e^{i(\theta_a + \theta_b - \theta_b)} = e^{i\theta_a} = a_i$$

This provides **exact recovery**: `unbind(bind(a, b), b) = a`.

### Bundling Operation

Bundling sums phasors and normalizes back to unit magnitude:

$$d = \text{normalize}\left(\sum_j v_j\right)$$

The result points in the "average" direction of the inputs.

### Fractional Power Encoding

FHRR uniquely supports fractional power encoding for continuous values:

$$z^\alpha = e^{i\alpha\theta}$$

This preserves metric structure: similar values produce similar vectors.

## Capacity Analysis

FHRR has the highest bundle capacity among VSA models due to its continuous phase space and near-zero noise floor.

Empirically measured (80% detection threshold):

$$n_{max} \approx 0.06 \cdot d$$

| Dimension | Max Items |
|-----------|-----------|
| 512 | ~50 |
| 1024 | ~70 |
| 2048 | ~120 |
| 4096 | ~250 |
| 10000 | ~600 |

## Code Examples

### Basic Usage

```python
from holovec import VSA

# Create FHRR model
model = VSA.create('FHRR', dim=2048)

# Generate random hypervectors
a = model.random(seed=1)
b = model.random(seed=2)

# Bind and recover
c = model.bind(a, b)
a_recovered = model.unbind(c, b)

print(model.similarity(a, a_recovered))  # 1.0 (exact)
```

### Fractional Power Encoding

```python
# Encode continuous values
base = model.random(seed=42)

# Encode value 2.5
encoded_2_5 = model.fractional_power(base, 2.5)
encoded_2_6 = model.fractional_power(base, 2.6)
encoded_5_0 = model.fractional_power(base, 5.0)

print(model.similarity(encoded_2_5, encoded_2_6))  # ~0.99 (similar)
print(model.similarity(encoded_2_5, encoded_5_0))  # ~0.80 (less similar)
```

### With GPU Acceleration

```python
# PyTorch with CUDA
model = VSA.create('FHRR', dim=2048, backend='torch', device='cuda')

# JAX with JIT
model = VSA.create('FHRR', dim=2048, backend='jax')
```

## Comparison with Similar Models

| vs Model | FHRR Advantage | FHRR Disadvantage |
|----------|---------------|-------------------|
| HRR | Exact inverse, better capacity | Requires complex arithmetic |
| MAP | Better capacity | More computation per operation |
| GHRR | Simpler, more efficient | No non-commutativity |

## Performance Notes

- Complex multiplication is ~2x slower than real multiplication
- PyTorch MPS (Apple Silicon) has full complex support
- JAX JIT works well with complex operations
- Vectorized operations are efficient on all backends

## References

- Plate, T. A. (2003). *Holographic Reduced Representations*
- Schlegel, K., et al. (2022). A comparison of vector symbolic architectures
- Frady, E. P., et al. (2021). Computing on Functions Using Randomized Vector Representations

## See Also

- [Models Overview](../models/index.md) — Compare all models
- [Encoder-FractionalPower](../encoders/fractional-power.md) — Continuous value encoding
- [Spaces](../architecture/spaces.md) — Complex space details
