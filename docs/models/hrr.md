**Holographic Reduced Representations**

HRR is the classic VSA model developed by Tony Plate. It uses circular convolution for binding and circular correlation for unbinding, providing an approximate inverse.

## Properties

| Property | Value |
|----------|-------|
| Binding | Circular convolution |
| Inverse | Approximate (correlation) |
| Commutative | Yes |
| Self-inverse | No |
| Space | Real-valued (Gaussian) |
| Complexity | O(d log d) via FFT |

## When to Use

- Academic research and comparison baselines
- When using established VSA literature methods
- Real-valued continuous data
- When approximate recovery is acceptable

## Theory

### Vector Space

HRR vectors are real-valued with Gaussian distribution:

$$v_i \sim N(0, 1/d)$$

The variance scaling 1/d ensures unit expected norm.

### Binding Operation

Binding uses circular convolution:

$$c = a \circledast b$$

where:

$$(a \circledast b)_k = \sum_{j=0}^{d-1} a_j \cdot b_{(k-j) \mod d}$$

Efficiently computed via FFT:

$$c = \text{IFFT}(\text{FFT}(a) \cdot \text{FFT}(b))$$

### Unbinding Operation

Unbinding uses circular correlation:

$$\hat{a} = c \star b = \text{IFFT}(\text{FFT}(c) \cdot \overline{\text{FFT}(b)})$$

This is an **approximate inverse**:

$$\text{unbind}(\text{bind}(a, b), b) \approx a + \text{noise}$$

Recovery similarity is typically 0.65-0.75.

### Why Approximate?

For random vectors:

$$\hat{a}_k = \sum_{j} c_j \cdot b_{(j-k) \mod d} = a_k \cdot ||b||^2 + \text{cross-terms}$$

The cross-terms don't cancel perfectly, leaving residual noise proportional to 1/√d.

### Bundling Operation

Bundling is simple addition (no normalization):

$$d = \sum_j v_j$$

HRR specifically does not normalize after bundling to preserve the magnitude relationships needed for correlation-based unbinding.

## Capacity Analysis

HRR has good capacity with near-zero noise floor (Gaussian similarity):

$$n_{max} \approx 0.04 \cdot d$$

| Dimension | Max Items |
|-----------|-----------|
| 1024 | ~40 |
| 2048 | ~80 |
| 4096 | ~160 |
| 10000 | ~400 |

!!! note
    HRR has approximate inverse (~0.65-0.75 recovery similarity), but bundled items are still detectable against random distractors due to the near-zero noise floor.

## Code Examples

### Basic Usage

```python
from holovec import VSA

# Create HRR model
model = VSA.create('HRR', dim=10000)

# Generate random hypervectors
a = model.random(seed=1)
b = model.random(seed=2)

# Bind and unbind
c = model.bind(a, b)
a_recovered = model.unbind(c, b)

# Approximate recovery
print(model.similarity(a, a_recovered))  # ~0.65-0.75
```

### Recovery Quality vs Dimension

```python
for dim in [1000, 5000, 10000]:
    model = VSA.create('HRR', dim=dim)
    a, b = model.random(seed=1), model.random(seed=2)
    c = model.bind(a, b)
    recovered = model.unbind(c, b)
    print(f"dim={dim}: similarity={model.similarity(a, recovered):.3f}")
```

### Using Cleanup for Better Recovery

```python
from holovec.retrieval import Codebook, ItemStore

# Create codebook of known items
items = {f"item_{i}": model.random(seed=i) for i in range(100)}
codebook = Codebook(items, backend=model.backend)
store = ItemStore(model).fit(codebook)

# Noisy recovery can be cleaned up
a = items["item_0"]
b = model.random(seed=999)
c = model.bind(a, b)
noisy_a = model.unbind(c, b)

# Clean up to nearest known item
result = store.query(noisy_a, k=1)
print(result[0][0])  # "item_0"
```

## Comparison with Similar Models

| vs Model | HRR Advantage | HRR Disadvantage |
|----------|--------------|------------------|
| FHRR | Real-valued, simpler | Lower capacity, approx inverse |
| MAP | More expressive | Not self-inverse |
| VTB | Commutative | Approximate inverse |

## Implementation Notes

### FFT-Based Binding

```python
# Efficient circular convolution
fa = np.fft.fft(a)
fb = np.fft.fft(b)
c = np.real(np.fft.ifft(fa * fb))
```

### FFT-Based Unbinding

```python
# Circular correlation
fa = np.fft.fft(a)
fb = np.fft.fft(b)
result = np.real(np.fft.ifft(fa * np.conj(fb)))
```

## References

- Plate, T. A. (1991). Holographic Reduced Representations
- Plate, T. A. (1995). Holographic reduced representations: Distributed representation for cognitive structures
- Plate, T. A. (2003). Holographic Reduced Representations (book)

## See Also

- [Models Overview](../models/index.md) — Compare all models
- [Model-FHRR](../models/fhrr.md) — Complex version with exact inverse
- [Retrieval-Overview](../retrieval/index.md) — Cleanup strategies
