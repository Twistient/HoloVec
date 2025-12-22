**Vector-derived Transformation Binding**

VTB implements non-commutative binding using vector-derived transformations. It applies a transformation derived from one vector to transform the other, creating directional associations.

## Properties

| Property | Value |
|----------|-------|
| Binding | Weighted transformation |
| Inverse | Approximate |
| Commutative | No |
| Self-inverse | No |
| Space | Real-valued (L2 normalized) |
| Complexity | O(d × K) for K bases |

## When to Use

- Order-sensitive relationships without matrix overhead
- Directional associations (A→B ≠ B→A)
- Memory-constrained non-commutative binding
- When approximate recovery is acceptable

## Theory

### Vector Space

VTB vectors are real-valued with L2 normalization:

$$||v|| = 1$$

### Binding Operation (MBAT-style)

VTB derives a transformation from vector `a` and applies it to vector `b`:

$$c = M(a) \cdot b \approx \sum_{k=1}^{K} w_k(a) \cdot \rho^{s_k}(b)$$

where:
- $w_k(a)$ = softmax weights derived from `a`
- $\rho^{s_k}(b)$ = circular shift of `b` by $s_k$ positions
- K = number of basis transformations (default: 4)

### Weight Computation

Weights are computed via softmax over projections:

$$w_k(a) = \text{softmax}(\tau \cdot \langle a, U_k \rangle)_k$$

where:
- $U_k$ = fixed code vectors (learned at initialization)
- $\tau$ = temperature parameter (default: 100)

### Non-Commutativity

Since $M(a)$ depends on `a`:

$$M(a) \cdot b \neq M(b) \cdot a$$

The binding order encodes directional information.

### Unbinding Operation

Approximate unbinding reverses the transformation:

$$\hat{b} = \sum_{k} w_k(a) \cdot \rho^{-s_k}(c)$$

!!! warning
    Due to non-commutativity: `unbind(c, a)` recovers `b`, not `a`.
    For `c = bind(a, b)`, use `unbind(c, a)` to get `b`.

## Capacity Analysis

VTB has moderate capacity similar to HRR:

$$n_{max} \approx 0.04 \cdot d$$

| Dimension | Max Items |
|-----------|-----------|
| 2048 | ~80 |
| 4096 | ~160 |
| 10000 | ~400 |

Recovery similarity depends on basis count and temperature.

## Code Examples

### Basic Usage

```python
from holovec import VSA

# Create VTB model
model = VSA.create('VTB', dim=10000)

# Generate random hypervectors
a = model.random(seed=1)
b = model.random(seed=2)

# Bind
c = model.bind(a, b)

# Test non-commutativity
ba = model.bind(b, a)
print(model.similarity(c, ba))  # ~0.3 (different)

# Unbind: recovers b, not a!
b_recovered = model.unbind(c, a)
print(model.similarity(b, b_recovered))  # ~0.8-0.9
```

### Non-Commutative Use Case

```python
# Directional relationship: "dog chases cat"
subject = model.random(seed=1)  # dog
action = model.random(seed=2)   # chases
obj = model.random(seed=3)      # cat

# Encode direction through binding order
dog_chases = model.bind(subject, action)
chases_cat = model.bind(action, obj)

# "dog chases" ≠ "chases dog"
chases_dog = model.bind(action, subject)
print(model.similarity(dog_chases, chases_dog))  # Low
```

### Comparing with GHRR

```python
# VTB: memory-efficient non-commutativity
vtb = VSA.create('VTB', dim=10000)

# GHRR: exact inverse but more memory
ghrr = VSA.create('GHRR', dim=100, matrix_size=3)  # 100 × 9 = 900 values

# VTB uses ~10000 values, GHRR uses ~900
# But GHRR has exact inverse
```

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_bases | 4 | Number of basis transformations |
| temperature | 100.0 | Softmax temperature |
| shifts | auto | Circular shift amounts per basis |

```python
# Custom parameters
model = VSA.create('VTB', dim=10000, n_bases=8, temperature=50.0)
```

## Comparison with Similar Models

| vs Model | VTB Advantage | VTB Disadvantage |
|----------|--------------|------------------|
| GHRR | Much less memory | Approximate inverse |
| HRR | Non-commutative | More complex |
| MAP | Non-commutative | Approximate inverse |

## References

- Gallant, S. I., & Okaywe, T. W. (2013). Representing Objects, Relations, and Sequences
- Gayler, R. W. (2003). Vector Symbolic Architectures answer Jackendoff's challenges
- Schlegel, K., et al. (2022). A comparison of vector symbolic architectures

## See Also

- [Models Overview](../models/index.md) — Compare all models
- [Model-GHRR](../models/ghrr.md) — Alternative with exact inverse
- [Patterns](../guides/patterns.md) — Directional binding patterns
