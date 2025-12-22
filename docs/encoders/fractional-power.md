**FractionalPowerEncoder**

The Fractional Power Encoder (FPE) encodes continuous scalar values while preserving metric structure. Similar values produce similar hypervectors, enabling interpolation and generalization.

## Overview

| Property | Value |
|----------|-------|
| Input | Scalar ∈ [min_val, max_val] |
| Output | Hypervector |
| Preserves | Metric/locality |
| Best Model | FHRR (native support) |
| Complexity | O(d) |

## When to Use

- Continuous sensor readings (temperature, pressure)
- Numeric features for ML
- Values where interpolation matters
- When similar values should match

## Theory

### Fractional Power Encoding

For a base hypervector b and scalar value x:

$$\text{encode}(x) = b^x$$

For FHRR (complex phasors):

$$b^x_i = e^{i \cdot x \cdot \theta_i}$$

where θ_i is the base phase at dimension i.

### Similarity Behavior

For values x and y:

$$\text{sim}(\text{encode}(x), \text{encode}(y)) \approx \text{sinc}(\pi \sigma |x - y|)$$

where σ is the bandwidth parameter.

Key properties:
- sim(x, x) = 1.0
- Similarity decreases smoothly with distance
- Rate of decrease controlled by bandwidth

### Bandwidth Parameter

The bandwidth σ controls the encoding resolution:

| Bandwidth | Effect |
|-----------|--------|
| Low (0.1) | Wide similarity kernel, smooth |
| Medium (1.0) | Balanced resolution |
| High (10.0) | Narrow kernel, precise |

## Code Examples

### Basic Usage

```python
from holovec import VSA
from holovec.encoders import FractionalPowerEncoder

# Create model and encoder
model = VSA.create('FHRR', dim=2048)
encoder = FractionalPowerEncoder(
    model,
    min_val=0,
    max_val=100,
    bandwidth=1.0
)

# Encode values
v50 = encoder.encode(50)
v51 = encoder.encode(51)
v60 = encoder.encode(60)
v100 = encoder.encode(100)

# Check similarity
print(model.similarity(v50, v51))   # ~0.99 (very similar)
print(model.similarity(v50, v60))   # ~0.90 (similar)
print(model.similarity(v50, v100))  # ~0.50 (different)
```

### Tuning Bandwidth

```python
# Compare bandwidths
for bw in [0.5, 1.0, 2.0, 5.0]:
    enc = FractionalPowerEncoder(model, min_val=0, max_val=100, bandwidth=bw)
    v0 = enc.encode(0)
    v10 = enc.encode(10)
    sim = model.similarity(v0, v10)
    print(f"bandwidth={bw}: sim(0, 10)={sim:.3f}")
```

### Interpolation

```python
# FPE naturally supports interpolation
v25 = encoder.encode(25)
v75 = encoder.encode(75)

# Encode 50 directly
v50_direct = encoder.encode(50)

# "Interpolate" via bundling (approximate midpoint)
v50_interp = model.bundle([v25, v75])

# Should be similar (not identical due to bundling)
print(model.similarity(v50_direct, v50_interp))  # ~0.8
```

### Multi-dimensional Values

```python
# Encode (x, y) coordinates
x_base = model.random(seed=1)
y_base = model.random(seed=2)

x_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, base=x_base)
y_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, base=y_base)

def encode_point(x, y):
    return model.bundle([x_enc.encode(x), y_enc.encode(y)])

# Nearby points are similar
p1 = encode_point(50, 50)
p2 = encode_point(51, 49)
p3 = encode_point(90, 10)

print(model.similarity(p1, p2))  # High (nearby)
print(model.similarity(p1, p3))  # Low (far)
```

### Decoding

```python
# Create decoder with resolution
values = list(range(0, 101, 5))  # [0, 5, 10, ..., 100]
codebook = {v: encoder.encode(v) for v in values}

def decode(vec):
    best_val = None
    best_sim = -1
    for v, enc in codebook.items():
        sim = model.similarity(vec, enc)
        if sim > best_sim:
            best_sim = sim
            best_val = v
    return best_val

# Test decoding
v47 = encoder.encode(47)
decoded = decode(v47)
print(f"Encoded 47, decoded to {decoded}")  # 45 or 50 (nearest)
```

## Use Cases

### Sensor Fusion

```python
# Temperature and humidity sensors
temp_enc = FractionalPowerEncoder(model, min_val=-20, max_val=50)
hum_enc = FractionalPowerEncoder(model, min_val=0, max_val=100)

TEMP_ROLE = model.random(seed=1)
HUM_ROLE = model.random(seed=2)

def encode_reading(temp, humidity):
    return model.bundle([
        model.bind(TEMP_ROLE, temp_enc.encode(temp)),
        model.bind(HUM_ROLE, hum_enc.encode(humidity))
    ])

# Similar conditions → similar vectors
hot_humid = encode_reading(35, 80)
hot_dry = encode_reading(35, 20)
cold_dry = encode_reading(5, 20)

print(model.similarity(hot_humid, hot_dry))  # Moderate
print(model.similarity(hot_dry, cold_dry))   # Moderate
print(model.similarity(hot_humid, cold_dry)) # Low
```

### Feature Scaling

```python
# Automatic normalization via encoder range
features = [
    (FractionalPowerEncoder(model, min_val=0, max_val=1e6), "price"),
    (FractionalPowerEncoder(model, min_val=0, max_val=5), "rating"),
    (FractionalPowerEncoder(model, min_val=1900, max_val=2024), "year"),
]

def encode_item(price, rating, year):
    parts = []
    for enc, name in features:
        role = model.random(seed=hash(name) % 1000)
        if name == "price":
            parts.append(model.bind(role, enc.encode(price)))
        elif name == "rating":
            parts.append(model.bind(role, enc.encode(rating)))
        else:
            parts.append(model.bind(role, enc.encode(year)))
    return model.bundle(parts)
```

## Comparison with Other Encoders

| vs Encoder | FPE Advantage | FPE Disadvantage |
|------------|--------------|------------------|
| Thermometer | Smoother similarity | Not for ordinals |
| Level | Continuous interpolation | Overkill for discrete |
| Random | Preserves locality | Requires FHRR for best results |

## References

- Frady, E. P., et al. (2021). Computing on Functions Using Randomized Vector Representations
- Plate, T. A. (2003). Holographic Reduced Representations

## See Also

- [Encoders Overview](../encoders/index.md) — All encoders
- [Model-FHRR](../models/fhrr.md) — Best model for FPE
- [Encoder-Thermometer-Level](../encoders/thermometer-level.md) — For ordinal data
