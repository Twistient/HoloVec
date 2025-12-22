**ImageEncoder** and **VectorEncoder**

These encoders handle multi-dimensional and spatial data, preserving structural relationships.

## ImageEncoder

### Overview

| Property | Value |
|----------|-------|
| Input | 2D grid (H × W) |
| Output | Single hypervector |
| Preserves | Spatial position |
| Best For | Images, heatmaps |

### How It Works

Encodes each pixel position with both its value and 2D location:

$$\text{image} = \sum_{i,j} \text{bind}(\text{value}_{i,j}, \text{pos}_{i,j})$$

Position encoding uses:
$$\text{pos}_{i,j} = \text{bind}(\rho^i(\text{row}), \rho^j(\text{col}))$$

### Code Example

```python
from holovec import VSA
from holovec.encoders import ImageEncoder
import numpy as np

model = VSA.create('FHRR', dim=4096)
encoder = ImageEncoder(model, height=8, width=8)

# Create simple images
img1 = np.zeros((8, 8))
img1[2:6, 2:6] = 1  # Center square

img2 = np.zeros((8, 8))
img2[2:6, 2:6] = 1  # Same square

img3 = np.zeros((8, 8))
img3[0:4, 0:4] = 1  # Different position

v1 = encoder.encode(img1)
v2 = encoder.encode(img2)
v3 = encoder.encode(img3)

print(model.similarity(v1, v2))  # 1.0 (identical)
print(model.similarity(v1, v3))  # Low (different position)
```

### Position-Aware Similarity

```python
# Small shift → moderate similarity
img_original = np.zeros((8, 8))
img_original[3:5, 3:5] = 1

img_shifted = np.zeros((8, 8))
img_shifted[4:6, 4:6] = 1  # Shifted by 1

v_orig = encoder.encode(img_original)
v_shift = encoder.encode(img_shifted)

# Partially overlapping → partial similarity
print(model.similarity(v_orig, v_shift))  # Moderate
```

### Querying by Position

```python
# What's at position (3, 3)?
row_vec = encoder.get_row_vector(3)
col_vec = encoder.get_col_vector(3)
pos_vec = model.bind(row_vec, col_vec)

query_result = model.unbind(v_orig, pos_vec)
# Similar to value vector for img_original[3, 3]
```

---

## VectorEncoder

### Overview

| Property | Value |
|----------|-------|
| Input | Feature vector (n-dim) |
| Output | Single hypervector |
| Preserves | Per-dimension values |
| Best For | ML features, embeddings |

### How It Works

Binds each dimension's value with a dimension-specific role:

$$\text{vec} = \sum_i \text{bind}(\text{dim}_i, \text{encode}(\text{value}_i))$$

Uses scalar encoders (FPE, Thermometer, Level) per dimension.

### Code Example

```python
from holovec import VSA
from holovec.encoders import VectorEncoder

model = VSA.create('FHRR', dim=2048)

# Create encoder for 3D vectors
encoder = VectorEncoder(
    model,
    n_dims=3,
    min_vals=[0, 0, 0],
    max_vals=[100, 100, 100]
)

# Encode feature vectors
v1 = encoder.encode([50, 50, 50])
v2 = encoder.encode([51, 49, 50])  # Nearby
v3 = encoder.encode([90, 10, 50])  # Different

print(model.similarity(v1, v2))  # High (nearby)
print(model.similarity(v1, v3))  # Moderate (partial match)
```

### Mixed Value Types

```python
# Different encoder types per dimension
from holovec.encoders import (
    FractionalPowerEncoder,
    ThermometerEncoder,
    LevelEncoder
)

# Manual setup for mixed types
dim_encoders = [
    ("age", FractionalPowerEncoder(model, min_val=0, max_val=100)),
    ("rating", ThermometerEncoder(model, levels=5)),
    ("color", LevelEncoder(model, levels=["red", "green", "blue"]))
]

dim_roles = [model.random(seed=i) for i in range(len(dim_encoders))]

def encode_record(age, rating, color):
    parts = []
    values = [age, rating, color]
    for i, ((name, enc), role) in enumerate(zip(dim_encoders, dim_roles)):
        parts.append(model.bind(role, enc.encode(values[i])))
    return model.bundle(parts)

# Encode records
r1 = encode_record(25, 4, "red")
r2 = encode_record(26, 4, "red")   # Similar
r3 = encode_record(25, 1, "blue")  # Different

print(model.similarity(r1, r2))  # High
print(model.similarity(r1, r3))  # Low
```

### Querying Dimensions

```python
# Query: What's the age?
age_role = dim_roles[0]
query_result = model.unbind(r1, age_role)

# Compare with encoded ages
age_25 = dim_encoders[0][1].encode(25)
age_50 = dim_encoders[0][1].encode(50)

print(model.similarity(query_result, age_25))  # High
print(model.similarity(query_result, age_50))  # Lower
```

---

## Use Cases

### Image Pattern Recognition

```python
# One-shot image classification
def create_class_prototype(images, encoder):
    encodings = [encoder.encode(img) for img in images]
    return model.bundle(encodings)

# Create prototypes
digit_0_proto = create_class_prototype(digit_0_samples, encoder)
digit_1_proto = create_class_prototype(digit_1_samples, encoder)

# Classify new image
def classify(img, prototypes):
    v = encoder.encode(img)
    best_class = max(prototypes.keys(),
                     key=lambda c: model.similarity(v, prototypes[c]))
    return best_class
```

### Feature Vector Similarity Search

```python
# Encode database of vectors
database = {}
for id, features in items.items():
    database[id] = encoder.encode(features)

# Find similar items
def find_similar(query_features, k=5):
    q = encoder.encode(query_features)
    scored = [(id, model.similarity(q, v)) for id, v in database.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
```

---

## Comparison

| Property | ImageEncoder | VectorEncoder |
|----------|--------------|---------------|
| Input structure | 2D grid | 1D array |
| Position encoding | 2D coordinates | Dimension index |
| Value encoding | Per-pixel | Per-dimension |
| Query support | By (row, col) | By dimension |

## References

- Plate, T. A. (2003). Holographic Reduced Representations
- Kleyko, D., et al. (2023). A Survey on Hyperdimensional Computing

## See Also

- [Encoders Overview](../encoders/index.md) — All encoders
- [Encoder-FractionalPower](../encoders/fractional-power.md) — Used for value encoding
- [Patterns](../guides/patterns.md) — Common encoding patterns
