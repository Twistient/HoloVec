**ThermometerEncoder** and **LevelEncoder**

These encoders handle ordinal and categorical data respectively.

## ThermometerEncoder

### Overview

| Property | Value |
|----------|-------|
| Input | Ordinal value (1, 2, 3, ...) |
| Output | Hypervector |
| Preserves | Order relationships |
| Best For | Rankings, ratings |

### How It Works

Thermometer encoding accumulates similarity as values increase:

For levels 1-5 with base vectors [b₁, b₂, b₃, b₄, b₅]:
- encode(1) = b₁
- encode(2) = bundle(b₁, b₂)
- encode(3) = bundle(b₁, b₂, b₃)
- encode(4) = bundle(b₁, b₂, b₃, b₄)
- encode(5) = bundle(b₁, b₂, b₃, b₄, b₅)

### Similarity Pattern

Adjacent values share most components:

```
sim(1, 2) = 0.71  (share 1 of 2)
sim(2, 3) = 0.82  (share 2 of 3)
sim(1, 5) = 0.45  (share 1 of 5)
```

### Code Example

```python
from holovec import VSA
from holovec.encoders import ThermometerEncoder

model = VSA.create('FHRR', dim=2048)

# Create thermometer encoder for ratings 1-5
encoder = ThermometerEncoder(model, levels=5)

# Encode ratings
r1 = encoder.encode(1)
r2 = encoder.encode(2)
r3 = encoder.encode(3)
r5 = encoder.encode(5)

# Adjacent ratings more similar
print(model.similarity(r1, r2))  # ~0.71
print(model.similarity(r2, r3))  # ~0.82
print(model.similarity(r1, r5))  # ~0.45
```

### Use Cases

- Star ratings (1-5 stars)
- Priority levels (low, medium, high)
- Education levels
- Size categories (S, M, L, XL)

---

## LevelEncoder

### Overview

| Property | Value |
|----------|-------|
| Input | Categorical label |
| Output | Hypervector |
| Preserves | Category identity |
| Best For | Discrete categories |

### How It Works

Each level gets an independent random vector:

- encode("cat") = random₁
- encode("dog") = random₂
- encode("bird") = random₃

All categories are approximately orthogonal.

### Similarity Pattern

```
sim("cat", "cat") = 1.0
sim("cat", "dog") ≈ 0.0
sim("dog", "bird") ≈ 0.0
```

### Code Example

```python
from holovec import VSA
from holovec.encoders import LevelEncoder

model = VSA.create('FHRR', dim=2048)

# Create level encoder with categories
categories = ["red", "green", "blue", "yellow"]
encoder = LevelEncoder(model, levels=categories)

# Encode categories
red = encoder.encode("red")
green = encoder.encode("green")
red2 = encoder.encode("red")

# Same category = identical
print(model.similarity(red, red2))    # 1.0

# Different categories = orthogonal
print(model.similarity(red, green))   # ~0.0
```

### Use Cases

- Color categories
- Species classification
- Country codes
- Product categories

---

## Choosing Between Them

| Scenario | Use |
|----------|-----|
| Values have natural order | ThermometerEncoder |
| Categories are unrelated | LevelEncoder |
| Rankings (1st, 2nd, 3rd) | ThermometerEncoder |
| Labels (A, B, C, D) | LevelEncoder |

### Combined Example

```python
# Product encoding with both types
model = VSA.create('FHRR', dim=2048)

# Ordinal: star rating
rating_enc = ThermometerEncoder(model, levels=5)

# Categorical: product category
category_enc = LevelEncoder(model, levels=["electronics", "clothing", "food"])

# Role vectors
RATING = model.random(seed=1)
CATEGORY = model.random(seed=2)

def encode_product(rating, category):
    return model.bundle([
        model.bind(RATING, rating_enc.encode(rating)),
        model.bind(CATEGORY, category_enc.encode(category))
    ])

# Similar ratings, same category → similar
phone_5star = encode_product(5, "electronics")
phone_4star = encode_product(4, "electronics")
shirt_5star = encode_product(5, "clothing")

print(model.similarity(phone_5star, phone_4star))  # High (same cat, close rating)
print(model.similarity(phone_5star, shirt_5star))  # Moderate (diff cat, same rating)
```

---

## Comparison

| Property | Thermometer | Level |
|----------|-------------|-------|
| Similarity structure | Gradual | Binary |
| Order preserved | Yes | No |
| Interpolation | Natural | Not applicable |
| Memory per level | Cumulative | Independent |

## References

- Kanerva, P. (2009). Hyperdimensional Computing

## See Also

- [Encoders Overview](../encoders/index.md) — All encoders
- [Encoder-FractionalPower](../encoders/fractional-power.md) — For continuous values
