**PositionBindingEncoder**, **NGramEncoder**, **TrajectoryEncoder**

These encoders handle ordered sequences of items, preserving positional or contextual information.

## PositionBindingEncoder

### Overview

| Property | Value |
|----------|-------|
| Input | List of hypervectors |
| Output | Single hypervector |
| Preserves | Absolute position |
| Best For | Fixed-length sequences |

### How It Works

Binds each item with its position vector:

$$\text{seq} = \sum_i \text{bind}(\text{item}_i, \rho^i(\text{pos}))$$

where ρⁱ is the i-th permutation of a base position vector.

### Code Example

```python
from holovec import VSA
from holovec.encoders import PositionBindingEncoder

model = VSA.create('FHRR', dim=2048)
encoder = PositionBindingEncoder(model, max_length=10)

# Create item vectors
items = {
    "cat": model.random(seed=1),
    "sat": model.random(seed=2),
    "mat": model.random(seed=3)
}

# Encode sequences
seq1 = encoder.encode([items["cat"], items["sat"], items["mat"]])
seq2 = encoder.encode([items["mat"], items["sat"], items["cat"]])

# Different order → different encoding
print(model.similarity(seq1, seq2))  # ~0.3 (some overlap from "sat" in middle)
```

### Querying by Position

```python
# Query: What's at position 0?
pos_0 = encoder.get_position_vector(0)
query_result = model.unbind(seq1, pos_0)

# Should be similar to "cat"
print(model.similarity(query_result, items["cat"]))  # High
print(model.similarity(query_result, items["mat"]))  # Low
```

---

## NGramEncoder

### Overview

| Property | Value |
|----------|-------|
| Input | Sequence of items |
| Output | Single hypervector |
| Preserves | Local context |
| Best For | Text, variable-length |

### How It Works

Creates and bundles all n-grams:

For sequence [A, B, C, D] with n=2:
- bigrams: [A,B], [B,C], [C,D]
- Each bigram: bind(A, permute(B))
- Result: bundle all bigrams

### Code Example

```python
from holovec import VSA
from holovec.encoders import NGramEncoder

model = VSA.create('FHRR', dim=2048)
encoder = NGramEncoder(model, n=3)  # trigrams

# Character-level encoding
def text_to_vectors(text, model):
    return [model.random(seed=ord(c)) for c in text]

# Encode words
cat_vecs = text_to_vectors("cat", model)
car_vecs = text_to_vectors("car", model)
dog_vecs = text_to_vectors("dog", model)

v_cat = encoder.encode(cat_vecs)
v_car = encoder.encode(car_vecs)
v_dog = encoder.encode(dog_vecs)

# Similar prefixes → some similarity
print(model.similarity(v_cat, v_car))  # Moderate (share "ca")
print(model.similarity(v_cat, v_dog))  # Low (different)
```

### Configurable N

```python
# Compare different n values
for n in [2, 3, 4]:
    enc = NGramEncoder(model, n=n)
    v1 = enc.encode(text_to_vectors("hello", model))
    v2 = enc.encode(text_to_vectors("hella", model))
    sim = model.similarity(v1, v2)
    print(f"n={n}: sim('hello', 'hella') = {sim:.3f}")
```

---

## TrajectoryEncoder

### Overview

| Property | Value |
|----------|-------|
| Input | Sequence of encoded points |
| Output | Single hypervector |
| Preserves | Temporal order |
| Best For | Paths, time series |

### How It Works

Uses sequential binding to encode trajectory:

$$\text{traj} = ((...((p_0 \otimes p_1) \otimes p_2)...) \otimes p_n)$$

or permutation-based:

$$\text{traj} = \sum_i \text{bind}(p_i, \rho^i(\text{time}))$$

### Code Example

```python
from holovec import VSA
from holovec.encoders import TrajectoryEncoder, FractionalPowerEncoder

model = VSA.create('FHRR', dim=2048)

# Position encoder
pos_enc = FractionalPowerEncoder(model, min_val=0, max_val=100)

# Trajectory encoder
traj_enc = TrajectoryEncoder(model)

# Encode a path: [10, 20, 30, 40, 50]
path1 = [pos_enc.encode(x) for x in [10, 20, 30, 40, 50]]
path2 = [pos_enc.encode(x) for x in [10, 20, 30, 40, 55]]  # Similar end
path3 = [pos_enc.encode(x) for x in [50, 40, 30, 20, 10]]  # Reversed

t1 = traj_enc.encode(path1)
t2 = traj_enc.encode(path2)
t3 = traj_enc.encode(path3)

# Similar paths → similar encodings
print(model.similarity(t1, t2))  # High
print(model.similarity(t1, t3))  # Low (reversed)
```

### Time Series Encoding

```python
# Encode sensor readings over time
def encode_timeseries(readings, model, traj_enc, value_enc):
    points = [value_enc.encode(r) for r in readings]
    return traj_enc.encode(points)

# Similar patterns → similar encodings
pattern1 = [1, 2, 3, 4, 5]
pattern2 = [1, 2, 3, 4, 6]  # Slight difference at end
pattern3 = [5, 4, 3, 2, 1]  # Reversed

value_enc = FractionalPowerEncoder(model, min_val=0, max_val=10)

t1 = encode_timeseries(pattern1, model, traj_enc, value_enc)
t2 = encode_timeseries(pattern2, model, traj_enc, value_enc)
t3 = encode_timeseries(pattern3, model, traj_enc, value_enc)
```

---

## Choosing a Sequence Encoder

| Scenario | Use |
|----------|-----|
| Fixed-length, query by position | PositionBindingEncoder |
| Variable-length text | NGramEncoder |
| Temporal order matters | TrajectoryEncoder |
| Need subsequence matching | NGramEncoder |

## Comparison

| Property | Position | NGram | Trajectory |
|----------|----------|-------|------------|
| Query by position | Yes | No | Partial |
| Variable length | Yes | Yes | Yes |
| Order sensitivity | Strong | Local | Sequential |
| Reversal detection | Yes | Partial | Yes |

## References

- Plate, T. A. (2003). Holographic Reduced Representations
- Kanerva, P. (1996). Binary Spatter-Coding of Ordered K-Tuples

## See Also

- [Encoders Overview](../encoders/index.md) — All encoders
- [Encoder-FractionalPower](../encoders/fractional-power.md) — For encoding sequence elements
- [Patterns](../guides/patterns.md) — Sequence encoding patterns
