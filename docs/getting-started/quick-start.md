This guide gets you running HoloVec in 5 minutes.

## Create a Model

```python
from holovec import VSA

# Create a VSA model with 2048 dimensions
model = VSA.create('FHRR', dim=2048)
```

`'FHRR'` is a good default choice—it has exact inverse binding and the best capacity. See [Models](../models/index.md) for alternatives.

## Generate Hypervectors

```python
# Random hypervectors
a = model.random()
b = model.random()

# Seeded for reproducibility
c = model.random(seed=42)
```

Each hypervector is a 2048-dimensional vector. Random vectors are nearly orthogonal to each other.

## Core Operations

### Binding (Association)

Binding creates an association between two vectors. The result is dissimilar to both inputs.

```python
# Bind a and b
c = model.bind(a, b)

# Recover a by unbinding with b
a_recovered = model.unbind(c, b)

# Verify recovery
print(model.similarity(a, a_recovered))  # 1.0 for FHRR
```

**Use case:** Key-value pairs, role-filler structures

### Bundling (Superposition)

Bundling combines multiple vectors into one. The result is similar to all inputs.

```python
# Bundle three vectors
combined = model.bundle([a, b, c])

# Check similarity to each
print(model.similarity(combined, a))  # ~0.57
print(model.similarity(combined, b))  # ~0.57
print(model.similarity(combined, c))  # ~0.57
```

**Use case:** Sets, prototypes, feature aggregation

### Permutation (Sequence)

Permutation shifts vector coordinates, encoding position or order.

```python
# Permute by 1 position
a_shifted = model.permute(a, k=1)

# Unpermute to recover
a_back = model.unpermute(a_shifted, k=1)

print(model.similarity(a, a_shifted))  # ~0.0 (dissimilar)
print(model.similarity(a, a_back))     # 1.0 (recovered)
```

**Use case:** Sequences, positional encoding, temporal order

## Similarity

Similarity measures how related two vectors are (0 = orthogonal, 1 = identical):

```python
# Same vector
print(model.similarity(a, a))  # 1.0

# Different random vectors
print(model.similarity(a, b))  # ~0.0

# After binding and unbinding (FHRR)
c = model.bind(a, b)
print(model.similarity(a, model.unbind(c, b)))  # 1.0
```

## Practical Example: Role-Filler Binding

Represent "the ball is red":

```python
from holovec import VSA

model = VSA.create('FHRR', dim=2048)

# Define roles and fillers
OBJECT = model.random(seed=1)
COLOR = model.random(seed=2)
ball = model.random(seed=3)
red = model.random(seed=4)

# Create structure: OBJECT→ball, COLOR→red
structure = model.bundle([
    model.bind(OBJECT, ball),
    model.bind(COLOR, red)
])

# Query: what is the color?
query_result = model.unbind(structure, COLOR)
print(model.similarity(query_result, red))   # ~0.7 (high)
print(model.similarity(query_result, ball))  # ~0.0 (low)
```

## Next Steps

- [Core Concepts](../getting-started/core-concepts.md) — Understand the theory
- [Models](../models/index.md) — Choose the right model
- [Encoders](../encoders/index.md) — Encode real data
- [Examples](https://github.com/Twistient/HoloVec/tree/master/examples) — Working code examples
