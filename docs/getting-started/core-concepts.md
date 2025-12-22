This page explains the foundational ideas behind hyperdimensional computing and Vector Symbolic Architectures.

## Why High Dimensions?

In high-dimensional spaces (~1,000-10,000 dimensions), random vectors are almost always nearly orthogonal. This creates a vast representational space where:

- **10,000 dimensions** can store ~1,000 distinct items with high reliability
- Adding noise barely affects retrieval
- Patterns emerge from algebraic operations, not learned weights

### The Orthogonality Property

In *d* dimensions, two random unit vectors have expected cosine similarity:

$$E[\cos(\theta)] \approx 0$$

with standard deviation:

$$\sigma \approx \frac{1}{\sqrt{d}}$$

For *d* = 10,000, random vectors have similarity within ±0.01 of zero—effectively orthogonal.

## Three Core Operations

All VSA models implement three fundamental operations:

| Operation | Input | Output | Property |
|-----------|-------|--------|----------|
| **Bind** | A, B | A ⊗ B | Dissimilar to both |
| **Bundle** | A, B | A + B | Similar to both |
| **Permute** | A | ρ(A) | Shifted, dissimilar to A |

### Binding (⊗)

**Creates an association between two vectors.**

| Property | Description |
|----------|-------------|
| Result | Dissimilar to both inputs |
| Purpose | Key-value pairs, associations |
| Inverse | Unbinding recovers operand |

```python
c = model.bind(a, b)      # c ≈ a ⊗ b
a = model.unbind(c, b)    # Recover a
```

The binding operation varies by model:
- **FHRR**: Element-wise complex multiplication
- **MAP**: Element-wise real multiplication
- **HRR**: Circular convolution
- **BSC**: XOR

### Bundling (+)

**Combines multiple vectors into a superposition.**

| Property | Description |
|----------|-------------|
| Result | Similar to all inputs |
| Purpose | Sets, prototypes, averaging |
| Inverse | None (information-preserving mix) |

```python
combined = model.bundle([a, b, c])
# combined is similar to a, b, and c
```

Bundling is typically element-wise addition followed by normalization. Each component contributes equally to the result.

### Permutation (ρ)

**Shifts vector coordinates to encode order.**

| Property | Description |
|----------|-------------|
| Result | Dissimilar to input |
| Purpose | Positions, sequences |
| Inverse | Unpermute reverses shift |

```python
a_pos1 = model.permute(a, k=1)
a_pos2 = model.permute(a, k=2)
# a_pos1 ≠ a_pos2 ≠ a (all dissimilar)
```

## Composing Operations

The power of VSA comes from combining operations to build complex structures.

### Example: Encoding a Sequence

Encode the sequence `[A, B, C]`:

```python
# Method 1: Position binding
seq = model.bundle([
    model.bind(pos_0, A),
    model.bind(pos_1, B),
    model.bind(pos_2, C)
])

# Method 2: Permutation
seq = model.bundle([
    A,
    model.permute(B, k=1),
    model.permute(C, k=2)
])
```

### Example: Encoding a Record

Encode `{name: "Alice", age: 30}`:

```python
# Role-filler binding
record = model.bundle([
    model.bind(ROLE_name, vec_alice),
    model.bind(ROLE_age, vec_30)
])

# Query: what is the name?
result = model.unbind(record, ROLE_name)
# result ≈ vec_alice
```

## Capacity and Noise

### Bundle Capacity

The number of items that can be bundled while maintaining reliable retrieval:

$$n_{max} \approx \frac{d}{2 \ln(d)}$$

For *d* = 10,000: ~500 items per bundle.

### Binding Depth

Nested bindings reduce similarity during recovery. For FHRR (exact inverse):
- Single binding: 100% recovery
- Nested bindings: Still 100% recovery

For approximate models (HRR, VTB):
- Single binding: ~70% recovery
- Nested bindings: Progressive degradation

### Noise Tolerance

HDC representations degrade gracefully:

| Noise Level | Expected Similarity |
|-------------|---------------------|
| 0% | 1.0 |
| 10% | ~0.90 |
| 30% | ~0.70 |
| 50% | ~0.50 |

## Mental Model

Think of hypervectors as:
- **Points in a vast space** where almost every random point is equally far from every other
- **Holographic** — each dimension carries partial information about the whole
- **Algebraic** — operations have well-defined properties (commutativity, associativity, distributivity)

## Key Differences from Neural Networks

| Aspect | Neural Networks | VSA/HDC |
|--------|-----------------|---------|
| Learning | Gradient descent | One-shot encoding |
| Representation | Learned embeddings | Algebraic composition |
| Interpretability | Black-box activations | Transparent operations |
| Training data | Large datasets | No training required |
| Noise handling | Overfitting risk | Built-in tolerance |

## Further Reading

- [Backends](../architecture/backends.md) — Computational infrastructure
- [Spaces](../architecture/spaces.md) — Vector types and similarity
- [Models](../models/index.md) — Specific VSA implementations
- [References](../reference/bibliography.md) — Academic papers
