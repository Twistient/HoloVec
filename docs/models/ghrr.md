**Generalized Holographic Reduced Representations**

GHRR extends FHRR from scalar phasors to m×m unitary matrices, enabling non-commutative binding for compositional structures like trees, graphs, and nested relationships.

## Properties

| Property | Value |
|----------|-------|
| Binding | Element-wise matrix product |
| Inverse | Exact (via conjugate transpose) |
| Commutative | No (tunable) |
| Self-inverse | No |
| Space | Matrix (unitary m×m) |
| Memory | O(d × m²) per vector |

## When to Use

- Order-sensitive relationships (A→B ≠ B→A)
- Hierarchical structures (parent-child, cause-effect)
- Knowledge graphs with directed edges
- When binding order must be recoverable
- State-of-the-art applications (2024)

## Theory

### Vector Space

GHRR vectors are arrays of m×m unitary matrices:

$$\mathbf{v} = [U_1, U_2, ..., U_d]$$

where each $U_j^\dagger U_j = I$ (unitary property).

### Binding Operation

Binding is element-wise matrix multiplication:

$$(\mathbf{a} \otimes \mathbf{b})_j = A_j \cdot B_j$$

Matrix multiplication is non-commutative: $A_j B_j \neq B_j A_j$ in general.

### Unbinding Operation

Unbinding uses conjugate transpose:

$$\text{unbind}(\mathbf{c}, \mathbf{b})_j = C_j \cdot B_j^\dagger$$

This provides exact recovery: `unbind(bind(a, b), b) = a`.

### Diagonality Control

GHRR's non-commutativity is controlled by diagonality parameter:

| Diagonality | Behavior |
|-------------|----------|
| 0.0 | Maximally non-commutative |
| 0.5 | Balanced (default for m=3) |
| 1.0 | Fully commutative (FHRR-like) |

When m=1, GHRR reduces to FHRR.

### Bundling Operation

Bundling sums matrices and projects back to unitary via polar decomposition (SVD):

$$\mathbf{d} = \text{normalize}\left(\sum_j \mathbf{v}_j\right)$$

## Capacity Analysis

GHRR has similar capacity to FHRR but trades scalar dimensions for matrix richness:

$$n_{max} \approx 0.06 \cdot d$$

*Note: GHRR uses effective dimensions = dim × m² (e.g., dim=100, m=3 → 900 effective). Capacity is measured per scalar dimension.

| Dimension | m | Effective | Max Items |
|-----------|---|-----------|-----------|
| 64 | 3 | 576 | ~35 |
| 100 | 3 | 900 | ~55 |
| 256 | 3 | 2304 | ~140 |

## Code Examples

### Basic Usage

```python
from holovec import VSA

# Create GHRR model with 3×3 matrices
model = VSA.create('GHRR', dim=100, matrix_size=3)

# Generate random hypervectors
a = model.random(seed=1)
b = model.random(seed=2)

# Test non-commutativity
ab = model.bind(a, b)
ba = model.bind(b, a)
print(model.similarity(ab, ba))  # ~0.0 (very different)

# Exact recovery still works
c = model.bind(a, b)
a_recovered = model.unbind(c, b)
print(model.similarity(a, a_recovered))  # 1.0
```

### Controlling Diagonality

```python
# Maximally non-commutative
model_nc = VSA.create('GHRR', dim=100, matrix_size=3, diagonality=0.0)

# Commutative (FHRR-like)
model_c = VSA.create('GHRR', dim=100, matrix_size=3, diagonality=1.0)

# Measure commutativity
a, b = model_nc.random(seed=1), model_nc.random(seed=2)
print(model_nc.test_non_commutativity(a, b))  # ~0.0
```

### Directional Relationships

```python
# Encode "A causes B" vs "B causes A"
cause = model.random(seed=1)
effect = model.random(seed=2)

# Order encodes direction
a_causes_b = model.bind(cause, effect)  # A → B
b_causes_a = model.bind(effect, cause)  # B → A

# These are different!
print(model.similarity(a_causes_b, b_causes_a))  # ~0.0
```

## Comparison with Similar Models

| vs Model | GHRR Advantage | GHRR Disadvantage |
|----------|---------------|-------------------|
| FHRR | Non-commutativity | O(m²) more memory |
| VTB | Exact inverse | More computation |
| MAP | Non-commutative, exact | Significantly more memory |

## Memory and Performance

| Matrix Size | Memory Factor | Non-comm Strength |
|-------------|---------------|-------------------|
| m=1 | 1× (= FHRR) | None |
| m=2 | 4× | Moderate |
| m=3 | 9× | Strong |
| m=4 | 16× | Very strong |

Typical usage: m=3 balances expressiveness and efficiency.

## References

- Yeung, W., Zou, A., & Imani, F. (2024). Generalized Holographic Reduced Representations. arXiv:2405.09689
- Plate, T. A. (2003). Holographic Reduced Representations

## See Also

- [Models Overview](../models/index.md) — Compare all models
- [Model-FHRR](../models/fhrr.md) — Scalar version
- [Model-VTB](../models/vtb.md) — Alternative non-commutative model
