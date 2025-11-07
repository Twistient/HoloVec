# Theory Guide: Hyperdimensional Computing & Vector Symbolic Architectures

**Last Updated**: November 5, 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [VSA Models Implemented](#vsa-models-implemented)
4. [Core Operations](#core-operations)
5. [Properties & Characteristics](#properties--characteristics)
6. [Model Comparison](#model-comparison)
7. [Capacity & Scalability](#capacity--scalability)
8. [Applications](#applications)
9. [References](#references)

---

## Introduction

### What is Hyperdimensional Computing?

**Hyperdimensional Computing (HDC)** / **Vector Symbolic Architectures (VSA)** is a computing paradigm inspired by the brain's use of high-dimensional distributed representations. The key idea is to represent concepts and data as large vectors (hypervectors) in very high-dimensional spaces (typically D = 1,000 to 10,000+ dimensions).

### Why High Dimensions?

In high-dimensional spaces, several remarkable properties emerge:

1. **Quasi-orthogonality**: Random vectors are very likely to be nearly orthogonal
2. **Robust representation**: Information is distributed across many dimensions
3. **Noise tolerance**: Small perturbations don't significantly affect similarity
4. **Compositional structure**: Complex structures can be encoded in fixed-size vectors

### Historical Context

- **1990s**: Tony Plate introduces Holographic Reduced Representations (HRR)
- **2009**: Pentti Kanerva formalizes Hyperdimensional Computing framework
- **2020s**: Multiple VSA variants developed for different application domains
- **2024**: Modern implementations with neural network integration

---

## Mathematical Foundations

### High-Dimensional Spaces

A hypervector $\mathbf{h} \in \mathbb{V}^D$ lives in a $D$-dimensional vector space $\mathbb{V}$.

**Common vector spaces**:

- **Real**: $\mathbb{V} = \mathbb{R}^D$ (continuous values)
- **Bipolar**: $\mathbb{V} = \{-1, +1\}^D$ (binary with sign)
- **Binary**: $\mathbb{V} = \{0, 1\}^D$ (unsigned binary)
- **Complex**: $\mathbb{V} = \mathbb{C}^D$ (complex numbers)
- **Sparse**: $\mathbb{V} = \{0, 1\}^D$ with $\|h\|_0 \ll D$ (few non-zeros)
- **Matrix**: $\mathbb{V} = \mathbb{C}^{D \times m \times m}$ (matrices per dimension)

### Quasi-Orthogonality

**Definition**: Random vectors $\mathbf{a}, \mathbf{b} \in \mathbb{V}^D$ sampled independently are approximately orthogonal with high probability.

**Mathematical statement**:
$$\mathbb{P}(\cos(\mathbf{a}, \mathbf{b}) \approx 0) \to 1 \quad \text{as } D \to \infty$$

**Intuition**: In high dimensions, "most of the space" is far from any given vector.

**Example** (Real vectors):

- For $D = 10,000$ and $\mathbf{a}, \mathbf{b} \sim \mathcal{N}(0, I)$:
- $\mathbb{P}(85° < \angle(\mathbf{a}, \mathbf{b}) < 95°) > 0.99$

### Similarity Measures

**Cosine Similarity** (Real/Bipolar):
$$\delta(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \cos(\mathbf{a}, \mathbf{b})$$

**Hamming Distance** (Binary):
$$d_H(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^D \mathbb{1}[a_i \neq b_i]$$

**Overlap Similarity** (Sparse Binary):
$$\delta(\mathbf{a}, \mathbf{b}) = \frac{|\{i : a_i = 1 \land b_i = 1\}|}{D \cdot p}$$

**Complex Angle Distance** (Complex):
$$\delta(\mathbf{a}, \mathbf{b}) = \frac{1}{D} \sum_{i=1}^D \cos(\arg(a_i) - \arg(b_i))$$

---

## VSA Models Implemented

### 1. MAP (Multiply-Add-Permute)

**Space**: Bipolar $\{-1, +1\}^D$ or Real $\mathbb{R}^D$

**Operations**:

- **Binding**: $\mathbf{c} = \mathbf{a} \odot \mathbf{b}$ (element-wise multiplication)
- **Unbinding**: $\mathbf{a} = \mathbf{c} \odot \mathbf{b}$ (self-inverse)
- **Bundling**: $\mathbf{s} = \sum_i \mathbf{a}_i$ (sum + normalization)

**Properties**:

- ✅ Self-inverse
- ✅ Commutative
- ✅ Associative
- ✅ Exact inverse (bipolar)
- ⚠️ Approximate inverse (real)

**Best for**: Simple operations, hardware implementations, self-inverse requirements

**Reference**: Gayler (1998), Schlegel et al. (2022)

---

### 2. FHRR (Fourier Holographic Reduced Representations)

**Space**: Complex unit circle $\{e^{i\theta} : \theta \in [0, 2\pi)\}^D$

**Operations**:

- **Binding**: $c_i = a_i \cdot b_i = e^{i(\theta_a + \theta_b)}$ (angle addition)
- **Unbinding**: $a_i = c_i \cdot b_i^* = e^{i(\theta_c - \theta_b)}$ (angle subtraction)
- **Bundling**: $s_i = \arg\left(\sum_j e^{i\theta_j}\right)$ (vector addition → angle)

**Properties**:

- ✅ Exact inverse
- ✅ Commutative
- ✅ Associative
- ✅ Best bundling capacity (~0.35D items)
- ✅ Element-wise operations (O(D))

**Best for**: High capacity requirements, exact recovery, kernel methods

**Reference**: Plate (2003), Schlegel et al. (2022)

---

### 3. HRR (Holographic Reduced Representations)

**Space**: Real $\mathbb{R}^D$

**Operations**:

- **Binding**: $\mathbf{c} = \mathbf{a} \circledast \mathbf{b}$ (circular convolution)
  $$c_j = \sum_{k=0}^{D-1} a_k b_{\text{mod}(j-k, D)}$$

- **Unbinding**: $\mathbf{a} \approx \mathbf{c} \circledast \tilde{\mathbf{b}}$ (circular correlation)
  $$a_j \approx \sum_{k=0}^{D-1} c_k b_{\text{mod}(k+j, D)}$$

- **Bundling**: $\mathbf{s} = \sum_i \mathbf{a}_i / \|\sum_i \mathbf{a}_i\|$

**Properties**:

- ⚠️ Approximate inverse (~0.70 similarity after one unbinding)
- ✅ Commutative
- ✅ Associative
- ⚠️ Degrades with depth (~0.05 after 40 operations)
- ⚠️ O(D log D) via FFT

**Best for**: Standard VSA tasks, good balance of properties

**Reference**: Plate (1995), Schlegel et al. (2022)

---

### 4. BSC (Binary Spatter Codes)

**Space**: Binary $\{0, 1\}^D$

**Operations**:

- **Binding**: $\mathbf{c} = \mathbf{a} \oplus \mathbf{b}$ (XOR)
- **Unbinding**: $\mathbf{a} = \mathbf{c} \oplus \mathbf{b}$ (XOR is self-inverse)
- **Bundling**: Majority vote

**Properties**:

- ✅ Self-inverse
- ✅ Exact inverse
- ✅ Commutative
- ✅ Associative
- ✅ Simple hardware (XOR gates)
- ⚠️ Lower capacity than FHRR

**Best for**: Hardware implementations, digital circuits, exact operations

**Reference**: Kanerva (1993), Schlegel et al. (2022)

---

### 5. GHRR (Generalized Holographic Reduced Representations)

**Space**: Unitary matrices $U(m)^D$ where each element is $m \times m$ unitary matrix

**Operations**:

- **Binding**: $[\mathbf{A}]_j \cdot [\mathbf{B}]_j$ (element-wise matrix multiplication)
- **Unbinding**: $[\mathbf{C}]_j \cdot [\mathbf{B}]_j^\dagger$ (conjugate transpose)
- **Bundling**: $\sum_i [\mathbf{A}_i]_j$ (element-wise matrix addition)

**Properties**:

- ✅ Exact inverse
- ✅ **Non-commutative** (tunable via diagonality)
- ⚠️ Non-associative
- ✅ Better capacity than FHRR for bound vectors
- ✅ No permutation needed for order
- ⚠️ Higher memory (D × m × m)

**Best for**: Compositional structures, trees, nested relationships

**Reference**: Yeung, Zou, & Imani (2024)

---

### 6. VTB (Vector-derived Transformation Binding)

**Space**: Real $\mathbb{R}^D$

**Operations** (implemented via circular convolution):

- **Binding**: $\mathbf{c} = \mathbf{a} \circledast \mathbf{b}$ (circular convolution)
- **Unbinding**: $\mathbf{a} \approx \mathbf{c} \circledast \tilde{\mathbf{b}}$ (circular correlation)
- **Bundling**: $\mathbf{s} = \sum_i \mathbf{a}_i / \|\sum_i \mathbf{a}_i\|$

**Properties**:

- ⚠️ Approximate inverse (~0.69 similarity)
- ✅ Commutative (via circular convolution)
- ✅ Better unbinding than HRR
- ⚠️ O(D log D) via FFT

**Best for**: Better approximate unbinding, similar to HRR with improvements

**Reference**: Gosmann & Eliasmith (2019), Schlegel et al. (2022)

**Note**: Our implementation uses FFT-based circular convolution (O(D log D)) instead of explicit circulant matrices (O(D²)) for efficiency.

---

### 7. BSDC (Binary Sparse Distributed Codes)

**Space**: Sparse binary $\{0, 1\}^D$ with $\|h\|_0 = pD$ where $p = 1/\sqrt{D}$

**Operations**:

- **Binding**: $\mathbf{c} = \mathbf{a} \oplus \mathbf{b}$ (XOR)
- **Unbinding**: $\mathbf{a} = \mathbf{c} \oplus \mathbf{b}$ (self-inverse)
- **Bundling**: Top-k selection (maintains sparsity)

**Properties**:

- ✅ Self-inverse
- ✅ Exact inverse
- ✅ Optimal sparsity formula
- ✅ **~50x memory savings** at D=10,000
- ✅ Excellent bundling capacity
- ⚠️ Density increases with binding

**Best for**: Memory-constrained applications, hardware implementations

**Reference**: Kanerva (2009), Rachkovskij (2001)

---

## Core Operations

### Bundling (+)

**Purpose**: Superimpose multiple hypervectors to create a representation similar to all inputs.

**Property**: If $\mathbf{h} = \mathbf{a} + \mathbf{b} + \mathbf{c}$, then:

- $\delta(\mathbf{h}, \mathbf{a}) > 0$ (similar)
- $\delta(\mathbf{h}, \mathbf{b}) > 0$ (similar)
- $\delta(\mathbf{h}, \mathbf{c}) > 0$ (similar)

**Cognitive interpretation**: Memory consolidation, prototype formation

**Example**:

```python
from holovec import VSA

model = VSA.create('FHRR', dim=10000)

# Create base concepts
dog = model.random(seed=1)
cat = model.random(seed=2)
bird = model.random(seed=3)

# Bundle into "pets" concept
pets = model.bundle([dog, cat, bird])

# pets is similar to all three
print(model.similarity(pets, dog))   # ~0.58
print(model.similarity(pets, cat))   # ~0.58
print(model.similarity(pets, bird))  # ~0.58
```

---

### Binding (⊗)

**Purpose**: Connect two hypervectors to create a representation dissimilar to both.

**Properties** (Plate 1997):

1. **Dissimilarity**: $\delta(\mathbf{a} \otimes \mathbf{b}, \mathbf{a}) \approx 0$
2. **Similarity preservation**: If $\mathbf{a}' \approx \mathbf{a}$ and $\mathbf{b}' \approx \mathbf{b}$, then $\mathbf{a}' \otimes \mathbf{b}' \approx \mathbf{a} \otimes \mathbf{b}$
3. **Invertible**: Can recover $\mathbf{a}$ from $\mathbf{c} = \mathbf{a} \otimes \mathbf{b}$ using $\mathbf{b}$

**Cognitive interpretation**: Association, role-filler binding

**Example** (Role-filler pairs):

```python
# Roles and fillers
role_subject = model.random(seed=10)
role_verb = model.random(seed=11)
role_object = model.random(seed=12)

filler_cat = model.random(seed=20)
filler_chased = model.random(seed=21)
filler_mouse = model.random(seed=22)

# Encode "cat chased mouse"
sentence = model.bundle([
    model.bind(role_subject, filler_cat),
    model.bind(role_verb, filler_chased),
    model.bind(role_object, filler_mouse)
])

# Query: What is the subject?
subject = model.unbind(sentence, role_subject)
print(model.similarity(subject, filler_cat))    # High!
print(model.similarity(subject, filler_mouse))  # Low
```

---

### Permutation (ρ)

**Purpose**: Reorder vector elements to encode position or sequence information.

**Property**: $\delta(\rho(\mathbf{h}), \mathbf{h}) \approx 0$ (dissimilar)

**Usage**: Encoding sequences without binding

**Example**:

```python
# Encode sequence [A, B, C]
A = model.random(seed=1)
B = model.random(seed=2)
C = model.random(seed=3)

sequence = model.bundle([
    A,                    # Position 0
    model.permute(B, 1),  # Position 1
    model.permute(C, 2)   # Position 2
])

# Different from [C, B, A]
sequence2 = model.bundle([
    C,
    model.permute(B, 1),
    model.permute(A, 2)
])

print(model.similarity(sequence, sequence2))  # Low!
```

---

## Properties & Characteristics

### Self-Inverse vs Non Self-Inverse

**Self-Inverse** (MAP, BSC, BSDC):

- Binding = Unbinding operation
- $\mathbf{a} \otimes \mathbf{b} \otimes \mathbf{b} = \mathbf{a}$
- Enables elegant solutions (e.g., "Dollar of Mexico" problem)

**Non Self-Inverse** (FHRR, HRR, GHRR, VTB):

- Separate unbinding operation
- More complex but often more powerful
- May require "readout machines" for hierarchical structures

---

### Exact vs Approximate Inverse

**Exact Inverse** (MAP-bipolar, FHRR, BSC, GHRR, BSDC):

- Perfect recovery: $\mathbf{a} = \mathbf{c} \oslash \mathbf{b}$ when $\mathbf{c} = \mathbf{a} \otimes \mathbf{b}$
- No degradation with multiple bind/unbind cycles
- Similarity after recovery = 1.0

**Approximate Inverse** (MAP-real, HRR, VTB):

- Imperfect recovery: $\mathbf{a} \approx \mathbf{c} \oslash \mathbf{b}$
- Degrades with sequence depth
- Similarity after recovery: 0.50-0.70 (single), 0.05-0.25 (after 40 ops)
- Often requires "clean-up memory" for practical use

---

### Commutativity

**Commutative** (MAP, FHRR, HRR, BSC, VTB, BSDC):

- $\mathbf{a} \otimes \mathbf{b} = \mathbf{b} \otimes \mathbf{a}$
- Cannot distinguish order without additional techniques (permutation)
- Simpler algebraic properties

**Non-Commutative** (GHRR):

- $\mathbf{a} \otimes \mathbf{b} \neq \mathbf{b} \otimes \mathbf{a}$
- Naturally encodes order
- Better for hierarchical/compositional structures
- Tunable via diagonality parameter

---

### Associativity

**Associative** (MAP, FHRR, HRR, BSC, BSDC):

- $(\mathbf{a} \otimes \mathbf{b}) \otimes \mathbf{c} = \mathbf{a} \otimes (\mathbf{b} \otimes \mathbf{c})$
- Order of operations doesn't matter
- Simplifies nested bindings

**Non-Associative** (GHRR, VTB for unbinding):

- Order of operations matters
- More expressive for structured data

---

## Model Comparison

### Quick Reference Table

| Model | Space | Self-Inv | Exact | Comm | Assoc | Capacity | Complexity |
|-------|-------|----------|-------|------|-------|----------|------------|
| MAP | Bipolar/Real | ✅ | ✅/⚠️ | ✅ | ✅ | Medium | O(D) |
| FHRR | Complex | ❌ | ✅ | ✅ | ✅ | **Best** | O(D) |
| HRR | Real | ❌ | ⚠️ | ✅ | ✅ | Medium | O(D log D) |
| BSC | Binary | ✅ | ✅ | ✅ | ✅ | Medium | O(D) |
| GHRR | Matrices | ❌ | ✅ | ❌ | ❌ | High | O(Dm²) |
| VTB | Real | ❌ | ⚠️ | ✅ | ❌ | Medium | O(D log D) |
| BSDC | Sparse | ✅ | ✅ | ✅ | ✅ | High | O(Dp) |

Legend:

- ✅ = Yes/Good
- ⚠️ = Approximate/Conditional
- ❌ = No

---

### When to Use Each Model

#### Use MAP when

- Need self-inverse binding
- Implementing on simple hardware
- Want exact recovery (bipolar mode)
- Prioritize simplicity

#### Use FHRR when

- Need maximum bundling capacity
- Want exact inverse
- Can work with complex numbers
- Kernel methods are relevant

#### Use HRR when

- Standard VSA tasks
- Good balance of properties needed
- Real-valued operations preferred
- Approximate recovery acceptable

#### Use BSC when

- Implementing on digital circuits
- Need exact XOR-based operations
- Binary operations required
- Simple hardware

#### Use GHRR when

- Encoding compositional structures
- Need non-commutative binding
- Working with trees/graphs
- Order is semantically important

#### Use VTB when

- Need slightly better unbinding than HRR
- Approximate recovery acceptable
- Real-valued operations preferred

#### Use BSDC when

- Memory is constrained
- Need sparse representations
- Exact recovery required
- Hardware efficiency critical

---

## Capacity & Scalability

### Bundling Capacity

**Definition**: Maximum number of hypervectors that can be bundled while maintaining retrieval accuracy.

**Empirical Results** (Schlegel et al. 2022):

For 99% retrieval accuracy:

| Model | Dimensions Needed (for k=15) | Efficiency |
|-------|------------------------------|------------|
| FHRR | ~330 | ⭐⭐⭐⭐⭐ Best |
| BSDC | ~320 | ⭐⭐⭐⭐⭐ Best |
| HRR | ~510 | ⭐⭐⭐⭐ Good |
| VTB | ~510 | ⭐⭐⭐⭐ Good |
| MAP-C | ~640 | ⭐⭐⭐ Adequate |
| BSC | ~750 | ⭐⭐⭐ Adequate |
| MAP-B | ~790 | ⭐⭐ Lower |

**Scaling**: Most models require dimensions to scale linearly with number of items: $D \propto k$

---

### Memory Efficiency

**Dense Models** (MAP, FHRR, HRR, BSC, VTB, GHRR):

- Memory: $O(D)$ for scalars, $O(Dm^2)$ for matrices
- Storage: 32-64 bits per element

**Sparse Models** (BSDC):

- Memory: $O(Dp)$ where $p = 1/\sqrt{D}$
- **Savings**: ~50x at D=10,000 (100 ones vs 5,000)
- Can use sparse data structures

---

### Computational Complexity

| Operation | MAP | FHRR | HRR | BSC | GHRR | VTB | BSDC |
|-----------|-----|------|-----|-----|------|-----|------|
| Random | O(D) | O(D) | O(D) | O(D) | O(Dm²) | O(D) | O(Dp) |
| Bind | O(D) | O(D) | O(D log D) | O(D) | O(Dm³) | O(D log D) | O(D) |
| Unbind | O(D) | O(D) | O(D log D) | O(D) | O(Dm³) | O(D log D) | O(D) |
| Bundle | O(kD) | O(kD) | O(kD) | O(kD) | O(kDm²) | O(kD) | O(kDp) |
| Similarity | O(D) | O(D) | O(D) | O(D) | O(Dm²) | O(D) | O(Dp) |

Where:

- $D$ = dimension
- $k$ = number of vectors to bundle
- $m$ = matrix size (GHRR)
- $p$ = sparsity (~$1/\sqrt{D}$)

---

### Capacity Formulas

**FHRR Capacity** (Plate 2003):
$$k_{\max} \approx 0.35D$$

**BSDC Optimal Sparsity** (Rachkovskij 2001):
$$p = \frac{1}{\sqrt{D}}$$

**Expected Ones**:
$$N_{\text{ones}} = pD = \sqrt{D}$$

---

## Applications

### Classic Applications

1. **Analogical Reasoning**
   - "What is the Dollar of Mexico?" → Peso
   - Self-inverse models excel

2. **Semantic Memory**
   - Store facts as bundled role-filler pairs
   - Query by unbinding

3. **Sequence Encoding**
   - Text, time series, trajectories
   - Use permutation or GHRR

4. **Image Similarity**
   - Encode spatial structure
   - Fast similarity search

5. **Cognitive Architectures**
   - Working memory
   - Reasoning systems
   - Symbol grounding

### Modern Applications

1. **Few-Shot Learning**
   - Fast adaptation with hypervectors
   - One-shot classification

2. **Neuromorphic Computing**
   - Brain-inspired hardware
   - Energy-efficient inference

3. **Edge AI**
   - Low-power devices
   - IoT sensors

4. **Bioinformatics**
   - DNA sequence analysis
   - Protein structure

5. **Robotic Navigation**
   - Place recognition
   - SLAM (Simultaneous Localization and Mapping)

---

## References

### Foundational Papers

1. **Plate, T. A. (2003)**
   "Holographic Reduced Representations"
   *IEEE Transactions on Neural Networks*
   - Original HRR and FHRR formulation

2. **Kanerva, P. (2009)**
   "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors"
   *Cognitive Computation*
   - Foundational HDC framework

3. **Gayler, R. W. (1998)**
   "Multiplicative Binding, Representation Operators & Analogy"
   - MAP architecture

4. **Kanerva, P. (1993)**
   "Sparse Distributed Memory and Related Models"
   - Binary spatter codes, SDM

### Modern Advances

5. **Schlegel, K., Neubert, P., & Protzel, P. (2022)**
   "A Comparison of Vector Symbolic Architectures"
   *Artificial Intelligence Review*
   - Comprehensive comparison of 11 VSA implementations

6. **Yeung, C., Zou, Z., & Imani, M. (2024)**
   "Generalized Holographic Reduced Representations"
   *arXiv:2405.09689v1*
   - GHRR with non-commutative binding

7. **Gosmann, J., & Eliasmith, C. (2019)**
   "Vector-Derived Transformation Binding"
   - VTB model

8. **Rachkovskij, D. A. (2001)**
   "Representation and Processing of Structures with Binary Sparse Distributed Codes"
   - Optimal sparsity for BSDC

### Survey Papers

9. **Kleyko, D., Osipov, E., & Gayler, R. W. (2023)**
   "A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations"
   *ACM Computing Surveys*
   - Comprehensive modern survey

10. **Frady, E. P., et al. (2021)**
    "Computing on Functions Using Randomized Vector Representations"
    *Physical Review Research*
    - Fractional power encoding, kernel methods

---

## Further Reading

### Implementation Guides

- **holovec Documentation**: Complete API reference and examples
- **VSA Toolbox** (Schlegel et al.): MATLAB implementation
- **Nengo**: Neural engineering framework with HDC support

### Tutorials

- **Kanerva (2009)**: Best introductory paper
- **Kleyko et al. (2023)**: Modern comprehensive survey
- **Plate (2003)**: Deep dive into HRR theory

### Applications Papers

- Robotics: Neubert et al. (2019)
- Language: Joshi et al. (2017)
- Classification: Imani et al. (2019)
- Cognitive Architecture: Eliasmith (2013)

---

## Appendix: Mathematical Notation

### Symbols

- $\mathbf{h}$: Hypervector
- $D$: Dimension
- $\otimes$: Binding operation
- $\oslash$: Unbinding operation
- $+$: Bundling operation
- $\rho$: Permutation operation
- $\delta(\cdot, \cdot)$: Similarity measure
- $\|\cdot\|$: Norm
- $\odot$: Element-wise (Hadamard) product
- $\oplus$: XOR operation
- $\circledast$: Circular convolution
- $\circledast$: Circular correlation

### Notation Conventions

- **Bold lowercase** ($\mathbf{a}$): Vectors
- **Bold uppercase** ($\mathbf{A}$): Matrices or hypervectors with matrix elements
- **Italic** ($D$, $k$, $m$): Scalars
- **Blackboard bold** ($\mathbb{R}$, $\mathbb{C}$): Number sets

---

**End of Theory Guide**

For practical usage examples, see the main README and examples directory.
For implementation details, see the API documentation.
For validation results, see `docs/validation_results.md`.
