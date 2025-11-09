<div align="center">
  <img src="docs/source/_static/logo.svg" alt="HoloVec Logo" width="400">
</div>

<div align="center">

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Compositional algebra in 10,000-D, with exact/approximate inverses and kernel-shaped encoders.**

[Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## Overview

**HoloVec** is a modern library for building and manipulating compositional representations in high-dimensional spaces using Vector Symbolic Architectures (VSA/HDC). It provides:

- **🎯 Compositional Power**: Bind structure, bundle sets, unbind components with exact or approximate inverses
- **🔬 Kernel-Aware Encoders**: FPE/RFF with rich phase distributions (Gaussian, Laplace, Cauchy, Student) plus multivariate and periodic encoders
- **⚡ Multi-Backend**: Run on NumPy, PyTorch (GPU), or JAX (JIT) with unified API
- **🧮 Modern VSA Models**: 7 validated models from HRR/FHRR (commutative) to GHRR/MBAT (non-commutative)
- **🎲 Sparse Codes**: BSDC-SEG with segment-wise operations and efficient search
- **🔍 Practical Retrieval**: Codebook/ItemStore APIs with resonator factorization

### Why Hyperdimensional Computing?

Hyperdimensional computing represents information as high-dimensional vectors (~1,000-10,000 dimensions) enabling:

- ✅ **One-shot learning** without gradient descent
- ✅ **Robust, noise-tolerant** representations
- ✅ **Explainable AI** with transparent operations
- ✅ **Efficient hardware** (neuromorphic chips, FPGAs)
- ✅ **Compositional reasoning** with symbolic structure

### What Makes HoloVec Different?

- **Non-commutative first-class**: GHRR with tunable diagonality, MBAT/VTB transform binding
- **Kernel theory integration**: FPE = RFF with multiple phase families and mixtures (M2)
- **Production-ready**: 480+ tests, 90-98% coverage, type-safe, zero dependencies beyond NumPy
- **Clean separation**: HoloVec (algebra + encoders), HoloMem (memories + training), HoloGraph (graph algebra)

<details>
<summary><b>📊 Technical Positioning & Scope</b></summary>

### What HoloVec Provides

**HoloVec = Algebra-first + Kernel-aware HDC core**

- **Compositional algebra**: HRR/FHRR (commutative), GHRR (exact non-commutative with diagonality), VTB/MBAT (non-commutative transform), BSC/BSDC + BSDC-SEG (sparse/segment)
- **Kernel encoders**: FPE = RFF with phase families + mixtures, multivariate (VectorFPE), periodic encoders; explicit link to kernel methods
- **Retrieval primitives**: Codebook/ItemStore, brute-force cleanup, resonator (hard/soft, temperature)

### Companion Libraries (Separate)

- **HoloMem**: Advanced memories + learning (SDM variants, attention/Hopfield cleanup, learned β/α, GPU batched retrieval)
- **HoloGraph**: Outer-product graph embeddings + algebra (tensor/outer-product, powers, subgraphs, homomorphism tests)

### Feature Matrix

| Category | Features |
|----------|----------|
| **Algebra (binding)** | HRR (conv), FHRR (phasor, exact inverse), GHRR (matrix-unitary, non-commutative, diagonality), VTB/MBAT (transform), BSC/BSDC (XOR), BSDC-SEG (segments) |
| **Encoders** | Scalar FPE (Gaussian/Laplace/Cauchy/Student/Uniform) + mixtures; VectorFPE (multivariate); Periodic/angle; Thermometer; Level |
| **Retrieval** | Codebook + ItemStore; Brute-force cleanup; Resonator (hard/soft, temperature, early-stop); batched similarity |
| **Sparse/Segment** | BSDC-SEG (exact 1-hot/segment); segment-wise masking/rotation/permutation; segment-pattern search |
| **Backends** | NumPy (base), PyTorch (GPU), JAX (JIT); helper parity (complex, FFT, softmax, angle, power, where, etc.) |

### Key Differentiators

**Already in HoloVec:**
- GHRR with diagonality → flexible non-commutativity and exact inverses
- VTB/MBAT transform binding → directional facts without heavy permutations
- FPE "done right" → multiple phase families, mixtures (M2), multivariate, periodic
- BSDC-SEG → exact segmented sparsity + segment search utilities
- Multi-backend (NumPy/Torch/JAX) + kernel framing approachable for ML engineers

**What we delegate:**
- Deep framework-specific training pipelines (→ HoloMem)
- Graph-specific algebra (→ HoloGraph)

</details>

---

## ⚡ Quick Start

```python
from holovec import VSA

# Create model (FHRR has best capacity)
model = VSA.create('FHRR', dim=2048)

# Bind and unbind vectors
a, b = model.random(), model.random()
c = model.bind(a, b)
a_recovered = model.unbind(c, b)
print(f"Similarity: {model.similarity(a, a_recovered):.3f}")  # ~0.99

# Encode continuous data with kernel-aware encoder
from holovec.encoders import VectorFPE
fpe = VectorFPE(model, input_dim=3, bandwidth=0.5, phase_dist='gaussian')
hx = fpe.encode([0.2, 1.4, -0.7])

# Store and retrieve
from holovec.retrieval import ItemStore, Codebook
cb = Codebook({f"item{i}": model.random(seed=100+i) for i in range(10)},
              backend=model.backend)
store = ItemStore(model).fit(cb)
print(store.query(cb._items['item3'], k=3))
```

---

## 📦 Installation

> [!IMPORTANT]
> HoloVec is not yet published to PyPI. Install from source for now.

### From Source (Current Method)

```bash
git clone https://github.com/Twistient/HoloVec.git
cd HoloVec

# Basic install (NumPy only)
pip install -e .

# Or with uv (faster)
uv pip install -e .
```

### Optional Backends

<details>
<summary><b>GPU Support (PyTorch)</b></summary>

```bash
pip install -e .[torch]
# or
uv pip install -e .[torch]
```
</details>

<details>
<summary><b>JIT Compilation (JAX)</b></summary>

```bash
pip install -e .[jax]
# or
uv pip install -e .[jax]
```
</details>

<details>
<summary><b>All Features (Development)</b></summary>

Includes all backends, dev tools (pytest, black, ruff, mypy), and docs (sphinx):

```bash
pip install -e .[all]
# or
uv pip install -e .[all]
```
</details>

### Future (when published to PyPI)

```bash
pip install holovec
# or
uv pip install holovec
```

---

## 🚀 Core Concepts

### Vector Symbolic Architectures (VSA)

VSAs represent information as high-dimensional vectors with three fundamental operations:

#### 1. Binding (⊗) - Associates two vectors → dissimilar result

```python
role_filler = model.bind(role, filler)
```

Creates structured representations like `color: red` or `position: 3`

#### 2. Bundling (+) - Superpose multiple vectors → similar to all inputs

```python
set_vector = model.bundle([item1, item2, item3])
```

Represents collections, prototypes, and averages

#### 3. Permutation (ρ) - Reorders coordinates → preserves similarity

```python
position_encoded = model.permute(vector, k)
```

Encodes sequences, positions, and temporal order

### VSA Model Comparison

Different models have different algebraic properties optimized for specific use cases:

| Model | Binding | Inverse | Capacity | Best For |
|-------|---------|---------|----------|----------|
| **FHRR** | Complex ∗ | Exact (conjugate) | Best (~330 dim) | Continuous data, highest accuracy |
| **GHRR** | Matrix product | Approximate | Excellent | State-of-the-art (2024), non-commutative |
| **MAP** | Element × | Self-inverse | Good (~510 dim) | Hardware, neuromorphic chips |
| **HRR** | Circular conv | Approximate | Good (~510 dim) | Classic baseline |
| **VTB** | Matrix transform | Learned | Excellent | Adaptive representations |
| **BSC** | XOR | Self-inverse | Good | Binary operations, FPGA |
| **BSDC** | Segment sample | Approximate | Very Good | Sparse data, memory efficient |

---

## 🎯 Features

### 🔧 VSA Models (7 Validated)

All models validated against academic literature with comprehensive property-based testing:

- **MAP** (Multiply-Add-Permute): Self-inverse, neuromorphic-friendly
- **FHRR** (Fourier HRR): Complex-valued, exact inverses, best capacity
- **HRR** (Holographic RR): Classic circular convolution
- **BSC** (Binary Spatter Codes): XOR-based for binary vectors
- **BSDC** (Binary Sparse DC): Sparse binary, memory-efficient
- **BSDC-SEG** (Segmented BSDC): Block-based sparse codes with segment operations
- **GHRR** (Generalized HRR): 2024 SOTA with matrix binding, non-commutative
- **VTB** (Vector-derived Transformation): Learned transformation matrices

### 🖥️ Computational Backends (3 Frameworks)

Write once, run anywhere with automatic backend selection:

```python
# Automatic best backend
model = VSA.create('FHRR', dim=10000)

# Explicit backend selection
model = VSA.create('FHRR', dim=10000, backend='torch', device='cuda')

# Query capabilities
from holovec import backend_info
info = backend_info()
print(info['available_backends'])  # ['numpy', 'torch', 'jax']
```

**Available backends:**
- **NumPy** (default): Pure CPU, zero dependencies, perfect for prototyping
- **PyTorch**: GPU acceleration (CUDA/Metal), neural network integration
- **JAX**: JIT compilation (10-100x speedup), TPU support, auto-differentiation

### 📊 Encoders (8 Production-Ready)

Transform diverse data types into hypervectors:

<details>
<summary><b>Scalar Encoders</b></summary>

- **FractionalPowerEncoder**: Continuous values with locality preservation
- **ThermometerEncoder**: Smooth ordinal encoding
- **LevelEncoder**: Discrete categorical values
</details>

<details>
<summary><b>Sequence Encoders</b></summary>

- **PositionBindingEncoder**: Order-sensitive sequences
- **NGramEncoder**: N-gram context windows
- **TrajectoryEncoder**: Temporal trajectories
</details>

<details>
<summary><b>Spatial & Structured Encoders</b></summary>

- **ImageEncoder**: 2D grids and images
- **VectorEncoder**: Multi-dimensional feature vectors
</details>

### 🔍 Cleanup & Retrieval

Recover clean vectors from noisy queries:

- **BruteForceCleanup**: Exhaustive search, guaranteed optimal
- **ResonatorCleanup**: 10-100x faster, iterative factorization
- **Codebook**: Persistent label→vector storage with JSON serialization
- **ItemStore**: High-level retrieval interface with batch operations

### 🛠️ Utilities

- **Search**: K-NN, threshold search, batch similarity
- **CPSE/CPSD**: Context-preserving encoding (Malits & Mendelson 2025)
- **Operations**: Top-k selection, noise injection, similarity matrices
- **Backend Detection**: Runtime capability probes (GPU, JIT, sparse, complex)

---

## 📚 Examples

### Example 1: Analogical Reasoning

Solve "King - Man + Woman = Queen":

```python
from holovec import VSA

model = VSA.create('FHRR', dim=10000)

# Create concept vectors
king = model.random(seed=1)
man = model.random(seed=2)
woman = model.random(seed=3)
queen = model.random(seed=4)

# Compute analogy: king - man + woman
result = model.bundle([
    king,
    model.unbind(king, man),  # Remove "maleness"
    woman                      # Add "femaleness"
])

# Should be similar to queen
similarity = model.similarity(result, queen)
print(f"King - Man + Woman ≈ Queen: {similarity:.3f}")
```

### Example 2: Role-Filler Binding

Represent structured knowledge: "The ball is red and large"

```python
from holovec import VSA

model = VSA.create('FHRR', dim=10000)

# Roles
color_role = model.random(seed=1)
size_role = model.random(seed=2)
object_role = model.random(seed=3)

# Fillers
red = model.random(seed=10)
large = model.random(seed=11)
ball = model.random(seed=12)

# Create structured representation
ball_representation = model.bundle([
    model.bind(object_role, ball),
    model.bind(color_role, red),
    model.bind(size_role, large)
])

# Query: What is the color?
color_query = model.unbind(ball_representation, color_role)
print(f"Color is red: {model.similarity(color_query, red):.3f}")  # ~0.99

# Query: What is the object?
object_query = model.unbind(ball_representation, object_role)
print(f"Object is ball: {model.similarity(object_query, ball):.3f}")  # ~0.99
```

### Example 3: Encoding Continuous Values

Encode scalars with locality preservation:

```python
from holovec import VSA
from holovec.encoders import FractionalPowerEncoder

model = VSA.create('FHRR', dim=10000)

# Create encoder for temperature range [-20°C, 40°C]
encoder = FractionalPowerEncoder(model, min_val=-20, max_val=40)

# Encode temperatures
temp_20 = encoder.encode(20.0)
temp_21 = encoder.encode(21.0)
temp_35 = encoder.encode(35.0)

# Similar temperatures → similar vectors
print(f"20°C vs 21°C: {model.similarity(temp_20, temp_21):.3f}")  # ~0.99
print(f"20°C vs 35°C: {model.similarity(temp_20, temp_35):.3f}")  # ~0.70

# Decode back to temperature
decoded = encoder.decode(temp_20)
print(f"Decoded: {decoded:.1f}°C")  # ~20.0°C
```

### Example 4: Sequence Encoding

Encode variable-length sequences:

```python
from holovec import VSA
from holovec.encoders import PositionBindingEncoder

model = VSA.create('MAP', dim=10000)
encoder = PositionBindingEncoder(model)

# Encode sequences
seq1 = encoder.encode(['hello', 'world'])
seq2 = encoder.encode(['hello', 'world', '!'])
seq3 = encoder.encode(['goodbye', 'world'])

# Shared prefixes → high similarity
print(f"['hello','world'] vs ['hello','world','!']: {model.similarity(seq1, seq2):.3f}")
print(f"['hello','world'] vs ['goodbye','world']: {model.similarity(seq1, seq3):.3f}")

# Order matters!
seq4 = encoder.encode(['world', 'hello'])
print(f"['hello','world'] vs ['world','hello']: {model.similarity(seq1, seq4):.3f}")  # Low
```

### Example 5: Multi-Dimensional Feature Vectors

Encode embeddings and feature vectors:

```python
from holovec import VSA
from holovec.encoders import FractionalPowerEncoder, VectorEncoder
import numpy as np

model = VSA.create('FHRR', dim=10000)

# Create composable encoder
scalar_enc = FractionalPowerEncoder(model, min_val=-1, max_val=1)
vec_encoder = VectorEncoder(model, scalar_encoder=scalar_enc, n_dimensions=128)

# Encode 128-dimensional embeddings
embedding1 = np.random.randn(128)
embedding2 = embedding1 + np.random.randn(128) * 0.1  # Similar embedding

hv1 = vec_encoder.encode(embedding1)
hv2 = vec_encoder.encode(embedding2)

# Similarity in hypervector space reflects embedding similarity
print(f"Hypervector similarity: {model.similarity(hv1, hv2):.3f}")

# Approximate decoding
decoded = vec_encoder.decode(hv1)
reconstruction_error = np.mean((embedding1 - decoded) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.4f}")
```

<details>
<summary><b>Example 6: Position-Aware Sequence Decoding</b></summary>

```python
from holovec import VSA

model = VSA.create('MAP', dim=10000)

# Elements
a = model.random(seed=1)
b = model.random(seed=2)
c = model.random(seed=3)

# Encode sequence: A + ρ(B) + ρ²(C)
sequence = model.bundle([
    a,
    model.permute(b, k=1),
    model.permute(c, k=2)
])

# Query position 1: should recover B
query_pos1 = model.unpermute(sequence, k=1)
similarity_b = model.similarity(query_pos1, b)
print(f"Recovered B with similarity: {similarity_b:.3f}")  # ~0.95+

# Order matters! Different permutations = different positions
```
</details>

---

## 🏗️ Architecture

HoloVec follows a clean layered architecture:

```
┌─────────────────────────────────────┐
│  High-Level API (VSA.create)        │  ← Simple interface
├─────────────────────────────────────┤
│  VSA Models (MAP, FHRR, HRR, ...)   │  ← Algebraic operations
├─────────────────────────────────────┤
│  Vector Spaces (Bipolar, Complex...)│  ← Random vectors, similarity
├─────────────────────────────────────┤
│  Backends (NumPy, PyTorch, JAX)     │  ← Computational primitives
└─────────────────────────────────────┘
```

### Project Structure

```
holovec/
├── backends/          # Backend implementations (NumPy, PyTorch, JAX)
├── spaces/           # Vector spaces (Bipolar, Binary, Real, Complex, Sparse, Matrix)
├── models/           # VSA models (MAP, FHRR, HRR, BSC, BSDC, GHRR, VTB)
├── encoders/         # Data encoders (scalars, sequences, images, vectors)
├── retrieval/        # Cleanup and retrieval (Codebook, ItemStore, strategies)
└── utils/            # Utilities (CPSE/CPSD, search, operations)

tests/                # 480+ test functions (90-98% coverage)
examples/             # Working examples and demos
docs/                 # Documentation and theory guides
```

---

## 🧪 Testing

HoloVec has rigorous testing with excellent coverage:

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=holovec --cov-report=html

# Specific test file
pytest tests/test_models.py

# Specific test
pytest tests/test_models.py::TestBindingOperation::test_unbinding_recovery

# Run tests for specific backend
pytest tests/test_models.py -k numpy
```

**Test Statistics:**

- ✅ **480+ test functions**
- ✅ **90-98% code coverage**
- ✅ **Property-based testing** with Hypothesis
- ✅ **Cross-backend consistency** validation
- ✅ **Numerical stability** verification

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/Twistient/HoloVec.git
cd HoloVec

# Install with development dependencies
pip install -e .[dev]
# or with uv (faster)
uv pip install -e .[dev]

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Code Quality Tools

```bash
# Format code with Black
black holovec tests examples

# Lint with Ruff
ruff check holovec tests examples

# Type check with Mypy
mypy holovec

# Run all checks
pre-commit run --all-files
```

### Development Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Write code, tests, documentation
3. **Run tests**: `pytest`
4. **Check quality**: `pre-commit run --all-files`
5. **Commit with DCO**: `git commit -s -m "Add feature: description"`
6. **Push and PR**: `git push origin feature/your-feature`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📖 Documentation

### Guides

- **Installation**: Multiple backend setup instructions
- **Quick Start**: Get running in 5 minutes
- **Core Concepts**: Understanding HDC/VSA fundamentals
- **Model Guide**: Choosing the right VSA model
- **Encoder Guide**: Encoding different data types
- **Backend Guide**: NumPy vs PyTorch vs JAX

### API Reference

- Comprehensive API documentation
- NumPy-style docstrings
- Type hints throughout
- Usage examples in every docstring

### Theory

- VSA model mathematics
- Encoder theory and validation
- Capacity analysis
- Performance benchmarks

---

## 🎓 Research Foundation

HoloVec is built on decades of academic research in hyperdimensional computing:

1. **Kanerva (1988, 1993)**: Sparse Distributed Memory - foundational SDM architecture
2. **Plate (1995, 2003)**: Holographic Reduced Representations - circular convolution binding
3. **Kanerva (2009)**: "Hyperdimensional Computing: An Introduction" - comprehensive HDC introduction
4. **Gayler (2003)**: Vector Symbolic Architectures - unifying framework
5. **Frady et al. (2021)**: "Computing on Functions" - fractional power encoding theory
6. **Schlegel et al. (2022)**: "A comparison of VSAs" - experimental model comparison
7. **Kleyko et al. (2023)**: "HDC/VSA Survey Part I" - most comprehensive survey to date
8. **Kymn et al. (2024)**: Resonator networks - fast iterative cleanup
9. **Malits & Mendelson (2025)**: Context-preserving encoding algorithms

---

## 📊 Project Status

### ✅ Phase 1: Core Foundation (COMPLETE)

- Backend abstraction (NumPy, PyTorch, JAX)
- Vector spaces (Bipolar, Binary, Real, Complex, Sparse, Matrix)
- MAP and FHRR models
- High-level API
- Comprehensive test suite

### ✅ Phase 2: Model Library (COMPLETE)

- HRR, BSC, VTB, BSDC, GHRR models
- All 7 models validated
- Property-based testing
- Cross-backend consistency

### ✅ Phase 3A-C: Encoders & Utilities (COMPLETE)

- Scalar encoders (FPE, Thermometer, Level)
- Sequence encoders (Position, NGram, Trajectory)
- Spatial encoder (Image)
- Structured encoder (Vector)
- Cleanup strategies (BruteForce, Resonator)
- Search utilities (K-NN, threshold, batch)
- CPSE/CPSD algorithms

### 🚧 Phase 3D: Memory Systems (IN PROGRESS)

- ItemMemory with cleanup strategies
- SequenceMemory with temporal binding
- Sparse Distributed Memory (SDM)
- Attention-based memory
- Probabilistic memory

### 📋 Future Phases

- Applications (classification, regression, clustering)
- Analysis and visualization tools
- Neural network integration
- Hardware acceleration guides

---

## 🧭 Roadmap (Near-Term)

<details>
<summary><b>P0 (within HoloVec)</b></summary>

- **Docs**: "HDC for ML People" and "GHRR for Researchers"; literature-backed notes
- **Bench**: factorization (3-5 factors) with convergence curves; bundling capacity curves; FPE kernel visualizations
- **sklearn helpers**: thin wrappers for FPE + ItemStore; UCI demo
- **Examples**: circular correlation HRR; role permutations; resonator factorization demo; FPE kernel shapes
- **API polish**: top-level factory finalize; stabilize public surface; batched ItemStore query (done)
</details>

<details>
<summary><b>P1 (HoloMem/HoloGraph scaffolds)</b></summary>

- **HoloMem**: SDM variants; attention/Hopfield cleanup; learned β/α trainers; GPU batched memory; evaluation harness; PyTorch modules
- **HoloGraph**: outer-product graph embeddings + algebra demos (subgraphs, homomorphisms, path powers)
</details>

<details>
<summary><b>P2 (Ecosystem/Adoption)</b></summary>

- **Website** with literate notebooks
- **Publish** 0.1.0-alpha
- **Blog posts**: Attention≈SDM; GHRR holography; BSDC-SEG capacity
- **Academic channels**: arXiv-style writeup linking library
</details>

---

## 📄 Citation

If you use HoloVec in your research, please cite:

```bibtex
@software{HoloVec2025,
  author       = {Brodie Schroeder},
  title        = {HoloVec: Holographic Vector Algebra for Compositional Machine Intelligence},
  organization = {Twistient Corp.},
  year         = {2025},
  version      = {0.1.0},
  url          = {https://github.com/Twistient/holovec},
  license      = {Apache-2.0}
}
```

---

## 🤝 Contributing

We welcome contributions! HoloVec is a community-driven project.

**Key areas for contribution:**

- Additional VSA models
- New encoder types
- Memory system implementations
- Performance optimizations
- Documentation and examples
- Bug reports and fixes

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.

---

## 📜 License

HoloVec is licensed under the **Apache License 2.0**.

```
Copyright 2025 Twistient Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

See [LICENSE](LICENSE) for full text.

---

## 🙏 Acknowledgments

Built with insights from:

- Decades of academic research in HDC/VSA
- The hyperdimensional computing community
- Open source contributors and users

Special thanks to the researchers whose work made this possible.

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Twistient/HoloVec/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Twistient/HoloVec/discussions)
- **Security**: <security@twistient.com>
- **General**: <brodie@twistient.com>

---

<div align="center">

**Ready to start?** Check out the [Installation](#-installation) section above, then dive into the [Quick Start](#-quick-start) guide!

</div>
