<div align="center">
  <img src="docs/source/_static/logo.svg" alt="HoloVec Logo" width="400">
</div>

<div align="center">

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Compositional algebra in 10,000‑D, with exact/approximate inverses and kernel‑shaped encoders.

</div>

## Holographic Vector Algebra for Compositional Machine Intelligence

HoloVec is a holographic vector algebra for building and manipulating compositional representations in high dimensions. It implements modern Vector Symbolic Architectures (VSA/HDC) with both commutative (HRR/FHRR) and non‑commutative (GHRR/MBAT) binding, kernel‑aware encoders (FPE/RFF with Gaussian, Laplace, Cauchy, Student distributions), and sparse/segment codes (BSDC‑SEG). You can bind structure, bundle sets, and unbind components with exact or approximate inverses — then factorize multi‑way compositions efficiently via resonator cleanup — on NumPy, PyTorch, or JAX.

## What is this?

- A high‑dimensional vector algebra (VSA/HDC) you can actually compute with: bind, bundle, permute, and unbind structured information using modern operations (FHRR/HRR/GHRR/MBAT) and kernel‑aware encoders.

## What does it do?

- Encodes continuous, discrete, periodic, and multivariate data (FPE/RFF, periodic angles).
- Binds structure (commutative and non‑commutative), bundles sets, and unbinds components with exact or Wiener‑style inverses.
- Factorizes multi‑factor compositions via resonator cleanup (hard/soft updates) with fast convergence.
- Supports sparse/segment codes (BSDC‑SEG) with segment‑wise search and utilities.
- Runs on NumPy, PyTorch, or JAX.

## Why should I use it?

- Compositional power: exact non‑commutative binding (GHRR) with tunable diagonality — encode direction/order without hacks.
- Kernel‑shaped encoders: FPE = RFF with richer phase distributions and mixtures; multivariate and periodic encoders built‑in.
- Practical retrieval: lean Codebook/ItemStore APIs, fast batched similarity, resonator factorization.
- Clean separation of concerns: HoloVec stays lean (algebra + encoders + retrieval); HoloMem (coming) adds learned memories/training; HoloGraph adds outer‑product graph algebra.

---

## Positioning & Scope

- **HoloVec = Algebra‑first + Kernel‑aware HDC core**
  - Compositional algebra: HRR/FHRR (commutative), GHRR (exact non‑commutative with diagonality), VTB/MBAT (non‑commutative transform), BSC/BSDC + BSDC‑SEG (sparse/segment).
  - Kernel encoders: FPE = RFF with phase families + mixtures, multivariate (VectorFPE), periodic encoders; explicit link to kernel methods.
- **HoloMem (separate)** = Advanced memories + learning
  - SDM variants, attention/Hopfield cleanup, learned β/α, GPU batched retrieval, evaluators.
- **HoloGraph (separate)** = Outer‑product graph embeddings + algebra
  - Tensor/outer‑product, powers, subgraphs, degrees, homomorphism tests, trace metrics.

### Feature Matrix (core)

| Category | Features |
|---|---|
| Algebra (binding) | HRR (conv), FHRR (phasor, exact inverse), GHRR (matrix‑unitary, non‑commutative, diagonality), VTB/MBAT (transform, non‑commutative), BSC/BSDC (XOR), BSDC‑SEG (segments) |
| Encoders | Scalar FPE (Gaussian/Laplace/Cauchy/Student/Uniform) + mixtures; VectorFPE (multivariate); Periodic/angle; Thermometer; Level |
| Retrieval | Codebook + ItemStore; Brute‑force cleanup; Resonator (hard/soft, temperature, early‑stop); batched similarity |
| Sparse/Segment | BSDC‑SEG (exact 1‑hot/segment); segment‑wise masking/rotation/permutation; segment‑pattern search |
| Backends | NumPy (base), PyTorch (GPU), JAX (JIT); helper parity (complex, FFT, softmax, angle, power, where, etc.) |

> HoloMem adds learned/attention memories and training; HoloGraph adds tensor/outer‑product graph algebra.

### How We Differ

HoloVec focuses on algebra and kernels across multiple backends. It emphasizes non‑commutative binding (GHRR/MBAT), kernel‑aware encoders (FPE/RFF) with rich phase families, and lean retrieval/factorization — keeping the core dependency‑light and theory‑aligned.

**Differentiators already in HoloVec**

- GHRR with diagonality → flexible non‑commutativity and exact inverses.
- VTB/MBAT transform binding → directional facts without heavy permutations.
- FPE “done right” → multiple phase families, mixtures (M2), multivariate, periodic.
- BSDC‑SEG → exact segmented sparsity + segment search utilities.
- Built‑in retrieval (Codebook/ItemStore), light resonator, segment utilities.
- Multi‑backend (NumPy/Torch/JAX) + kernel framing approachable for ML engineers.

**What we don’t duplicate in core**

- Deep, framework‑specific training pipelines and dataset loaders (HoloMem/HoloGraph and the website will cover more applied flows).

### Differentiators

- Non‑commutative binding as first‑class (GHRR diagonality sweeps; MBAT VTB). Side‑by‑side decoding quality/speed vs HRR.
- Kernel shaping (FPE=RFF): phase distributions, mixtures, multivariate, periodic; kernel visualizations; UCI tasks.
- Sparse/segment codes: BSDC‑SEG capacity/robustness curves; biologically plausible angle.
- Resonator factorization: convergence plots; ablations (hard vs soft, temperature, top‑K).
- Graph algebra (HoloGraph): tensor/outer‑product and matrix algebra for graphs.

## ⚡ 10‑second code

```python
from holovec import VSA  # post-rename: from holovec import VSA

model = VSA.create('FHRR', dim=2048)
a, b = model.random(), model.random()
c = model.bind(a, b)
a_hat = model.unbind(c, b)
print("sim(a, a_hat) =", model.similarity(a, a_hat))

from holovec.encoders import VectorFPE
fpe = VectorFPE(model, input_dim=3, bandwidth=0.5, phase_dist='gaussian')
hx = fpe.encode([0.2, 1.4, -0.7])

from holovec.retrieval import ItemStore, Codebook
cb = Codebook({f"item{i}": model.random(seed=100+i) for i in range(10)}, backend=model.backend)
store = ItemStore(model).fit(cb)
print(store.query(cb._items['item3'], k=3))
```

## 🌟 Why HoloVec?

Hyperdimensional computing represents information as high-dimensional vectors (~1000-10000 dimensions) that enable:

- **One-shot learning** without gradient descent
- **Robust, noise-tolerant** representations
- **Explainable AI** with transparent operations
- **Efficient hardware** implementation (neuromorphic chips, FPGAs)
- **Compositional reasoning** with symbolic structure

HoloVec makes HDC accessible with:

- ✅ **7 validated VSA models** from academic literature
- ✅ **3 computational backends** (NumPy, PyTorch, JAX)
- ✅ **8 production-ready encoders** for diverse data types
- ✅ **480+ tests** with 90-98% coverage
- ✅ **Zero dependencies** beyond NumPy for base install
- ✅ **Type-safe** with comprehensive documentation

---

## 📦 Installation

> **Note**: HoloVec is not yet published to PyPI. For now, install from source using the development installation below.

### Development Installation (Recommended)

Clone and install in editable mode:

```bash
git clone https://github.com/Twistient/HoloVec.git
cd HoloVec

# With pip
pip install -e .

# Or with uv (faster)
uv pip install -e .
```

### Quick Start (when published to PyPI)

```bash
pip install holovec
```

Or using [uv](https://github.com/astral-sh/uv) (recommended for faster installs):

```bash
uv pip install holovec
```

### Optional Backends

#### GPU Support (PyTorch)

```bash
# With pip
pip install -e .[torch]

# With uv
uv pip install -e .[torch]
```

#### JIT Compilation (JAX)

```bash
# With pip
pip install -e .[jax]

# With uv
uv pip install -e .[jax]
```

#### All Features (Development)

```bash
# With pip
pip install -e .[all]

# With uv
uv pip install -e .[all]
```

This installs all backends, development tools (pytest, black, ruff, mypy), and documentation tools (sphinx).

---

## 🚀 Quick Start

```python
from holovec import VSA

# Create a VSA model (FHRR has best capacity)
model = VSA.create('FHRR', dim=10000)

# Generate random hypervectors
country = model.random(seed=1)
capital = model.random(seed=2)
currency = model.random(seed=3)

# Bind vectors to create associations
usa = model.bind_multiple([country, capital, currency])

# Query: What is the capital? (unbind country)
capital_query = model.unbind(usa, country)
similarity = model.similarity(capital_query, capital)
print(f"Similarity: {similarity:.3f}")  # ~0.99 for FHRR

# Bundle multiple items into a set
countries = model.bundle([usa, model.random(), model.random()])
```

---

## 🧠 Core Concepts

### Vector Symbolic Architectures

VSAs represent information as high-dimensional vectors with three fundamental operations:

#### 1. **Binding (⊗)** - Associates two vectors → dissimilar result

```python
role_filler = model.bind(role, filler)
```

Creates structured representations like "color: red" or "position: 3"

#### 2. **Bundling (+)** - Superpose multiple vectors → similar to all inputs

```python
set_vector = model.bundle([item1, item2, item3])
```

Represents collections, prototypes, and averages

#### 3. **Permutation (ρ)** - Reorders coordinates → preserves similarity

```python
position_encoded = model.permute(vector, k)
```

Encodes sequences, positions, and temporal order

### Key Properties

Different VSA models have different algebraic properties optimized for specific use cases:

| Model | Binding | Inverse | Capacity | Best For |
|-------|---------|---------|----------|----------|
| **FHRR** | Complex ∗ | Exact (conjugate) | Best (~330 dim) | Continuous data, highest accuracy |
| **MAP** | Element × | Self-inverse | Good (~510 dim) | Hardware, neuromorphic chips |
| **HRR** | Circular conv | Approximate | Good (~510 dim) | Classic baseline |
| **BSC** | XOR | Self-inverse | Good | Binary operations, FPGA |
| **BSDC** | Segment sample | Approximate | Very Good | Sparse data, memory efficient |
| **VTB** | Matrix transform | Learned | Excellent | Adaptive representations |
| **GHRR** | Matrix product | Approximate | Excellent | State-of-the-art (2024) |

---

## 🎯 Features

### 🔧 VSA Models (7 Validated Implementations)

All models validated against academic literature with comprehensive property-based testing:

- **MAP (Multiply-Add-Permute)**: Self-inverse, neuromorphic-friendly
- **FHRR (Fourier HRR)**: Complex-valued, exact inverses, best capacity
- **HRR (Holographic RR)**: Classic circular convolution
- **BSC (Binary Spatter Codes)**: XOR-based for binary vectors
- **BSDC (Binary Sparse DC)**: Sparse binary, memory-efficient
- **GHRR (Generalized HRR)**: 2024 SOTA with matrix binding
- **VTB (Vector-derived Transformation)**: Learned transformation matrices

### 🖥️ Computational Backends (3 Frameworks)

Write once, run anywhere with automatic backend selection:

- **NumPy** (default): Pure CPU, zero dependencies, perfect for prototyping
- **PyTorch**: GPU acceleration (CUDA/Metal), neural network integration
- **JAX**: JIT compilation (10-100x speedup), TPU support, auto-differentiation

```python
# Automatic best backend
model = VSA.create('FHRR', dim=10000)

# Explicit backend selection
model = VSA.create('FHRR', dim=10000, backend='torch', device='cuda')

# Query backend capabilities
from holovec import backend_info
info = backend_info()
print(info['available_backends'])  # ['numpy', 'torch', 'jax']
```

### 📊 Encoders (8 Production-Ready)

Transform diverse data types into hypervectors:

**Scalar Encoders:**

- `FractionalPowerEncoder`: Continuous values with locality preservation
- `ThermometerEncoder`: Smooth ordinal encoding
- `LevelEncoder`: Discrete categorical values

**Sequence Encoders:**

- `PositionBindingEncoder`: Order-sensitive sequences
- `NGramEncoder`: N-gram context windows
- `TrajectoryEncoder`: Temporal trajectories

**Spatial Encoder:**

- `ImageEncoder`: 2D grids and images

**Structured Encoder:**

- `VectorEncoder`: Multi-dimensional feature vectors

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

### Example 2: Sequence Encoding

Encode ordered sequences with position information:

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

### Example 3: Role-Filler Binding

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
similarity_red = model.similarity(color_query, red)
print(f"Color is red: {similarity_red:.3f}")  # ~0.99

# Query: What is the object?
object_query = model.unbind(ball_representation, object_role)
similarity_ball = model.similarity(object_query, ball)
print(f"Object is ball: {similarity_ball:.3f}")  # ~0.99
```

### Example 4: Encoding Continuous Values

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
sim_20_21 = model.similarity(temp_20, temp_21)
print(f"20°C vs 21°C: {sim_20_21:.3f}")  # ~0.99 (very similar)

sim_20_35 = model.similarity(temp_20, temp_35)
print(f"20°C vs 35°C: {sim_20_35:.3f}")  # ~0.70 (somewhat similar)

# Decode back to temperature
decoded = encoder.decode(temp_20)
print(f"Decoded: {decoded:.1f}°C")  # ~20.0°C
```

### Example 5: Symbolic Sequences

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
sim_1_2 = model.similarity(seq1, seq2)
print(f"['hello','world'] vs ['hello','world','!']: {sim_1_2:.3f}")

sim_1_3 = model.similarity(seq1, seq3)
print(f"['hello','world'] vs ['goodbye','world']: {sim_1_3:.3f}")

# Order matters!
seq4 = encoder.encode(['world', 'hello'])
sim_1_4 = model.similarity(seq1, seq4)
print(f"['hello','world'] vs ['world','hello']: {sim_1_4:.3f}")  # Low
```

### Example 6: Multi-Dimensional Feature Vectors

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
sim = model.similarity(hv1, hv2)
print(f"Hypervector similarity: {sim:.3f}")  # High for similar embeddings

# Approximate decoding
decoded = vec_encoder.decode(hv1)
reconstruction_error = np.mean((embedding1 - decoded) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.4f}")
```

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

- 480+ test functions
- 90-98% code coverage
- Property-based testing with Hypothesis
- Cross-backend consistency validation
- Numerical stability verification

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/Twistient/HoloVec.git
cd HoloVec

# Install with development dependencies
pip install -e .[dev]

# Or with uv (faster)
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

**Ready to start?** Check out the [Installation](#-installation) section above to get HoloVec, then dive into the [Quick Start](#-quick-start) guide!

---

## 🧭 Roadmap (near‑term)

**P0 (within HoloVec)**

- Docs: “HDC for ML People” and “GHRR for Researchers”; literature‑backed notes.
- Bench: factorization (3–5 factors) with convergence curves; bundling capacity curves; FPE kernel visualizations.
- sklearn helpers: thin wrappers for FPE + ItemStore; UCI demo.
- Examples: circular correlation HRR; role permutations; resonator factorization demo; FPE kernel shapes.
- API polish: top‑level factory finalize; stabilize public surface; batched ItemStore query (done).

**P1 (HoloMem/HoloGraph scaffolds)**

- HoloMem: SDM variants; attention/Hopfield cleanup; learned β/α trainers; GPU batched memory; evaluation harness; PyTorch modules.
- HoloGraph: outer‑product graph embeddings + algebra demos (subgraphs, homomorphisms, path powers).

**P2 (Ecosystem/Adoption)**

- Website with literate notebooks; publish 0.1.0‑alpha; blog posts (Attention≈SDM; GHRR holography; BSDC‑SEG capacity); academic channels (arXiv‑style writeup linking library).
