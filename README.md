<div align="center">
  <img src="docs/assets/logo.svg" alt="HoloVec Logo" width="400">
</div>

<div align="center">

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Vector Symbolic Architectures for compositional, high-dimensional computing.**

[Documentation](https://twistient.github.io/HoloVec/) • [Installation](#installation) • [Quick Start](#quick-start) • [Examples](examples/)

</div>

---

## What is HoloVec?

HoloVec is a Python library for hyperdimensional computing (HDC) and Vector Symbolic Architectures (VSA). It represents data as high-dimensional vectors (~1,000-10,000 dimensions) that can be composed using algebraic operations—binding creates associations, bundling creates sets, and permutation encodes order.

This approach enables one-shot learning, noise-tolerant representations, and transparent symbolic reasoning without gradient descent.

---

## Installation

```bash
git clone https://github.com/Twistient/HoloVec.git
cd HoloVec
uv sync --extra dev
```

For an editable install outside the project workflow, use `uv pip install -e .[dev]`.
For GPU support add `.[torch]`, for JIT compilation add `.[jax]`, or use `.[all]` for everything.

---

## Quick Start

```python
from holovec import VSA

# Create a model
model = VSA.create('FHRR', dim=2048)

# Generate random hypervectors
a = model.random()
b = model.random()

# Bind them together (creates an association)
c = model.bind(a, b)

# Unbind to recover the original
a_recovered = model.unbind(c, b)
print(model.similarity(a, a_recovered))  # 1.0 (exact recovery)
```

---

## Concepts

HoloVec has four layers: **Backends** provide numerical primitives, **Spaces** define vector types and similarity measures, **Models** implement algebraic operations, and **Encoders** transform data into hypervectors.

### Operations

Every VSA model supports three core operations:

| Operation | What it does | Use case |
|-----------|--------------|----------|
| **Bind** | Associates two vectors → result dissimilar to both | Key-value pairs, role-filler structures |
| **Bundle** | Superposes vectors → result similar to all inputs | Sets, prototypes, averaging |
| **Permute** | Shifts coordinates → preserves similarity | Sequences, positions, temporal order |

### Models

Different models have different algebraic properties:

| Model | Binding method | Inverse | Notes |
|-------|---------------|---------|-------|
| **FHRR** | Complex multiply | Exact | Best capacity, continuous data |
| **GHRR** | Matrix product | Exact | Non-commutative, 2024 state-of-the-art |
| **MAP** | Element multiply | Self | Hardware-friendly, neuromorphic |
| **HRR** | Circular convolution | Approximate | Classic baseline |
| **VTB** | Matrix transform | Approximate | Non-commutative, directional |
| **BSC** | XOR | Self | Binary, FPGA-friendly |
| **BSDC** | Sparse XOR | Approximate | Memory efficient |
| **BSDC-SEG** | Segment XOR | Self | Segment-sparse, fast search |

Choose FHRR for general use, GHRR when order matters, MAP for hardware deployment, or BSDC variants for memory constraints.

### Spaces

Each model operates on a specific vector space that determines how random vectors are generated and how similarity is measured:

| Space | Values | Similarity | Used by |
|-------|--------|------------|---------|
| **Complex** | Unit phasors (e^iθ) | Cosine | FHRR |
| **Bipolar** | {-1, +1} | Cosine | MAP, HRR |
| **Binary** | {0, 1} | Hamming | BSC |
| **Sparse** | Sparse binary | Overlap | BSDC |
| **Matrix** | Unitary matrices | Trace | GHRR, VTB |

You typically don't interact with spaces directly—`VSA.create()` selects the appropriate space for each model.

### Backends

HoloVec runs on NumPy (default), PyTorch (GPU), or JAX (JIT). The API is identical:

```python
# NumPy (default)
model = VSA.create('FHRR', dim=2048)

# PyTorch with GPU
model = VSA.create('FHRR', dim=2048, backend='torch', device='cuda')

# JAX with JIT compilation
model = VSA.create('FHRR', dim=2048, backend='jax')
```

NumPy is the release-blocking backend. PyTorch and JAX remain optional backends and should be
treated as conditional support paths unless their backend-specific tests are enabled in your
environment.

### Encoders

Encoders transform data into hypervectors:

| Encoder | Input | Preserves |
|---------|-------|-----------|
| **FractionalPowerEncoder** | Scalars | Locality (similar values → similar vectors) |
| **ThermometerEncoder** | Ordinals | Order relationships |
| **PositionBindingEncoder** | Sequences | Position information |
| **NGramEncoder** | Text | Local context |
| **ImageEncoder** | 2D grids | Spatial relationships |
| **VectorEncoder** | Feature vectors | Per-dimension encoding |

---

## Examples

### Role-Filler Binding

Represent structured knowledge like "the ball is red":

```python
model = VSA.create('FHRR', dim=2048)

# Create role and filler vectors
color = model.random(seed=1)
red = model.random(seed=2)
ball = model.random(seed=3)

# Bind role to filler, bundle into a structure
ball_repr = model.bundle([
    model.bind(color, red),
    model.bind(model.random(seed=4), ball)  # object role
])

# Query: what color?
query = model.unbind(ball_repr, color)
print(model.similarity(query, red))  # High similarity
```

### Encoding Continuous Values

```python
from holovec.encoders import FractionalPowerEncoder

model = VSA.create('FHRR', dim=2048)
encoder = FractionalPowerEncoder(model, min_val=0, max_val=100)

# Similar values produce similar vectors
v50 = encoder.encode(50)
v51 = encoder.encode(51)
v90 = encoder.encode(90)

print(model.similarity(v50, v51))  # ~0.99
print(model.similarity(v50, v90))  # ~0.70
```

### Cleanup and Retrieval

```python
from holovec.retrieval import Codebook, ItemStore

model = VSA.create('FHRR', dim=2048)

# Create a codebook of known items
cb = Codebook({
    "apple": model.random(seed=1),
    "banana": model.random(seed=2),
    "cherry": model.random(seed=3)
}, backend=model.backend)

# Query with a noisy vector
store = ItemStore(model).fit(cb)
noisy_apple = cb["apple"] + model.random() * 0.1
result = store.query(noisy_apple, k=1)
print(result)  # [('apple', ~0.99)]
```

See [examples/](examples/) for more.

---

## Project Structure

```
holovec/
├── backends/     NumPy, PyTorch, JAX implementations
├── spaces/       Vector spaces (Bipolar, Complex, Sparse, etc.)
├── models/       VSA models (MAP, FHRR, HRR, BSC, GHRR, VTB, etc.)
├── encoders/     Scalar, sequence, spatial, structured encoders
├── retrieval/    Codebook, ItemStore, cleanup strategies
└── utils/        Search, similarity, helper functions
```

---

## Testing

```bash
uv run --extra dev pytest
uv run --extra dev pytest --cov=holovec --cov-report=html
uv run --extra dev ruff check holovec tests
uv run --extra dev mypy holovec
```

The engine CI enforces tests, Ruff, and source-only mypy on every push and pull request.

---

## References

HoloVec implements ideas from:

- Kanerva (1988) — Sparse Distributed Memory
- Plate (2003) — Holographic Reduced Representations
- Gayler (2003) — Vector Symbolic Architectures
- Frady et al. (2021) — Fractional power encoding
- Schlegel et al. (2022) — VSA model comparison
- Kleyko et al. (2023) — HDC/VSA survey
- Kymn et al. (2024) — Resonator networks

---

## Citation

```bibtex
@software{HoloVec2025,
  author       = {Brodie Schroeder},
  title        = {HoloVec: Vector Symbolic Architectures for Python},
  year         = {2025},
  version      = {0.3.0},
  url          = {https://github.com/Twistient/HoloVec},
  license      = {Apache-2.0}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Key areas: new models, encoders, documentation, and performance.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Contact

[GitHub Issues](https://github.com/Twistient/HoloVec/issues) • [Discussions](https://github.com/Twistient/HoloVec/discussions) • brodie@twistient.com
