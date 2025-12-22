# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HoloVec** is a Python library for Hyperdimensional Computing (HDC) / Vector Symbolic Architectures (VSA). It provides compositional algebra in high-dimensional spaces (~1,000-10,000 dimensions) with support for 7 VSA models, 3 computational backends, and kernel-aware encoders for various data types.

**Key Philosophy**: Algebra-first + kernel-aware HDC core with production-ready code (480+ tests, 90-98% coverage, type-safe).

## Essential Commands

### Development Setup
```bash
# Install for development (includes all backends and dev tools)
uv pip install -e .[all]

# Install specific configurations
uv pip install -e .              # NumPy only (base)
uv pip install -e .[torch]       # Add PyTorch backend
uv pip install -e .[jax]         # Add JAX backend
uv pip install -e .[dev]         # Add dev tools (pytest, black, ruff, mypy)
uv pip install -e .[docs]        # Add documentation tools (sphinx)
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=holovec --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test class or function
pytest tests/test_models.py::TestBindingOperation::test_unbinding_recovery

# Run tests for specific backend
pytest tests/test_models.py -k numpy
pytest tests/test_models.py -k torch
pytest tests/test_models.py -k jax

# Run single test quickly (no coverage)
pytest tests/test_models.py::test_specific_function -v
```

### Code Quality
```bash
# Format code (line length: 100)
black holovec tests examples

# Lint code
ruff check holovec tests examples

# Type check (strict mode enabled)
mypy holovec

# Run all quality checks
pre-commit run --all-files
```

### Documentation
```bash
# Build documentation (uses Sphinx)
cd docs
make html

# View docs
open build/html/index.html
```

## Architecture

### Layered Design

HoloVec follows a clean separation of concerns with 4 layers:

```
┌────────────────────────────────┐
│  High-Level API (VSA.create)   │  ← User-facing factory
├────────────────────────────────┤
│  VSA Models (MAP, FHRR, ...)   │  ← Algebraic operations
├────────────────────────────────┤
│  Vector Spaces (Bipolar, ...)  │  ← Random vectors, similarity
├────────────────────────────────┤
│  Backends (NumPy, Torch, JAX)  │  ← Computational primitives
└────────────────────────────────┘
```

### Core Modules

**holovec/backends/** - Computational backends with unified interface
- `base.py`: Abstract `Backend` interface defining all required operations
- `numpy_backend.py`: Pure NumPy implementation (always available, default)
- `torch_backend.py`: PyTorch for GPU acceleration (CUDA/Metal)
- `jax_backend.py`: JAX for JIT compilation and TPU support
- Backend selection is automatic but can be overridden via `VSA.create(..., backend='torch', device='cuda')`

**holovec/spaces/** - Vector space implementations
- `base.py`: Abstract `VectorSpace` class defining random vector generation and similarity
- `spaces.py`: Concrete implementations (Bipolar, Binary, Real, Complex, Sparse, Matrix)
- Each space is paired with a specific model type (e.g., Bipolar ↔ MAP, Complex ↔ FHRR)

**holovec/models/** - 7 VSA model implementations
- `base.py`: Abstract `VSAModel` with core operations (bind, bundle, permute, etc.)
- `map.py`: Multiply-Add-Permute (bipolar, self-inverse)
- `fhrr.py`: Fourier HRR (complex, exact inverse, best capacity ~330 dim)
- `hrr.py`: Holographic RR (real, circular convolution, approximate inverse)
- `bsc.py`: Binary Spatter Codes (binary, XOR, self-inverse)
- `bsdc.py`: Binary Sparse Distributed Codes (sparse binary, memory-efficient)
- `bsdc_seg.py`: Segmented BSDC with block-based operations
- `ghrr.py`: Generalized HRR (matrix, non-commutative, SOTA 2024)
- `vtb.py`: Vector-derived Transformation Binding (non-commutative)

**holovec/encoders/** - Data type encoders
- `scalar.py`: FractionalPowerEncoder (FPE), ThermometerEncoder, LevelEncoder
- `vector.py`: VectorFPE for multivariate encoding with phase distributions (Gaussian/Laplace/Cauchy/Student)
- `periodic.py`: PeriodicAngleEncoder for cyclic data (time, angles)
- `sequence.py`: PositionBindingEncoder, NGramEncoder, TrajectoryEncoder
- `structured.py`: VectorEncoder for multi-dimensional feature vectors
- `spatial.py`: ImageEncoder for 2D grids

**holovec/retrieval/** - Memory and retrieval systems
- `codebook.py`: Persistent label→vector storage with JSON serialization
- `itemstore.py`: High-level retrieval interface with batch operations
- `assocstore.py`: Associative memory with key-value binding

**holovec/utils/** - Supporting utilities
- `cleanup.py`: BruteForceCleanup, ResonatorCleanup (iterative factorization)
- `search.py`: K-NN, threshold search, batch similarity
- `operations.py`: Top-k selection, noise injection, similarity matrices
- `cpse.py`: Context-Preserving Encoding (CPSE/CPSD algorithms)
- `decode.py`: Decoding utilities for retrieval

### Main Entry Point

Users primarily interact via `holovec.VSA.create()`:

```python
from holovec import VSA

# Create model (factory handles backend, space, and model initialization)
model = VSA.create('FHRR', dim=2048, backend='torch', device='cuda')

# Core operations
a, b = model.random(), model.random()
c = model.bind(a, b)                    # Binding (association)
d = model.bundle([a, b])                # Bundling (superposition)
e = model.permute(a, k=1)               # Permutation (sequence)
a_recovered = model.unbind(c, b)        # Unbinding (query)
sim = model.similarity(a, a_recovered)  # Similarity (cosine-like)
```

## Key Design Patterns

### Backend Abstraction
All numerical operations go through the backend interface. Never call NumPy/PyTorch/JAX directly in model code. Instead:

```python
# Good: Backend-agnostic
result = self.backend.multiply(a, b)
result = self.backend.fft(vector)

# Bad: Direct NumPy usage (breaks backend abstraction)
result = np.multiply(a, b)
```

### Model-Space-Backend Coordination
Models don't own random generation or similarity—the `VectorSpace` does:

```python
# Model delegates to space
def random(self, seed=None):
    return self.space.random(seed=seed)

def similarity(self, a, b):
    return self.space.similarity(a, b)
```

### Property-Based Testing
Tests use Hypothesis for property-based testing to validate algebraic properties across all backends:

```python
# Example: Verify binding is invertible
@given(st.integers(min_value=100, max_value=1000))
def test_unbinding_recovery(self, dim):
    model = VSA.create('FHRR', dim=dim)
    a, b = model.random(), model.random()
    c = model.bind(a, b)
    a_recovered = model.unbind(c, b)
    assert model.similarity(a, a_recovered) > 0.95
```

## Testing Strategy

### Coverage Requirements
- Minimum 90% coverage for all modules
- Core modules (models, backends, spaces) aim for 95-98%
- Use `pytest --cov=holovec --cov-report=html` to generate reports

### Backend Consistency
All tests should pass with NumPy, PyTorch, and JAX backends. Many tests are parameterized:

```python
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_operation(backend):
    if backend == "jax" and not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    # Test implementation
```

### Numerical Stability
Test edge cases: very small/large dimensions, extreme similarity values, sparse vectors, near-zero magnitudes.

## Code Style & Standards

### Type Annotations
- **Strict typing enabled**: All functions must have full type annotations
- Use `from __future__ import annotations` for forward references
- Mypy configuration is strict (no untyped defs, no incomplete defs)
- Array types use `Array` type alias from backends

### Docstrings
- **NumPy-style docstrings** for all public APIs
- Include Parameters, Returns, Examples, and Notes sections
- Reference academic papers where applicable (e.g., "See Plate (2003) for details")
- Examples should be runnable (use `>>>` prompt)

### Line Length
- Black formatter with 100 character line length
- Applies to code, comments, and docstrings

### Imports
- Ruff handles import sorting (isort-compatible)
- Group imports: stdlib, third-party, local
- Use absolute imports within holovec: `from holovec.backends import Backend`

## Common Development Workflows

### Adding a New VSA Model
1. Create `holovec/models/mymodel.py` inheriting from `VSAModel`
2. Implement required methods: `bind()`, `unbind()`, `bundle()`, `permute()`, `unpermute()`
3. Define properties: `is_self_inverse`, `is_commutative`, `is_exact_inverse`
4. Add to `holovec/models/__init__.py` exports
5. Register in `VSA._MODELS` dict in `holovec/__init__.py`
6. Write tests in `tests/test_models.py` with property-based checks
7. Test with all three backends
8. Add example to `docs/source/examples/`

### Adding a New Encoder
1. Create encoder class in appropriate file (`scalar.py`, `sequence.py`, etc.)
2. Inherit from base class: `ScalarEncoder`, `SequenceEncoder`, or `StructuredEncoder`
3. Implement `encode()` method (and optionally `decode()`)
4. Add comprehensive tests in `tests/test_encoders_*.py`
5. Document theory and parameters in docstring
6. Add to `holovec/encoders/__init__.py` exports

### Adding Backend Support
1. Create `holovec/backends/mybackend_backend.py` inheriting from `Backend`
2. Implement all abstract methods from `base.py`
3. Add capability checks: `supports_gpu()`, `supports_sparse()`, etc.
4. Add tests in `tests/test_backends.py`
5. Register in `holovec/backends/__init__.py`
6. Update README.md installation section

## Development Best Practices

### Commit Messages
Use DCO sign-off (`git commit -s`) with conventional commit format:
- `Add feature: ...` for new features
- `Fix: ...` for bug fixes
- `Update: ...` for enhancements
- `Refactor: ...` for code improvements
- `Docs: ...` for documentation
- `Test: ...` for tests

### Pre-commit Hooks
Install with `pip install pre-commit && pre-commit install`. Runs:
- Black (formatting)
- Ruff (linting)
- Mypy (type checking)
- Trailing whitespace removal

### Branching
- Feature branches: `feature/your-feature-name`
- Bug fixes: `fix/issue-description`
- Main branch: `master` (note: not `main`)

## Performance Considerations

### Backend Selection
- **NumPy**: CPU-only, ~1-10ms for dim=10000 operations
- **PyTorch**: GPU acceleration, ~0.1-1ms on CUDA (10-100x speedup for large batches)
- **JAX**: JIT compilation, ~0.01-0.1ms after warmup (10-100x speedup, especially for repeated ops)

### Dimension Scaling
- Typical range: 512-10000 dimensions
- FHRR has best capacity: ~330 dim for single binding
- Larger dimensions = better capacity but slower operations
- Sparse models (BSDC) can use 10000+ dimensions efficiently

### Batch Operations
When processing multiple vectors, use batched operations:
```python
# Good: Batched similarity (vectorized)
similarities = model.backend.batch_similarity(query, codebook_vectors)

# Bad: Loop over individual similarities
similarities = [model.similarity(query, v) for v in codebook_vectors]
```

## Important Caveats

### Backend Limitations
- JAX requires explicit device placement, no dynamic control flow in JIT
- PyTorch MPS (Apple Silicon GPU) has limited complex number support
- Some operations (e.g., matrix binding in GHRR) are memory-intensive

### Numerical Precision
- Default dtype is float32 for all backends (sufficient for HDC)
- Complex models (FHRR) use complex64
- Very high dimensions (>50000) may encounter numerical instability

### Model Selection Trade-offs
- **FHRR**: Best capacity but requires complex arithmetic (slower on some hardware)
- **MAP**: Fastest but lower capacity, good for hardware deployment
- **GHRR**: Non-commutative (order matters), useful for directional relationships
- **BSDC**: Sparse and memory-efficient but approximate inverse

## Repository Conventions

### File Organization
- One model per file in `holovec/models/`
- Encoders grouped by type: scalar, sequence, spatial, structured
- Tests mirror source structure: `test_models.py` ↔ `models/`
- Examples use numbered prefixes: `00_quickstart.py`, `01_basic_operations.py`

### Naming Conventions
- Classes: PascalCase (e.g., `FHRRModel`, `VectorSpace`)
- Functions/methods: snake_case (e.g., `bind()`, `create_model()`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_DIM`)
- Private members: leading underscore (e.g., `_validate()`)

### Documentation Structure
- `README.md`: User-facing overview, quick start, examples
- `CONTRIBUTING.md`: Development guidelines, code standards
- `docs/`: Sphinx documentation with API reference and theory guides
- `examples/`: Runnable examples (also used in docs via sphinx-gallery)
