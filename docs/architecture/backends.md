HoloVec abstracts computational operations through a backend interface. This enables the same code to run on CPUs, GPUs, and TPUs without modification.

## Available Backends

| Backend | Hardware | Key Features |
|---------|----------|--------------|
| **NumPy** | CPU only | Always available, zero dependencies |
| **PyTorch** | CPU, CUDA, MPS | GPU acceleration, neural integration |
| **JAX** | CPU, GPU, TPU | JIT compilation, autodiff |

## Selecting a Backend

```python
from holovec import VSA

# NumPy (default)
model = VSA.create('FHRR', dim=2048)

# PyTorch with CPU
model = VSA.create('FHRR', dim=2048, backend='torch')

# PyTorch with NVIDIA GPU
model = VSA.create('FHRR', dim=2048, backend='torch', device='cuda')

# PyTorch with Apple Silicon GPU
model = VSA.create('FHRR', dim=2048, backend='torch', device='mps')

# JAX
model = VSA.create('FHRR', dim=2048, backend='jax')
```

## Backend Comparison

### NumPy

**Best for:** Development, small-scale experiments, maximum compatibility.

```python
model = VSA.create('FHRR', dim=2048, backend='numpy')
```

**Characteristics:**
- Always available (no extra dependencies)
- Consistent behavior across platforms
- Optimized for modern CPUs via BLAS/LAPACK
- ~1-10ms per operation at dim=10,000

### PyTorch

**Best for:** GPU acceleration, integration with neural networks, production deployment.

```python
# Auto-select available device
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VSA.create('FHRR', dim=2048, backend='torch', device=device)
```

**Characteristics:**
- CUDA support for NVIDIA GPUs
- MPS support for Apple Silicon
- 10-100x speedup for large batches on GPU
- Seamless integration with PyTorch models
- Requires `pip install torch`

**Device Options:**
- `'cpu'` — CPU computation
- `'cuda'` — First NVIDIA GPU
- `'cuda:0'`, `'cuda:1'` — Specific GPU
- `'mps'` — Apple Metal Performance Shaders

### JAX

**Best for:** JIT compilation, TPU deployment, automatic differentiation research.

```python
model = VSA.create('FHRR', dim=2048, backend='jax')
```

**Characteristics:**
- JIT compilation for 10-100x speedup after warmup
- TPU support for large-scale computation
- Functional programming model
- Automatic differentiation
- Requires `pip install jax jaxlib`

!!! note
    JAX JIT compilation happens on first call. Subsequent calls are significantly faster.

## Performance Benchmarks

Approximate performance for dim=10,000, 1000 operations:

| Backend | bind() | bundle(10) | similarity() |
|---------|--------|------------|--------------|
| NumPy | 2ms | 4ms | 0.5ms |
| PyTorch CPU | 2ms | 4ms | 0.5ms |
| PyTorch CUDA | 0.1ms | 0.2ms | 0.05ms |
| JAX (first call) | 100ms | 200ms | 50ms |
| JAX (JIT warmed) | 0.05ms | 0.1ms | 0.02ms |

!!! tip
    For one-off operations, NumPy is fastest. For repeated operations or batches, use JAX with JIT or PyTorch with GPU.

## Backend Capabilities

Check what's available:

```python
from holovec import backend_info

info = backend_info()
print(info)
# {'numpy': True, 'torch': True, 'jax': False}
```

Check specific capabilities:

```python
from holovec.backends import TorchBackend

backend = TorchBackend()
print(backend.supports_gpu())      # True if CUDA/MPS available
print(backend.supports_complex())  # True
print(backend.supports_sparse())   # True
```

## Backend Interface

All backends implement the same interface:

```python
class Backend(ABC):
    # Array creation
    def zeros(self, shape, dtype) -> Array
    def ones(self, shape, dtype) -> Array
    def random(self, shape, dtype) -> Array

    # Element-wise operations
    def add(self, a, b) -> Array
    def multiply(self, a, b) -> Array
    def conj(self, a) -> Array  # Complex conjugate

    # Reductions
    def sum(self, a, axis=None) -> Array
    def dot(self, a, b) -> Scalar

    # Linear algebra
    def fft(self, a) -> Array
    def ifft(self, a) -> Array
    def norm(self, a) -> Scalar

    # Utilities
    def to_numpy(self, a) -> np.ndarray
    def from_numpy(self, arr) -> Array
```

## Cross-Backend Compatibility

HoloVec ensures consistent behavior across backends:

```python
# Same operations, same results
for backend in ['numpy', 'torch', 'jax']:
    model = VSA.create('FHRR', dim=1024, backend=backend, seed=42)
    a, b = model.random(seed=1), model.random(seed=2)
    c = model.bind(a, b)
    sim = model.similarity(a, model.unbind(c, b))
    print(f"{backend}: {sim:.6f}")

# Output:
# numpy: 1.000000
# torch: 1.000000
# jax: 1.000000
```

## When to Use Each Backend

| Scenario | Recommended Backend |
|----------|---------------------|
| Development/debugging | NumPy |
| Small experiments (dim < 5000) | NumPy |
| GPU available, batch processing | PyTorch |
| Neural network integration | PyTorch |
| Repeated identical operations | JAX |
| TPU deployment | JAX |
| Maximum portability | NumPy |
| Edge device deployment | NumPy or PyTorch |

## See Also

- [Installation](../getting-started/installation.md) — Installing backend dependencies
- [Performance](../guides/performance.md) — Optimization guidelines
- [Architecture Overview](../architecture/index.md) — How backends fit in
