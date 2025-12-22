Guidelines for optimizing HoloVec applications.

## Backend Selection

### When to Use Each Backend

| Scenario | Backend | Reason |
|----------|---------|--------|
| Development/debugging | NumPy | Simple, fast startup |
| Single operations | NumPy | Lowest overhead |
| GPU available | PyTorch | Hardware acceleration |
| Batch processing | PyTorch (GPU) | Parallel execution |
| Repeated operations | JAX | JIT compilation |
| TPU deployment | JAX | Native support |

### Backend Performance Comparison

Operations on dim=10,000 vectors:

| Operation | NumPy | PyTorch CPU | PyTorch CUDA | JAX (JIT) |
|-----------|-------|-------------|--------------|-----------|
| bind() | 0.1 ms | 0.1 ms | 0.01 ms | 0.01 ms |
| bundle(10) | 0.3 ms | 0.3 ms | 0.02 ms | 0.02 ms |
| similarity() | 0.05 ms | 0.05 ms | 0.005 ms | 0.005 ms |

!!! note
    JAX times are after JIT warmup. First call is slower (~100ms).

---

## Dimension Selection

### Capacity vs Speed Trade-off

| Dimension | Capacity | bind() time | Memory |
|-----------|----------|-------------|--------|
| 512 | ~80 items | 0.01 ms | 4 KB |
| 2048 | ~330 items | 0.05 ms | 16 KB |
| 10000 | ~1600 items | 0.2 ms | 80 KB |
| 50000 | ~8000 items | 1.0 ms | 400 KB |

### Recommendations

- **Prototype/testing**: 512-1024
- **Production (moderate)**: 2048-4096
- **High capacity needs**: 10000+
- **Memory constrained**: Use sparse models (BSDC)

---

## Model Performance

### Operation Speed (dim=2048)

| Model | bind() | bundle(10) | similarity() |
|-------|--------|------------|--------------|
| MAP | 0.03 ms | 0.15 ms | 0.02 ms |
| FHRR | 0.05 ms | 0.20 ms | 0.03 ms |
| HRR | 0.10 ms | 0.20 ms | 0.02 ms |
| BSC | 0.02 ms | 0.10 ms | 0.02 ms |
| BSDC | 0.02 ms | 0.15 ms | 0.01 ms |
| GHRR | 0.50 ms | 1.00 ms | 0.20 ms |
| VTB | 0.30 ms | 0.50 ms | 0.02 ms |

### Model Selection by Speed

1. **Fastest**: BSC, BSDC (binary ops)
2. **Fast**: MAP (element-wise multiply)
3. **Moderate**: FHRR, HRR (FFT-based)
4. **Slower**: VTB, GHRR (matrix ops)

---

## Batch Operations

Always prefer batch operations over loops:

```python
# SLOW: Loop over similarities
similarities = []
for v in codebook_vectors:
    similarities.append(model.similarity(query, v))

# FAST: Batch similarity
similarities = model.backend.batch_similarity(query, codebook_vectors)
```

Speedup: 10-100× for large codebooks

### Batch Encoding

```python
# SLOW: Encode one at a time
vectors = [encoder.encode(x) for x in values]

# FAST: Batch encode (if encoder supports it)
vectors = encoder.batch_encode(values)
```

---

## Memory Optimization

### Dense vs Sparse

| Type | Memory (dim=10000) | Use Case |
|------|-------------------|----------|
| Dense float32 | 40 KB | Most models |
| Dense float16 | 20 KB | GPU inference |
| Sparse (1%) | 0.4 KB | BSDC |

### Reducing Memory

```python
# Use smaller dtype (GPU)
model = VSA.create('FHRR', dim=2048, backend='torch', dtype='float16')

# Use sparse model
model = VSA.create('BSDC', dim=50000, sparsity=0.01)  # Only 500 active bits

# Clear unused vectors
del old_vectors
torch.cuda.empty_cache()  # For PyTorch GPU
```

---

## GPU Optimization

### PyTorch CUDA

```python
import torch

# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Create GPU model
model = VSA.create('FHRR', dim=4096, backend='torch', device='cuda')

# Move existing tensors to GPU
tensor_gpu = tensor.to('cuda')
```

### Apple Silicon (MPS)

```python
import torch

# Check MPS availability
print(torch.backends.mps.is_available())

# Create MPS model
model = VSA.create('FHRR', dim=4096, backend='torch', device='mps')
```

!!! note
    MPS has full complex number support as of PyTorch 2.0.

---

## JAX JIT Compilation

### Warmup Pattern

```python
import jax

model = VSA.create('FHRR', dim=2048, backend='jax')

# First call triggers compilation (slow)
a, b = model.random(), model.random()
c = model.bind(a, b)  # ~100ms

# Subsequent calls are fast
for _ in range(1000):
    c = model.bind(a, b)  # ~0.01ms each
```

### JIT Custom Functions

```python
from jax import jit

@jit
def encode_and_bind(encoder, values, roles):
    encoded = [encoder.encode(v) for v in values]
    bound = [model.bind(e, r) for e, r in zip(encoded, roles)]
    return model.bundle(bound)
```

---

## Profiling

### Timing Operations

```python
import time

def benchmark(fn, iterations=100):
    # Warmup
    for _ in range(10):
        fn()

    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms

# Benchmark binding
ms = benchmark(lambda: model.bind(a, b))
print(f"bind(): {ms:.3f} ms")
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# Your code here
vectors = [model.random() for _ in range(1000)]

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024:.1f} KB, Peak: {peak / 1024:.1f} KB")
tracemalloc.stop()
```

---

## Best Practices

1. **Start simple**: Use NumPy for development
2. **Batch when possible**: Avoid Python loops
3. **Profile first**: Find actual bottlenecks
4. **Match model to task**: Don't use GHRR if MAP suffices
5. **Consider sparsity**: BSDC for memory constraints
6. **Warm up JAX**: Account for JIT compilation time
7. **Use appropriate dimension**: Balance capacity vs speed

---

## See Also

- [Backends](../architecture/backends.md) — Backend details
- [Models](../models/index.md) — Model comparison
- [Tutorials](../guides/patterns.md) — Performance benchmarks example
