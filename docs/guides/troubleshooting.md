Common issues and solutions.

## Installation Issues

### Backend Not Available

**Symptom**: `Backend 'torch' not available`

**Solution**:
```bash
# Install PyTorch
pip install torch

# Or install with HoloVec
pip install holovec[torch]
```

### JAX Installation Failed

**Symptom**: `No module named 'jax'`

**Solution**:
```bash
# CPU-only JAX
pip install jax jaxlib

# GPU JAX (CUDA)
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Import Error

**Symptom**: `ImportError: cannot import name 'VSA' from 'holovec'`

**Solution**: Ensure proper installation:
```bash
pip install -e .  # From source
# or
pip install holovec  # From PyPI
```

---

## Runtime Errors

### Dimension Mismatch

**Symptom**: `ValueError: Shape mismatch: (2048,) vs (1024,)`

**Cause**: Mixing vectors from different models.

**Solution**: Use vectors from the same model:
```python
# Wrong
model1 = VSA.create('FHRR', dim=2048)
model2 = VSA.create('FHRR', dim=1024)
model1.bind(model1.random(), model2.random())  # Error!

# Correct
a = model1.random()
b = model1.random()
model1.bind(a, b)
```

### Empty Bundle

**Symptom**: `ValueError: Cannot bundle empty sequence`

**Solution**: Check your input:
```python
# Wrong
model.bundle([])

# Correct
if vectors:
    result = model.bundle(vectors)
```

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch size
batch_size = 32  # Instead of 1024

# Clear cache
import torch
torch.cuda.empty_cache()

# Use smaller dimension
model = VSA.create('FHRR', dim=1024)  # Instead of 10000
```

---

## Unexpected Results

### Low Similarity After Unbinding

**Symptom**: `unbind(bind(a, b), b)` gives similarity ~0.7, not 1.0

**Cause**: Using model with approximate inverse (HRR, VTB, BSDC).

**Solution**: Use exact inverse model or cleanup:
```python
# Use FHRR for exact recovery
model = VSA.create('FHRR', dim=2048)

# Or use cleanup for approximate models
from holovec.utils.cleanup import BruteForceCleanup
cleanup = BruteForceCleanup(model)
cleaned = cleanup.cleanup(noisy_result, codebook)
```

### Random Vectors Not Orthogonal

**Symptom**: `similarity(random1, random2) ≈ 0.3` instead of ~0.0

**Cause**: Dimension too low.

**Solution**: Increase dimension:
```python
# Too low dimension
model = VSA.create('FHRR', dim=100)  # Low orthogonality

# Better
model = VSA.create('FHRR', dim=1000)  # Good orthogonality
```

### Binding Returns Same Result

**Symptom**: `bind(a, b) ≈ a`

**Cause**: One vector is near identity (all 1s for MAP/BSC).

**Solution**: Verify vectors are random:
```python
# Check vector values
print(a.min(), a.max(), a.mean())
```

### Bundle Doesn't Preserve Components

**Symptom**: Components not retrievable from bundle.

**Cause**: Too many items bundled (exceeds capacity).

**Solution**: Reduce bundle size or increase dimension:
```python
# Check capacity
max_items = dimension / 50  # Rough estimate for FHRR

# Split into smaller bundles
chunk_size = 100
chunks = [model.bundle(items[i:i+chunk_size])
          for i in range(0, len(items), chunk_size)]
```

---

## Performance Issues

### JAX First Call Slow

**Symptom**: First operation takes 100ms, then 0.1ms.

**Cause**: JIT compilation happens on first call.

**Solution**: Warm up before timing:
```python
# Warmup
model.bind(model.random(), model.random())

# Now time
import time
start = time.time()
for _ in range(100):
    model.bind(a, b)
print(f"Avg: {(time.time() - start) * 10:.2f} ms")
```

### GPU Not Being Used

**Symptom**: CUDA available but operations slow.

**Cause**: Vectors not on GPU.

**Solution**: Verify device:
```python
import torch

# Check device
print(a.device)  # Should be cuda:0

# Move to GPU
model = VSA.create('FHRR', dim=2048, backend='torch', device='cuda')
```

### Memory Growing Over Time

**Symptom**: Memory usage increases in loop.

**Solution**: Clear references:
```python
# Use context manager for temporary vectors
import gc

for i in range(10000):
    temp = model.random()
    # ... use temp ...

    if i % 1000 == 0:
        gc.collect()
        torch.cuda.empty_cache()  # If using CUDA
```

---

## Model-Specific Issues

### GHRR High Memory Usage

**Symptom**: GHRR uses much more memory than other models.

**Cause**: Matrix space uses O(d × m²) memory.

**Solution**: Reduce matrix size or dimension:
```python
# Default: 100 × 3 × 3 = 900 values per vector
model = VSA.create('GHRR', dim=100, matrix_size=3)

# Smaller: 50 × 2 × 2 = 200 values per vector
model = VSA.create('GHRR', dim=50, matrix_size=2)
```

### BSDC Sparsity Drift

**Symptom**: After operations, vectors are denser than expected.

**Solution**: Use rehash:
```python
# After multiple operations
c = model.bind(model.bind(a, b), c)
c_rehashed = model.rehash(c)
```

### BSC/BSDC Similarity Always 0 or 1

**Symptom**: Binary similarity gives extreme values.

**Cause**: Hamming distance on small overlap.

**Solution**: Increase dimension for finer granularity:
```python
model = VSA.create('BSC', dim=10000)  # More granular similarity
```

---

## Getting Help

If your issue isn't covered here:

1. **Search existing issues**: [GitHub Issues](https://github.com/Twistient/HoloVec/issues)
2. **Ask a question**: [GitHub Discussions](https://github.com/Twistient/HoloVec/discussions)
3. **Report a bug**: Include minimal reproduction code
4. **Contact**: brodie@twistient.com

---

## See Also

- [Installation](../getting-started/installation.md) — Setup guide
- [Performance](../guides/performance.md) — Optimization tips
- [Models](../models/index.md) — Model characteristics
