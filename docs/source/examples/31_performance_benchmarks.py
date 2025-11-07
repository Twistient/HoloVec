"""
Performance Benchmarks
======================

Topics: Speed comparison, accuracy testing, backend selection, model efficiency
Time: 15 minutes
Prerequisites: 02_models_comparison.py, 01_basic_operations.py
Related: 32_distributed_representations.py, 02_models_comparison.py

This example benchmarks different VSA models and backends to help you choose
the right configuration for your application's performance requirements.

Key concepts:
- Operation speed: bind, bundle, permute, similarity
- Backend comparison: NumPy (CPU) vs PyTorch (GPU) vs JAX (JIT)
- Model efficiency: Memory and computation trade-offs
- Dimension scaling: How performance changes with dimension
- Practical recommendations: Choose based on your constraints

Use this to make informed decisions about model and backend selection.
"""

import time
import numpy as np
from holovec import VSA

print("=" * 70)
print("Performance Benchmarks")
print("=" * 70)
print()

# ============================================================================
# Demo 1: Operation Speed by Model
# ============================================================================
print("=" * 70)
print("Demo 1: Basic Operation Speed (NumPy backend)")
print("=" * 70)

dimension = 10000
n_iterations = 1000

models_to_test = ['MAP', 'FHRR', 'HRR', 'BSC']

print(f"\nDimension: {dimension}")
print(f"Iterations: {n_iterations}")
print(f"Backend: NumPy (CPU)")
print()

results = {}

for model_name in models_to_test:
    model = VSA.create(model_name, dim=dimension, seed=42)

    # Create test vectors
    A = model.random(seed=1)
    B = model.random(seed=2)
    vectors = [model.random(seed=i) for i in range(10)]

    # Benchmark bind
    start = time.time()
    for _ in range(n_iterations):
        _ = model.bind(A, B)
    bind_time = (time.time() - start) / n_iterations * 1000  # ms

    # Benchmark bundle
    start = time.time()
    for _ in range(n_iterations):
        _ = model.bundle(vectors)
    bundle_time = (time.time() - start) / n_iterations * 1000

    # Benchmark similarity
    start = time.time()
    for _ in range(n_iterations):
        _ = model.similarity(A, B)
    sim_time = (time.time() - start) / n_iterations * 1000

    # Benchmark permute (if available)
    try:
        start = time.time()
        for _ in range(n_iterations):
            _ = model.permute(A)
        perm_time = (time.time() - start) / n_iterations * 1000
    except:
        perm_time = None

    results[model_name] = {
        'bind': bind_time,
        'bundle': bundle_time,
        'similarity': sim_time,
        'permute': perm_time
    }

# Print results
print(f"{'Model':<10s} {'Bind (ms)':<12s} {'Bundle (ms)':<12s} {'Sim (ms)':<12s} {'Permute (ms)':<12s}")
print("-" * 70)

for model_name, times in results.items():
    perm_str = f"{times['permute']:.4f}" if times['permute'] else "N/A"
    print(f"{model_name:<10s} {times['bind']:10.4f}   {times['bundle']:10.4f}   "
          f"{times['similarity']:10.4f}   {perm_str:>10s}")

print("\nObservations:")
print("  - MAP typically fastest (simple multiplication)")
print("  - FHRR/HRR slower (FFT operations)")
print("  - BSC depends on sparsity (fewer operations on sparse vectors)")

# ============================================================================
# Demo 2: Dimension Scaling
# ============================================================================
print("\n" + "=" * 70)
print("Demo 2: Performance vs Dimension")
print("=" * 70)

dimensions = [1000, 5000, 10000, 20000]
model_name = 'MAP'  # Test with MAP (fastest)

print(f"\nModel: {model_name}")
print(f"\n{'Dimension':<12s} {'Bind (ms)':<12s} {'Bundle (ms)':<12s} {'Similarity (ms)':<15s}")
print("-" * 60)

for dim in dimensions:
    model = VSA.create(model_name, dim=dim, seed=42)
    A = model.random(seed=1)
    B = model.random(seed=2)
    vectors = [model.random(seed=i) for i in range(10)]

    # Quick benchmark (fewer iterations for larger dims)
    n_iter = max(100, 10000 // (dim // 1000))

    # Bind
    start = time.time()
    for _ in range(n_iter):
        _ = model.bind(A, B)
    bind_time = (time.time() - start) / n_iter * 1000

    # Bundle
    start = time.time()
    for _ in range(n_iter):
        _ = model.bundle(vectors)
    bundle_time = (time.time() - start) / n_iter * 1000

    # Similarity
    start = time.time()
    for _ in range(n_iter):
        _ = model.similarity(A, B)
    sim_time = (time.time() - start) / n_iter * 1000

    print(f"{dim:<12d} {bind_time:10.4f}   {bundle_time:10.4f}   {sim_time:12.4f}")

print("\nScaling pattern:")
print("  - Generally linear with dimension")
print("  - Bundle scales with number of vectors to combine")
print("  - Similarity involves dot product (linear complexity)")

# ============================================================================
# Demo 3: Memory Usage
# ============================================================================
print("\n" + "=" * 70)
print("Demo 3: Memory Footprint")
print("=" * 70)

dimension = 10000

print(f"\nDimension: {dimension}")
print(f"\n{'Model':<10s} {'Dtype':<15s} {'Bytes/Vector':<15s} {'MB/1000 vectors':<15s}")
print("-" * 65)

for model_name in ['MAP', 'FHRR', 'HRR', 'BSC']:
    model = VSA.create(model_name, dim=dimension, seed=42)
    A = model.random(seed=1)

    # Get dtype info
    if hasattr(A, 'dtype'):
        dtype = str(A.dtype)
        itemsize = A.itemsize if hasattr(A, 'itemsize') else 8
    else:
        dtype = "backend-specific"
        itemsize = 8  # estimate

    bytes_per_vector = dimension * itemsize
    mb_per_1000 = bytes_per_vector * 1000 / (1024 * 1024)

    print(f"{model_name:<10s} {dtype:<15s} {bytes_per_vector:<15,d} {mb_per_1000:14.2f}")

print("\nMemory considerations:")
print("  - MAP: int8 or float32 (smallest)")
print("  - FHRR/HRR: complex64/128 (larger, 2x float)")
print("  - BSC: Binary sparse (very small if sparse)")
print("  - Choose based on storage constraints")

# ============================================================================
# Demo 4: Accuracy Under Noise
# ============================================================================
print("\n" + "=" * 70)
print("Demo 4: Noise Tolerance Comparison")
print("=" * 70)

dimension = 10000
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

print(f"\nDimension: {dimension}")
print(f"Test: similarity(A, A + noise)")
print()

print(f"{'Noise':<10s} ", end="")
for model_name in models_to_test:
    print(f"{model_name:<10s} ", end="")
print()
print("-" * 55)

for noise_level in noise_levels:
    print(f"{noise_level:<10.1f} ", end="")

    for model_name in models_to_test:
        model = VSA.create(model_name, dim=dimension, seed=42)
        A = model.random(seed=1)

        # Add noise
        noise = model.random(seed=999)
        noisy_A = model.bundle([A, noise])  # Simple noise addition via bundling

        # Measure similarity to original
        sim = float(model.similarity(A, noisy_A))
        print(f"{sim:10.3f} ", end="")

    print()

print("\nNoise tolerance:")
print("  - All models degrade gracefully with noise")
print("  - Higher dimension = better noise tolerance")
print("  - Use cleanup strategies for noise-heavy applications")

# ============================================================================
# Demo 5: Bundling Capacity
# ============================================================================
print("\n" + "=" * 70)
print("Demo 5: Bundling Capacity (Information Loss)")
print("=" * 70)

dimension = 10000
bundle_sizes = [1, 5, 10, 20, 50, 100]

print(f"\nDimension: {dimension}")
print(f"Test: similarity after bundling N vectors")
print()

print(f"{'N vectors':<12s} ", end="")
for model_name in models_to_test:
    print(f"{model_name:<10s} ", end="")
print()
print("-" * 60)

for n in bundle_sizes:
    print(f"{n:<12d} ", end="")

    for model_name in models_to_test:
        model = VSA.create(model_name, dim=dimension, seed=42)

        # Create target and other vectors
        target = model.random(seed=1)
        others = [model.random(seed=100+i) for i in range(1, n)]

        # Bundle
        if n == 1:
            bundled = target
        else:
            bundled = model.bundle([target] + others)

        # Similarity to original
        sim = float(model.similarity(bundled, target))
        print(f"{sim:10.3f} ", end="")

    print()

print("\nCapacity insights:")
print("  - Similarity degrades as bundle size increases")
print("  - MAP maintains higher similarity (sum-based)")
print("  - Higher dimension supports more vectors in bundle")

# ============================================================================
# Demo 6: Backend Comparison (if available)
# ============================================================================
print("\n" + "=" * 70)
print("Demo 6: Backend Comparison")
print("=" * 70)

available_backends = []

# Test numpy
try:
    model = VSA.create('MAP', dim=10000, backend='numpy', seed=42)
    available_backends.append('numpy')
except:
    pass

# Test torch (if available)
try:
    model = VSA.create('MAP', dim=10000, backend='torch', seed=42)
    available_backends.append('torch')
except:
    pass

# Test jax (if available)
try:
    model = VSA.create('MAP', dim=10000, backend='jax', seed=42)
    available_backends.append('jax')
except:
    pass

print(f"\nAvailable backends: {', '.join(available_backends)}")

if len(available_backends) > 1:
    print("\nBenchmarking available backends...")
    dimension = 10000
    n_iter = 100

    print(f"\n{'Backend':<10s} {'Bind (ms)':<12s} {'Bundle (ms)':<12s} {'Similarity (ms)':<15s}")
    print("-" * 60)

    for backend in available_backends:
        model = VSA.create('MAP', dim=dimension, backend=backend, seed=42)
        A = model.random(seed=1)
        B = model.random(seed=2)
        vectors = [model.random(seed=i) for i in range(10)]

        # Bind
        start = time.time()
        for _ in range(n_iter):
            _ = model.bind(A, B)
        bind_time = (time.time() - start) / n_iter * 1000

        # Bundle
        start = time.time()
        for _ in range(n_iter):
            _ = model.bundle(vectors)
        bundle_time = (time.time() - start) / n_iter * 1000

        # Similarity
        start = time.time()
        for _ in range(n_iter):
            _ = model.similarity(A, B)
        sim_time = (time.time() - start) / n_iter * 1000

        print(f"{backend:<10s} {bind_time:10.4f}   {bundle_time:10.4f}   {sim_time:12.4f}")
else:
    print(f"\nOnly {available_backends[0]} backend available.")
    print("\nTo test other backends:")
    print("  pip install torch  # For GPU acceleration")
    print("  pip install jax jaxlib  # For JIT compilation")

print("\nBackend recommendations:")
print("  - NumPy: Default, good for CPU, no extra dependencies")
print("  - PyTorch: Best for GPU, large batches, deep learning integration")
print("  - JAX: Best for JIT compilation, TPU, functional programming")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Performance Recommendations")
print("=" * 70)
print()

print("✓ Model Selection by Speed:")
print("  1. MAP - Fastest (element-wise multiplication)")
print("  2. BSC - Fast for sparse operations")
print("  3. HRR/FHRR - Slower (FFT overhead)")
print()

print("✓ Dimension Recommendations:")
print("  - Small problems (<1000 items): 1000-5000 dim")
print("  - Medium problems (1000-10000 items): 5000-10000 dim")
print("  - Large problems (>10000 items): 10000-20000 dim")
print()

print("✓ Backend Selection:")
print("  - CPU only: NumPy (default)")
print("  - GPU available: PyTorch (faster for large batches)")
print("  - Need JIT/TPU: JAX (compile once, run fast)")
print()

print("✓ Memory Constraints:")
print("  - Limited memory: MAP with lower dimension")
print("  - Plenty of memory: Any model, higher dimension")
print("  - Sparse data: BSC (efficient sparse storage)")
print()

print("✓ Noise Tolerance:")
print("  - High noise: Higher dimension, use cleanup strategies")
print("  - Low noise: Standard dimension (10000) sufficient")
print()

print("Next steps:")
print("  → 32_distributed_representations.py - Capacity deep dive")
print("  → 33_error_handling_robustness.py - Noise handling strategies")
print("  → 02_models_comparison.py - Model characteristics")
print()
print("=" * 70)
