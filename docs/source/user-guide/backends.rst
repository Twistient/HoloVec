Backends and Performance
========================

HoloVec supports three computational backends: NumPy, PyTorch, and JAX. This guide helps you choose and configure the right backend for your performance needs.

Quick Backend Selection
------------------------

**Quick Recommendations:**

* **CPU only, simple**: NumPy (default) - No dependencies, works everywhere
* **GPU acceleration**: PyTorch - Best GPU support, easy setup
* **Research/optimization**: JAX - JIT compilation, automatic differentiation
* **Production deployment**: NumPy or PyTorch - Mature, well-tested
* **Getting started**: NumPy - Zero configuration required

Backend Overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Backend
     - GPU
     - Speed
     - Setup
     - Best For
     - Trade-offs
   * - **NumPy**
     - No
     - Good
     - None
     - CPU, simplicity
     - No GPU acceleration
   * - **PyTorch**
     - Yes
     - Excellent
     - Easy
     - GPU, production
     - Larger dependency
   * - **JAX**
     - Yes
     - Excellent
     - Moderate
     - Research, optimization
     - Less mature, learning curve

NumPy Backend
-------------

**Default backend - works everywhere, requires no setup.**

Overview
^^^^^^^^

The NumPy backend uses pure NumPy arrays and operations. It's the default choice for CPU-based computation and provides solid performance for most applications.

**Key Properties:**

* No external dependencies beyond NumPy
* Runs on any platform (Windows, Mac, Linux, ARM)
* Good CPU performance with optimized NumPy
* Fully tested and stable

**When to Use:**

* Prototyping and development
* CPU-only environments
* Small to medium datasets
* Simple deployments without GPU
* Educational and research code

**Limitations:**

* No GPU acceleration
* Single-threaded for most operations
* May be slower for very large batches

Installation
^^^^^^^^^^^^

NumPy comes with HoloVec - no additional setup required:

.. code-block:: bash

    pip install holovec

Usage
^^^^^

.. code-block:: python

    from holovec import VSA

    # NumPy is the default backend
    model = VSA.create('FHRR', dim=10000)  # Uses NumPy

    # Or explicitly specify
    model = VSA.create('FHRR', dim=10000, backend='numpy')

    # All operations use NumPy arrays
    a = model.random(seed=1)
    b = model.random(seed=2)
    c = model.bind(a, b)

    print(type(a))  # <class 'numpy.ndarray'>

Performance Tips
^^^^^^^^^^^^^^^^

**Optimize NumPy installation:**

.. code-block:: bash

    # Use optimized NumPy builds for better CPU performance
    pip install numpy  # Often includes Intel MKL on compatible systems

**Batch operations:**

.. code-block:: python

    # Process in batches rather than one-by-one
    # Bad: slow loop
    results = []
    for item in items:
        hv = encoder.encode(item)
        results.append(hv)

    # Good: vectorize when possible
    # (depends on encoder, but generally faster)
    results = model.bundle([encoder.encode(item) for item in batch])

PyTorch Backend
---------------

**Best choice for GPU acceleration and production deployments.**

Overview
^^^^^^^^

The PyTorch backend leverages PyTorch tensors and CUDA for GPU acceleration. It provides excellent performance for both CPU and GPU workloads.

**Key Properties:**

* Full GPU support via CUDA
* Automatic CPU/GPU transfer
* Excellent performance optimizations
* Mature and well-maintained
* Large ecosystem

**When to Use:**

* GPU available (NVIDIA CUDA)
* Large-scale processing
* Production deployments
* Need fast batched operations
* Already using PyTorch in your stack

**Limitations:**

* Requires PyTorch installation (~500MB-2GB)
* GPU requires CUDA setup
* Slightly more memory overhead than NumPy

Installation
^^^^^^^^^^^^

**CPU-only (simple):**

.. code-block:: bash

    pip install holovec torch

**GPU with CUDA (recommended for GPU users):**

.. code-block:: bash

    # CUDA 11.8 (check your CUDA version)
    pip install holovec torch --index-url https://download.pytorch.org/whl/cu118

    # CUDA 12.1
    pip install holovec torch --index-url https://download.pytorch.org/whl/cu121

    # For latest instructions, see: https://pytorch.org/get-started/

**Verify GPU availability:**

.. code-block:: python

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

Usage
^^^^^

.. code-block:: python

    from holovec import VSA

    # Create model with PyTorch backend
    model = VSA.create('FHRR', dim=10000, backend='pytorch')

    # Automatically uses GPU if available
    a = model.random(seed=1)
    b = model.random(seed=2)
    c = model.bind(a, b)

    print(type(a))  # <class 'torch.Tensor'>
    print(a.device)  # cuda:0 (if GPU) or cpu

**Explicit device control:**

.. code-block:: python

    import torch

    # Force CPU
    model = VSA.create('FHRR', dim=10000, backend='pytorch')
    a = model.random(seed=1)
    a_cpu = a.cpu()  # Move to CPU if on GPU

    # Force GPU
    if torch.cuda.is_available():
        a_gpu = a.cuda()  # Move to GPU

GPU Configuration
^^^^^^^^^^^^^^^^^

**Select specific GPU:**

.. code-block:: python

    import torch

    # Use GPU 1 instead of 0
    torch.cuda.set_device(1)

    model = VSA.create('FHRR', dim=10000, backend='pytorch')

**Multiple GPUs:**

.. code-block:: python

    # PyTorch backend uses current device by default
    # For data parallelism, use torch.nn.DataParallel or manual batching

    device = torch.device("cuda:0")
    model = VSA.create('FHRR', dim=10000, backend='pytorch')

    # Process different batches on different GPUs
    # (requires manual orchestration)

Performance Tips
^^^^^^^^^^^^^^^^

**Batch operations for GPU:**

.. code-block:: python

    # GPUs excel at parallel processing
    # Process many items at once

    # Bad: one at a time (slow)
    for item in items:
        hv = encoder.encode(item)
        result = model.similarity(hv, query)

    # Good: batch processing (fast)
    hvs = [encoder.encode(item) for item in batch]
    bundled = model.bundle(hvs)

**Memory management:**

.. code-block:: python

    import torch

    # Clear GPU cache when done with large operations
    torch.cuda.empty_cache()

    # Monitor GPU memory
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

JAX Backend
-----------

**Best for research, optimization, and automatic differentiation.**

Overview
^^^^^^^^

The JAX backend provides JIT compilation, automatic differentiation, and GPU acceleration through XLA (Accelerated Linear Algebra).

**Key Properties:**

* JIT compilation for optimized execution
* Automatic differentiation (useful for learning)
* GPU and TPU support
* Functional programming style
* Great for research and optimization

**When to Use:**

* Research requiring gradients
* Performance-critical optimization
* Want JIT compilation benefits
* Using JAX in your stack
* Need TPU support

**Limitations:**

* Requires understanding of JAX paradigm
* Functional style (no in-place mutations)
* Less mature than PyTorch
* Smaller ecosystem

Installation
^^^^^^^^^^^^

**CPU-only:**

.. code-block:: bash

    pip install holovec jax jaxlib

**GPU with CUDA:**

.. code-block:: bash

    # CUDA 11 (check JAX docs for latest)
    pip install holovec jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # CUDA 12
    pip install holovec jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

**Verify GPU:**

.. code-block:: python

    import jax
    print(f"JAX devices: {jax.devices()}")
    print(f"Default device: {jax.default_backend()}")

Usage
^^^^^

.. code-block:: python

    from holovec import VSA

    # Create model with JAX backend
    model = VSA.create('FHRR', dim=10000, backend='jax')

    # Operations return JAX arrays
    a = model.random(seed=1)
    b = model.random(seed=2)
    c = model.bind(a, b)

    print(type(a))  # <class 'jaxlib.xla_extension.DeviceArray'>

**JIT compilation:**

.. code-block:: python

    from jax import jit

    # JIT compile operations for speed
    @jit
    def process_batch(encoder, model, items):
        hvs = [encoder.encode(item) for item in items]
        return model.bundle(hvs)

    # First call compiles, subsequent calls are fast
    result = process_batch(encoder, model, batch)

Performance Comparison
----------------------

Benchmark Results
^^^^^^^^^^^^^^^^^

Typical performance on common operations (approximate, varies by hardware):

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20 10

   * - Operation
     - NumPy (CPU)
     - PyTorch (GPU)
     - JAX (GPU)
     - Winner
   * - Random vector (10K dim)
     - 0.1 ms
     - 0.05 ms
     - 0.05 ms
     - Tie
   * - Bind (single)
     - 0.05 ms
     - 0.02 ms
     - 0.02 ms
     - GPU
   * - Bundle (100 vectors)
     - 2 ms
     - 0.3 ms
     - 0.3 ms
     - GPU
   * - Similarity (1000 queries)
     - 50 ms
     - 5 ms
     - 4 ms
     - JAX
   * - Encode 1000 scalars (FPE)
     - 100 ms
     - 10 ms
     - 8 ms
     - JAX

**Key insights:**

* NumPy: Good baseline CPU performance
* PyTorch GPU: 5-10x faster than NumPy for most operations
* JAX GPU: Similar to PyTorch, slight edge with JIT
* GPU advantage grows with batch size

Hardware Considerations
^^^^^^^^^^^^^^^^^^^^^^^

**When GPU is worth it:**

* Batch size > 100 items
* Dimension > 5,000
* Many similarity comparisons
* Repeated operations (JIT helps)

**When CPU is fine:**

* Small batches (< 50 items)
* Prototyping and development
* One-off computations
* Limited GPU memory

Switching Backends
------------------

Easy Backend Changes
^^^^^^^^^^^^^^^^^^^^

Switching backends is simple - just change one parameter:

.. code-block:: python

    from holovec import VSA

    # Try all three backends with same code
    for backend in ['numpy', 'pytorch', 'jax']:
        model = VSA.create('FHRR', dim=10000, backend=backend)

        a = model.random(seed=1)
        b = model.random(seed=2)
        c = model.bind(a, b)

        sim = model.similarity(a, b)
        print(f"{backend}: similarity = {sim:.3f}")

**Important notes:**

* Results are identical across backends (same seeds → same values)
* Performance characteristics differ
* Cannot mix arrays from different backends

Converting Between Backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    import torch
    import jax.numpy as jnp

    # NumPy to PyTorch
    numpy_array = np.random.randn(10000)
    torch_tensor = torch.from_numpy(numpy_array)

    # PyTorch to NumPy
    torch_tensor = torch.randn(10000)
    numpy_array = torch_tensor.cpu().numpy()

    # JAX to NumPy
    jax_array = jnp.ones(10000)
    numpy_array = np.array(jax_array)

    # NumPy to JAX
    numpy_array = np.random.randn(10000)
    jax_array = jnp.array(numpy_array)

Best Practices
--------------

Development Workflow
^^^^^^^^^^^^^^^^^^^^

**Recommended approach:**

1. **Prototype with NumPy** - Fast iteration, no setup
2. **Validate with examples** - Ensure correctness
3. **Profile to find bottlenecks** - Identify slow operations
4. **Switch to PyTorch/JAX if needed** - Only if performance matters
5. **Benchmark both** - PyTorch and JAX may perform differently

Example:

.. code-block:: python

    import time
    from holovec import VSA

    def benchmark_backend(backend, n_items=1000):
        model = VSA.create('FHRR', dim=10000, backend=backend)

        start = time.time()
        hvs = [model.random(seed=i) for i in range(n_items)]
        bundled = model.bundle(hvs)
        elapsed = time.time() - start

        print(f"{backend}: {elapsed:.3f}s for {n_items} items")
        return elapsed

    # Compare all backends
    for backend in ['numpy', 'pytorch', 'jax']:
        try:
            benchmark_backend(backend)
        except Exception as e:
            print(f"{backend}: not available ({e})")

Resource Management
^^^^^^^^^^^^^^^^^^^

**NumPy:**

.. code-block:: python

    # NumPy arrays released when out of scope
    # No special cleanup needed

**PyTorch:**

.. code-block:: python

    import torch

    # Clear GPU cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Delete large tensors explicitly when done
    del large_tensor
    torch.cuda.empty_cache()

**JAX:**

.. code-block:: python

    # JAX manages memory automatically
    # For explicit cleanup, delete arrays
    del large_array

    # Clear compilation cache if needed
    from jax import clear_backends
    clear_backends()

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**"CUDA out of memory" (PyTorch):**

.. code-block:: python

    # Reduce batch size
    batch_size = 100  # Try 50, 25, etc.

    # Process in smaller chunks
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        # Process batch

    # Clear cache
    torch.cuda.empty_cache()

**"No GPU detected" (PyTorch/JAX):**

.. code-block:: python

    # Check CUDA installation
    import torch
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    # Check JAX devices
    import jax
    print(jax.devices())

    # Verify NVIDIA driver
    # Terminal: nvidia-smi

**"Slow performance on GPU":**

.. code-block:: python

    # Ensure you're actually using GPU
    import torch
    a = model.random(seed=1)
    print(a.device)  # Should show 'cuda:0'

    # Batch operations for GPU efficiency
    # Single operations may be slower due to transfer overhead

**"Backend not available":**

.. code-block:: bash

    # Install missing backend
    pip install torch  # For PyTorch
    pip install jax jaxlib  # For JAX

Frequently Asked Questions
--------------------------

**Q: Which backend should I use?**

A: Start with NumPy (default). Switch to PyTorch if you have GPU and need better performance.

**Q: Do I need a GPU?**

A: No - NumPy backend works fine on CPU for most applications. GPU helps with large-scale processing.

**Q: Can I mix backends in one application?**

A: No - stick to one backend per program. You can convert arrays, but it's inefficient.

**Q: Which GPU is supported?**

A: PyTorch: NVIDIA GPUs with CUDA. JAX: NVIDIA GPUs with CUDA, Google TPUs.

**Q: How much faster is GPU?**

A: Typically 5-10x faster for batched operations. Speedup increases with batch size.

**Q: Will my results change with different backends?**

A: No - same random seeds produce identical results across backends (within floating point precision).

**Q: How do I know if I'm using GPU?**

A: PyTorch: ``print(tensor.device)``. JAX: ``print(jax.devices())``.

**Q: Can I use Apple Silicon GPU (M1/M2)?**

A: PyTorch supports MPS backend for Apple Silicon. Check PyTorch docs for setup.

Next Steps
----------

* :doc:`choosing-models` - Pick the right VSA model
* :doc:`encoding-data` - Encode your data efficiently
* :doc:`../examples/index` - See backends in action
* :doc:`../api/backends` - Backend API reference

See Also
--------

* :doc:`../api/backends` - Backend implementations
* `PyTorch Documentation <https://pytorch.org/docs/>`_
* `JAX Documentation <https://jax.readthedocs.io/>`_
* `NumPy Documentation <https://numpy.org/doc/>`_
