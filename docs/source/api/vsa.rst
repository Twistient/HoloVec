VSA - Main API
==============

The :class:`~holovec.VSA` class is the main entry point for creating and using VSA models in HoloVec.

.. currentmodule:: holovec

VSA Factory Class
-----------------

.. autoclass:: VSA
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~VSA.create
      ~VSA.available_models
      ~VSA.available_backends

Quick Start
-----------

The simplest way to create a VSA model:

.. code-block:: python

    from holovec import VSA

    # Create a model with defaults
    model = VSA.create('FHRR', dim=10000, seed=42)

    # Generate random vectors
    a = model.random(seed=1)
    b = model.random(seed=2)

    # Bind (associate)
    c = model.bind(a, b)

    # Unbind (recover)
    a_recovered = model.unbind(c, b)

    # Check similarity
    similarity = model.similarity(a, a_recovered)
    print(f"Similarity: {similarity:.3f}")  # ~1.0

Model Selection
---------------

Choose a model based on your requirements:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - Model
     - Best For
     - Inverse
     - Space
     - Key Feature
   * - **MAP**
     - Hardware, speed
     - Self-inverse
     - Bipolar
     - Simple multiplication
   * - **FHRR**
     - Capacity, accuracy
     - Exact inverse
     - Complex
     - FFT-based, best capacity
   * - **HRR**
     - General purpose
     - Approximate
     - Real
     - Circular convolution
   * - **BSC**
     - Memory efficiency
     - Self-inverse
     - Binary
     - Sparse binary
   * - **BSDC**
     - High capacity
     - Self-inverse
     - Sparse
     - Block-structured
   * - **GHRR**
     - Custom geometry
     - Tunable
     - Matrix
     - Generalized algebra
   * - **VTB**
     - Transform binding
     - Learnable
     - Real
     - Vector-derived

See :doc:`../user-guide/choosing-models` for detailed guidance.

Backend Selection
-----------------

Choose a backend for different hardware:

.. code-block:: python

    # NumPy (CPU, default)
    model = VSA.create('FHRR', dim=10000, backend='numpy')

    # PyTorch (GPU)
    model = VSA.create('FHRR', dim=10000, backend='torch', device='cuda')

    # JAX (JIT/TPU)
    model = VSA.create('FHRR', dim=10000, backend='jax')

See :doc:`../user-guide/backends` for performance comparisons.

Creating Models
---------------

**Basic creation:**

.. code-block:: python

    model = VSA.create('MAP', dim=10000, seed=42)

**With specific backend:**

.. code-block:: python

    model = VSA.create('FHRR', dim=10000, backend='torch', device='cuda')

**With custom vector space:**

.. code-block:: python

    from holovec.spaces import create_space

    space = create_space('bipolar', dim=10000, backend='numpy')
    model = VSA.create('MAP', space=space)

**Check available options:**

.. code-block:: python

    # List all models
    print(VSA.available_models())
    # ['MAP', 'FHRR', 'HRR', 'BSC', 'BSDC', 'GHRR', 'VTB']

    # List available backends
    print(VSA.available_backends())
    # ['numpy', 'torch', 'jax']

Common Operations
-----------------

Once you have a model, these are the most common operations:

**Binding (association):**

.. code-block:: python

    # Bind two vectors (like key-value pairs)
    bound = model.bind(key, value)

**Unbinding (recovery):**

.. code-block:: python

    # Recover value given key
    recovered = model.unbind(bound, key)

**Bundling (superposition):**

.. code-block:: python

    # Combine multiple vectors
    bundle = model.bundle([vec1, vec2, vec3])

**Similarity:**

.. code-block:: python

    # Measure similarity (cosine-like)
    sim = model.similarity(vec1, vec2)  # Returns float in [-1, 1]

**Permutation (for sequences):**

.. code-block:: python

    # Shift/rotate vector (model-dependent)
    permuted = model.permute(vec, k=1)

See :class:`~holovec.models.VSAModel` for the complete API.

Functional Interface
--------------------

For direct model creation without the factory:

.. code-block:: python

    from holovec import create_model

    model = create_model('FHRR', dim=10000, seed=42)

This is equivalent to ``VSA.create()`` but as a function.

.. autofunction:: create_model

Examples
--------

See the :doc:`../examples/index` for 28+ examples including:

* :ref:`sphx_glr_examples_00_quickstart.py` - 5-minute introduction
* :ref:`sphx_glr_examples_01_basic_operations.py` - Core operations
* :ref:`sphx_glr_examples_02_models_comparison.py` - Model comparison

See Also
--------

* :doc:`models` - All model implementations
* :doc:`../user-guide/choosing-models` - Model selection guide
* :doc:`../user-guide/backends` - Backend performance guide
* :doc:`../theory/vsa_models` - Theoretical foundations
