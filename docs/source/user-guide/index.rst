User Guide
==========

Complete guides for building applications with HoloVec.

These guides help you make informed decisions about models, encoders, and backends for your specific use case.

.. toctree::
   :maxdepth: 2

   choosing-models
   encoding-data
   backends

Overview
--------

The user guide is organized into three main topics:

1. **Choosing Models** - Pick the right VSA model for your application
2. **Encoding Data** - Transform your data into hypervectors
3. **Backends** - Optimize performance with NumPy, PyTorch, or JAX

Quick Navigation
----------------

Choosing a VSA Model
^^^^^^^^^^^^^^^^^^^^

:doc:`choosing-models`

**Learn how to:**

* Select the right model for your use case
* Understand trade-offs between models
* Compare performance characteristics
* Match models to hardware constraints

**Key decision factors:**

* Do you need exact unbinding? → FHRR, HRR, GHRR, VTB
* Need maximum speed? → MAP, BSC
* Working with continuous data? → FHRR (with FractionalPowerEncoder)
* Hardware constraints? → MAP, BSC (binary operations)

Encoding Your Data
^^^^^^^^^^^^^^^^^^

:doc:`encoding-data`

**Learn how to:**

* Choose encoders for different data types
* Encode scalars, sequences, images, and vectors
* Combine multiple features
* Handle unknown values

**Quick encoder selection:**

* **Numbers (continuous)**: FractionalPowerEncoder
* **Numbers (ordinal)**: ThermometerEncoder
* **Sequences (text)**: NGramEncoder
* **Trajectories**: TrajectoryEncoder
* **Images**: ImageEncoder
* **Embeddings**: VectorEncoder

Optimizing Performance
^^^^^^^^^^^^^^^^^^^^^^

:doc:`backends`

**Learn how to:**

* Choose between NumPy, PyTorch, and JAX
* Set up GPU acceleration
* Benchmark performance
* Scale to production

**Quick backend selection:**

* **CPU only**: NumPy (default, no setup)
* **GPU available**: PyTorch (best GPU support)
* **Research/JIT**: JAX (compilation + autodiff)

Getting Started
---------------

**New to HoloVec?**

1. Start with :doc:`../getting-started/quickstart` for a quick introduction
2. Try the :doc:`../tutorials/index` for complete applications
3. Browse :doc:`../examples/index` for focused examples
4. Then dive into these user guides for deeper understanding

**Already familiar with basics?**

Jump directly to the guide you need:

* Building a classifier? → :doc:`choosing-models` + :doc:`encoding-data`
* Optimizing performance? → :doc:`backends`
* Working with specific data types? → :doc:`encoding-data`

Best Practices
--------------

**Development Workflow:**

1. **Prototype with defaults**: Start with FHRR model, NumPy backend
2. **Choose encoders**: Match encoders to your data types
3. **Validate**: Test with examples to ensure correctness
4. **Optimize**: Switch backends or tune parameters if needed

**Production Deployment:**

1. **Select production model**: Based on user guide recommendations
2. **Benchmark**: Test performance with your actual data
3. **Choose backend**: NumPy for CPU, PyTorch for GPU
4. **Document decisions**: Note why you chose specific models/encoders

See Also
--------

* :doc:`../tutorials/index` - Complete end-to-end tutorials
* :doc:`../api/index` - API reference documentation
* :doc:`../examples/index` - Code examples
* :doc:`../theory/vsa_models` - Theoretical foundations
