API Reference
=============

Complete API documentation for HoloVec.

.. contents:: Contents
   :local:
   :depth: 2

Main API
--------

The primary interface for creating and using VSA models.

.. toctree::
   :maxdepth: 2

   vsa

Models
------

All VSA model implementations.

.. toctree::
   :maxdepth: 2

   models

Encoders
--------

Encoders for different data types.

.. toctree::
   :maxdepth: 2

   encoders

Retrieval & Memory
------------------

Cleanup, retrieval, and associative storage.

.. toctree::
   :maxdepth: 2

   retrieval

Backends
--------

Backend implementations for different hardware.

.. toctree::
   :maxdepth: 2

   backends

Vector Spaces
-------------

Vector space definitions and operations.

.. toctree::
   :maxdepth: 2

   spaces

Utilities
---------

Helper functions and utilities.

.. toctree::
   :maxdepth: 2

   utils

Quick Links
-----------

**Most Common:**

* :class:`~holovec.VSA` - Factory for creating models
* :meth:`~holovec.VSA.create` - Create a VSA model
* :class:`~holovec.models.VSAModel` - Base class for all models
* :class:`~holovec.encoders.FractionalPowerEncoder` - Continuous value encoding
* :class:`~holovec.retrieval.ItemStore` - Associative memory

**Models:**

* :class:`~holovec.models.MAPModel` - Multiply-Add-Permute
* :class:`~holovec.models.FHRRModel` - Fourier HRR
* :class:`~holovec.models.HRRModel` - Holographic Reduced Representations
* :class:`~holovec.models.BSCModel` - Binary Spatter Codes
* :class:`~holovec.models.BSDCModel` - Block-Structured Distributed Codes
* :class:`~holovec.models.GHRRModel` - Generalized HRR
* :class:`~holovec.models.VTBModel` - Vector-derived Transformation Binding

**Encoders:**

* :class:`~holovec.encoders.FractionalPowerEncoder` - Smooth scalar encoding
* :class:`~holovec.encoders.ThermometerEncoder` - Ordinal encoding
* :class:`~holovec.encoders.LevelEncoder` - Discrete bins
* :class:`~holovec.encoders.PositionBindingEncoder` - Sequence encoding
* :class:`~holovec.encoders.NGramEncoder` - N-gram text encoding
* :class:`~holovec.encoders.TrajectoryEncoder` - Motion path encoding
* :class:`~holovec.encoders.ImageEncoder` - 2D spatial encoding
* :class:`~holovec.encoders.VectorEncoder` - Multivariate encoding

**Backends:**

* :func:`~holovec.backends.get_backend` - Get a backend by name
* :class:`~holovec.backends.Backend` - Backend interface
* :class:`~holovec.backends.NumPyBackend` - NumPy (CPU) backend
* :class:`~holovec.backends.TorchBackend` - PyTorch (GPU) backend
* :class:`~holovec.backends.JAXBackend` - JAX (JIT/TPU) backend

**Retrieval:**

* :class:`~holovec.retrieval.Codebook` - Simple lookup table
* :class:`~holovec.retrieval.ItemStore` - Similarity-based retrieval
* :class:`~holovec.retrieval.AssocStore` - Associative memory
* :class:`~holovec.utils.cleanup.BruteForceCleanup` - Nearest-neighbor cleanup
* :class:`~holovec.utils.cleanup.ResonatorCleanup` - Iterative cleanup

See Also
--------

* :doc:`../user-guide/index` - User guide and tutorials
* :doc:`../examples/index` - Example gallery
* :doc:`../theory/index` - Theoretical foundations
