Theory & Foundations
====================

Theoretical foundations of hyperdimensional computing and vector symbolic architectures.

This section provides mathematical foundations, algorithm descriptions, and theoretical analysis of HDC/VSA systems.

Contents
--------

.. toctree::
   :maxdepth: 2

   vsa_models
   encoders
   foundations
   models-deep-dive
   capacity-analysis

VSA Models
----------

:doc:`vsa_models` provides comprehensive mathematical descriptions of all 7 VSA models implemented in HoloVec:

* **MAP** (Multiply-Add-Permute)
* **FHRR** (Fourier Holographic Reduced Representations)
* **HRR** (Holographic Reduced Representations)
* **BSC** (Binary Spatter Codes)
* **BSDC** (Block-Structured Distributed Codes)
* **GHRR** (Generalized HRR)
* **VTB** (Vector-derived Transformation Binding)

Each model includes:

* Mathematical definition
* Binding and unbinding operations
* Bundling semantics
* Theoretical properties
* Computational complexity

Encoders
--------

:doc:`encoders` covers the theory and algorithms for all encoder types:

**Scalar Encoders:**

* **Fractional Power Encoder (FPE)** - Smooth similarity via complex exponentials
* **Thermometer Encoder** - Ordinal encoding with monotonic similarity
* **Level Encoder** - Discrete bin encoding

**Sequence Encoders:**

* **Position Binding** - Order-sensitive sequence encoding
* **N-gram** - Overlapping subsequence patterns
* **Trajectory** - Continuous motion path encoding

**Spatial Encoders:**

* **Image Encoder** - 2D spatial data encoding
* **Vector Encoder** - Multivariate feature vectors

Each encoder includes:

* Algorithm description
* Mathematical formulation
* Similarity properties
* Reversibility analysis
* Use case guidance

Additional Topics
-----------------

:doc:`foundations`
   Core concepts and mathematical foundations of HDC/VSA

:doc:`models-deep-dive`
   Detailed analysis of model properties and trade-offs

:doc:`capacity-analysis`
   Information capacity, bundling limits, and dimensionality analysis

See Also
--------

* :doc:`../validation/index` - Empirical validation and benchmarks
* :doc:`../api/models` - Model API reference
* :doc:`../api/encoders` - Encoder API reference
