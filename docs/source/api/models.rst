Models
======

All VSA model implementations in HoloVec.

.. currentmodule:: holovec.models

Base Model
----------

All models inherit from the :class:`VSAModel` base class.

.. autoclass:: VSAModel
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

MAP Model
---------

Multiply-Add-Permute: Fast, self-inverse, hardware-friendly.

.. autoclass:: MAPModel
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

**Key Properties:**

* **Space**: Bipolar ({-1, +1})
* **Binding**: Element-wise multiplication
* **Unbinding**: Self-inverse (multiply again)
* **Best For**: Hardware implementations, speed, simplicity

FHRR Model
----------

Fourier Holographic Reduced Representations: Exact inverse, best capacity.

.. autoclass:: FHRRModel
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

**Key Properties:**

* **Space**: Complex (unit phasors)
* **Binding**: Element-wise multiplication
* **Unbinding**: Exact inverse (complex conjugate)
* **Best For**: High capacity, accurate retrieval

HRR Model
---------

Holographic Reduced Representations: Circular convolution, general purpose.

.. autoclass:: HRRModel
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

**Key Properties:**

* **Space**: Real (Gaussian)
* **Binding**: Circular convolution
* **Unbinding**: Circular correlation (approximate inverse)
* **Best For**: General purpose, proven track record

BSC Model
---------

Binary Spatter Codes: Memory-efficient, sparse binary.

.. autoclass:: BSCModel
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

**Key Properties:**

* **Space**: Binary ({0, 1})
* **Binding**: XOR
* **Unbinding**: Self-inverse (XOR again)
* **Best For**: Memory efficiency, binary operations

BSDC, GHRR, VTB Models
-----------------------

Additional specialized models:

.. autoclass:: BSDCModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GHRRModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: VTBModel
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

* :doc:`vsa` - Main VSA API
* :doc:`../user-guide/choosing-models` - Model selection guide
* :doc:`../theory/vsa_models` - Mathematical foundations
