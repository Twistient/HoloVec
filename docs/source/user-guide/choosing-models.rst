Choosing a VSA Model
====================

HoloVec provides 7 different VSA models. This guide helps you choose the right model for your use case.

Quick Selection Guide
---------------------

**Quick Recommendations:**

* **Most versatile**: :class:`~holovec.models.FHRRModel` - Works everywhere, smooth encodings
* **Fastest**: :class:`~holovec.models.MAPModel` - Binary, ultra-fast, lowest memory
* **Best for complex reasoning**: :class:`~holovec.models.FHRRModel` or :class:`~holovec.models.HRRModel`
* **Best for sparse data**: :class:`~holovec.models.BSCModel` or :class:`~holovec.models.BSDCModel`
* **Best for hardware**: :class:`~holovec.models.MAPModel` or :class:`~holovec.models.BSCModel` (binary operations)

Decision Tree
-------------

.. code-block:: text

    Do you need exact similarity reversal (unbinding)?
    ├─ YES → Consider FHRR, HRR, GHRR, VTB
    │   │
    │   ├─ Need smooth encodings (continuous data)?
    │   │   └─ YES → FHRR (best choice)
    │   │
    │   ├─ Working with symbolic data only?
    │   │   └─ YES → HRR (classic, well-studied)
    │   │
    │   └─ Need advanced features?
    │       ├─ Exact encoding/decoding → GHRR
    │       └─ Variable binding strength → VTB
    │
    └─ NO → Consider MAP, BSC, BSDC
        │
        ├─ Need maximum speed?
        │   └─ YES → MAP (binary, ultra-fast)
        │
        ├─ Working with sparse data?
        │   └─ YES → BSC or BSDC
        │
        └─ Need hardware efficiency?
            └─ YES → MAP or BSC (binary operations)

Model Comparison Table
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 12 12 12 12 12 12 13

   * - Feature
     - MAP
     - FHRR
     - HRR
     - BSC
     - BSDC
     - GHRR
     - VTB
   * - **Data Type**
     - Binary
     - Complex
     - Real
     - Binary
     - Binary
     - Complex
     - Real
   * - **Unbinding**
     - Approx
     - Exact
     - Exact
     - Approx
     - Approx
     - Exact
     - Exact
   * - **Speed**
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐
     - ⭐⭐
   * - **Memory**
     - Lowest
     - Medium
     - Medium
     - Low
     - Low
     - Medium
     - Medium
   * - **Hardware**
     - Best
     - Good
     - Good
     - Best
     - Best
     - Good
     - Good
   * - **FPE Support**
     - No
     - Yes
     - Yes
     - No
     - No
     - Yes
     - Yes
   * - **Best For**
     - Speed
     - General
     - Classic
     - Sparse
     - Sparse
     - Advanced
     - Research

Detailed Model Descriptions
----------------------------

MAP (Multiply-Add-Permute)
^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.models.MAPModel`

**Type**: Binary hypervectors

**Key Properties:**

* Ultra-fast operations (binary XOR, shifts)
* Lowest memory footprint (1 bit per dimension)
* Hardware-friendly (FPGA, ASIC implementations)
* Approximate unbinding

**Best For:**

* Real-time applications
* Edge devices with limited resources
* Maximum speed requirements
* Hardware implementations

**Limitations:**

* Cannot use :class:`~holovec.encoders.FractionalPowerEncoder`
* Less precise similarity measurements
* Approximate rather than exact unbinding

**Example Use Cases:**

* Gesture recognition on wearables
* Real-time sensor fusion
* Embedded systems
* Mobile applications

**See Also:**

* :doc:`../examples/01_basic_operations` - MAP examples
* :doc:`../api/models` - Full API reference

FHRR (Fourier Holographic Reduced Representations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.models.FHRRModel`

**Type**: Complex-valued hypervectors

**Key Properties:**

* Exact unbinding via complex conjugate
* Smooth similarity for continuous values
* Supports FractionalPowerEncoder
* Well-suited for numerical data

**Best For:**

* Continuous data (temperature, time, coordinates)
* Applications requiring exact reversal
* General-purpose VSA tasks
* Symbolic + numeric fusion

**Limitations:**

* Higher memory than binary models
* Complex arithmetic overhead
* May need normalization

**Example Use Cases:**

* Time series analysis
* Sensor data processing
* Multimodal data fusion
* Scientific computing

**See Also:**

* :doc:`../examples/11_encoders_fractional_power` - FPE with FHRR
* :doc:`../examples/22_app_gesture_recognition` - Trajectory encoding
* :doc:`../api/models` - Full API reference

HRR (Holographic Reduced Representations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.models.HRRModel`

**Type**: Real-valued hypervectors

**Key Properties:**

* Classic model (well-studied since 1990s)
* Circular convolution for binding
* Exact unbinding via correlation
* Supports FractionalPowerEncoder

**Best For:**

* Classic VSA applications
* Research with established baselines
* Symbolic reasoning
* Applications with existing HRR code

**Limitations:**

* Slower than FHRR for some operations
* Higher memory than binary models

**Example Use Cases:**

* Natural language processing
* Knowledge representation
* Analogical reasoning
* Cognitive modeling

**See Also:**

* :doc:`../examples/03_binding_and_unbinding` - Binding examples
* :doc:`../api/models` - Full API reference

BSC (Binary Spatter Codes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.models.BSCModel`

**Type**: Binary hypervectors

**Key Properties:**

* Very sparse representations
* Fast binary operations
* Low memory footprint
* Hardware-friendly

**Best For:**

* Sparse data
* Fast similarity search
* Large-scale storage
* Hardware implementations

**Limitations:**

* Cannot use FractionalPowerEncoder
* Approximate unbinding
* Requires careful capacity management

**Example Use Cases:**

* Document retrieval
* Large vocabulary applications
* Recommendation systems
* Content-based search

**See Also:**

* :doc:`../api/models` - Full API reference

BSDC (Binary Spatter Codes with Discrete Cleanup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.models.BSDCModel`

**Type**: Binary hypervectors with cleanup

**Key Properties:**

* Enhanced BSC with cleanup mechanism
* Better error correction
* Improved similarity discrimination

**Best For:**

* BSC applications requiring better accuracy
* Noisy environments
* Complex retrieval tasks

**See Also:**

* :doc:`../api/models` - Full API reference

GHRR (Generalized Holographic Reduced Representations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.models.GHRRModel`

**Type**: Complex-valued with advanced features

**Key Properties:**

* Generalization of HRR/FHRR
* Flexible encoding/decoding
* Research-oriented features

**Best For:**

* Advanced VSA research
* Custom encoding schemes
* Specialized applications

**See Also:**

* :doc:`../api/models` - Full API reference

VTB (Variable Binding)
^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.models.VTBModel`

**Type**: Real-valued with variable binding strength

**Key Properties:**

* Tunable binding strength
* Research model for variable binding
* Flexible unbinding

**Best For:**

* Research on binding mechanisms
* Applications requiring variable binding strength

**See Also:**

* :doc:`../api/models` - Full API reference

Use Case Recommendations
------------------------

By Application Domain
^^^^^^^^^^^^^^^^^^^^^

**Text Classification / NLP**

* **Primary**: :class:`~holovec.models.FHRRModel` or :class:`~holovec.models.HRRModel`
* **Alternative**: :class:`~holovec.models.MAPModel` for speed
* **Reason**: Symbolic data with optional continuous features (word embeddings)
* **Example**: :doc:`../examples/20_app_text_classification`

**Image Recognition**

* **Primary**: :class:`~holovec.models.FHRRModel`
* **Alternative**: :class:`~holovec.models.MAPModel` for edge devices
* **Reason**: Spatial encoding benefits from smooth similarity
* **Example**: :doc:`../examples/21_app_image_recognition`

**Gesture Recognition / Motion**

* **Primary**: :class:`~holovec.models.FHRRModel`
* **Alternative**: :class:`~holovec.models.MAPModel` for wearables
* **Reason**: Continuous trajectories, real-time requirements
* **Example**: :doc:`../examples/22_app_gesture_recognition`

**Time Series / Sensor Fusion**

* **Primary**: :class:`~holovec.models.FHRRModel`
* **Reason**: Continuous temporal data, exact unbinding helpful
* **Example**: :doc:`../examples/15_encoders_trajectory`

**Symbolic Reasoning**

* **Primary**: :class:`~holovec.models.HRRModel` or :class:`~holovec.models.FHRRModel`
* **Reason**: Classic applications, exact unbinding essential
* **Example**: :doc:`../examples/16_compositional_structures`

**Large-Scale Retrieval**

* **Primary**: :class:`~holovec.models.BSCModel` or :class:`~holovec.models.MAPModel`
* **Reason**: Fast similarity search, low memory
* **Example**: :doc:`../examples/05_similarity_and_distance`

By Hardware Platform
^^^^^^^^^^^^^^^^^^^^

**Desktop / Server (No Constraints)**

* **Recommended**: :class:`~holovec.models.FHRRModel`
* **Reason**: Best general-purpose capabilities

**Mobile / Edge Devices**

* **Recommended**: :class:`~holovec.models.MAPModel`
* **Reason**: Lowest memory and compute requirements

**FPGA / ASIC Implementation**

* **Recommended**: :class:`~holovec.models.MAPModel` or :class:`~holovec.models.BSCModel`
* **Reason**: Binary operations map directly to hardware

**GPU Acceleration**

* **Recommended**: :class:`~holovec.models.FHRRModel` with PyTorch backend
* **Reason**: Complex operations parallelize well
* **See**: :doc:`backends`

By Performance Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Maximum Speed**

1. :class:`~holovec.models.MAPModel` - Fastest
2. :class:`~holovec.models.BSCModel` - Very fast
3. :class:`~holovec.models.FHRRModel` - Fast enough for most applications

**Minimum Memory**

1. :class:`~holovec.models.MAPModel` - 1 bit/dimension
2. :class:`~holovec.models.BSCModel` - 1 bit/dimension
3. :class:`~holovec.models.FHRRModel` - 16 bytes/dimension (complex float64)

**Best Accuracy**

1. :class:`~holovec.models.FHRRModel` - Smooth similarity, exact unbinding
2. :class:`~holovec.models.HRRModel` - Exact unbinding, well-studied
3. :class:`~holovec.models.BSDCModel` - Enhanced cleanup

Performance Trade-offs
-----------------------

Speed vs Accuracy
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Model
     - Speed Characteristics
     - Accuracy Characteristics
   * - MAP
     - Ultra-fast (binary XOR)
     - Approximate (discrete similarity)
   * - BSC/BSDC
     - Very fast (binary ops)
     - Good (sparse codes)
   * - FHRR
     - Fast (optimized FFT)
     - Excellent (smooth similarity)
   * - HRR
     - Moderate (convolution)
     - Excellent (exact unbinding)
   * - GHRR/VTB
     - Moderate to slow
     - Research models

Memory vs Capacity
^^^^^^^^^^^^^^^^^^

**Memory per dimension:**

* Binary (MAP, BSC, BSDC): 1 bit
* Real (HRR, VTB): 8 bytes (float64)
* Complex (FHRR, GHRR): 16 bytes (complex128)

**Capacity (number of distinct items):**

* All models: ~dimension / 100 for good orthogonality
* Example: 10,000 dimensions → ~100 items

**Trade-off:**

* Higher dimensions → More capacity but more memory
* Binary models → Less memory per dimension
* Choose dimension based on application needs

Switching Models
----------------

Changing models is easy - just modify the ``create()`` call:

.. code-block:: python

    from holovec import VSA

    # Try different models with same code
    model = VSA.create('FHRR', dim=10000)  # Complex-valued
    # model = VSA.create('MAP', dim=10000)   # Binary
    # model = VSA.create('HRR', dim=10000)   # Real-valued

    # Rest of code stays the same
    a = model.random(seed=1)
    b = model.random(seed=2)
    c = model.bind(a, b)

**Important**: Some encoders only work with specific models:

* :class:`~holovec.encoders.FractionalPowerEncoder` requires FHRR, HRR, GHRR, or VTB
* :class:`~holovec.encoders.ThermometerEncoder` and :class:`~holovec.encoders.LevelEncoder` work with all models

See :doc:`encoding-data` for encoder selection guidance.

Frequently Asked Questions
--------------------------

**Q: Which model should I start with?**

A: :class:`~holovec.models.FHRRModel` - It's the most versatile and works for most applications.

**Q: When should I use MAP instead of FHRR?**

A: When speed and memory are critical (embedded systems, real-time applications, hardware implementations).

**Q: Can I mix models in one application?**

A: No - hypervectors from different models are incompatible. Choose one model for your entire application.

**Q: How do I know if my model has enough capacity?**

A: Rule of thumb: ``dimension / 100`` items. Use :doc:`../examples/06_cleanup_and_retrieval` to test retrieval accuracy.

**Q: Does model choice affect encoding?**

A: Yes - some encoders (like FractionalPowerEncoder) only work with certain models. See :doc:`encoding-data`.

**Q: Can I change models after development?**

A: Switching is usually straightforward, but you'll need to re-encode all data. Test thoroughly.

**Q: Which model has the best theoretical foundations?**

A: :class:`~holovec.models.HRRModel` (classic, well-studied) and :class:`~holovec.models.FHRRModel` (modern, mathematically elegant).

Next Steps
----------

* :doc:`encoding-data` - Learn about encoding different data types
* :doc:`backends` - Choose the right backend for performance
* :doc:`../examples/index` - See models in action
* :doc:`../api/models` - Complete model API reference
* :doc:`../theory/vsa_models` - Theoretical foundations

See Also
--------

* :doc:`../api/vsa` - VSA factory class
* :doc:`../api/models` - Model implementations
* :doc:`../examples/02_models_comparison` - Hands-on comparison
