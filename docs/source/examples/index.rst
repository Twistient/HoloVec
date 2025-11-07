:orphan:

================
HoloVec Examples
================

**Demonstrations and tutorials for hyperdimensional computing with HoloVec**

This directory contains comprehensive, well-documented examples covering everything from 5-minute quickstarts to advanced theoretical validation. All examples are designed to be intuitive, executable, and suitable for integration into tutorials and documentation.

----

Quick Start
===========

**New to HoloVec?** Start here with these foundational examples:

- ``00_quickstart.py`` - Get started in 5 minutes
- ``01_basic_operations.py`` - Understand core VSA operations
- ``02_models_comparison.py`` - Learn when to use each model

Learning Paths
==============

Path 1: Absolute Beginner
--------------------------

**Time: ~30 minutes**

1. ``00_quickstart.py`` - Get started in 5 minutes
2. ``01_basic_operations.py`` - Understand core VSA operations
3. ``02_models_comparison.py`` - Learn when to use each model
4. ``10_encoders_scalar.py`` - Encode continuous values
5. ``13_encoders_position_binding.py`` - Encode sequences

Path 2: Application Developer
------------------------------

**Time: ~1-2 hours**

1. Start with Path 1 basics
2. Choose your domain:

   - **Text/NLP**: ``14_encoders_ngram.py`` → ``20_app_text_classification.py``
   - **Images**: ``17_encoders_image.py`` → ``21_app_image_recognition.py``
   - **Sequences**: ``15_encoders_trajectory.py`` → ``22_app_gesture_recognition.py``

Path 3: Researcher / Advanced User
-----------------------------------

**Time: ~2-3 hours**

1. Complete Path 1 and 2
2. Explore advanced topics:

   - **Theory**: ``30_theory_fpe_validation.py``, ``32_distributed_representations.py``
   - **Performance**: ``31_performance_benchmarks.py``
   - **Memory**: ``27_cleanup_strategies.py``, ``28_factorization_methods.py``

----

Browse All Examples
===================

Below you'll find all available examples organized by category.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Installation, basic workflow, encoding, binding, retrieval Time: 5 minutes Prerequisites: None Related: 01_basic_operations.py, 02_models_comparison.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_00_quickstart_thumb.png
    :alt:

  :ref:`sphx_glr_examples_00_quickstart.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">HoloVec Quickstart Guide</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the core operations of VSA models: - Binding (association) - Unbinding (recovery) - Bundling (superposition) - Permutation (sequence encoding)">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_01_basic_operations_thumb.png
    :alt:

  :ref:`sphx_glr_examples_01_basic_operations.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Demo: Basic VSA Operations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: MAP, FHRR, HRR, BSC model selection and characteristics Time: 15 minutes Prerequisites: 00_quickstart.py, 01_basic_operations.py Related: 40_model_hrr_correlation.py, 41_model_ghrr_diagonality.py, 42_model_bsdc_seg.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_02_models_comparison_thumb.png
    :alt:

  :ref:`sphx_glr_examples_02_models_comparison.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">VSA Models Comparison Guide</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates: 1. FractionalPowerEncoder - Continuous scalar encoding with smooth similarity 2. ThermometerEncoder - Ordinal encoding with monotonic similarity 3. LevelEncoder - Discrete level encoding with exact recovery">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_10_encoders_scalar_thumb.png
    :alt:

  :ref:`sphx_glr_examples_10_encoders_scalar.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comprehensive demo of scalar encoders in holovec.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: FPE theory, bandwidth tuning, similarity profiles, decoding accuracy Time: 15 minutes Prerequisites: 10_encoders_scalar.py, 01_basic_operations.py Related: 12_encoders_thermometer_level.py, 30_theory_fpe_validation.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_11_encoders_fractional_power_thumb.png
    :alt:

  :ref:`sphx_glr_examples_11_encoders_fractional_power.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fractional Power Encoder Deep Dive</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Ordinal encoding, discrete bins, model compatibility, use cases Time: 10 minutes Prerequisites: 10_encoders_scalar.py, 01_basic_operations.py Related: 11_encoders_fractional_power.py, 02_models_comparison.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_12_encoders_thermometer_level_thumb.png
    :alt:

  :ref:`sphx_glr_examples_12_encoders_thermometer_level.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Thermometer and Level Encoders Deep Dive</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: PositionBindingEncoder, order sensitivity, sequence similarity Time: 15 minutes Prerequisites: 00_quickstart.py, 01_basic_operations.py Related: 14_encoders_ngram.py, 15_encoders_trajectory.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_13_encoders_position_binding_thumb.png
    :alt:

  :ref:`sphx_glr_examples_13_encoders_position_binding.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Position-Based Sequence Encoding</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This demo showcases the NGramEncoder, which captures local patterns in sequences using sliding windows (n-grams). This is particularly useful for:">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_14_encoders_ngram_thumb.png
    :alt:

  :ref:`sphx_glr_examples_14_encoders_ngram.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Demonstration of N-gram Encoder for local sequence pattern encoding.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This demo showcases the TrajectoryEncoder, which encodes continuous sequences like time series, paths, and motion trajectories into hypervectors. This is particularly useful for:">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_15_encoders_trajectory_thumb.png
    :alt:

  :ref:`sphx_glr_examples_15_encoders_trajectory.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Demonstration of Trajectory Encoder for continuous sequence encoding.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: VectorEncoder, scalar encoder composition, high-dimensional data Time: 15 minutes Prerequisites: 00_quickstart.py, 10_encoders_scalar.py Related: 17_encoders_image.py, 21_app_image_recognition.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_16_encoders_vector_thumb.png
    :alt:

  :ref:`sphx_glr_examples_16_encoders_vector.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multivariate Vector Encoding</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This demo showcases the ImageEncoder, which encodes 2D images (grayscale, RGB, RGBA) into hypervectors by binding spatial positions with pixel values. This is particularly useful for:">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_17_encoders_image_thumb.png
    :alt:

  :ref:`sphx_glr_examples_17_encoders_image.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Demonstration of Image Encoder for 2D spatial data encoding.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Graph structures, knowledge graphs, semantic triples, role-filler binding Time: 20 minutes Prerequisites: 00_quickstart.py, 01_basic_operations.py Related: 23_app_symbolic_reasoning.py, 24_app_working_memory.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_18_encoders_graph_thumb.png
    :alt:

  :ref:`sphx_glr_examples_18_encoders_graph.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compositional Graph Encoding</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Text classification, n-gram encoding, supervised learning, NLP Time: 15 minutes Prerequisites: 14_encoders_ngram.py, 26_retrieval_basics.py Related: 23_app_symbolic_reasoning.py, 25_app_integration_patterns.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_20_app_text_classification_thumb.png
    :alt:

  :ref:`sphx_glr_examples_20_app_text_classification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Document Classification with N-grams</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Image classification, spatial encoding, pattern matching, computer vision Time: 15 minutes Prerequisites: 17_encoders_image.py, 16_encoders_vector.py Related: 20_app_text_classification.py, 25_app_integration_patterns.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_21_app_image_recognition_thumb.png
    :alt:

  :ref:`sphx_glr_examples_21_app_image_recognition.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Image Pattern Recognition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Trajectory encoding, motion classification, time series, HCI Time: 15 minutes Prerequisites: 15_encoders_trajectory.py, 10_encoders_scalar.py Related: 20_app_text_classification.py, 21_app_image_recognition.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_22_app_gesture_recognition_thumb.png
    :alt:

  :ref:`sphx_glr_examples_22_app_gesture_recognition.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gesture Recognition from Motion Trajectories</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Role-filler binding, structured representations, reasoning, analogy Time: 20 minutes Prerequisites: 00_quickstart.py, 18_encoders_graph.py Related: 24_app_working_memory.py, 25_app_integration_patterns.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_23_app_symbolic_reasoning_thumb.png
    :alt:

  :ref:`sphx_glr_examples_23_app_symbolic_reasoning.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Symbolic Reasoning with Role-Filler Binding</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Working memory, cleanup, resonator, factorization, noisy retrieval Time: 20 minutes Prerequisites: 23_app_symbolic_reasoning.py, 26_retrieval_basics.py Related: 27_cleanup_strategies.py, 28_factorization_methods.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_24_app_working_memory_thumb.png
    :alt:

  :ref:`sphx_glr_examples_24_app_working_memory.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Working Memory with Cleanup Strategies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Multimodal fusion, encoder integration, design patterns, hierarchical encoding Time: 25 minutes Prerequisites: 10_encoders_scalar.py, 14_encoders_ngram.py, 23_app_symbolic_reasoning.py Related: 20-22_app_*.py (domain-specific applications)">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_25_app_integration_patterns_thumb.png
    :alt:

  :ref:`sphx_glr_examples_25_app_integration_patterns.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Integration Patterns and Multimodal Fusion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run (optional):   python -m examples.retrieval_demo">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_26_retrieval_basics_thumb.png
    :alt:

  :ref:`sphx_glr_examples_26_retrieval_basics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Retrieval demo using ItemStore + Codebook</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: BruteForce cleanup, Resonator cleanup, performance comparison Time: 15 minutes Prerequisites: 24_app_working_memory.py, 26_retrieval_basics.py Related: 28_factorization_methods.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_27_cleanup_strategies_thumb.png
    :alt:

  :ref:`sphx_glr_examples_27_cleanup_strategies.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cleanup Strategies Comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Factorization, multi-factor unbinding, composite structures, iterative cleanup Time: 15 minutes Prerequisites: 27_cleanup_strategies.py, 24_app_working_memory.py Related: 23_app_symbolic_reasoning.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_28_factorization_methods_thumb.png
    :alt:

  :ref:`sphx_glr_examples_28_factorization_methods.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-Factor Unbinding and Factorization Methods</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script empirically tests the key theoretical claims: 1. Inner product convergence to sinc kernel for uniform phase distribution 2. Dimensionality dependence of convergence 3. Bandwidth scaling effects 4. Similarity properties (symmetry, self-similarity)">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_30_theory_fpe_validation_thumb.png
    :alt:

  :ref:`sphx_glr_examples_30_theory_fpe_validation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Validate FractionalPowerEncoder against theoretical predictions from Frady et al. (2021).</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Speed comparison, accuracy testing, backend selection, model efficiency Time: 15 minutes Prerequisites: 02_models_comparison.py, 01_basic_operations.py Related: 32_distributed_representations.py, 02_models_comparison.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_31_performance_benchmarks_thumb.png
    :alt:

  :ref:`sphx_glr_examples_31_performance_benchmarks.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Performance Benchmarks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Bundling capacity, dimension effects, information limits, cleanup Time: 15 minutes Prerequisites: 01_basic_operations.py, 02_models_comparison.py Related: 31_performance_benchmarks.py, 27_cleanup_strategies.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_32_distributed_representations_thumb.png
    :alt:

  :ref:`sphx_glr_examples_32_distributed_representations.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Distributed Representations and Capacity Analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Topics: Noise tolerance, error propagation, fault recovery, graceful degradation Time: 15 minutes Prerequisites: 01_basic_operations.py, 27_cleanup_strategies.py Related: 32_distributed_representations.py, 31_performance_benchmarks.py">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_33_error_handling_robustness_thumb.png
    :alt:

  :ref:`sphx_glr_examples_33_error_handling_robustness.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Error Handling and Robustness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shows how circular correlation approximates unbinding for HRR.">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_40_model_hrr_correlation_thumb.png
    :alt:

  :ref:`sphx_glr_examples_40_model_hrr_correlation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">HRR correlation vs convolution demo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run (optional):   python -m examples.ghrr_diagonality_sweep">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_41_model_ghrr_diagonality_thumb.png
    :alt:

  :ref:`sphx_glr_examples_41_model_ghrr_diagonality.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GHRR usage example: diagonality/m sweeps and non-commutativity trends</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run (optional):   python -m examples.bsdc_seg_demo">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_42_model_bsdc_seg_thumb.png
    :alt:

  :ref:`sphx_glr_examples_42_model_bsdc_seg.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BSDC-SEG demo: segment-sparse codes, bundling, and segment-wise search</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/00_quickstart
   /examples/01_basic_operations
   /examples/02_models_comparison
   /examples/10_encoders_scalar
   /examples/11_encoders_fractional_power
   /examples/12_encoders_thermometer_level
   /examples/13_encoders_position_binding
   /examples/14_encoders_ngram
   /examples/15_encoders_trajectory
   /examples/16_encoders_vector
   /examples/17_encoders_image
   /examples/18_encoders_graph
   /examples/20_app_text_classification
   /examples/21_app_image_recognition
   /examples/22_app_gesture_recognition
   /examples/23_app_symbolic_reasoning
   /examples/24_app_working_memory
   /examples/25_app_integration_patterns
   /examples/26_retrieval_basics
   /examples/27_cleanup_strategies
   /examples/28_factorization_methods
   /examples/30_theory_fpe_validation
   /examples/31_performance_benchmarks
   /examples/32_distributed_representations
   /examples/33_error_handling_robustness
   /examples/40_model_hrr_correlation
   /examples/41_model_ghrr_diagonality
   /examples/42_model_bsdc_seg



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
