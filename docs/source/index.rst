.. This is a hidden title to maintain proper heading hierarchy
.. rst-class:: hidden-title

HoloVec Documentation
=====================

.. raw:: html

   <div style="max-width: 1200px; margin: 0 auto;">
      <div style="display: flex; align-items: center; gap: 2rem; margin: 3rem 0;">
         <div style="flex: 1;">
            <h1 style="font-size: clamp(2.5rem, 5vw, 3.75rem); font-weight: 800; line-height: 1.1; margin: 0 0 1rem 0; letter-spacing: -0.04em;">
               Brain-Inspired Computing with <span style="background: linear-gradient(135deg, #00d9ff 0%, #0066ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Hyperdimensional Vectors</span>
            </h1>
            <p style="font-size: clamp(1.125rem, 2vw, 1.25rem); line-height: 1.6; color: rgba(255, 255, 255, 0.8); margin: 0 0 2rem 0;">
               A high-performance Python library for building cognitive AI systems using vector symbolic architectures. Encode any data type into high-dimensional space for fast, robust, and explainable machine learning.
            </p>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
               <a href="getting-started/quickstart.html" style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.875rem 1.75rem; background: linear-gradient(135deg, #00d9ff 0%, #0066ff 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3); transition: all 0.2s;">
                  <span>Get Started</span>
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                     <path d="M7.5 5L12.5 10L7.5 15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
               </a>
               <a href="examples/index.html" style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.875rem 1.75rem; background: rgba(255, 255, 255, 0.1); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; border: 1px solid rgba(255, 255, 255, 0.2); transition: all 0.2s;">
                  Browse Examples
               </a>
            </div>
         </div>
         <div style="flex: 0 0 300px; display: flex; justify-content: center; align-items: center;">
            <svg width="300" height="300" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
               <!-- Abstract geometric pattern -->
               <defs>
                  <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                     <stop offset="0%" style="stop-color:#00d9ff;stop-opacity:0.8" />
                     <stop offset="100%" style="stop-color:#0066ff;stop-opacity:0.8" />
                  </linearGradient>
                  <filter id="glow">
                     <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                     <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                     </feMerge>
                  </filter>
               </defs>
               <!-- Hyperdimensional space visualization -->
               <circle cx="150" cy="150" r="120" fill="none" stroke="url(#grad1)" stroke-width="2" opacity="0.3"/>
               <circle cx="150" cy="150" r="90" fill="none" stroke="url(#grad1)" stroke-width="2" opacity="0.4"/>
               <circle cx="150" cy="150" r="60" fill="none" stroke="url(#grad1)" stroke-width="2" opacity="0.5"/>
               <!-- Vector lines -->
               <line x1="150" y1="150" x2="230" y2="100" stroke="url(#grad1)" stroke-width="3" filter="url(#glow)"/>
               <line x1="150" y1="150" x2="70" y2="80" stroke="url(#grad1)" stroke-width="3" filter="url(#glow)"/>
               <line x1="150" y1="150" x2="200" y2="220" stroke="url(#grad1)" stroke-width="3" filter="url(#glow)"/>
               <line x1="150" y1="150" x2="90" y2="200" stroke="url(#grad1)" stroke-width="3" filter="url(#glow)"/>
               <!-- Dots at vector endpoints -->
               <circle cx="230" cy="100" r="6" fill="#00d9ff" filter="url(#glow)"/>
               <circle cx="70" cy="80" r="6" fill="#00d9ff" filter="url(#glow)"/>
               <circle cx="200" cy="220" r="6" fill="#0066ff" filter="url(#glow)"/>
               <circle cx="90" cy="200" r="6" fill="#0066ff" filter="url(#glow)"/>
               <!-- Center point -->
               <circle cx="150" cy="150" r="8" fill="#00d9ff" filter="url(#glow)"/>
            </svg>
         </div>
      </div>
   </div>

----

Why HoloVec?
------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card::
        :class-header: bg-light
        :class-body: docutils
        :class-card: sd-shadow-sm

        🧠 **Brain-Inspired**
        ^^^

        Hyperdimensional computing mimics how the brain represents and processes information using high-dimensional distributed representations.

        **Bio-plausible operations** that map naturally to neural circuits.

    .. grid-item-card::
        :class-header: bg-light
        :class-body: docutils
        :class-card: sd-shadow-sm

        ⚡ **Fast & Efficient**
        ^^^

        Vectorized operations on 10,000-dimensional vectors are surprisingly efficient. **No neural network training required.**

        Process data in real-time with minimal computational overhead.

    .. grid-item-card::
        :class-header: bg-light
        :class-body: docutils
        :class-card: sd-shadow-sm

        🛡️ **Robust to Noise**
        ^^^

        Gracefully handles noise, partial corruption, and approximate data. **Ideal for edge devices** and fault-tolerant systems.

        Performance degrades gracefully under hardware errors.

    .. grid-item-card::
        :class-header: bg-light
        :class-body: docutils
        :class-card: sd-shadow-sm

        🔍 **Explainable**
        ^^^

        Operations are transparent and interpretable. You can **visualize and understand** what the model is doing.

        No black box. Every operation has clear semantics.

    .. grid-item-card::
        :class-header: bg-light
        :class-body: docutils
        :class-card: sd-shadow-sm

        🎓 **One-Shot Learning**
        ^^^

        Learn from very few examples. **No gradient descent required.** Perfect for online learning and rapid adaptation.

        Add new knowledge incrementally without retraining.

    .. grid-item-card::
        :class-header: bg-light
        :class-body: docutils
        :class-card: sd-shadow-sm

        📦 **Versatile**
        ^^^

        Works with **any data type** - text, images, sensors, symbolic structures. Perfect for **multimodal fusion**.

        Unify different modalities in the same representational space.

----

What You Can Build
------------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card::
        :class-card: sd-shadow-sm

        **📝 Text & NLP**

        Sentiment analysis, document classification, chatbot intent recognition, semantic search

    .. grid-item-card::
        :class-card: sd-shadow-sm

        **🖼️ Computer Vision**

        Real-time image recognition, gesture detection, visual similarity search, anomaly detection

    .. grid-item-card::
        :class-card: sd-shadow-sm

        **🎯 Recommender Systems**

        Personalized recommendations, content filtering, collaborative filtering with explainable results

    .. grid-item-card::
        :class-card: sd-shadow-sm

        **📊 Sensor & IoT**

        Time-series classification, sensor fusion, predictive maintenance, edge device inference

    .. grid-item-card::
        :class-card: sd-shadow-sm

        **🔗 Multimodal AI**

        Text + image fusion, audio-visual processing, cross-modal retrieval and reasoning

    .. grid-item-card::
        :class-card: sd-shadow-sm

        **🧪 Research & Prototyping**

        Fast experimentation, one-shot learning tasks, cognitive modeling, neurosymbolic AI

----

HoloVec Features
----------------

A comprehensive library for hyperdimensional computing with everything you need from research to production.

.. raw:: html

   <details class="sd-dropdown feature-dropdown-title" open>
   <summary class="sd-summary-title sd-font-weight-bold">
      <svg class="octicon lucide-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10 22V7a1 1 0 0 0-1-1H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5a1 1 0 0 0-1-1H2"/><rect x="14" y="2" width="8" height="8" rx="1"/></svg>
      <div class="dropdown-title-content">
         <div class="dropdown-headline">Multiple VSA Models</div>
         <div class="dropdown-subheadline">Choose different mathematical frameworks for representing and manipulating information in high-dimensional space.</div>
      </div>
      <svg class="dropdown-chevron" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 8 10 12 14 8"></polyline></svg>
   </summary>
   <div class="sd-summary-content">

**Available Models:**

- **MAP** (Multiply-Add-Permute) — Binary vectors, XOR operations, fast and simple
- **FHRR** (Fourier Holographic Reduced Representations) — Complex-valued, perfect for structured data
- **HRR** (Holographic Reduced Representations) — Real-valued circular convolution, original VSA model
- **BSC** (Binary Spatter Codes) — Sparse binary vectors for extreme efficiency
- **BSDC** (Binary Sparse Distributed Codes) — Block-based sparse representations
- **GHRR** (Generalized HRR) — Extended HRR with flexible algebras
- **VTB** (Vector-derived Transformation Binding) — Learned binding transformations

**Why it matters:** Each model has different trade-offs in accuracy, speed, and memory. Choose based on your application needs.

.. raw:: html

   </div></details>

.. raw:: html

   <details class="sd-dropdown feature-dropdown-title">
   <summary class="sd-summary-title sd-font-weight-bold">
      <svg class="octicon lucide-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m19 5 3-3"/><path d="m2 22 3-3"/><path d="M6.3 20.3a2.4 2.4 0 0 0 3.4 0L12 18l-6-6-2.3 2.3a2.4 2.4 0 0 0 0 3.4Z"/><path d="M7.5 13.5 10 11"/><path d="M10.5 16.5 13 14"/><path d="m12 6 6 6 2.3-2.3a2.4 2.4 0 0 0 0-3.4l-2.6-2.6a2.4 2.4 0 0 0-3.4 0Z"/></svg>
      <div class="dropdown-title-content">
         <div class="dropdown-headline">Flexible Backends</div>
         <div class="dropdown-subheadline">Run the same code on different hardware with zero code changes.</div>
      </div>
      <svg class="dropdown-chevron" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 8 10 12 14 8"></polyline></svg>
   </summary>
   <div class="sd-summary-content">

**Available Backends:**

- **NumPy (CPU)** — Default backend, no dependencies, perfect for prototyping and small-scale applications
- **PyTorch (GPU)** — CUDA acceleration for large-scale processing, batch operations, gradient computation
- **JAX (JIT/TPU)** — Just-in-time compilation for extreme performance, TPU support for massive scale

**Why it matters:** Start on your laptop, scale to GPU clusters without rewriting code. Switch backends with a single parameter.

.. raw:: html

   </div></details>

.. raw:: html

   <details class="sd-dropdown feature-dropdown-title">
   <summary class="sd-summary-title sd-font-weight-bold">
      <svg class="octicon lucide-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="1"/><path d="M18.944 12.33a1 1 0 0 0 0-.66 7.5 7.5 0 0 0-13.888 0 1 1 0 0 0 0 .66 7.5 7.5 0 0 0 13.888 0"/></svg>
      <div class="dropdown-title-content">
         <div class="dropdown-headline">Comprehensive Encoders</div>
         <div class="dropdown-subheadline">Convert raw data (numbers, text, images, graphs) into high-dimensional vectors that preserve semantic relationships.</div>
      </div>
      <svg class="dropdown-chevron" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 8 10 12 14 8"></polyline></svg>
   </summary>
   <div class="sd-summary-content">

**Available Encoders:**

- **Scalar Encoders:**
    - Fractional Power Encoding (FPE) — Smooth encoding of continuous values
    - Thermometer/Level Encoding — Threshold-based encoding
    - Position Binding — Positional information in sequences

- **Sequence Encoders:**
    - N-gram Encoding — Text and sequential patterns
    - Trajectory Encoding — Temporal sequences and paths

- **Structured Encoders:**
    - Image Encoding — Spatial patterns and visual data
    - Graph Encoding — Relational structures and networks
    - Vector Encoding — Pre-computed embeddings from other models

**Why it matters:** The right encoder preserves the structure of your data. Similar inputs produce similar vectors, enabling semantic operations.

.. raw:: html

   </div></details>

.. raw:: html

   <details class="sd-dropdown feature-dropdown-title">
   <summary class="sd-summary-title sd-font-weight-bold">
      <svg class="octicon lucide-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m10.852 14.772-.383.923"/><path d="m10.852 9.228-.383-.923"/><path d="m13.148 14.772.382.924"/><path d="m13.531 8.305-.383.923"/><path d="m14.772 10.852.923-.383"/><path d="m14.772 13.148.923.383"/><path d="M17.598 6.5A3 3 0 1 0 12 5a3 3 0 0 0-5.63-1.446 3 3 0 0 0-.368 1.571 4 4 0 0 0-2.525 5.771"/><path d="M17.998 5.125a4 4 0 0 1 2.525 5.771"/><path d="M19.505 10.294a4 4 0 0 1-1.5 7.706"/><path d="M4.032 17.483A4 4 0 0 0 11.464 20c.18-.311.892-.311 1.072 0a4 4 0 0 0 7.432-2.516"/><path d="M4.5 10.291A4 4 0 0 0 6 18"/><path d="M6.002 5.125a3 3 0 0 0 .4 1.375"/><path d="m9.228 10.852-.923-.383"/><path d="m9.228 13.148-.923.383"/><circle cx="12" cy="12" r="3"/></svg>
      <div class="dropdown-title-content">
         <div class="dropdown-headline">Memory & Retrieval</div>
         <div class="dropdown-subheadline">Build associative memories that work like human memory—noisy, robust, content-addressable.</div>
      </div>
      <svg class="dropdown-chevron" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 8 10 12 14 8"></polyline></svg>
   </summary>
   <div class="sd-summary-content">

**Available Systems:**

- **Cleanup Memory** — Remove noise and reconstruct clean representations
- **Item Memory** — Store and retrieve individual hypervectors
- **Associative Storage** — Create key-value associations, bind concepts together
- **Codebook** — Maintain vocabularies of known symbols and patterns
- **Resonator Networks** — Iteratively refine and factorize complex representations

**Why it matters:** Build systems that can store knowledge, answer queries, and reason with partial or noisy information—just like human cognition.

.. raw:: html

   </div></details>

.. raw:: html

   <details class="sd-dropdown feature-dropdown-title">
   <summary class="sd-summary-title sd-font-weight-bold">
      <svg class="octicon lucide-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m10.852 14.772-.383.923"/><path d="M13.148 14.772a3 3 0 1 0-2.296-5.544l-.383-.923"/><path d="m13.148 9.228.383-.923"/><path d="m13.53 15.696-.382-.924a3 3 0 1 1-2.296-5.544"/><path d="m14.772 10.852.923-.383"/><path d="m14.772 13.148.923.383"/><path d="M4.5 10H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2h-.5"/><path d="M4.5 14H4a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-4a2 2 0 0 0-2-2h-.5"/><path d="M6 18h.01"/><path d="M6 6h.01"/><path d="m9.228 10.852-.923-.383"/><path d="m9.228 13.148-.923.383"/></svg>
      <div class="dropdown-title-content">
         <div class="dropdown-headline">Production-Ready Architecture</div>
         <div class="dropdown-subheadline">Professional software engineering practices, not research code.</div>
      </div>
      <svg class="dropdown-chevron" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 8 10 12 14 8"></polyline></svg>
   </summary>
   <div class="sd-summary-content">

**Features:**

- **Type-Safe APIs** — Full type hints, IDE autocomplete, catch errors before runtime
- **Comprehensive Testing** — 95%+ code coverage, validated against theoretical properties
- **Complete Documentation** — API reference, tutorials, 28+ examples, theory guides
- **Extensible Design** — Add custom models, encoders, and backends
- **Performance Optimized** — Vectorized operations, memory-efficient representations
- **Dependency Minimal** — Core library requires only NumPy

**Why it matters:** Go from research prototype to production deployment with confidence. Code that works today will work tomorrow.

.. raw:: html

   </div></details>

.. raw:: html

   <details class="sd-dropdown feature-dropdown-title">
   <summary class="sd-summary-title sd-font-weight-bold">
      <svg class="octicon lucide-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><path d="M9 17c2 0 2.8-1 2.8-2.8V10c0-2 1-3.3 3.2-3"/><path d="M9 11.2h5.7"/></svg>
      <div class="dropdown-title-content">
         <div class="dropdown-headline">Validation & Benchmarks</div>
         <div class="dropdown-subheadline">Every model and encoder validated against both mathematical theory and real-world performance.</div>
      </div>
      <svg class="dropdown-chevron" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 8 10 12 14 8"></polyline></svg>
   </summary>
   <div class="sd-summary-content">

**Validation:**

- Property-based testing (all VSA algebraic properties verified)
- Capacity analysis (how many items can be stored reliably)
- Noise robustness measurements (graceful degradation under errors)
- Cross-backend consistency (same results on CPU/GPU/TPU)

**Benchmarks:**

- Encoding speed across different data types
- Memory usage and scalability
- Retrieval accuracy vs. storage capacity
- Performance comparisons between models

**Why it matters:** Know exactly what to expect. No surprises in production.

.. raw:: html

   </div></details>

----

Quick Start
-----------

.. rubric:: Installation

Install holovec using uv (recommended) or pip:

.. code-block:: bash

    # Using uv (recommended)
    uv pip install holovec

    # Or using pip
    pip install holovec

.. rubric:: Optional Backends

For GPU acceleration or specialized hardware, install backend-specific dependencies:

.. code-block:: bash

    # GPU support with PyTorch
    uv pip install holovec[torch]

    # JIT compilation and TPU support with JAX
    uv pip install holovec[jax]

----

Example Usage
-------------

Explore how HoloVec handles different types of data and use cases. Each example demonstrates complete workflows from encoding to querying.

.. tab-set::

    .. tab-item:: Symbolic Reasoning

        .. code-block:: python

            from holovec import VSA

            # Create model
            model = VSA.create('FHRR', dim=10000, seed=42)

            # Encode symbols
            alice = model.random(seed=1)
            bob = model.random(seed=2)
            loves = model.random(seed=3)

            # Create "Alice loves Bob"
            statement = model.bind(model.bind(alice, loves), bob)

            # Query: Who does Alice love?
            query = model.unbind(model.unbind(statement, alice), loves)
            print(f"Similarity to Bob: {model.similarity(query, bob):.3f}")
            # Output: 1.0

    .. tab-item:: Continuous Values

        .. code-block:: python

            from holovec import VSA
            from holovec.encoders import FractionalPowerEncoder

            # Create model
            model = VSA.create('FHRR', dim=10000, seed=42)

            # Create encoder for temperatures (0-100°C)
            temp_encoder = FractionalPowerEncoder(
                model,
                min_val=0,
                max_val=100,
                bandwidth=0.1
            )

            # Encode temperatures
            temp_25 = temp_encoder.encode(25.0)
            temp_26 = temp_encoder.encode(26.0)
            temp_50 = temp_encoder.encode(50.0)

            # Similar temperatures have high similarity
            print(f"25°C vs 26°C: {model.similarity(temp_25, temp_26):.3f}")
            # Output: 0.95

            print(f"25°C vs 50°C: {model.similarity(temp_25, temp_50):.3f}")
            # Output: 0.32

    .. tab-item:: Text Classification

        .. code-block:: python

            from holovec import VSA
            from holovec.encoders import NGramEncoder

            # Create model
            model = VSA.create('FHRR', dim=10000, seed=42)

            # Create text encoder
            text_encoder = NGramEncoder(model, n=3)

            # Encode training documents
            doc_positive = text_encoder.encode("I love this product!")
            doc_negative = text_encoder.encode("This is terrible")

            # Encode new review
            new_review = text_encoder.encode("I really love it")

            # Classify by similarity
            sim_pos = model.similarity(new_review, doc_positive)
            sim_neg = model.similarity(new_review, doc_negative)

            print(f"Positive: {sim_pos:.3f}, Negative: {sim_neg:.3f}")
            # Output: Positive: 0.87, Negative: 0.12

----

Documentation Hub
-----------------

.. grid:: 2 2 2 2
    :gutter: 3

    .. grid-item-card::
        :link: getting-started/quickstart
        :link-type: doc
        :class-header: bg-light
        :class-body: docutils
        :class-card: sd-shadow-md

        :octicon:`rocket;1em` **Getting Started**
        ^^^

        **New to HoloVec?**

        Start here with a 5-minute introduction to hyperdimensional computing and vector symbolic architectures.

    .. grid-item-card::
        :link: tutorials/index
        :link-type: doc
        :class-header: bg-light
        :class-card: sd-shadow-md

        :octicon:`book;1em` **Tutorials**
        ^^^

        **Step-by-Step Guides**

        Build complete applications: text classification, recommender systems, and more.

    .. grid-item-card::
        :link: user-guide/index
        :link-type: doc
        :class-header: bg-light
        :class-card: sd-shadow-md

        :octicon:`file-directory;1em` **User Guide**
        ^^^

        **In-Depth Documentation**

        Learn how to choose models, encode data, configure backends, and optimize performance.

    .. grid-item-card::
        :link: examples/index
        :link-type: doc
        :class-header: bg-light
        :class-card: sd-shadow-md

        :octicon:`code-square;1em` **Examples**
        ^^^

        **28+ Code Examples**

        Browse examples from quickstart to advanced applications, encoders, and model comparisons.

    .. grid-item-card::
        :link: api/index
        :link-type: doc
        :class-header: bg-light
        :class-card: sd-shadow-md

        :octicon:`terminal;1em` **API Reference**
        ^^^

        **Complete API Docs**

        Detailed documentation for all modules, classes, functions, and parameters.

    .. grid-item-card::
        :link: theory/vsa_models
        :link-type: doc
        :class-header: bg-light
        :class-card: sd-shadow-md

        :octicon:`mortar-board;1.5em` **Theory**
        ^^^

        **Mathematical Foundations**

        Understand the theory behind VSA models, encoders, and hyperdimensional operations.

----

Community & Support
-------------------

.. grid:: 1 1 3 3
    :gutter: 2

    .. grid-item-card::
        :link: https://github.com/twistient/holovec/discussions
        :class-header: bg-light
        :class-body: sd-text-center
        :class-card: sd-text-center

        :octicon:`comment-discussion;1.5em` **GitHub Discussions**
        ^^^

        Ask questions and share ideas with the community

    .. grid-item-card::
        :link: https://github.com/twistient/holovec/issues
        :class-header: bg-light
        :class-body: sd-text-center
        :class-card: sd-text-center

        :octicon:`bug;1.5em` **Issue Tracker**
        ^^^

        Report bugs and request features

    .. grid-item-card::
        :class-header: bg-light
        :class-body: sd-text-center
        :class-card: sd-text-center

        :octicon:`mail;1.5em` **Contact**
        ^^^

        Email: support@twistient.com

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/index
   tutorials/text-classification
   tutorials/recommender-system

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user-guide/index
   user-guide/choosing-models
   user-guide/encoding-data
   user-guide/backends

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/vsa
   api/models
   api/encoders
   api/retrieval
   api/backends
   api/spaces
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Theory
   :hidden:

   theory/vsa_models

----

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
