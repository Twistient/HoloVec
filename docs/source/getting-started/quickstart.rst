Quickstart
==========

A 5-minute introduction to HoloVec and hyperdimensional computing.

What is Hyperdimensional Computing?
------------------------------------

Hyperdimensional computing (HDC) is a brain-inspired computing paradigm that represents information using high-dimensional vectors (typically 10,000 dimensions). These vectors are:

* **Quasi-orthogonal**: Random vectors are nearly perpendicular to each other
* **Distributed**: Information is spread across all dimensions
* **Robust**: Tolerant to noise and partial corruption
* **Composable**: Can be combined using simple operations

Basic Workflow
--------------

1. Create a VSA model
2. Encode data as hypervectors  
3. Compose representations using bind/bundle
4. Retrieve information via similarity

Example: Symbolic Relationships
--------------------------------

.. code-block:: python

    from holovec import VSA

    # 1. Create a model
    model = VSA.create('FHRR', dim=10000, seed=42)

    # 2. Encode symbols as random hypervectors
    alice = model.random(seed=1)
    bob = model.random(seed=2)
    loves = model.random(seed=3)
    hates = model.random(seed=4)

    # 3. Bind to create "Alice loves Bob"
    alice_loves_bob = model.bind(model.bind(alice, loves), bob)

    # 4. Query: Who does Alice love?
    query = model.unbind(model.unbind(alice_loves_bob, alice), loves)
    
    # Check similarity
    print(f"Similarity to Bob: {model.similarity(query, bob):.3f}")    # ~1.0
    print(f"Similarity to Alice: {model.similarity(query, alice):.3f}") # ~0.0

Example: Encoding Continuous Values
------------------------------------

.. code-block:: python

    from holovec.encoders import FractionalPowerEncoder

    # Create encoder for temperature (0-100°C)
    encoder = FractionalPowerEncoder(model, min_val=0, max_val=100, bandwidth=0.1)

    # Encode temperatures
    temp_25 = encoder.encode(25.0)
    temp_26 = encoder.encode(26.0)
    temp_50 = encoder.encode(50.0)

    # Similar values have high similarity
    print(f"Similarity (25°C, 26°C): {model.similarity(temp_25, temp_26):.3f}")  # ~0.95
    print(f"Similarity (25°C, 50°C): {model.similarity(temp_25, temp_50):.3f}")  # ~0.40

    # Reversible encoding
    decoded = encoder.decode(temp_25)
    print(f"Decoded: {decoded:.1f}°C")  # ~25.0°C

Example: Bundling (Superposition)
----------------------------------

.. code-block:: python

    # Create multiple hypervectors
    A = model.random(seed=10)
    B = model.random(seed=11)
    C = model.random(seed=12)

    # Bundle them together
    bundle = model.bundle([A, B, C])

    # All items remain retrievable
    print(f"Similarity to A: {model.similarity(bundle, A):.3f}")  # ~0.6
    print(f"Similarity to B: {model.similarity(bundle, B):.3f}")  # ~0.6
    print(f"Similarity to C: {model.similarity(bundle, C):.3f}")  # ~0.6

Next Steps
----------

* :doc:`first-steps` - Learn core operations
* :doc:`key-concepts` - Understand HDC fundamentals
* :doc:`../examples/index` - Browse 28+ examples
* :doc:`../user-guide/index` - Comprehensive user guide
