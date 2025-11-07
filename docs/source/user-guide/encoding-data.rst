Encoding Data
=============

Encoders transform different data types into hypervectors. This guide helps you choose the right encoder for your data.

Quick Encoder Selection
------------------------

**By Data Type:**

* **Numbers (continuous)**: :class:`~holovec.encoders.FractionalPowerEncoder` (smooth similarity)
* **Numbers (ordinal/rankings)**: :class:`~holovec.encoders.ThermometerEncoder`
* **Numbers (categories)**: :class:`~holovec.encoders.LevelEncoder`
* **Sequences (text, DNA)**: :class:`~holovec.encoders.NGramEncoder`
* **Positions in sequence**: :class:`~holovec.encoders.PositionBindingEncoder`
* **2D/3D trajectories**: :class:`~holovec.encoders.TrajectoryEncoder`
* **Images**: :class:`~holovec.encoders.ImageEncoder`
* **Dense vectors**: :class:`~holovec.encoders.VectorEncoder`

**By Model Compatibility:**

* **All models**: Thermometer, Level, NGram, PositionBinding, Image, Vector
* **FHRR/HRR/GHRR/VTB only**: FractionalPowerEncoder

Encoder Overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 15 30

   * - Encoder
     - Data Type
     - Reversible
     - Models
     - Best For
   * - **FractionalPowerEncoder**
     - Continuous scalars
     - Yes
     - FHRR, HRR, GHRR, VTB
     - Temperature, time, coordinates
   * - **ThermometerEncoder**
     - Ordinal scalars
     - No
     - All
     - Rankings, ratings, scores
   * - **LevelEncoder**
     - Discrete bins
     - Yes
     - All
     - Categories, grade levels
   * - **NGramEncoder**
     - Sequences
     - Partial
     - All
     - Text, DNA, tokens
   * - **PositionBindingEncoder**
     - Sequence positions
     - Yes
     - All
     - Ordered sequences
   * - **TrajectoryEncoder**
     - Multi-D paths
     - No
     - All (best: FHRR)
     - Motion, gestures, time series
   * - **ImageEncoder**
     - 2D grids
     - No
     - All (best: FHRR)
     - Images, patterns
   * - **VectorEncoder**
     - Dense vectors
     - Yes
     - All
     - Embeddings, features

Scalar Encoders
---------------

For encoding single numerical values.

FractionalPowerEncoder
^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.encoders.FractionalPowerEncoder`

**Best encoder for continuous numerical data with smooth similarity.**

**How it works:**

* Uses fractional binding powers to encode values
* Creates smooth, continuous similarity gradients
* Exact reversal through decoding

**When to use:**

* Continuous measurements (temperature, time, distance)
* Geographic coordinates (latitude, longitude)
* Any numeric value where similar values should have high similarity
* When you need exact value recovery

**Requirements:**

* Only works with FHRR, HRR, GHRR, or VTB models
* Requires complex or real-valued hypervectors

**Parameters:**

* ``min_val``, ``max_val``: Range of values to encode
* ``bandwidth``: Controls similarity spread (0.05-0.2 typical)
* Smaller bandwidth = sharper similarity peaks
* Larger bandwidth = broader similarity

**Example:**

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import FractionalPowerEncoder

    model = VSA.create('FHRR', dim=10000)

    # Encode temperature range
    temp_encoder = FractionalPowerEncoder(
        model,
        min_val=-20.0,
        max_val=40.0,
        bandwidth=0.1,
        seed=42
    )

    # Encode values
    temp_15 = temp_encoder.encode(15.0)
    temp_16 = temp_encoder.encode(16.0)
    temp_25 = temp_encoder.encode(25.0)

    # Similar temperatures have high similarity
    sim_close = model.similarity(temp_15, temp_16)  # ~0.95
    sim_far = model.similarity(temp_15, temp_25)    # ~0.3

    # Decode to recover value
    decoded = temp_encoder.decode(temp_15)  # ~15.0

**See Also:**

* :doc:`../examples/11_encoders_fractional_power` - Complete tutorial
* :doc:`../examples/10_encoders_scalar` - Comparison with other encoders

ThermometerEncoder
^^^^^^^^^^^^^^^^^^

:class:`~holovec.encoders.ThermometerEncoder`

**For ordinal data where order matters (rankings, levels).**

**How it works:**

* Divides range into discrete bins
* Each value activates all bins up to its level (cumulative)
* Creates monotonic similarity: farther values → lower similarity

**When to use:**

* Ordinal data (rankings, priority levels)
* When you need model compatibility (works with MAP, BSC, etc.)
* Don't need exact value recovery
* Want gradual similarity decay with distance

**Works with all models** (MAP, BSC, FHRR, HRR, etc.)

**Parameters:**

* ``min_val``, ``max_val``: Range of values
* ``n_bins``: Number of discrete bins (rule of thumb: dim/200)

**Example:**

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import ThermometerEncoder

    model = VSA.create('MAP', dim=10000)  # Works with binary models

    # Product ratings (1-5 stars)
    rating_encoder = ThermometerEncoder(
        model,
        min_val=1.0,
        max_val=5.0,
        n_bins=20,  # Finer granularity than 5 levels
        seed=42
    )

    rating_4 = rating_encoder.encode(4.0)
    rating_5 = rating_encoder.encode(5.0)
    rating_2 = rating_encoder.encode(2.0)

    # Monotonic similarity
    sim_close = model.similarity(rating_4, rating_5)  # High
    sim_far = model.similarity(rating_4, rating_2)    # Low

**See Also:**

* :doc:`../examples/12_encoders_thermometer_level` - Complete tutorial
* :doc:`../examples/10_encoders_scalar` - Scalar encoder comparison

LevelEncoder
^^^^^^^^^^^^

:class:`~holovec.encoders.LevelEncoder`

**For categorical bins with sharp boundaries.**

**How it works:**

* Divides range into discrete levels
* Each level is a distinct random hypervector
* Sharp boundaries between categories
* Reversible (decodes to level center)

**When to use:**

* Categorical data with numeric ranges (grade levels A/B/C)
* Discrete bins (age groups, income brackets)
* Need reversibility (can decode which bin)
* Want sharp category boundaries

**Works with all models**

**Parameters:**

* ``min_val``, ``max_val``: Range
* ``n_levels``: Number of categories

**Example:**

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import LevelEncoder

    model = VSA.create('FHRR', dim=10000)

    # Age groups
    age_encoder = LevelEncoder(
        model,
        min_val=0,
        max_val=100,
        n_levels=5,  # 0-20, 20-40, 40-60, 60-80, 80-100
        seed=42
    )

    age_25 = age_encoder.encode(25)  # Falls in level 1 (20-40)
    age_38 = age_encoder.encode(38)  # Also level 1
    age_65 = age_encoder.encode(65)  # Falls in level 3 (60-80)

    # Same level = high similarity
    sim_same = model.similarity(age_25, age_38)  # ~1.0
    sim_diff = model.similarity(age_25, age_65)  # Low

    # Decode to level center
    decoded = age_encoder.decode(age_25)  # Returns 30.0 (center of 20-40)

**See Also:**

* :doc:`../examples/12_encoders_thermometer_level` - Complete tutorial

Sequence Encoders
-----------------

For encoding sequences of items (text, DNA, tokens).

NGramEncoder
^^^^^^^^^^^^

:class:`~holovec.encoders.NGramEncoder`

**For encoding sequences using n-grams.**

**How it works:**

* Breaks sequence into overlapping n-grams
* Each item has a random hypervector
* Bundles all n-grams from the sequence
* Captures local order information

**When to use:**

* Text classification (words, characters)
* DNA/protein sequences
* Any sequence where local patterns matter
* Don't need exact position information

**Works with all models**

**Parameters:**

* ``item_to_hv``: Dict mapping items → hypervectors
* ``n``: N-gram size (2-3 typical for text, 3-5 for DNA)
* ``mode``: How to combine n-grams

  * ``'bundle'`` (default): Bundle all n-grams
  * ``'bind'``: Bind n-grams sequentially
  * ``'permute-bind'``: Use permutations to encode position

**Example:**

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import NGramEncoder

    model = VSA.create('FHRR', dim=10000)

    # Create vocabulary
    vocab = ['the', 'cat', 'sat', 'on', 'mat', 'dog']
    item_hvs = {word: model.random(seed=hash(word)) for word in vocab}

    # Bigram encoder
    ngram_encoder = NGramEncoder(
        model,
        item_to_hv=item_hvs,
        n=2,
        mode='bundle'
    )

    # Encode sentences
    sent1 = ngram_encoder.encode(['the', 'cat', 'sat'])
    sent2 = ngram_encoder.encode(['the', 'cat', 'on', 'mat'])
    sent3 = ngram_encoder.encode(['the', 'dog', 'sat'])

    # Similar sequences have high similarity
    sim_12 = model.similarity(sent1, sent2)  # Share 'the cat'
    sim_13 = model.similarity(sent1, sent3)  # Share 'the' and 'sat'

**See Also:**

* :doc:`../examples/13_encoders_ngram` - Complete tutorial
* :doc:`../examples/20_app_text_classification` - Text classification application

PositionBindingEncoder
^^^^^^^^^^^^^^^^^^^^^^

:class:`~holovec.encoders.PositionBindingEncoder`

**For encoding sequences with explicit position information.**

**How it works:**

* Each item bound to its position in sequence
* Position encoded using position encoder (FPE or Thermometer)
* Preserves exact order information
* Reversible through unbinding

**When to use:**

* Order is critical (mathematical expressions, code)
* Need to query "what's at position N?"
* Want to preserve exact positional relationships
* Symbolic sequences with structure

**Works with all models** (position encoder determines compatibility)

**Parameters:**

* ``item_to_hv``: Dict mapping items → hypervectors
* ``position_encoder``: Encoder for positions (FPE or Thermometer)

**Example:**

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import PositionBindingEncoder, FractionalPowerEncoder

    model = VSA.create('FHRR', dim=10000)

    # Vocabulary
    vocab = ['(', ')', '+', '*', 'x', 'y', '2']
    item_hvs = {token: model.random(seed=hash(token)) for token in vocab}

    # Position encoder
    pos_encoder = FractionalPowerEncoder(
        model, min_val=0, max_val=20, bandwidth=0.1
    )

    # Create encoder
    seq_encoder = PositionBindingEncoder(
        model,
        item_to_hv=item_hvs,
        position_encoder=pos_encoder
    )

    # Encode: (x + 2) * y
    expr = seq_encoder.encode(['(', 'x', '+', '2', ')', '*', 'y'])

    # Can unbind to query positions
    # What's at position 3? → '+'

**See Also:**

* :doc:`../examples/14_encoders_position_binding` - Complete tutorial
* :doc:`../examples/16_compositional_structures` - Symbolic structures

Spatial Encoders
----------------

For encoding spatial and temporal data.

TrajectoryEncoder
^^^^^^^^^^^^^^^^^

:class:`~holovec.encoders.TrajectoryEncoder`

**For encoding multi-dimensional continuous paths (motion, gestures).**

**How it works:**

* Encodes path as sequence of multi-dimensional coordinates
* Each time step binds coordinates to position
* Uses single scalar encoder for all dimensions
* Captures spatial and temporal patterns

**When to use:**

* Motion trajectories (gestures, handwriting)
* 2D/3D paths (robot motion, object tracking)
* Time series with multiple dimensions
* Continuous sensor data over time

**Works with all models** (best with FHRR for smooth encoding)

**Parameters:**

* ``scalar_encoder``: Encoder for coordinate values (usually FPE)
* ``n_dimensions``: Number of spatial dimensions (2 for 2D, 3 for 3D)
* ``time_range``: Optional time interval encoding

**Example:**

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import TrajectoryEncoder, FractionalPowerEncoder

    model = VSA.create('FHRR', dim=10000)

    # Scalar encoder for coordinates
    scalar_encoder = FractionalPowerEncoder(
        model,
        min_val=-1.0,
        max_val=1.0,
        bandwidth=0.1
    )

    # 2D trajectory encoder
    traj_encoder = TrajectoryEncoder(
        model,
        scalar_encoder=scalar_encoder,
        n_dimensions=2,
        seed=42
    )

    # Encode circular gesture (20 time steps, 2D)
    import numpy as np
    t = np.linspace(0, 2*np.pi, 20)
    x = 0.5 * np.cos(t)
    y = 0.5 * np.sin(t)
    trajectory = np.column_stack([x, y])  # Shape: (20, 2)

    circle_hv = traj_encoder.encode(trajectory)

**See Also:**

* :doc:`../examples/15_encoders_trajectory` - Complete tutorial
* :doc:`../examples/22_app_gesture_recognition` - Gesture recognition application

ImageEncoder
^^^^^^^^^^^^

:class:`~holovec.encoders.ImageEncoder`

**For encoding 2D images and spatial patterns.**

**How it works:**

* Encodes each pixel value and binds to spatial position
* Uses position encoders for x and y coordinates
* Can use different strategies (binding, random projection)
* Captures spatial relationships

**When to use:**

* Image classification
* Pattern recognition
* 2D spatial data
* Grid-based representations

**Works with all models** (best with FHRR for continuous pixels)

**Parameters:**

* ``pixel_encoder``: Encoder for pixel values
* ``x_encoder``, ``y_encoder``: Position encoders for coordinates
* ``strategy``: Encoding strategy

**Example:**

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import ImageEncoder, FractionalPowerEncoder

    model = VSA.create('FHRR', dim=10000)

    # Encoders for pixels and positions
    pixel_encoder = FractionalPowerEncoder(
        model, min_val=0, max_val=255, bandwidth=0.1
    )
    x_encoder = FractionalPowerEncoder(
        model, min_val=0, max_val=28, bandwidth=0.5
    )
    y_encoder = FractionalPowerEncoder(
        model, min_val=0, max_val=28, bandwidth=0.5
    )

    # Create image encoder
    img_encoder = ImageEncoder(
        model,
        pixel_encoder=pixel_encoder,
        x_encoder=x_encoder,
        y_encoder=y_encoder
    )

    # Encode 28x28 grayscale image
    import numpy as np
    image = np.random.randint(0, 256, (28, 28))
    image_hv = img_encoder.encode(image)

**See Also:**

* :doc:`../examples/17_encoders_image` - Complete tutorial
* :doc:`../examples/21_app_image_recognition` - Image classification application

VectorEncoder
^^^^^^^^^^^^^

:class:`~holovec.encoders.VectorEncoder`

**For encoding dense numerical vectors (embeddings, feature vectors).**

**How it works:**

* Maps each dimension to a random hypervector
* Binds dimension HVs to their scalar values
* Bundles all dimensions together
* Preserves vector similarity

**When to use:**

* Pre-trained embeddings (word2vec, BERT)
* Feature vectors from neural networks
* Any dense numerical representation
* Want to combine with symbolic VSA operations

**Works with all models**

**Parameters:**

* ``dimension_hvs``: Random HVs for each dimension
* ``value_encoder``: Scalar encoder for values (FPE or Thermometer)

**Example:**

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import VectorEncoder, FractionalPowerEncoder

    model = VSA.create('FHRR', dim=10000)

    # Encode 300-dimensional word embeddings
    embedding_dim = 300

    # Create dimension hypervectors
    dim_hvs = [model.random(seed=i) for i in range(embedding_dim)]

    # Value encoder
    value_encoder = FractionalPowerEncoder(
        model, min_val=-1.0, max_val=1.0, bandwidth=0.1
    )

    # Create vector encoder
    vec_encoder = VectorEncoder(
        model,
        dimension_hvs=dim_hvs,
        value_encoder=value_encoder
    )

    # Encode word embedding
    import numpy as np
    word_embedding = np.random.randn(300)  # From word2vec
    word_hv = vec_encoder.encode(word_embedding)

**See Also:**

* :doc:`../examples/18_encoders_vector` - Complete tutorial

Encoder Selection Guide
-----------------------

Decision Tree
^^^^^^^^^^^^^

.. code-block:: text

    What type of data do you have?
    │
    ├─ Single numbers (scalars)
    │   ├─ Continuous (temperature, time, coordinates)
    │   │   └─ → FractionalPowerEncoder (if FHRR/HRR)
    │   │       → ThermometerEncoder (if MAP/BSC)
    │   ├─ Ordinal (rankings, ratings)
    │   │   └─ → ThermometerEncoder
    │   └─ Categories (age groups, bins)
    │       └─ → LevelEncoder
    │
    ├─ Sequences (text, DNA, tokens)
    │   ├─ Local patterns important (n-grams)
    │   │   └─ → NGramEncoder
    │   └─ Exact positions important
    │       └─ → PositionBindingEncoder
    │
    ├─ Paths / Trajectories (gestures, motion)
    │   └─ → TrajectoryEncoder
    │
    ├─ Images (2D grids)
    │   └─ → ImageEncoder
    │
    └─ Dense vectors (embeddings, features)
        └─ → VectorEncoder

By Model Type
^^^^^^^^^^^^^

**Using MAP or BSC (binary models)?**

* ✓ ThermometerEncoder, LevelEncoder
* ✓ NGramEncoder, PositionBindingEncoder (with Thermometer positions)
* ✓ ImageEncoder, VectorEncoder (with Thermometer values)
* ✗ FractionalPowerEncoder (requires FHRR/HRR/GHRR/VTB)

**Using FHRR or HRR (complex/real models)?**

* ✓ All encoders work
* ✓ Best: FractionalPowerEncoder for smooth similarity

Common Patterns
---------------

Combining Multiple Encoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create rich representations by binding multiple encoded features:

.. code-block:: python

    from holovec import VSA
    from holovec.encoders import FractionalPowerEncoder, NGramEncoder

    model = VSA.create('FHRR', dim=10000)

    # Product representation with multiple features
    # Feature 1: Price
    price_encoder = FractionalPowerEncoder(
        model, min_val=0, max_val=1000, bandwidth=0.1
    )
    PRICE = model.random(seed=1)

    # Feature 2: Category (as sequence)
    categories = ['electronics', 'books', 'clothing']
    cat_hvs = {c: model.random(seed=hash(c)) for c in categories}
    CATEGORY = model.random(seed=2)

    # Feature 3: Rating
    rating_encoder = FractionalPowerEncoder(
        model, min_val=1, max_val=5, bandwidth=0.1
    )
    RATING = model.random(seed=3)

    # Encode product
    def encode_product(price, category, rating):
        price_hv = model.bind(PRICE, price_encoder.encode(price))
        cat_hv = model.bind(CATEGORY, cat_hvs[category])
        rating_hv = model.bind(RATING, rating_encoder.encode(rating))

        # Bundle all features
        return model.bundle([price_hv, cat_hv, rating_hv])

    # Example products
    laptop = encode_product(price=899.99, category='electronics', rating=4.5)
    novel = encode_product(price=12.99, category='books', rating=4.8)

    # Similar products (same category, similar price/rating) will have
    # higher similarity

Hierarchical Encoding
^^^^^^^^^^^^^^^^^^^^^

Encode complex structures hierarchically:

.. code-block:: python

    # Encode sentence with word and sentence levels

    # Word level: encode each word
    word_hvs = [word_encoder.encode(word) for word in sentence]

    # Bind words to positions
    positioned_words = [
        model.bind(pos_encoder.encode(i), word_hv)
        for i, word_hv in enumerate(word_hvs)
    ]

    # Bundle to get sentence
    sentence_hv = model.bundle(positioned_words)

Best Practices
--------------

Parameter Selection
^^^^^^^^^^^^^^^^^^^

**Dimension Count:**

* Rule of thumb: ``capacity ≈ dimension / 100``
* Example: 10,000 dimensions → ~100 distinct items
* More items needed? Use higher dimension or fewer bundles

**Bandwidth (FractionalPowerEncoder):**

* Smaller (0.05-0.1): Sharp similarity peaks, precise encoding
* Medium (0.1-0.2): Balanced, most common
* Larger (0.2-0.5): Broad similarity, more robust to noise

**N-gram Size:**

* Text: n=2 (bigrams) or n=3 (trigrams)
* DNA: n=3-5 (k-mers)
* Larger n → More specific patterns, less generalization

**Bin Count (Thermometer/Level):**

* Rule: ``bins ≤ dimension / 200``
* More bins → finer granularity but need higher dimension
* Fewer bins → coarser but works with lower dimension

Testing Encoders
^^^^^^^^^^^^^^^^

Always validate your encoder choice:

.. code-block:: python

    # Test similarity behavior
    val1 = encoder.encode(10.0)
    val2 = encoder.encode(10.5)
    val3 = encoder.encode(15.0)

    sim_close = model.similarity(val1, val2)  # Should be high
    sim_far = model.similarity(val1, val3)    # Should be lower

    print(f"Close values: {sim_close:.3f}")
    print(f"Far values: {sim_far:.3f}")

    # Test reversibility (if encoder supports decode)
    if hasattr(encoder, 'decode'):
        decoded = encoder.decode(val1)
        print(f"Original: 10.0, Decoded: {decoded:.2f}")

Error Handling
^^^^^^^^^^^^^^

**Value out of range:**

.. code-block:: python

    # Encoders will clip or raise errors for out-of-range values
    encoder = FractionalPowerEncoder(model, min_val=0, max_val=100)

    # Safe: clip values to range
    safe_value = np.clip(value, 0, 100)
    hv = encoder.encode(safe_value)

**Unknown items in sequences:**

.. code-block:: python

    # NGramEncoder: use UNK token for unknown items
    vocab_hvs = {word: model.random(seed=hash(word)) for word in vocab}
    vocab_hvs['<UNK>'] = model.random(seed=999)

    def safe_encode(sequence):
        safe_seq = [word if word in vocab_hvs else '<UNK>'
                    for word in sequence]
        return ngram_encoder.encode(safe_seq)

Frequently Asked Questions
--------------------------

**Q: Which encoder should I start with for numbers?**

A: :class:`~holovec.encoders.FractionalPowerEncoder` if using FHRR/HRR, otherwise :class:`~holovec.encoders.ThermometerEncoder`.

**Q: Can I use multiple encoders in one application?**

A: Yes! Bind different encoded features together. See "Combining Multiple Encoders" above.

**Q: How do I encode text?**

A: Use :class:`~holovec.encoders.NGramEncoder` for bag-of-ngrams approach, or :class:`~holovec.encoders.PositionBindingEncoder` if position matters.

**Q: What if my data doesn't fit these categories?**

A: You can create custom encoders or combine existing ones. All encoders just need to return hypervectors.

**Q: How do I choose bandwidth for FractionalPowerEncoder?**

A: Start with 0.1. Decrease if you need sharper discrimination, increase for more robustness to noise.

**Q: Can I change encoders after development?**

A: You'll need to re-encode all data, but the rest of your code can stay the same.

**Q: What's the difference between NGramEncoder and PositionBindingEncoder?**

A: NGram captures local patterns without exact positions (good for text classification). PositionBinding preserves exact order (good for structured sequences).

Next Steps
----------

* :doc:`choosing-models` - Pick the right VSA model for your encoder
* :doc:`backends` - Optimize performance with different backends
* :doc:`../examples/index` - See encoders in action
* :doc:`../api/encoders` - Complete encoder API reference

See Also
--------

* :doc:`../api/encoders` - Encoder implementations
* :doc:`../examples/10_encoders_scalar` - Scalar encoding tutorial
* :doc:`../examples/13_encoders_ngram` - Sequence encoding tutorial
* :doc:`../examples/20_app_text_classification` - Real application example
