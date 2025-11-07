Tutorial: Text Classification
==============================

Build a complete text classifier from scratch using HoloVec's hyperdimensional computing approach.

**In this tutorial**, you'll create a sentiment analysis system that classifies movie reviews as positive or negative. We'll use n-gram encoding to capture local word patterns and build prototypes for each class.

**Time**: 20-30 minutes

**What you'll learn:**

* How to encode text with n-grams for classification
* Building class prototypes from training examples
* Classifying new text with similarity matching
* Evaluating and optimizing classifier performance
* Best practices for text classification with HDC

Prerequisites
-------------

* Basic Python programming
* Understanding of text classification concepts
* HoloVec installed (``pip install holovec``)

Overview
--------

Text classification with HDC works differently from traditional approaches:

**Traditional ML:**
1. Extract features (TF-IDF, word embeddings)
2. Train a classifier (SVM, neural network)
3. Complex optimization and many parameters

**HDC Approach:**
1. Encode text as hypervectors (n-grams)
2. Bundle examples into class prototypes
3. Classify by similarity matching
4. Fast, simple, interpretable

**Advantages:**

* No gradient descent or complex optimization
* Few hyperparameters
* Fast training (single pass)
* Incremental learning (add examples anytime)
* Interpretable (similarity scores)

Step 1: Setup and Imports
--------------------------

First, let's import everything we need:

.. code-block:: python

    import numpy as np
    from holovec import VSA
    from holovec.encoders import NGramEncoder
    from collections import Counter

    # For reproducibility
    np.random.seed(42)

    print("HoloVec Text Classification Tutorial")
    print("=" * 50)

Step 2: Choose Model and Parameters
------------------------------------

Select a VSA model and configure encoding parameters:

.. code-block:: python

    # Create VSA model
    # FHRR: Good for general-purpose classification
    # 10000 dimensions: Balance between capacity and speed
    model = VSA.create('FHRR', dim=10000, seed=42)

    print(f"\nModel: {model.model_name}")
    print(f"Dimension: {model.dimension}")
    print(f"Capacity: ~{model.dimension // 100} distinct items")

**Why these choices?**

* **FHRR model**: Supports smooth similarity, good for text
* **10,000 dimensions**: Enough capacity for vocabulary + n-grams
* **Seed**: Ensures reproducible results

Step 3: Prepare Training Data
------------------------------

Create a simple movie review dataset:

.. code-block:: python

    # Training examples: (text, label)
    training_data = [
        # Positive reviews
        ("This movie was excellent and entertaining", "positive"),
        ("I loved this film it was amazing", "positive"),
        ("Great acting and wonderful story", "positive"),
        ("Best movie I have seen this year", "positive"),
        ("Fantastic film highly recommended", "positive"),
        ("Brilliant performance truly outstanding", "positive"),

        # Negative reviews
        ("This movie was terrible and boring", "negative"),
        ("I hated this film it was awful", "negative"),
        ("Poor acting and weak story", "negative"),
        ("Worst movie I have seen this year", "negative"),
        ("Horrible film do not recommend", "negative"),
        ("Terrible performance very disappointing", "negative"),
    ]

    print(f"\nTraining examples: {len(training_data)}")
    print(f"  Positive: {sum(1 for _, label in training_data if label == 'positive')}")
    print(f"  Negative: {sum(1 for _, label in training_data if label == 'negative')}")

**Real-world datasets:**

For production use, consider:

* IMDB movie reviews (50K reviews)
* Amazon product reviews
* Twitter sentiment datasets
* Your own labeled data

Step 4: Text Preprocessing
---------------------------

Simple preprocessing to normalize text:

.. code-block:: python

    def preprocess(text):
        """Basic text preprocessing."""
        # Lowercase and split into words
        words = text.lower().split()

        # Remove punctuation (simple approach)
        words = [w.strip('.,!?;:()[]"\'') for w in words]

        # Remove empty strings
        words = [w for w in words if w]

        return words

    # Test preprocessing
    sample_text = "This movie was excellent and entertaining!"
    sample_words = preprocess(sample_text)
    print(f"\nPreprocessing example:")
    print(f"  Original: {sample_text}")
    print(f"  Processed: {sample_words}")

**Extension ideas:**

* Remove stop words ('the', 'a', 'is')
* Stemming/lemmatization
* Handle special characters
* N-gram at character level for misspellings

Step 5: Build Vocabulary
-------------------------

Extract all unique words from training data:

.. code-block:: python

    # Extract vocabulary from training data
    all_words = []
    for text, _ in training_data:
        all_words.extend(preprocess(text))

    # Get unique words and their frequencies
    word_freq = Counter(all_words)
    vocabulary = list(word_freq.keys())

    print(f"\nVocabulary statistics:")
    print(f"  Unique words: {len(vocabulary)}")
    print(f"  Total words: {len(all_words)}")
    print(f"  Most common: {word_freq.most_common(5)}")

    # Create hypervector for each word
    word_hvs = {
        word: model.random(seed=hash(word) % 100000)
        for word in vocabulary
    }

    print(f"  Word hypervectors created: {len(word_hvs)}")

**Important notes:**

* Each word gets a unique random hypervector
* Using ``hash(word)`` ensures same word → same HV across runs
* For large vocabularies (>1000 words), consider filtering rare words

Step 6: Create N-gram Encoder
------------------------------

Set up an encoder to capture word sequences:

.. code-block:: python

    # Create bigram encoder (n=2)
    # Captures pairs of consecutive words
    encoder = NGramEncoder(
        model,
        item_to_hv=word_hvs,
        n=2,  # Bigrams: "this movie", "movie was", etc.
        mode='bundle'  # Bundle all bigrams together
    )

    print(f"\nN-gram Encoder:")
    print(f"  N-gram size: 2 (bigrams)")
    print(f"  Mode: bundle")
    print(f"  Example bigrams from '{sample_text}':")

    # Show example bigrams
    words = preprocess(sample_text)
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        print(f"    {bigram}")

**N-gram size selection:**

* **n=1** (unigrams): Bag-of-words, no order information
* **n=2** (bigrams): Captures local word pairs (recommended)
* **n=3** (trigrams): More specific patterns, needs more data
* Higher n → more specific but requires more training examples

Step 7: Encode Training Examples
---------------------------------

Convert each text into a hypervector:

.. code-block:: python

    # Encode all training examples
    encoded_examples = []

    print("\nEncoding training examples...")
    for text, label in training_data:
        words = preprocess(text)
        hv = encoder.encode(words)
        encoded_examples.append((hv, label))

    print(f"  Encoded: {len(encoded_examples)} examples")

    # Check encoding
    ex_text, ex_label = training_data[0]
    ex_hv, _ = encoded_examples[0]
    print(f"\nExample encoding:")
    print(f"  Text: '{ex_text}'")
    print(f"  Label: {ex_label}")
    print(f"  HV shape: {ex_hv.shape}")
    print(f"  HV type: {type(ex_hv)}")

Step 8: Build Class Prototypes
-------------------------------

Create a prototype for each class by bundling examples:

.. code-block:: python

    # Group examples by class
    class_hvs = {}
    for label in ['positive', 'negative']:
        # Get all hypervectors for this class
        hvs = [hv for hv, lbl in encoded_examples if lbl == label]

        # Bundle them into a prototype
        class_hvs[label] = model.bundle(hvs)

        print(f"\n{label.capitalize()} prototype:")
        print(f"  Examples bundled: {len(hvs)}")
        print(f"  Prototype shape: {class_hvs[label].shape}")

**What is bundling?**

Bundling (superposition) combines multiple hypervectors into one that is similar to all of them. It's like averaging but preserves the high-dimensional structure.

* Input: N hypervectors representing positive reviews
* Output: 1 prototype hypervector that captures "positive-ness"

Step 9: Classify New Text
--------------------------

Test the classifier on new examples:

.. code-block:: python

    def classify(text):
        """Classify a text string."""
        # Preprocess and encode
        words = preprocess(text)

        # Handle unknown words gracefully
        known_words = [w for w in words if w in word_hvs]
        if not known_words:
            return None, 0.0  # Cannot classify

        test_hv = encoder.encode(known_words)

        # Find most similar class
        best_label = None
        best_sim = float('-inf')

        for label, prototype in class_hvs.items():
            sim = float(model.similarity(test_hv, prototype))
            if sim > best_sim:
                best_sim = sim
                best_label = label

        return best_label, best_sim

    # Test examples
    test_reviews = [
        "This film was amazing and wonderful",
        "Terrible movie very disappointing",
        "Great story and excellent acting",
        "Awful film worst ever",
    ]

    print("\n" + "=" * 50)
    print("Classification Results")
    print("=" * 50)

    for text in test_reviews:
        label, sim = classify(text)
        print(f"\nReview: '{text}'")
        print(f"  Predicted: {label}")
        print(f"  Confidence: {sim:.3f}")

Step 10: Evaluate Performance
------------------------------

Test on held-out data and compute accuracy:

.. code-block:: python

    # Create test set
    test_data = [
        # Positive
        ("Excellent movie highly enjoyable", "positive"),
        ("Loved the story and acting", "positive"),
        ("Outstanding film wonderful experience", "positive"),

        # Negative
        ("Poor film very boring", "negative"),
        ("Hated this movie terrible", "negative"),
        ("Disappointing and awful story", "negative"),
    ]

    # Evaluate
    correct = 0
    total = len(test_data)

    print("\n" + "=" * 50)
    print("Evaluation on Test Set")
    print("=" * 50)

    for text, true_label in test_data:
        pred_label, confidence = classify(text)
        is_correct = (pred_label == true_label)
        correct += is_correct

        marker = "✓" if is_correct else "✗"
        print(f"\n{marker} '{text}'")
        print(f"   True: {true_label}, Predicted: {pred_label} ({confidence:.3f})")

    accuracy = correct / total
    print(f"\n" + "=" * 50)
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")
    print("=" * 50)

**Typical results:**

* **Small dataset** (like ours): 70-90% accuracy
* **Medium dataset** (hundreds of examples): 85-95% accuracy
* **Large dataset** (thousands of examples): 90-98% accuracy

Step 11: Analyze Class Similarities
------------------------------------

Understand the learned representations:

.. code-block:: python

    # Similarity between class prototypes
    pos_neg_sim = float(model.similarity(
        class_hvs['positive'],
        class_hvs['negative']
    ))

    print(f"\nClass Analysis:")
    print(f"  Positive-Negative similarity: {pos_neg_sim:.3f}")
    print(f"  (Close to 0 = well-separated classes)")

    # Most confident classifications
    print(f"\nConfidence analysis:")
    for text in test_reviews:
        label, sim = classify(text)
        print(f"  {label:8s}: {sim:.3f} - '{text[:40]}...'")

**Good separation indicators:**

* Class prototypes have low similarity (< 0.1)
* Confident predictions have high similarity (> 0.5)
* Wrong predictions often have low confidence

Step 12: Extensions and Improvements
-------------------------------------

Ways to improve the classifier:

**1. Add more training data:**

.. code-block:: python

    # More examples → better prototypes
    # Aim for 50-100+ examples per class

**2. Tune n-gram size:**

.. code-block:: python

    # Try trigrams for more context
    encoder_3gram = NGramEncoder(
        model,
        item_to_hv=word_hvs,
        n=3,  # Trigrams
        mode='bundle'
    )

**3. Combine multiple n-gram sizes:**

.. code-block:: python

    def encode_multi_ngram(words):
        """Encode with multiple n-gram sizes."""
        hv_bigram = encoder_2gram.encode(words)
        hv_trigram = encoder_3gram.encode(words)
        # Bundle both representations
        return model.bundle([hv_bigram, hv_trigram])

**4. Add confidence threshold:**

.. code-block:: python

    def classify_with_threshold(text, threshold=0.3):
        """Classify with confidence threshold."""
        label, sim = classify(text)
        if sim < threshold:
            return "uncertain", sim
        return label, sim

**5. Handle unknown words:**

.. code-block:: python

    # Add <UNK> token for unknown words
    word_hvs['<UNK>'] = model.random(seed=999)

    def encode_with_unk(words):
        safe_words = [w if w in word_hvs else '<UNK>' for w in words]
        return encoder.encode(safe_words)

**6. Use larger vocabulary:**

.. code-block:: python

    # Pre-trained word lists
    # Common English words, domain-specific terms, etc.

**7. Incremental learning:**

.. code-block:: python

    def add_training_example(text, label):
        """Add new example to existing prototype."""
        words = preprocess(text)
        new_hv = encoder.encode(words)

        # Update prototype by bundling with new example
        class_hvs[label] = model.bundle([
            class_hvs[label],
            new_hv
        ])

Complete Code
-------------

Here's the full classifier in one place:

.. code-block:: python

    import numpy as np
    from holovec import VSA
    from holovec.encoders import NGramEncoder
    from collections import Counter

    # Setup
    np.random.seed(42)
    model = VSA.create('FHRR', dim=10000, seed=42)

    # Training data
    training_data = [
        ("This movie was excellent and entertaining", "positive"),
        ("I loved this film it was amazing", "positive"),
        ("Great acting and wonderful story", "positive"),
        ("Best movie I have seen this year", "positive"),
        ("Fantastic film highly recommended", "positive"),
        ("Brilliant performance truly outstanding", "positive"),
        ("This movie was terrible and boring", "negative"),
        ("I hated this film it was awful", "negative"),
        ("Poor acting and weak story", "negative"),
        ("Worst movie I have seen this year", "negative"),
        ("Horrible film do not recommend", "negative"),
        ("Terrible performance very disappointing", "negative"),
    ]

    # Preprocessing
    def preprocess(text):
        words = text.lower().split()
        words = [w.strip('.,!?;:()[]"\'') for w in words]
        return [w for w in words if w]

    # Build vocabulary
    all_words = []
    for text, _ in training_data:
        all_words.extend(preprocess(text))

    vocabulary = list(set(all_words))
    word_hvs = {word: model.random(seed=hash(word) % 100000)
                for word in vocabulary}

    # Create encoder
    encoder = NGramEncoder(model, item_to_hv=word_hvs, n=2, mode='bundle')

    # Encode training data
    encoded_examples = []
    for text, label in training_data:
        words = preprocess(text)
        hv = encoder.encode(words)
        encoded_examples.append((hv, label))

    # Build class prototypes
    class_hvs = {}
    for label in ['positive', 'negative']:
        hvs = [hv for hv, lbl in encoded_examples if lbl == label]
        class_hvs[label] = model.bundle(hvs)

    # Classifier
    def classify(text):
        words = preprocess(text)
        known_words = [w for w in words if w in word_hvs]
        if not known_words:
            return None, 0.0

        test_hv = encoder.encode(known_words)

        best_label = None
        best_sim = float('-inf')
        for label, prototype in class_hvs.items():
            sim = float(model.similarity(test_hv, prototype))
            if sim > best_sim:
                best_sim = sim
                best_label = label

        return best_label, best_sim

    # Test
    test_text = "This film was amazing and wonderful"
    label, confidence = classify(test_text)
    print(f"Text: '{test_text}'")
    print(f"Predicted: {label} (confidence: {confidence:.3f})")

Best Practices Summary
----------------------

**Model Selection:**

* Use FHRR or HRR for text classification
* 10,000 dimensions for medium vocabularies (<1000 words)
* 20,000+ dimensions for large vocabularies (>1000 words)

**Encoding:**

* Start with bigrams (n=2)
* Use trigrams (n=3) if you have enough data
* Consider combining multiple n-gram sizes

**Training:**

* Need 20-50 examples minimum per class
* More examples = better prototypes
* Balanced classes help (equal positive/negative)

**Evaluation:**

* Always test on held-out data
* Check confidence scores for uncertainty
* Analyze failure cases to improve

**Production Deployment:**

* Save prototypes (``word_hvs``, ``class_hvs``)
* Preprocess consistently
* Handle unknown words gracefully
* Set confidence thresholds

Common Issues and Solutions
---------------------------

**Problem**: Low accuracy (< 60%)

**Solutions**:

* Add more training examples
* Check class balance
* Try different n-gram sizes
* Ensure good preprocessing

**Problem**: High confidence on wrong predictions

**Solutions**:

* Classes may be too similar
* Need more distinctive training examples
* Try larger dimension

**Problem**: Unknown word errors

**Solutions**:

* Add <UNK> token to vocabulary
* Filter rare words before encoding
* Use more training data to expand vocabulary

**Problem**: Slow classification

**Solutions**:

* Use MAP or BSC model (faster)
* Reduce vocabulary size
* Use PyTorch backend with GPU

Next Steps
----------

**Explore more:**

* :doc:`../examples/20_app_text_classification` - Extended text classification example
* :doc:`../user-guide/encoding-data` - Deep dive on encoders
* :doc:`../user-guide/choosing-models` - Model selection guide

**Try these datasets:**

* IMDB reviews (50K examples)
* 20 Newsgroups (18K documents)
* AG News (120K articles)

**Advanced topics:**

* Multi-class classification (>2 classes)
* Hierarchical classification
* Online learning (update prototypes dynamically)
* Ensemble methods (combine multiple encoders)

Conclusion
----------

You've built a complete text classifier using hyperdimensional computing!

**Key takeaways:**

* HDC provides a simple, fast approach to text classification
* No complex optimization or gradient descent needed
* Prototypes capture class characteristics through bundling
* Classification is just similarity matching
* Easy to extend and adapt to new data

**Advantages of HDC for text:**

* Fast training (single pass)
* Incremental learning
* Interpretable similarity scores
* Few hyperparameters
* Works well with limited data

The same principles apply to many other classification tasks - try applying this to your own text data!