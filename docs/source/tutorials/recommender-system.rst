Tutorial: Building a Recommender System
========================================

Create a content-based recommendation engine using hyperdimensional computing to find similar items based on multiple features.

**In this tutorial**, you'll build a product recommendation system that suggests similar items based on price, category, ratings, and features. We'll use HoloVec's binding and bundling operations to create rich product representations.

**Time**: 20-30 minutes

**What you'll learn:**

* Encoding multi-feature items (categorical + numerical)
* Building product representations with binding
* Finding similar items with similarity search
* Implementing "customers who bought X also bought Y"
* Scaling recommendations to large catalogs

Prerequisites
-------------

* Basic Python programming
* Understanding of recommendation systems
* HoloVec installed (``pip install holovec``)

Overview
--------

Recommendation with HDC differs from traditional collaborative filtering:

**Traditional approach:**
1. User-item interaction matrix
2. Matrix factorization or neural networks
3. Requires lots of interaction data

**HDC Approach:**
1. Encode item features as hypervectors
2. Combine features with binding
3. Find similar items with nearest neighbors
4. Fast, interpretable, works with limited data

**Advantages:**

* Works immediately (no training period)
* Handles new items easily (cold-start problem)
* Combines multiple feature types naturally
* Interpretable similarity scores
* Efficient similarity search

Step 1: Setup and Imports
--------------------------

Import required libraries:

.. code-block:: python

    import numpy as np
    from holovec import VSA
    from holovec.encoders import FractionalPowerEncoder, LevelEncoder
    from holovec.retrieval import ItemStore

    # For reproducibility
    np.random.seed(42)

    print("HoloVec Recommender System Tutorial")
    print("=" * 60)

Step 2: Choose Model and Create Encoders
-----------------------------------------

Set up the VSA model and encoders for different feature types:

.. code-block:: python

    # Create VSA model
    model = VSA.create('FHRR', dim=10000, seed=42)

    print(f"\nModel: {model.model_name}, dimension={model.dimension}")

    # Encoder for continuous features (price, rating)
    price_encoder = FractionalPowerEncoder(
        model,
        min_val=0.0,
        max_val=1000.0,
        bandwidth=0.1,
        seed=10
    )

    rating_encoder = FractionalPowerEncoder(
        model,
        min_val=1.0,
        max_val=5.0,
        bandwidth=0.1,
        seed=11
    )

    # Encoder for discrete features (brand tier)
    # Low=1, Medium=2, Premium=3
    tier_encoder = LevelEncoder(
        model,
        min_val=1,
        max_val=3,
        n_levels=3,
        seed=12
    )

    print("\nEncoders created:")
    print("  - Price: $0-$1000 (continuous)")
    print("  - Rating: 1-5 stars (continuous)")
    print("  - Brand tier: Low/Medium/Premium (discrete)")

**Why multiple encoders?**

Different feature types need different encodings:

* **Continuous** (price, rating): FractionalPowerEncoder for smooth similarity
* **Categorical** (brand, color): Random hypervectors for discrete choices
* **Ordinal** (tier levels): LevelEncoder for discrete bins

Step 3: Define Product Catalog
-------------------------------

Create a sample product catalog with multiple features:

.. code-block:: python

    # Product catalog
    # Each product: (id, name, price, category, rating, brand_tier, features)
    products = [
        # Electronics
        {
            'id': 'laptop_1',
            'name': 'UltraBook Pro',
            'price': 899.99,
            'category': 'electronics',
            'rating': 4.5,
            'brand_tier': 3,  # Premium
            'features': ['lightweight', 'touchscreen', 'ssd']
        },
        {
            'id': 'laptop_2',
            'name': 'Budget Laptop',
            'price': 399.99,
            'category': 'electronics',
            'rating': 3.8,
            'brand_tier': 1,  # Low
            'features': ['budget', 'basic', 'hdd']
        },
        {
            'id': 'laptop_3',
            'name': 'Gaming Laptop',
            'price': 1299.99,
            'category': 'electronics',
            'rating': 4.7,
            'brand_tier': 3,  # Premium
            'features': ['gaming', 'powerful', 'rgb', 'ssd']
        },
        {
            'id': 'tablet_1',
            'name': 'Pro Tablet',
            'price': 799.99,
            'category': 'electronics',
            'rating': 4.6,
            'brand_tier': 3,  # Premium
            'features': ['lightweight', 'touchscreen', 'stylus']
        },

        # Books
        {
            'id': 'book_1',
            'name': 'Python Programming',
            'price': 39.99,
            'category': 'books',
            'rating': 4.8,
            'brand_tier': 2,  # Medium
            'features': ['programming', 'python', 'beginner']
        },
        {
            'id': 'book_2',
            'name': 'Machine Learning Basics',
            'price': 49.99,
            'category': 'books',
            'rating': 4.7,
            'brand_tier': 2,  # Medium
            'features': ['ml', 'ai', 'intermediate']
        },
        {
            'id': 'book_3',
            'name': 'Data Science Guide',
            'price': 44.99,
            'category': 'books',
            'rating': 4.6,
            'brand_tier': 2,  # Medium
            'features': ['data-science', 'python', 'intermediate']
        },

        # Home & Kitchen
        {
            'id': 'blender_1',
            'name': 'Power Blender Pro',
            'price': 129.99,
            'category': 'home',
            'rating': 4.4,
            'brand_tier': 2,  # Medium
            'features': ['powerful', 'durable', 'easy-clean']
        },
        {
            'id': 'coffee_1',
            'name': 'Premium Coffee Maker',
            'price': 199.99,
            'category': 'home',
            'rating': 4.5,
            'brand_tier': 3,  # Premium
            'features': ['programmable', 'premium', 'thermal']
        },
        {
            'id': 'coffee_2',
            'name': 'Basic Coffee Maker',
            'price': 29.99,
            'category': 'home',
            'rating': 3.9,
            'brand_tier': 1,  # Low
            'features': ['budget', 'basic', 'compact']
        },
    ]

    print(f"\nProduct catalog: {len(products)} items")
    print(f"  Categories: {len(set(p['category'] for p in products))}")
    print(f"  Price range: ${min(p['price'] for p in products):.2f} - "
          f"${max(p['price'] for p in products):.2f}")

**Catalog structure:**

Each product has:

* **Numerical features**: price, rating
* **Categorical features**: category, features (tags)
* **Ordinal features**: brand_tier (quality level)

Step 4: Create Role Hypervectors
---------------------------------

Create hypervectors to represent feature "roles":

.. code-block:: python

    # Role hypervectors for binding
    # Each role labels what a value represents
    PRICE = model.random(seed=100)
    CATEGORY = model.random(seed=101)
    RATING = model.random(seed=102)
    BRAND_TIER = model.random(seed=103)
    FEATURE = model.random(seed=104)

    print("\nRole hypervectors created:")
    print("  PRICE, CATEGORY, RATING, BRAND_TIER, FEATURE")

**What are role hypervectors?**

Role hypervectors label what each feature represents. Think of them as "keys" in a key-value store:

* ``PRICE ⊗ <encoded_price>`` = "this is the price"
* ``CATEGORY ⊗ <category_hv>`` = "this is the category"

Binding different values to the same role makes them comparable.

Step 5: Create Category and Feature Hypervectors
-------------------------------------------------

Assign random hypervectors to categories and features:

.. code-block:: python

    # Extract all unique categories and features
    all_categories = set(p['category'] for p in products)
    all_features = set()
    for p in products:
        all_features.update(p['features'])

    # Create hypervectors for categories
    category_hvs = {
        cat: model.random(seed=hash(cat) % 100000)
        for cat in all_categories
    }

    # Create hypervectors for features
    feature_hvs = {
        feat: model.random(seed=hash(feat) % 100000)
        for feat in all_features
    }

    print(f"\nSymbolic hypervectors:")
    print(f"  Categories: {len(category_hvs)}")
    print(f"  Features: {len(feature_hvs)}")
    print(f"\nCategories: {list(category_hvs.keys())}")
    print(f"Feature tags (sample): {list(all_features)[:10]}")

**Hash-based seeding:**

Using ``hash(name)`` ensures:

* Same name → same hypervector (across sessions)
* Different names → different hypervectors
* Reproducible but pseudo-random

Step 6: Encode Product Representations
---------------------------------------

Combine all features into a single hypervector per product:

.. code-block:: python

    def encode_product(product):
        """Encode a product as a hypervector."""

        # Encode numerical features
        price_hv = model.bind(PRICE, price_encoder.encode(product['price']))
        rating_hv = model.bind(RATING, rating_encoder.encode(product['rating']))
        tier_hv = model.bind(BRAND_TIER, tier_encoder.encode(product['brand_tier']))

        # Encode categorical features
        category_hv = model.bind(CATEGORY, category_hvs[product['category']])

        # Encode feature tags (bundle all features)
        feature_tag_hvs = [feature_hvs[f] for f in product['features']]
        if feature_tag_hvs:
            features_hv = model.bind(FEATURE, model.bundle(feature_tag_hvs))
        else:
            features_hv = model.zero()

        # Bundle all components together
        product_hv = model.bundle([
            price_hv,
            rating_hv,
            tier_hv,
            category_hv,
            features_hv
        ])

        return product_hv

    # Encode all products
    product_hvs = {}
    for product in products:
        product_hvs[product['id']] = encode_product(product)

    print(f"\nEncoded {len(product_hvs)} products")

    # Show example
    sample_product = products[0]
    sample_hv = product_hvs[sample_product['id']]
    print(f"\nExample: {sample_product['name']}")
    print(f"  Price: ${sample_product['price']}")
    print(f"  Category: {sample_product['category']}")
    print(f"  Rating: {sample_product['rating']}")
    print(f"  Tier: {sample_product['brand_tier']}")
    print(f"  Features: {sample_product['features']}")
    print(f"  Encoded HV shape: {sample_hv.shape}")

**Encoding structure:**

.. code-block:: text

    Product HV = Bundle(
        PRICE ⊗ encode(price),
        CATEGORY ⊗ category_hv,
        RATING ⊗ encode(rating),
        BRAND_TIER ⊗ encode(tier),
        FEATURE ⊗ Bundle(feature_hvs)
    )

This creates a distributed representation where all features contribute equally.

Step 7: Build Recommendation Index
-----------------------------------

Use ItemStore for efficient similarity search:

.. code-block:: python

    # Create item store for fast retrieval
    store = ItemStore(model)

    # Add all products
    for product in products:
        product_id = product['id']
        product_hv = product_hvs[product_id]
        store.add(product_id, product_hv)

    print(f"\nItemStore built with {len(store)} products")
    print("Ready for similarity queries!")

**What is ItemStore?**

ItemStore is a vector database optimized for HDC:

* Stores hypervector → item_id mapping
* Fast k-nearest neighbor search
* Returns similarity scores
* Handles cleanup and thresholding

Step 8: Find Similar Products
------------------------------

Implement the core recommendation function:

.. code-block:: python

    def recommend_similar(product_id, k=3):
        """Find k most similar products."""
        # Get product hypervector
        query_hv = product_hvs[product_id]

        # Find similar items
        results = store.query(query_hv, k=k+1)  # +1 to exclude self

        # Filter out the query product itself
        recommendations = [
            (item_id, sim) for item_id, sim in results
            if item_id != product_id
        ][:k]

        return recommendations

    # Test recommendations
    print("\n" + "=" * 60)
    print("Product Recommendations")
    print("=" * 60)

    # Find similar products for UltraBook Pro
    query_product = products[0]  # UltraBook Pro
    print(f"\nQuery: {query_product['name']}")
    print(f"  Price: ${query_product['price']}")
    print(f"  Category: {query_product['category']}")
    print(f"  Features: {query_product['features']}")

    recommendations = recommend_similar(query_product['id'], k=3)

    print(f"\nTop {len(recommendations)} similar products:")
    for i, (rec_id, similarity) in enumerate(recommendations, 1):
        # Find product details
        rec_product = next(p for p in products if p['id'] == rec_id)
        print(f"\n{i}. {rec_product['name']} (similarity: {similarity:.3f})")
        print(f"   Price: ${rec_product['price']}")
        print(f"   Category: {rec_product['category']}")
        print(f"   Rating: {rec_product['rating']}")
        print(f"   Features: {rec_product['features']}")

**Why is this similar?**

Similarity is based on:

* Similar price range
* Same category
* Similar ratings
* Overlapping features
* Same brand tier

Step 9: Category-Specific Recommendations
------------------------------------------

Filter recommendations by category:

.. code-block:: python

    def recommend_in_category(product_id, category, k=3):
        """Recommend similar products within a category."""
        query_hv = product_hvs[product_id]
        results = store.query(query_hv, k=len(products))

        # Filter by category
        recommendations = []
        for item_id, sim in results:
            if item_id == product_id:
                continue  # Skip self

            item_product = next(p for p in products if p['id'] == item_id)
            if item_product['category'] == category:
                recommendations.append((item_id, sim))

            if len(recommendations) >= k:
                break

        return recommendations

    # Test
    print("\n" + "=" * 60)
    print("Category-Specific Recommendations")
    print("=" * 60)

    laptop = next(p for p in products if 'laptop' in p['id'].lower())
    print(f"\nQuery: {laptop['name']} (electronics)")

    recs = recommend_in_category(laptop['id'], 'electronics', k=3)
    print(f"\nRecommended electronics:")
    for i, (rec_id, sim) in enumerate(recs, 1):
        rec = next(p for p in products if p['id'] == rec_id)
        print(f"  {i}. {rec['name']} (similarity: {sim:.3f})")

**Use cases:**

* "More laptops like this"
* "Similar books"
* "Other electronics"

Step 10: Feature-Based Search
------------------------------

Find products with specific features:

.. code-block:: python

    def find_by_features(desired_features, k=5):
        """Find products matching desired features."""

        # Create query from features only
        feature_tag_hvs = [feature_hvs[f] for f in desired_features
                           if f in feature_hvs]

        if not feature_tag_hvs:
            return []

        query_hv = model.bind(FEATURE, model.bundle(feature_tag_hvs))

        # Search
        results = store.query(query_hv, k=k)
        return results

    # Test
    print("\n" + "=" * 60)
    print("Feature-Based Search")
    print("=" * 60)

    desired = ['lightweight', 'touchscreen']
    print(f"\nSearching for products with: {desired}")

    results = find_by_features(desired, k=3)
    print(f"\nTop {len(results)} matches:")
    for i, (item_id, sim) in enumerate(results, 1):
        item = next(p for p in products if p['id'] == item_id)
        print(f"\n{i}. {item['name']} (similarity: {sim:.3f})")
        print(f"   Features: {item['features']}")
        # Check which desired features are present
        matching = [f for f in desired if f in item['features']]
        print(f"   Matching: {matching}")

**Applications:**

* "Find lightweight touchscreen devices"
* "Show me budget gaming laptops"
* "Premium coffee makers with thermal carafes"

Step 11: Price Range Filtering
-------------------------------

Recommend within a price range:

.. code-block:: python

    def recommend_in_price_range(product_id, min_price, max_price, k=3):
        """Recommend similar products in price range."""
        query_hv = product_hvs[product_id]
        results = store.query(query_hv, k=len(products))

        # Filter by price
        recommendations = []
        for item_id, sim in results:
            if item_id == product_id:
                continue

            item = next(p for p in products if p['id'] == item_id)
            if min_price <= item['price'] <= max_price:
                recommendations.append((item_id, sim))

            if len(recommendations) >= k:
                break

        return recommendations

    # Test
    print("\n" + "=" * 60)
    print("Price Range Recommendations")
    print("=" * 60)

    query_prod = products[0]  # UltraBook Pro ($899)
    print(f"\nQuery: {query_prod['name']} (${query_prod['price']})")
    print(f"Finding similar products in $500-$1000 range:")

    recs = recommend_in_price_range(query_prod['id'], 500, 1000, k=3)
    for i, (rec_id, sim) in enumerate(recs, 1):
        rec = next(p for p in products if p['id'] == rec_id)
        print(f"  {i}. {rec['name']}: ${rec['price']} (sim: {sim:.3f})")

Step 12: Collaborative Patterns
--------------------------------

Implement "customers who bought X also bought Y":

.. code-block:: python

    # Simulate purchase history
    purchase_history = [
        ('user_1', ['laptop_1', 'book_1']),
        ('user_2', ['laptop_1', 'book_2', 'tablet_1']),
        ('user_3', ['laptop_2', 'book_1']),
        ('user_4', ['laptop_3', 'book_2']),
        ('user_5', ['book_1', 'book_3']),
        ('user_6', ['coffee_1', 'blender_1']),
    ]

    def customers_also_bought(product_id, k=3):
        """Find products often bought together."""

        # Find users who bought this product
        users_who_bought = [
            user_id for user_id, items in purchase_history
            if product_id in items
        ]

        # Collect all other products they bought
        co_purchased = {}
        for user_id in users_who_bought:
            items = next(items for uid, items in purchase_history
                        if uid == user_id)
            for item_id in items:
                if item_id != product_id:
                    co_purchased[item_id] = co_purchased.get(item_id, 0) + 1

        # Sort by frequency
        sorted_items = sorted(co_purchased.items(),
                             key=lambda x: x[1], reverse=True)

        # Get similarities
        results = []
        query_hv = product_hvs[product_id]

        for item_id, count in sorted_items[:k]:
            item_hv = product_hvs[item_id]
            sim = float(model.similarity(query_hv, item_hv))
            results.append((item_id, sim, count))

        return results

    # Test
    print("\n" + "=" * 60)
    print("Customers Also Bought")
    print("=" * 60)

    laptop = products[0]  # UltraBook Pro
    print(f"\nCustomers who bought '{laptop['name']}' also bought:")

    also_bought = customers_also_bought(laptop['id'], k=3)
    for i, (item_id, sim, count) in enumerate(also_bought, 1):
        item = next(p for p in products if p['id'] == item_id)
        print(f"\n{i}. {item['name']}")
        print(f"   Purchased together: {count} times")
        print(f"   Content similarity: {sim:.3f}")

**Hybrid approach:**

Combining:

* **Collaborative filtering**: Purchase frequency
* **Content similarity**: Feature similarity

Provides better recommendations than either alone.

Step 13: Evaluation Metrics
----------------------------

Measure recommendation quality:

.. code-block:: python

    def evaluate_recommendations(test_cases):
        """
        Evaluate recommendation quality.

        test_cases: list of (query_id, expected_similar_ids)
        """
        metrics = {
            'precision': [],
            'recall': [],
            'avg_similarity': []
        }

        for query_id, expected in test_cases:
            # Get recommendations
            recs = recommend_similar(query_id, k=len(expected))
            rec_ids = [item_id for item_id, _ in recs]

            # Precision: How many recommended items are relevant?
            relevant_count = sum(1 for rid in rec_ids if rid in expected)
            precision = relevant_count / len(rec_ids) if rec_ids else 0

            # Recall: How many relevant items were recommended?
            recall = relevant_count / len(expected) if expected else 0

            # Average similarity
            avg_sim = np.mean([sim for _, sim in recs]) if recs else 0

            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['avg_similarity'].append(avg_sim)

        # Aggregate
        return {
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'avg_similarity': np.mean(metrics['avg_similarity'])
        }

    # Test cases: (query, expected similar items)
    test_cases = [
        ('laptop_1', ['laptop_3', 'tablet_1']),  # Similar premium electronics
        ('book_1', ['book_2', 'book_3']),        # Similar books
        ('coffee_1', ['blender_1', 'coffee_2']), # Similar home items
    ]

    metrics = evaluate_recommendations(test_cases)

    print("\n" + "=" * 60)
    print("Recommendation Quality Metrics")
    print("=" * 60)
    print(f"\nPrecision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"Avg Similarity: {metrics['avg_similarity']:.3f}")

**Metrics explained:**

* **Precision**: Fraction of recommendations that are relevant
* **Recall**: Fraction of relevant items that were recommended
* **Avg Similarity**: Mean similarity score (higher = better matches)

Step 14: Scaling to Large Catalogs
-----------------------------------

Optimize for production scale:

.. code-block:: python

    # For large catalogs (10,000+ items):

    # 1. Use higher dimensions
    model_large = VSA.create('FHRR', dim=20000)

    # 2. Pre-compute and cache product hypervectors
    # Save to disk for fast loading
    import pickle

    def save_catalog(product_hvs, filename='product_hvs.pkl'):
        """Save encoded products to disk."""
        with open(filename, 'wb') as f:
            pickle.dump(product_hvs, f)

    def load_catalog(filename='product_hvs.pkl'):
        """Load encoded products from disk."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    # 3. Use PyTorch backend for GPU acceleration
    model_gpu = VSA.create('FHRR', dim=10000, backend='pytorch')

    # 4. Batch similarity computation
    def batch_recommend(query_ids, k=5):
        """Recommend for multiple queries at once."""
        results = {}
        for query_id in query_ids:
            results[query_id] = recommend_similar(query_id, k=k)
        return results

    print("\n" + "=" * 60)
    print("Scaling Strategies")
    print("=" * 60)
    print("\nFor production deployment:")
    print("  1. Use 20,000+ dimensions for large catalogs")
    print("  2. Cache encoded product hypervectors")
    print("  3. Use GPU backend for fast similarity")
    print("  4. Batch process recommendations")
    print("  5. Use approximate nearest neighbors for very large catalogs")

**Performance tips:**

* **10-100 items**: NumPy backend is fine
* **100-10K items**: Consider PyTorch backend
* **10K+ items**: Use GPU + approximate nearest neighbors (FAISS, Annoy)

Complete Recommender System
----------------------------

Here's the full system in one place:

.. code-block:: python

    import numpy as np
    from holovec import VSA
    from holovec.encoders import FractionalPowerEncoder, LevelEncoder
    from holovec.retrieval import ItemStore

    # Setup
    np.random.seed(42)
    model = VSA.create('FHRR', dim=10000, seed=42)

    # Encoders
    price_encoder = FractionalPowerEncoder(model, 0, 1000, 0.1)
    rating_encoder = FractionalPowerEncoder(model, 1, 5, 0.1)
    tier_encoder = LevelEncoder(model, 1, 3, 3)

    # Role hypervectors
    PRICE = model.random(seed=100)
    CATEGORY = model.random(seed=101)
    RATING = model.random(seed=102)
    BRAND_TIER = model.random(seed=103)
    FEATURE = model.random(seed=104)

    # Product encoding
    def encode_product(product, category_hvs, feature_hvs):
        price_hv = model.bind(PRICE, price_encoder.encode(product['price']))
        rating_hv = model.bind(RATING, rating_encoder.encode(product['rating']))
        tier_hv = model.bind(BRAND_TIER, tier_encoder.encode(product['brand_tier']))
        category_hv = model.bind(CATEGORY, category_hvs[product['category']])

        feature_tag_hvs = [feature_hvs[f] for f in product['features']]
        features_hv = model.bind(FEATURE, model.bundle(feature_tag_hvs))

        return model.bundle([price_hv, rating_hv, tier_hv,
                           category_hv, features_hv])

    # Recommendation
    def recommend_similar(product_id, product_hvs, store, k=3):
        query_hv = product_hvs[product_id]
        results = store.query(query_hv, k=k+1)
        return [(id, sim) for id, sim in results if id != product_id][:k]

    # Use with your product catalog!

Best Practices Summary
----------------------

**Encoding:**

* Use FractionalPowerEncoder for continuous features (price, rating)
* Use LevelEncoder for ordinal features (quality tiers)
* Use random hypervectors for categorical features
* Bind each feature to a role hypervector
* Bundle all features together

**Model Selection:**

* FHRR for smooth similarity on numerical features
* 10,000 dimensions for small catalogs (<1000 items)
* 20,000+ dimensions for large catalogs (>1000 items)

**Performance:**

* Cache encoded product hypervectors
* Use ItemStore for efficient k-NN search
* Consider GPU backend for large catalogs
* Pre-compute for batch recommendations

**Quality:**

* Balance feature importance with careful encoding
* Test with held-out items
* Monitor precision/recall metrics
* Combine with collaborative filtering

Common Extensions
-----------------

**1. User Profiles:**

.. code-block:: python

    def create_user_profile(purchase_history):
        """Build user preference vector from history."""
        purchased_hvs = [product_hvs[item_id]
                        for item_id in purchase_history]
        return model.bundle(purchased_hvs)

    def recommend_for_user(user_profile_hv, k=5):
        """Recommend based on user preferences."""
        return store.query(user_profile_hv, k=k)

**2. Multi-Criteria Filtering:**

.. code-block:: python

    def recommend_multi_criteria(product_id, filters, k=3):
        """Filter by multiple criteria."""
        recs = recommend_similar(product_id, k=len(products))

        filtered = []
        for item_id, sim in recs:
            item = get_product(item_id)

            # Check all filters
            if all(check_filter(item, f) for f in filters):
                filtered.append((item_id, sim))

            if len(filtered) >= k:
                break

        return filtered

**3. Temporal Recommendations:**

.. code-block:: python

    def trending_recommendations(time_window, k=5):
        """Recommend based on recent popularity."""
        # Combine content similarity with recency
        recent_purchases = get_recent_purchases(time_window)
        popular_items = Counter(recent_purchases).most_common(k)
        return popular_items

**4. Diversity:**

.. code-block:: python

    def diverse_recommendations(product_id, k=5, diversity_weight=0.3):
        """Recommend diverse set of similar items."""
        results = recommend_similar(product_id, k=k*2)

        diverse_set = []
        for item_id, sim in results:
            # Check diversity with existing recommendations
            if is_diverse(item_id, diverse_set, diversity_weight):
                diverse_set.append(item_id)

            if len(diverse_set) >= k:
                break

        return diverse_set

Next Steps
----------

**Explore more:**

* :doc:`../examples/26_retrieval_basics` - ItemStore and retrieval
* :doc:`../user-guide/encoding-data` - Advanced encoding techniques
* :doc:`../examples/25_app_integration_patterns` - Integration patterns

**Try these datasets:**

* Amazon product reviews
* MovieLens ratings
* E-commerce catalogs
* Music recommendations

**Advanced topics:**

* Hybrid collaborative + content filtering
* Real-time recommendation updates
* A/B testing recommendation strategies
* Explainable recommendations

Conclusion
----------

You've built a complete content-based recommender system!

**Key takeaways:**

* HDC naturally combines multiple feature types
* Binding and bundling create rich representations
* No training needed - works immediately
* Easy to add new items (cold-start solution)
* Interpretable similarity scores

**Advantages for recommendations:**

* Handles multi-modal features (price, category, tags)
* Fast similarity search with ItemStore
* Easy to filter and combine criteria
* Works with sparse data
* Explainable results

Apply these patterns to build recommendations for your products, content, or services!