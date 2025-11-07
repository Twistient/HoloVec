"""
HoloVec Quickstart Guide
========================

Topics: Installation, basic workflow, encoding, binding, retrieval
Time: 5 minutes
Prerequisites: None
Related: 01_basic_operations.py, 02_models_comparison.py

This quickstart demonstrates the core HoloVec workflow in under 100 lines.
You'll encode data, compose representations, and retrieve information using
hyperdimensional computing - brain-inspired computing with 10,000-dimensional
vectors.
"""

from holovec import VSA

print("=" * 70)
print("HoloVec Quickstart - Hyperdimensional Computing in 5 Minutes")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create a VSA Model
# ============================================================================
print("Step 1: Creating a VSA model")
print("-" * 70)

# Create a FHRR model (Fourier Holographic Reduced Representations)
# FHRR provides exact inverses and works well with continuous encoders
model = VSA.create('FHRR', dim=10000, seed=42)

print(f"Model: {model.model_name}")
print(f"Dimension: {model.dimension}")
print(f"Backend: {model.backend.name}")
print()

# ============================================================================
# Step 2: Encode Symbolic Data
# ============================================================================
print("Step 2: Encoding symbolic data")
print("-" * 70)

# Create random hypervectors for symbols
# Each symbol gets a unique 10,000-dimensional vector
alice = model.random(seed=1)
bob = model.random(seed=2)
loves = model.random(seed=3)
hates = model.random(seed=4)

print("Encoded symbols:")
print(f"  alice → 10000-dim hypervector")
print(f"  bob   → 10000-dim hypervector")
print(f"  loves → 10000-dim hypervector")
print(f"  hates → 10000-dim hypervector")
print()

# ============================================================================
# Step 3: Compose Representations with Binding
# ============================================================================
print("Step 3: Composing structured representations")
print("-" * 70)

# Bind vectors to create structured representations
# Binding creates a new vector that is dissimilar to both operands
alice_loves_bob = model.bind(model.bind(alice, loves), bob)
alice_hates_bob = model.bind(model.bind(alice, hates), bob)

print("Created compositional structures:")
print(f"  'Alice loves Bob' → single hypervector")
print(f"  'Alice hates Bob' → single hypervector")
print()

# These two statements are completely different
similarity = model.similarity(alice_loves_bob, alice_hates_bob)
print(f"Similarity between statements: {similarity:.3f}")
print("  (Low similarity confirms they represent different facts)")
print()

# ============================================================================
# Step 4: Query and Retrieve Information
# ============================================================================
print("Step 4: Querying compositional structures")
print("-" * 70)

# Query: Who does Alice love?
# Unbind 'alice' and 'loves' from 'alice_loves_bob'
query_result = model.unbind(model.unbind(alice_loves_bob, alice), loves)

# Check similarity to answer
similarity_bob = model.similarity(query_result, bob)
similarity_alice = model.similarity(query_result, alice)

print("Query: Who does Alice love?")
print(f"  Similarity to 'bob':   {similarity_bob:.3f}  ← Answer!")
print(f"  Similarity to 'alice': {similarity_alice:.3f}")
print()

# ============================================================================
# Step 5: Encode Continuous Values
# ============================================================================
print("Step 5: Encoding continuous values")
print("-" * 70)

from holovec.encoders import FractionalPowerEncoder

# Create an encoder for temperatures (0-100°C)
temp_encoder = FractionalPowerEncoder(model, min_val=0, max_val=100, bandwidth=0.1)

# Encode temperatures
temp_25 = temp_encoder.encode(25.0)
temp_26 = temp_encoder.encode(26.0)
temp_50 = temp_encoder.encode(50.0)

print("Encoded temperatures:")
print(f"  25°C → hypervector")
print(f"  26°C → hypervector")
print(f"  50°C → hypervector")
print()

# Similar values have high similarity
sim_25_26 = model.similarity(temp_25, temp_26)
sim_25_50 = model.similarity(temp_25, temp_50)

print(f"Similarity (25°C, 26°C): {sim_25_26:.3f}  ← Close values, high similarity")
print(f"Similarity (25°C, 50°C): {sim_25_50:.3f}  ← Distant values, low similarity")
print()

# ============================================================================
# Step 6: Build Associative Memory
# ============================================================================
print("Step 6: Building associative memory")
print("-" * 70)

# Create a simple associative memory
# Bind objects to their properties
hot = model.random(seed=10)
cold = model.random(seed=11)

fire_is_hot = model.bind(model.random(seed=20), hot)  # fire → hot
ice_is_cold = model.bind(model.random(seed=21), cold)  # ice → cold

# Bundle related facts into knowledge base
knowledge = model.bundle([fire_is_hot, ice_is_cold])

print("Created knowledge base:")
print(f"  'fire is hot' + 'ice is cold' → single hypervector")
print()

# Query: What is fire?
fire = model.random(seed=20)
fire_property = model.unbind(knowledge, fire)

sim_hot = model.similarity(fire_property, hot)
sim_cold = model.similarity(fire_property, cold)

print("Query: What property does fire have?")
print(f"  Similarity to 'hot':  {sim_hot:.3f}  ← Answer!")
print(f"  Similarity to 'cold': {sim_cold:.3f}")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("Summary: What You've Learned")
print("=" * 70)
print()
print("✓ Created a 10,000-dimensional VSA model")
print("✓ Encoded symbolic data as hypervectors")
print("✓ Composed structured representations with binding")
print("✓ Retrieved information through unbinding")
print("✓ Encoded continuous values with similarity preservation")
print("✓ Built a simple associative memory")
print()
print("Next steps:")
print("  → 01_basic_operations.py - Deep dive into VSA operations")
print("  → 02_models_comparison.py - Compare different VSA models")
print("  → 10_encoders_scalar.py - Master continuous value encoding")
print()
print("Full documentation: https://docs.holovecai.com")
print("=" * 70)
