"""
Working Memory with Cleanup Strategies
======================================

Topics: Working memory, cleanup, resonator, factorization, noisy retrieval
Time: 20 minutes
Prerequisites: 23_app_symbolic_reasoning.py, 26_retrieval_basics.py
Related: 27_cleanup_strategies.py, 28_factorization_methods.py

This example demonstrates working memory systems using hyperdimensional
computing, with a focus on cleanup strategies for retrieving information
from noisy bundled representations.

Key concepts:
- Working memory: Bundled representation of active information
- Cleanup: Recovering clean symbols from noisy hypervectors
- BruteForce cleanup: Exhaustive nearest-neighbor search
- Resonator cleanup: Iterative refinement using codebook
- Multi-factor unbinding: Decompose bundled representations
- Noise tolerance: Graceful degradation with increasing bundle size

Working memory in HDC mimics human working memory: limited capacity,
distributed representation, and content-addressable retrieval.
"""

from holovec import VSA
from holovec.retrieval import ItemStore, Codebook
from holovec.utils.cleanup import BruteForceCleanup, ResonatorCleanup

print("=" * 70)
print("Working Memory with Cleanup Strategies")
print("=" * 70)
print()

# Create model
model = VSA.create('FHRR', dim=10000, seed=42)

# ============================================================================
# Demo 1: Basic Working Memory - Bundled Active Items
# ============================================================================
print("=" * 70)
print("Demo 1: Basic Working Memory")
print("=" * 70)

# Simulate working memory with 5 active items
print("\nEncoding 5 active items in working memory:")

items = {}
items["book"] = model.random(seed=100)
items["pen"] = model.random(seed=101)
items["coffee"] = model.random(seed=102)
items["phone"] = model.random(seed=103)
items["keys"] = model.random(seed=104)

print("  Items: book, pen, coffee, phone, keys")

# Bundle into working memory
working_memory = model.bundle([items["book"], items["pen"], items["coffee"],
                                items["phone"], items["keys"]])

print("\nWorking memory: bundled all 5 items")

# Query: Is "book" in working memory?
print("\n" + "=" * 70)
print("Query: Check if items are in working memory")
print("=" * 70)

print("\nSimilarity to active items:")
for name, vec in items.items():
    sim = float(model.similarity(working_memory, vec))
    print(f"  {name:10s}: {sim:.3f}")

# Check items NOT in working memory
print("\nSimilarity to inactive items:")
laptop = model.random(seed=200)
mouse = model.random(seed=201)
print(f"  laptop:     {float(model.similarity(working_memory, laptop)):.3f}")
print(f"  mouse:      {float(model.similarity(working_memory, mouse)):.3f}")

print("\nKey observation:")
print("  - Active items have high similarity (~0.45)")
print("  - Inactive items have low similarity (~0)")
print("  - Working memory acts as distributed content-addressable store")

# ============================================================================
# Demo 2: Cleanup with BruteForce Strategy
# ============================================================================
print("\n" + "=" * 70)
print("Demo 2: BruteForce Cleanup Strategy")
print("=" * 70)

# Create noisy query by adding noise
print("\nSimulating noisy query:")
print("  Original: book")

# Add noise to book vector
noise = model.random(seed=999)
noisy_book = model.bundle([items["book"], noise])  # 50% book, 50% noise

print(f"  Similarity noisy_book → book: {float(model.similarity(noisy_book, items['book'])):.3f}")
print(f"  (Pure book would be 1.000, random would be ~0)")

# Create codebook for cleanup
codebook = Codebook(items, backend=model.backend)

# BruteForce cleanup: find nearest neighbor
print("\n" + "=" * 70)
print("Cleanup with BruteForce (nearest neighbor)")
print("=" * 70)

cleanup_bf = BruteForceCleanup()
labels, sims = cleanup_bf.factorize(noisy_book, items, model, n_factors=3)

print("\nTop 3 matches:")
for i, (label, sim) in enumerate(zip(labels, sims), 1):
    marker = "  ← Correct!" if label == "book" else ""
    print(f"  {i}. {label:10s}: {sim:.3f}{marker}")

print("\nKey observation:")
print("  - BruteForce finds nearest neighbor via exhaustive search")
print("  - Correct even with significant noise")
print("  - Fast for small codebooks, O(N) complexity")

# ============================================================================
# Demo 3: Resonator Cleanup Strategy
# ============================================================================
print("\n" + "=" * 70)
print("Demo 3: Resonator Cleanup Strategy")
print("=" * 70)

# Resonator uses iterative refinement
print("\nResonator cleanup (iterative refinement):")

cleanup_res = ResonatorCleanup()

# Create moderately noisy query
noise_moderate = model.random(seed=888)
noisy_pen = model.bundle([items["pen"], items["pen"], items["pen"], noise_moderate])  # 75% pen

print(f"  Starting similarity to pen: {float(model.similarity(noisy_pen, items['pen'])):.3f}")

labels_res, sims_res = cleanup_res.factorize(noisy_pen, items, model, n_factors=1,
                                              max_iterations=10, threshold=0.99)

print(f"\nResonator result:")
print(f"  Best match: {labels_res[0]} (similarity={sims_res[0]:.3f})")

print("\nKey observation:")
print("  - Resonator iteratively refines noisy input")
print("  - Uses codebook to project onto valid subspace")
print("  - More robust to noise than single nearest-neighbor")

# ============================================================================
# Demo 4: Multi-Factor Unbinding - Decomposing Bundled Representations
# ============================================================================
print("\n" + "=" * 70)
print("Demo 4: Multi-Factor Unbinding")
print("=" * 70)

# Create bundle of multiple items
print("\nBundling 3 items:")
item1 = items["book"]
item2 = items["coffee"]
item3 = items["phone"]

bundle = model.bundle([item1, item2, item3])

print("  Bundle: book ⊕ coffee ⊕ phone")

# Factorize to recover all items
print("\n" + "=" * 70)
print("Factorizing bundle to recover all items:")
print("=" * 70)

labels_fact, sims_fact = cleanup_bf.factorize(bundle, items, model, n_factors=5)

print("\nRecovered factors:")
for i, (label, sim) in enumerate(zip(labels_fact, sims_fact), 1):
    in_bundle = "✓" if label in ["book", "coffee", "phone"] else "✗"
    print(f"  {i}. {label:10s}: {sim:.3f}  [{in_bundle}]")

print("\nKey observation:")
print("  - Factorization recovers multiple bundled items")
print("  - Top factors are the original bundled items")
print("  - Similarity degrades but items still identifiable")

# ============================================================================
# Demo 5: Capacity Limits - How Many Items Can Working Memory Hold?
# ============================================================================
print("\n" + "=" * 70)
print("Demo 5: Working Memory Capacity Limits")
print("=" * 70)

# Test with increasing bundle sizes
print("\nTesting working memory capacity:")

# Create larger item set
large_items = {}
for i in range(20):
    large_items[f"item_{i}"] = model.random(seed=300 + i)

# Test different bundle sizes
bundle_sizes = [3, 5, 7, 10, 15]

print("\nRecovery accuracy vs. bundle size:")
print("Size | Top-1 | Top-3 | Top-5")
print("-" * 40)

for size in bundle_sizes:
    # Bundle first 'size' items
    bundled_items = [large_items[f"item_{i}"] for i in range(size)]
    bundle_test = model.bundle(bundled_items)

    # Factorize to recover
    recovered, _ = cleanup_bf.factorize(bundle_test, large_items, model, n_factors=5)

    # Count correct in top-k
    correct_labels = {f"item_{i}" for i in range(size)}
    top1_correct = 1 if recovered[0] in correct_labels else 0
    top3_correct = sum(1 for r in recovered[:3] if r in correct_labels) / min(3, size)
    top5_correct = sum(1 for r in recovered[:5] if r in correct_labels) / min(5, size)

    print(f"  {size:2d} | {top1_correct:5.2f} | {top3_correct:5.2f} | {top5_correct:5.2f}")

print("\nKey observation:")
print("  - Accuracy degrades with bundle size")
print("  - Working memory capacity: ~5-7 items (like human WM!)")
print("  - Distributed representation naturally limits capacity")

# ============================================================================
# Demo 6: Structured Working Memory - Role-Filler Binding
# ============================================================================
print("\n" + "=" * 70)
print("Demo 6: Structured Working Memory")
print("=" * 70)

# Working memory with structure: current task context
print("\nEncoding task context:")
print("  Task: 'Write email to Bob about meeting'")

# Define roles
task_type = model.random(seed=400)
recipient = model.random(seed=401)
topic = model.random(seed=402)

# Define fillers
write_email = model.random(seed=500)
bob_entity = model.random(seed=501)
meeting_topic = model.random(seed=502)

# Create structured working memory
wm_structure = model.bundle([
    model.bind(task_type, write_email),
    model.bind(recipient, bob_entity),
    model.bind(topic, meeting_topic)
])

print("\nRole-filler bindings in working memory:")
print("  task_type ⊗ write_email")
print("  recipient ⊗ bob")
print("  topic ⊗ meeting")

# Query roles
print("\n" + "=" * 70)
print("Querying working memory by role:")
print("=" * 70)

# What is the task type?
task_query = model.unbind(wm_structure, task_type)
print(f"\nWhat is the task?")
print(f"  Similarity to write_email: {float(model.similarity(task_query, write_email)):.3f}  ← Match")

# Who is the recipient?
recip_query = model.unbind(wm_structure, recipient)
print(f"\nWho is the recipient?")
print(f"  Similarity to bob: {float(model.similarity(recip_query, bob_entity)):.3f}  ← Match")

# What is the topic?
topic_query = model.unbind(wm_structure, topic)
print(f"\nWhat is the topic?")
print(f"  Similarity to meeting: {float(model.similarity(topic_query, meeting_topic)):.3f}  ← Match")

print("\nKey observation:")
print("  - Structured working memory supports role-based queries")
print("  - Combines bundling (multiple bindings) with binding (role-filler)")
print("  - Enables cognitive architectures with WM component")

# ============================================================================
# Demo 7: Working Memory with ItemStore
# ============================================================================
print("\n" + "=" * 70)
print("Demo 7: Working Memory with ItemStore")
print("=" * 70)

# Create item store for semantic memory (long-term)
print("\nBuilding semantic memory (long-term store):")

semantic_memory = ItemStore(model)
semantic_memory.add("alice", model.random(seed=600))
semantic_memory.add("bob", model.random(seed=601))
semantic_memory.add("charlie", model.random(seed=602))
semantic_memory.add("email", model.random(seed=603))
semantic_memory.add("meeting", model.random(seed=604))
semantic_memory.add("document", model.random(seed=605))

print("  Stored: alice, bob, charlie, email, meeting, document")

# Working memory holds current focus
print("\nWorking memory (current focus):")
wm_current = model.bundle([
    semantic_memory.codebook._items["bob"],
    semantic_memory.codebook._items["meeting"]
])

print("  Active: bob, meeting")

# Query: What's in working memory?
print("\n" + "=" * 70)
print("Retrieving from working memory:")
print("=" * 70)

# Use ItemStore to query working memory
results = semantic_memory.query(wm_current, k=6)

print("\nTop matches in semantic memory:")
for i, (label, sim) in enumerate(results, 1):
    in_wm = "✓" if label in ["bob", "meeting"] else " "
    print(f"  {i}. {label:10s}: {sim:.3f}  [{in_wm}]")

print("\nKey observation:")
print("  - ItemStore enables fast retrieval from semantic memory")
print("  - Working memory acts as query to semantic memory")
print("  - Models interaction between WM and long-term memory")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Working Memory Key Takeaways")
print("=" * 70)
print()
print("✓ Working memory: Bundled representation of active items")
print("✓ Content-addressable: Query by similarity, not address")
print("✓ Cleanup strategies: Recover clean symbols from noise")
print("✓ BruteForce: Fast O(N) nearest-neighbor search")
print("✓ Resonator: Iterative refinement for robustness")
print("✓ Multi-factor unbinding: Decompose bundled items")
print("✓ Capacity limits: ~5-7 items (mirrors human WM!)")
print()
print("Cleanup strategy comparison:")
print("  BruteForce:")
print("    - Fast for small codebooks (O(N))")
print("    - Single nearest-neighbor lookup")
print("    - Good for clean or moderately noisy queries")
print()
print("  Resonator:")
print("    - Iterative refinement (O(k*N), k iterations)")
print("    - Projects onto valid subspace using codebook")
print("    - Robust to higher noise levels")
print("    - Can recover from very corrupted inputs")
print()
print("Working memory applications:")
print("  - Cognitive architectures: Active information maintenance")
print("  - Attention mechanisms: Focus on relevant items")
print("  - Task context: Maintain current goals and parameters")
print("  - Short-term buffers: Temporary storage before consolidation")
print()
print("Next steps:")
print("  → 27_cleanup_strategies.py - Detailed cleanup comparison")
print("  → 28_factorization_methods.py - Advanced unbinding techniques")
print("  → 25_app_integration_patterns.py - Integrate WM in larger systems")
print()
print("=" * 70)
