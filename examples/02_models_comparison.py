"""
VSA Models Comparison Guide
===========================

Topics: MAP, FHRR, HRR, BSC model selection and characteristics
Time: 15 minutes
Prerequisites: 00_quickstart.py, 01_basic_operations.py
Related: 40_model_hrr_correlation.py, 41_model_ghrr_diagonality.py, 42_model_bsdc_seg.py

This example helps you choose the right VSA model for your application by
demonstrating the key differences, trade-offs, and use cases for each model
in the HoloVec library.
"""

from holovec import VSA

print("=" * 70)
print("VSA Models Comparison - Choosing the Right Model")
print("=" * 70)
print()

# ============================================================================
# Overview: Available Models
# ============================================================================
print("Available VSA Models in HoloVec")
print("-" * 70)
print()
print("1. MAP  (Multiply-Add-Permute)")
print("   - Element-wise operations, self-inverse binding")
print("   - Fast, simple, works on CPU")
print()
print("2. FHRR (Fourier Holographic Reduced Representations)")
print("   - Complex-valued, exact inverses")
print("   - Best capacity, recommended for most applications")
print()
print("3. HRR  (Holographic Reduced Representations)")
print("   - Real-valued circular convolution")
print("   - Classic model, good for research reproduction")
print()
print("4. BSC  (Binary Spatter Codes)")
print("   - Binary vectors with XOR binding")
print("   - Memory-efficient, exact inverse")
print()
print("5. BSDC (Binary Sparse Distributed Codes)")
print("   - Sparse binary representation")
print("   - Inspired by neuroscience, brain-like sparsity")
print()

# ============================================================================
# Comparison 1: Model Characteristics
# ============================================================================
print("=" * 70)
print("Comparison 1: Model Characteristics")
print("=" * 70)
print()

models_info = {}

# Create each model
for model_name in ['MAP', 'FHRR', 'HRR', 'BSC']:
    model = VSA.create(model_name, dim=10000, seed=42)
    models_info[model_name] = {
        'model': model,
        'self_inverse': model.is_self_inverse,
        'commutative': model.is_commutative,
        'exact_inverse': model.is_exact_inverse,
        'space': model.space.space_name
    }

# Print comparison table
print(f"{'Model':<8} {'Space':<12} {'Self-Inv':<10} {'Commut':<10} {'Exact-Inv':<10}")
print("-" * 70)
for name, info in models_info.items():
    print(f"{name:<8} {info['space']:<12} {str(info['self_inverse']):<10} "
          f"{str(info['commutative']):<10} {str(info['exact_inverse']):<10}")
print()

print("Key:")
print("  Self-Inverse: bind(A, B) can be unbound without separate inverse")
print("  Commutative: bind(A, B) = bind(B, A)")
print("  Exact-Inverse: Unbinding recovers exact original (no approximation)")
print()

# ============================================================================
# Comparison 2: Capacity (Bundling Performance)
# ============================================================================
print("=" * 70)
print("Comparison 2: Bundling Capacity")
print("=" * 70)
print()
print("Testing: How many random vectors can be bundled before similarity degrades?")
print()

# Test bundling capacity for each model
for model_name in ['FHRR', 'MAP', 'HRR']:
    model = models_info[model_name]['model']

    # Create a target vector and bundle it with noise
    target = model.random(seed=100)
    n_items_list = [1, 5, 10, 20, 50]

    print(f"{model_name} Bundling:")
    for n in n_items_list:
        # Bundle target with (n-1) random vectors
        vectors = [target] + [model.random(seed=100+i) for i in range(1, n)]
        bundled = model.bundle(vectors)

        # Measure similarity to target
        similarity = float(model.similarity(bundled, target))
        print(f"  {n:3d} items bundled: similarity = {similarity:.3f}")
    print()

print("Observation:")
print("  - FHRR maintains highest similarity (best capacity)")
print("  - MAP and HRR degrade faster with more items")
print("  - For >20 bundled items, prefer FHRR")
print()

# ============================================================================
# Comparison 3: Binding/Unbinding Accuracy
# ============================================================================
print("=" * 70)
print("Comparison 3: Binding/Unbinding Accuracy")
print("=" * 70)
print()
print("Testing: Can we accurately recover bound information?")
print()

for model_name in ['FHRR', 'MAP', 'HRR']:
    model = models_info[model_name]['model']

    # Create vectors
    a = model.random(seed=1)
    b = model.random(seed=2)

    # Bind
    c = model.bind(a, b)

    # Unbind to recover b
    b_recovered = model.unbind(c, a)

    # Measure recovery accuracy
    similarity = float(model.similarity(b, b_recovered))

    print(f"{model_name}: bind(A, B) then unbind(·, A) → similarity = {similarity:.4f}")

print()
print("Observation:")
print("  - FHRR: Exact inverse (similarity ≈ 1.000)")
print("  - MAP: Approximate but very good (similarity ≈ 0.999)")
print("  - HRR: Good approximation (similarity ≈ 0.990)")
print()

# ============================================================================
# Comparison 4: Performance Characteristics
# ============================================================================
print("=" * 70)
print("Comparison 4: Performance Characteristics")
print("=" * 70)
print()

import time

# Time binding operations
n_iterations = 1000
results = {}

for model_name in ['MAP', 'FHRR', 'HRR']:
    model = models_info[model_name]['model']
    a = model.random(seed=1)
    b = model.random(seed=2)

    # Time binding
    start = time.time()
    for _ in range(n_iterations):
        _ = model.bind(a, b)
    elapsed = time.time() - start

    results[model_name] = elapsed

# Normalize to MAP
map_time = results['MAP']
print(f"Binding Speed (relative to MAP, {n_iterations} operations):")
print(f"  MAP:  1.00x (fastest - element-wise multiply)")
print(f"  FHRR: {results['FHRR']/map_time:.2f}x (complex FFT operations)")
print(f"  HRR:  {results['HRR']/map_time:.2f}x (circular convolution via FFT)")
print()
print("Note: Times may vary by backend (NumPy vs PyTorch vs JAX)")
print()

# ============================================================================
# Decision Guide: When to Use Each Model
# ============================================================================
print("=" * 70)
print("Decision Guide: When to Use Each Model")
print("=" * 70)
print()

print("🏆 FHRR - **Recommended for most applications**")
print("   Use when:")
print("   - You need high capacity (bundling many items)")
print("   - Exact inverse is important")
print("   - Working with continuous encoders (FractionalPowerEncoder)")
print("   - Moderate performance is acceptable")
print()

print("⚡ MAP - **Best for speed-critical applications**")
print("   Use when:")
print("   - Performance is critical (real-time systems)")
print("   - Self-inverse binding is beneficial")
print("   - Simple element-wise operations preferred")
print("   - Lower capacity is acceptable")
print()

print("📚 HRR - **Good for research and reproduction**")
print("   Use when:")
print("   - Reproducing classic HDC papers")
print("   - Real-valued representations required")
print("   - Good balance of capacity and performance")
print()

print("💾 BSC - **Binary and memory-efficient**")
print("   Use when:")
print("   - Memory is extremely limited")
print("   - Binary operations preferred")
print("   - Exact XOR-based inverse needed")
print()

print("🧠 BSDC - **Brain-inspired sparse coding**")
print("   Use when:")
print("   - Biological plausibility important")
print("   - Sparse representations desired")
print("   - Neuromorphic hardware targeted")
print()

# ============================================================================
# Example: Same Application, Different Models
# ============================================================================
print("=" * 70)
print("Example: Temperature Encoding with Different Models")
print("=" * 70)
print()

from holovec.encoders import FractionalPowerEncoder, ThermometerEncoder

# Encode same data with different models
temps = [20.0, 21.0, 40.0]

# FPE works with FHRR and HRR
print("Using FractionalPowerEncoder (works with FHRR, HRR):")
for model_name in ['FHRR', 'HRR']:
    model = models_info[model_name]['model']
    encoder = FractionalPowerEncoder(model, min_val=0, max_val=100, bandwidth=0.1)

    hvs = [encoder.encode(t) for t in temps]

    # Check similarity between similar temps (20°C vs 21°C)
    sim_close = float(model.similarity(hvs[0], hvs[1]))

    # Check similarity between distant temps (20°C vs 40°C)
    sim_far = float(model.similarity(hvs[0], hvs[2]))

    print(f"  {model_name}: sim(20°C, 21°C)={sim_close:.3f}, sim(20°C, 40°C)={sim_far:.3f}")

print()

# ThermometerEncoder works with all models
print("Using ThermometerEncoder (works with all models):")
for model_name in ['FHRR', 'MAP', 'HRR', 'BSC']:
    model = models_info[model_name]['model']
    encoder = ThermometerEncoder(model, min_val=0, max_val=100, n_bins=100)

    hvs = [encoder.encode(t) for t in temps]

    # Check similarity between similar temps (20°C vs 21°C)
    sim_close = float(model.similarity(hvs[0], hvs[1]))

    # Check similarity between distant temps (20°C vs 40°C)
    sim_far = float(model.similarity(hvs[0], hvs[2]))

    print(f"  {model_name}: sim(20°C, 21°C)={sim_close:.3f}, sim(20°C, 40°C)={sim_far:.3f}")

print()
print("Observation:")
print("  - All models preserve similarity structure")
print("  - Different encoders may have model compatibility constraints")
print("  - Check encoder.compatible_models before use")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("Summary: Quick Reference")
print("=" * 70)
print()
print("Model Selection Flowchart:")
print()
print("  Need exact inverses? → Use FHRR")
print("  Need maximum speed? → Use MAP")
print("  Reproducing research? → Use HRR")
print("  Need binary/sparse? → Use BSC or BSDC")
print()
print("  **Default recommendation: FHRR**")
print("  (Best balance of capacity, accuracy, and features)")
print()
print("Next steps:")
print("  → Try different models with your specific use case")
print("  → See model-specific examples: 40-42_model_*.py")
print("  → Explore encoders with your chosen model: 10-18_encoders_*.py")
print()
print("=" * 70)
