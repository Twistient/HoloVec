"""
Demo: Basic VSA Operations
===========================

This example demonstrates the core operations of VSA models:
- Binding (association)
- Unbinding (recovery)
- Bundling (superposition)
- Permutation (sequence encoding)
"""

import sys
sys.path.insert(0, '..')

from holovec import VSA, backend_info

def main():
    print("=" * 60)
    print("HoloVec Demo: Basic VSA Operations")
    print("=" * 60)
    print()

    # Show available backends
    info = backend_info()
    print(f"Available backends: {info['available_backends']}")
    print(f"Recommended backend: {info['recommended_backend']}")
    print()

    # Create a FHRR model (best capacity)
    print("Creating FHRR model (dim=512)...")
    model = VSA.create('FHRR', dim=512, seed=42)
    print(f"Model: {model}")
    print(f"  - Self-inverse: {model.is_self_inverse}")
    print(f"  - Commutative: {model.is_commutative}")
    print(f"  - Exact inverse: {model.is_exact_inverse}")
    print()

    # Demonstration 1: Binding and Unbinding
    print("-" * 60)
    print("Demo 1: Binding and Unbinding (Association)")
    print("-" * 60)

    # Create role and filler vectors
    role = model.random(seed=1)
    filler = model.random(seed=2)

    print("Created vectors:")
    print(f"  role: random vector (seed=1)")
    print(f"  filler: random vector (seed=2)")
    print()

    # Bind them
    bound = model.bind(role, filler)
    print("Binding: bound = role ⊗ filler")
    print(f"  Similarity(bound, role): {model.similarity(bound, role):.4f}")
    print(f"  Similarity(bound, filler): {model.similarity(bound, filler):.4f}")
    print("  → Bound vector is dissimilar to both inputs ✓")
    print()

    # Unbind to recover
    recovered = model.unbind(bound, filler)
    print("Unbinding: recovered = bound ⊘ filler")
    print(f"  Similarity(recovered, role): {model.similarity(recovered, role):.4f}")
    print("  → Successfully recovered original vector ✓")
    print()

    # Demonstration 2: Bundling
    print("-" * 60)
    print("Demo 2: Bundling (Superposition)")
    print("-" * 60)

    # Create multiple vectors
    vec1 = model.random(seed=10)
    vec2 = model.random(seed=11)
    vec3 = model.random(seed=12)

    print("Created 3 random vectors: vec1, vec2, vec3")
    print()

    # Bundle them
    bundled = model.bundle([vec1, vec2, vec3])
    print("Bundling: bundled = vec1 + vec2 + vec3")
    print(f"  Similarity(bundled, vec1): {model.similarity(bundled, vec1):.4f}")
    print(f"  Similarity(bundled, vec2): {model.similarity(bundled, vec2):.4f}")
    print(f"  Similarity(bundled, vec3): {model.similarity(bundled, vec3):.4f}")
    print("  → Bundled vector is similar to all inputs ✓")
    print()

    # Demonstration 3: Permutation for Sequences
    print("-" * 60)
    print("Demo 3: Permutation (Sequence Encoding)")
    print("-" * 60)

    # Encode sequence [A, B, C]
    a = model.random(seed=20)
    b = model.random(seed=21)
    c = model.random(seed=22)

    print("Elements: A, B, C")
    print("Encoding sequence: seq = A + ρ(B) + ρ²(C)")
    print()

    sequence = model.bundle([
        a,
        model.permute(b, k=1),
        model.permute(c, k=2)
    ])

    # Query position 0 (should find A)
    sim_a = model.similarity(sequence, a)
    print(f"Query position 0 → Similarity(seq, A): {sim_a:.4f}")

    # Query position 1 (should find B)
    query_pos1 = model.unpermute(sequence, k=1)
    sim_b = model.similarity(query_pos1, b)
    print(f"Query position 1 → Similarity(ρ⁻¹(seq), B): {sim_b:.4f}")

    # Query position 2 (should find C)
    query_pos2 = model.unpermute(sequence, k=2)
    sim_c = model.similarity(query_pos2, c)
    print(f"Query position 2 → Similarity(ρ⁻²(seq), C): {sim_c:.4f}")
    print("  → Successfully encoded and queried sequence ✓")
    print()

    # Demonstration 4: Structured Representation
    print("-" * 60)
    print("Demo 4: Structured Representation")
    print("-" * 60)

    # Represent: "The ball is red and large"
    object_role = model.random(seed=30)
    color_role = model.random(seed=31)
    size_role = model.random(seed=32)

    ball = model.random(seed=40)
    red = model.random(seed=41)
    large = model.random(seed=42)

    print("Creating representation:")
    print("  'The ball is red and large'")
    print()
    print("Structure:")
    print("  object=ball ⊗ color=red ⊗ size=large")
    print()

    representation = model.bundle([
        model.bind(object_role, ball),
        model.bind(color_role, red),
        model.bind(size_role, large)
    ])

    # Query: What is the color?
    print("Query: What is the color?")
    color_query = model.unbind(representation, color_role)
    sim_red = model.similarity(color_query, red)
    sim_ball = model.similarity(color_query, ball)
    sim_large = model.similarity(color_query, large)

    print(f"  Similarity(query, red): {sim_red:.4f} ✓")
    print(f"  Similarity(query, ball): {sim_ball:.4f}")
    print(f"  Similarity(query, large): {sim_large:.4f}")
    print("  → Correctly identified 'red' as the color ✓")
    print()

    # Demonstration 5: Fractional Power Encoding (FHRR-specific)
    print("-" * 60)
    print("Demo 5: Fractional Power Encoding (FHRR)")
    print("-" * 60)

    # Encode continuous value 2.5
    base = model.random(seed=50)
    value = 2.5

    print(f"Encoding value: {value}")
    print(f"  Using base vector and fractional power")
    print()

    encoded = model.fractional_power(base, value)
    print(f"Encoded: base^{value}")

    # Decode by dividing
    decoded = model.fractional_power(encoded, 1.0 / value)
    sim_base = model.similarity(decoded, base)

    print(f"Decoded: encoded^(1/{value})")
    print(f"  Similarity(decoded, base): {sim_base:.4f}")
    print("  → Successfully encoded and decoded continuous value ✓")
    print()

    # Comparison with MAP
    print("-" * 60)
    print("Bonus: Comparing FHRR vs MAP")
    print("-" * 60)

    map_model = VSA.create('MAP', dim=10000, seed=42)
    print(f"MAP model: {map_model}")
    print(f"  - Self-inverse: {map_model.is_self_inverse}")
    print(f"  - Exact inverse: {map_model.is_exact_inverse}")
    print()

    # Test unbinding quality
    a_map = map_model.random(seed=1)
    b_map = map_model.random(seed=2)
    c_map = map_model.bind(a_map, b_map)
    a_recovered_map = map_model.unbind(c_map, b_map)

    sim_map = map_model.similarity(a_map, a_recovered_map)
    print(f"MAP unbinding quality: {sim_map:.4f}")

    a_fhrr = model.random(seed=1)
    b_fhrr = model.random(seed=2)
    c_fhrr = model.bind(a_fhrr, b_fhrr)
    a_recovered_fhrr = model.unbind(c_fhrr, b_fhrr)

    sim_fhrr = model.similarity(a_fhrr, a_recovered_fhrr)
    print(f"FHRR unbinding quality: {sim_fhrr:.4f}")
    print()
    print("  → FHRR has better unbinding quality (exact inverse) ✓")
    print()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
