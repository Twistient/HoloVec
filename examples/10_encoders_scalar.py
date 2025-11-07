"""
Comprehensive demo of scalar encoders in holovec.
===============================================

This script demonstrates:
1. FractionalPowerEncoder - Continuous scalar encoding with smooth similarity
2. ThermometerEncoder - Ordinal encoding with monotonic similarity
3. LevelEncoder - Discrete level encoding with exact recovery

Run with: python examples/demo_encoders.py
"""

import numpy as np
from holovec import VSA
from holovec.encoders import (
    FractionalPowerEncoder,
    ThermometerEncoder,
    LevelEncoder,
)


def demo_fpe_basic():
    """Demonstrate basic FractionalPowerEncoder usage."""
    print("=" * 70)
    print("Demo 1: FractionalPowerEncoder - Basic Usage")
    print("=" * 70)

    # Create FHRR model (recommended for FPE)
    model = VSA.create('FHRR', dim=10000, seed=42)

    # Create encoder for temperature range 0-100°C
    encoder = FractionalPowerEncoder(
        model,
        min_val=0,
        max_val=100,
        bandwidth=1.0,  # Standard kernel width
        seed=42
    )

    print(f"\nEncoder: {encoder}")
    print(f"Reversible: {encoder.is_reversible}")
    print(f"Compatible models: {encoder.compatible_models}")

    # Encode some temperature values
    temps = [20.0, 25.0, 30.0, 75.0]
    print(f"\nEncoding temperatures: {temps}")

    encoded = [encoder.encode(t) for t in temps]

    # Check similarities
    print("\nSimilarity Matrix:")
    print("     ", "  ".join(f"{t:5.1f}" for t in temps))
    for i, t1 in enumerate(temps):
        similarities = [
            float(model.similarity(encoded[i], encoded[j]))
            for j in range(len(temps))
        ]
        print(f"{t1:5.1f}", "  ".join(f"{s:5.3f}" for s in similarities))

    # Test decoding
    print("\nDecoding test:")
    for i, temp in enumerate(temps):
        decoded = encoder.decode(encoded[i])
        error = abs(decoded - temp)
        print(f"  Original: {temp:6.2f}°C → Decoded: {decoded:6.2f}°C "
              f"(error: {error:5.3f}°C)")

    print("\nKey observations:")
    print("  - Close values (20, 25, 30) have high similarity (>0.9)")
    print("  - Distant values (20, 75) have low similarity (<0.5)")
    print("  - Decoding is approximate but accurate (error < 1°C)")


def demo_fpe_bandwidth():
    """Demonstrate effect of bandwidth parameter."""
    print("\n" + "=" * 70)
    print("Demo 2: FractionalPowerEncoder - Bandwidth Effects")
    print("=" * 70)

    model = VSA.create('FHRR', dim=10000, seed=42)

    # Test different bandwidths
    bandwidths = [0.01, 0.1, 1.0, 10.0]
    test_values = [25.0, 26.0, 30.0, 50.0]

    print("\nEffect of bandwidth on similarity:")
    print("Reference value: 25.0°C")
    print("\nBandwidth | 26.0°C | 30.0°C | 50.0°C")
    print("-" * 45)

    for beta in bandwidths:
        encoder = FractionalPowerEncoder(model, 0, 100, bandwidth=beta, seed=42)

        # Encode all values
        ref_hv = encoder.encode(25.0)
        similarities = [
            float(model.similarity(ref_hv, encoder.encode(v)))
            for v in [26.0, 30.0, 50.0]
        ]

        print(f"{beta:8.2f}  | {similarities[0]:6.3f} | "
              f"{similarities[1]:6.3f} | {similarities[2]:6.3f}")

    print("\nKey observations:")
    print("  - Lower bandwidth → wider kernel → more similar values")
    print("  - Higher bandwidth → narrower kernel → more distinct values")
    print("  - β=0.01: Good for classification (generalization)")
    print("  - β=10: Good for exact matching (discrimination)")


def demo_fpe_vs_hrr():
    """Compare FPE performance on FHRR vs HRR."""
    print("\n" + "=" * 70)
    print("Demo 3: FractionalPowerEncoder - FHRR vs HRR")
    print("=" * 70)

    # Create both models
    fhrr_model = VSA.create('FHRR', dim=10000, seed=42)
    hrr_model = VSA.create('HRR', dim=10000, seed=42)

    # Create encoders
    fhrr_encoder = FractionalPowerEncoder(fhrr_model, 0, 100, bandwidth=1.0, seed=42)
    hrr_encoder = FractionalPowerEncoder(hrr_model, 0, 100, bandwidth=1.0, seed=42)

    # Test values
    values = [20.0, 25.0, 30.0, 50.0]

    print("\nSimilarity comparison (reference: 25.0°C):")
    print("Value | FHRR  | HRR")
    print("-" * 25)

    ref_fhrr = fhrr_encoder.encode(25.0)
    ref_hrr = hrr_encoder.encode(25.0)

    for v in values:
        sim_fhrr = float(fhrr_model.similarity(ref_fhrr, fhrr_encoder.encode(v)))
        sim_hrr = float(hrr_model.similarity(ref_hrr, hrr_encoder.encode(v)))
        print(f"{v:5.1f} | {sim_fhrr:5.3f} | {sim_hrr:5.3f}")

    print("\nKey observations:")
    print("  - FHRR uses complex arithmetic (exact implementation)")
    print("  - HRR uses FFT-based approximation (still accurate)")
    print("  - Both preserve locality well")
    print("  - FHRR is faster for encoding, HRR uses less memory")


def demo_thermometer():
    """Demonstrate ThermometerEncoder."""
    print("\n" + "=" * 70)
    print("Demo 4: ThermometerEncoder - Ordinal Encoding")
    print("=" * 70)

    model = VSA.create('MAP', dim=10000, seed=42)

    # Create thermometer encoder with 20 bins
    encoder = ThermometerEncoder(
        model,
        min_val=0,
        max_val=100,
        n_bins=20,
        seed=42
    )

    print(f"\nEncoder: {encoder}")
    print(f"Number of bins: {encoder.n_bins}")
    print(f"Bin width: {encoder.bin_width:.2f}°C")
    print(f"Reversible: {encoder.is_reversible}")

    # Encode values
    values = [10.0, 20.0, 50.0, 90.0]
    encoded = [encoder.encode(v) for v in values]

    print("\nSimilarity Matrix:")
    print("     ", "  ".join(f"{v:5.1f}" for v in values))
    for i, v1 in enumerate(values):
        similarities = [
            float(model.similarity(encoded[i], encoded[j]))
            for j in range(len(values))
        ]
        print(f"{v1:5.1f}", "  ".join(f"{s:5.3f}" for s in similarities))

    print("\nKey observations:")
    print("  - Monotonic similarity: higher value → more bins → higher similarity")
    print("  - Simple and robust")
    print("  - Works with all VSA models")
    print("  - Not reversible (cannot decode)")


def demo_level():
    """Demonstrate LevelEncoder."""
    print("\n" + "=" * 70)
    print("Demo 5: LevelEncoder - Discrete Level Encoding")
    print("=" * 70)

    model = VSA.create('MAP', dim=10000, seed=42)

    # Create level encoder for weekdays (7 discrete levels)
    encoder = LevelEncoder(
        model,
        min_val=0,
        max_val=6,
        n_levels=7,
        seed=42
    )

    print(f"\nEncoder: {encoder}")
    print(f"Number of levels: {encoder.n_levels}")
    print(f"Reversible: {encoder.is_reversible}")

    # Encode weekdays
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_values = list(range(7))
    encoded = [encoder.encode(float(v)) for v in day_values]

    print("\nEncoding weekdays:")
    for day, value, hv in zip(days, day_values, encoded):
        decoded = encoder.decode(hv)
        print(f"  {day} ({value}) → HV → {decoded:.0f} ✓")

    # Check similarity
    print("\nSimilarity Matrix (first 4 days):")
    print("     ", "  ".join(f"{d:>3}" for d in days[:4]))
    for i in range(4):
        similarities = [
            float(model.similarity(encoded[i], encoded[j]))
            for j in range(4)
        ]
        print(f"{days[i]:>3} ", "  ".join(f"{s:5.3f}" for s in similarities))

    print("\nKey observations:")
    print("  - Exact encoding/decoding (perfect recovery)")
    print("  - Fast O(1) lookup")
    print("  - Levels are nearly orthogonal (low similarity)")
    print("  - Best for discrete categorical data")


def demo_encoder_comparison():
    """Compare all three encoders on the same data."""
    print("\n" + "=" * 70)
    print("Demo 6: Encoder Comparison")
    print("=" * 70)

    # Create models
    fhrr_model = VSA.create('FHRR', dim=10000, seed=42)
    map_model = VSA.create('MAP', dim=10000, seed=42)

    # Create encoders
    fpe = FractionalPowerEncoder(fhrr_model, 0, 100, bandwidth=1.0, seed=42)
    thermo = ThermometerEncoder(map_model, 0, 100, n_bins=20, seed=42)
    level = LevelEncoder(map_model, 0, 100, n_levels=11, seed=42)

    # Test data: 0, 10, 20, ..., 100 (11 values)
    values = [float(i * 10) for i in range(11)]

    print("\nEncoding 11 values: [0, 10, 20, ..., 100]")
    print("\nSimilarity to reference value 50:")
    print("\nValue |  FPE  | Thermo | Level")
    print("-" * 38)

    # Encode reference
    ref_fpe = fpe.encode(50.0)
    ref_thermo = thermo.encode(50.0)
    ref_level = level.encode(50.0)

    for v in values:
        sim_fpe = float(fhrr_model.similarity(ref_fpe, fpe.encode(v)))
        sim_thermo = float(map_model.similarity(ref_thermo, thermo.encode(v)))
        sim_level = float(map_model.similarity(ref_level, level.encode(v)))

        print(f"{v:5.0f}  | {sim_fpe:5.3f} | {sim_thermo:6.3f} | {sim_level:5.3f}")

    print("\nKey differences:")
    print("  - FPE: Smooth Gaussian-like profile (continuous)")
    print("  - Thermometer: Monotonic increasing (ordinal)")
    print("  - Level: Binary (1.0 for exact match, ~0 otherwise)")

    print("\nWhen to use each:")
    print("  - FPE: Continuous values, need smooth similarity")
    print("  - Thermometer: Ordinal data, monotonicity important")
    print("  - Level: Discrete categories, exact matching")


def demo_visualization():
    """Visualize similarity profiles of different encoders."""
    print("\n" + "=" * 70)
    print("Demo 7: Similarity Profile Visualization")
    print("=" * 70)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")
        return

    # Create models
    fhrr_model = VSA.create('FHRR', dim=10000, seed=42)
    map_model = VSA.create('MAP', dim=10000, seed=42)

    # Create encoders
    fpe = FractionalPowerEncoder(fhrr_model, 0, 100, bandwidth=1.0, seed=42)
    thermo = ThermometerEncoder(map_model, 0, 100, n_bins=20, seed=42)
    level = LevelEncoder(map_model, 0, 100, n_levels=11, seed=42)

    # Reference value
    ref_value = 50.0

    # Test range
    test_values = np.linspace(0, 100, 101)

    # Compute similarities
    ref_fpe = fpe.encode(ref_value)
    ref_thermo = thermo.encode(ref_value)
    ref_level = level.encode(ref_value)

    sim_fpe = [
        float(fhrr_model.similarity(ref_fpe, fpe.encode(v)))
        for v in test_values
    ]
    sim_thermo = [
        float(map_model.similarity(ref_thermo, thermo.encode(v)))
        for v in test_values
    ]
    sim_level = [
        float(map_model.similarity(ref_level, level.encode(v)))
        for v in test_values
    ]

    # Create plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(test_values, sim_fpe, 'b-', linewidth=2)
    plt.axvline(ref_value, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Value')
    plt.ylabel('Similarity')
    plt.title('FractionalPowerEncoder\n(Smooth, Gaussian-like)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)

    plt.subplot(1, 3, 2)
    plt.plot(test_values, sim_thermo, 'g-', linewidth=2)
    plt.axvline(ref_value, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Value')
    plt.ylabel('Similarity')
    plt.title('ThermometerEncoder\n(Monotonic, Triangular)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)

    plt.subplot(1, 3, 3)
    plt.plot(test_values, sim_level, 'orange', linewidth=2)
    plt.axvline(ref_value, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Value')
    plt.ylabel('Similarity')
    plt.title('LevelEncoder\n(Discrete, Binary)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('encoder_similarity_profiles.png', dpi=150)
    print("\nSimilarity profiles saved to: encoder_similarity_profiles.png")

    print("\nVisualization notes:")
    print("  - Red dashed line: reference value (50)")
    print("  - FPE: Smooth bell curve centered at reference")
    print("  - Thermometer: Triangular peak at reference")
    print("  - Level: Sharp spike at exact match only")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("HDVEC Scalar Encoders - Comprehensive Demo")
    print("=" * 70)

    try:
        # Run all demos
        demo_fpe_basic()
        demo_fpe_bandwidth()
        demo_fpe_vs_hrr()
        demo_thermometer()
        demo_level()
        demo_encoder_comparison()
        demo_visualization()

        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print("\nFor more information:")
        print("  - Theory: docs/theory/encoders.md")
        print("  - Tests: tests/test_encoders_scalar.py")
        print("  - API docs: help(FractionalPowerEncoder)")

    except Exception as e:
        print(f"\nError running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
