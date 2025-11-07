"""
Image Pattern Recognition
=========================

Topics: Image classification, spatial encoding, pattern matching, computer vision
Time: 15 minutes
Prerequisites: 17_encoders_image.py, 16_encoders_vector.py
Related: 20_app_text_classification.py, 25_app_integration_patterns.py

This example demonstrates practical image pattern recognition using
hyperdimensional computing. Learn how to classify simple image patterns
using spatial encoding techniques.

Key concepts:
- Image encoding: Flatten pixels + VectorEncoder
- Pattern prototypes: Bundle examples per class
- Classification: Similarity-based matching
- Practical considerations: Real-world trade-offs

Image recognition with HDC is efficient and works well for simple patterns,
edge devices, and situations with limited training data.
"""

import numpy as np
from holovec import VSA
from holovec.encoders import VectorEncoder, FractionalPowerEncoder
from holovec.retrieval import ItemStore

print("=" * 70)
print("Image Pattern Recognition")
print("=" * 70)
print()

# Create model and encoder
model = VSA.create('FHRR', dim=10000, seed=42)

# For 8x8 images (64 pixels)
scalar_encoder = FractionalPowerEncoder(model, min_val=0, max_val=255, seed=42)
image_encoder = VectorEncoder(model, scalar_encoder, n_dimensions=64, seed=43)

print(f"Model: {model.model_name}, dimension={model.dimension}")
print(f"Image encoder: 8x8 grayscale (64D flattened)")
print()

# ============================================================================
# Dataset: Simple Shape Patterns
# ============================================================================
print("=" * 70)
print("Dataset: Simple 8x8 Shapes")
print("=" * 70)

np.random.seed(42)

# Create simple synthetic patterns
def create_vertical_line():
    """Vertical line pattern."""
    img = np.zeros((8, 8), dtype=np.uint8)
    img[:, 3:5] = 200  # Vertical line in middle
    return img.flatten()

def create_horizontal_line():
    """Horizontal line pattern."""
    img = np.zeros((8, 8), dtype=np.uint8)
    img[3:5, :] = 200  # Horizontal line in middle
    return img.flatten()

def create_cross():
    """Cross pattern."""
    img = np.zeros((8, 8), dtype=np.uint8)
    img[3:5, :] = 200  # Horizontal
    img[:, 3:5] = 200  # Vertical
    return img.flatten()

def create_square():
    """Square pattern."""
    img = np.zeros((8, 8), dtype=np.uint8)
    img[2:6, 2:6] = 200  # Square
    return img.flatten()

# Generate training examples with variations (noise)
print("\nGenerating training examples (4 classes, 5 examples each):")

training_data = []
noise_level = 15  # pixel noise

# Vertical lines
for i in range(5):
    img = create_vertical_line() + np.random.randint(-noise_level, noise_level, 64)
    img = np.clip(img, 0, 255)
    training_data.append((img, "vertical"))

# Horizontal lines
for i in range(5):
    img = create_horizontal_line() + np.random.randint(-noise_level, noise_level, 64)
    img = np.clip(img, 0, 255)
    training_data.append((img, "horizontal"))

# Crosses
for i in range(5):
    img = create_cross() + np.random.randint(-noise_level, noise_level, 64)
    img = np.clip(img, 0, 255)
    training_data.append((img, "cross"))

# Squares
for i in range(5):
    img = create_square() + np.random.randint(-noise_level, noise_level, 64)
    img = np.clip(img, 0, 255)
    training_data.append((img, "square"))

print(f"  vertical:   5 examples")
print(f"  horizontal: 5 examples")
print(f"  cross:      5 examples")
print(f"  square:     5 examples")
print(f"\nTotal: {len(training_data)} training examples")

# ============================================================================
# Training: Build Pattern Prototypes
# ============================================================================
print("\n" + "=" * 70)
print("Training: Building Pattern Prototypes")
print("=" * 70)

# Group by class
classes = {}
for img, label in training_data:
    if label not in classes:
        classes[label] = []
    classes[label].append(img)

# Encode and bundle per class
pattern_prototypes = {}

print("\nEncoding patterns:")
for label, images in classes.items():
    encoded = [image_encoder.encode(img.astype(float)) for img in images]
    prototype = model.bundle(encoded)
    pattern_prototypes[label] = prototype
    print(f"  {label:12s}: {len(images)} examples → prototype")

print(f"\nPattern prototypes created: {len(pattern_prototypes)}")

# ============================================================================
# Classification: Test on New Patterns
# ============================================================================
print("\n" + "=" * 70)
print("Classification: Testing on New Patterns")
print("=" * 70)

# Create test patterns with noise
test_patterns = [
    (create_vertical_line() + np.random.randint(-10, 10, 64), "vertical"),
    (create_horizontal_line() + np.random.randint(-10, 10, 64), "horizontal"),
    (create_cross() + np.random.randint(-10, 10, 64), "cross"),
    (create_square() + np.random.randint(-10, 10, 64), "square"),
]

test_patterns = [(np.clip(img, 0, 255), label) for img, label in test_patterns]

print("\nClassifying test patterns:")
print()

correct = 0
for i, (img, expected) in enumerate(test_patterns, 1):
    # Encode test pattern
    test_hv = image_encoder.encode(img.astype(float))

    # Find most similar prototype
    best_label = None
    best_sim = float('-inf')

    for label, prototype in pattern_prototypes.items():
        sim = float(model.similarity(test_hv, prototype))
        if sim > best_sim:
            best_sim = sim
            best_label = label

    is_correct = (best_label == expected)
    correct += (1 if is_correct else 0)
    marker = "✓" if is_correct else "✗"

    print(f"{i}. Pattern: {expected:12s}")
    print(f"   Predicted: {best_label:12s} (similarity={best_sim:.3f}) {marker}")
    print()

accuracy = correct / len(test_patterns)
print(f"Accuracy: {correct}/{len(test_patterns)} = {accuracy:.1%}")

# ============================================================================
# Analysis: Similarity Scores
# ============================================================================
print("\n" + "=" * 70)
print("Analysis: Pattern Similarity Matrix")
print("=" * 70)

print("\nSimilarity between pattern prototypes:")
labels = sorted(pattern_prototypes.keys())

# Print header
print(f"{'':12s}", end="")
for label in labels:
    print(f" {label:>10s}", end="")
print()

# Print matrix
for label1 in labels:
    print(f"{label1:12s}", end="")
    for label2 in labels:
        sim = float(model.similarity(pattern_prototypes[label1],
                                      pattern_prototypes[label2]))
        print(f" {sim:10.3f}", end="")
    print()

print("\nKey observation:")
print("  - Diagonal = 1.0 (self-similarity)")
print("  - Off-diagonal shows inter-class confusion")
print("  - Cross similar to both vertical and horizontal")

# ============================================================================
# Practical Considerations
# ============================================================================
print("\n" + "=" * 70)
print("Practical Considerations")
print("=" * 70)

print("\n✓ Advantages of HDC Image Recognition:")
print("  - Fast: No backpropagation or gradient descent")
print("  - Small: Works with limited training data")
print("  - Efficient: Low memory and compute requirements")
print("  - Interpretable: Similarity scores show confidence")
print("  - Robust: Tolerant to noise and distortions")
print()

print("✗ Limitations:")
print("  - Accuracy: Not as good as CNNs for complex images")
print("  - Scale: Best for small images (8x8, 16x16, 28x28)")
print("  - Features: Doesn't learn features like deep learning")
print("  - Flattening: Loses 2D spatial structure")
print()

print("Best use cases:")
print("  - Edge devices (limited compute/memory)")
print("  - Few-shot learning (< 10 examples per class)")
print("  - Simple patterns (icons, symbols, digits)")
print("  - Rapid prototyping (baseline before CNN)")
print("  - Explainable AI (similarity scores)")
print()

# ============================================================================
# Extension: Efficient Multi-Pattern Recognition
# ============================================================================
print("=" * 70)
print("Extension: ItemStore for Efficient Recognition")
print("=" * 70)

# Build pattern store
pattern_store = ItemStore(model)
for label, prototype in pattern_prototypes.items():
    pattern_store.add(label, prototype)

print(f"\nPattern store built with {len(pattern_prototypes)} classes")

# Test pattern
test_img = create_cross() + np.random.randint(-15, 15, 64)
test_img = np.clip(test_img, 0, 255)
test_hv = image_encoder.encode(test_img.astype(float))

# Query returns ranked results
results = pattern_store.query(test_hv, k=4)

print("\nTest pattern: cross (with noise)")
print("\nTop predictions:")
for i, (label, sim) in enumerate(results, 1):
    print(f"  {i}. {label:12s}: {sim:.3f}")

print("\nKey observation:")
print("  - ItemStore enables fast k-nearest pattern retrieval")
print("  - Can examine confidence of predictions")
print("  - Scales to many pattern classes")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Image Pattern Recognition with HDC")
print("=" * 70)
print()
print("Complete workflow:")
print("  1. Setup: Create model + VectorEncoder for flattened pixels")
print("  2. Training: Encode images + bundle per pattern class")
print("  3. Recognition: Encode test image + find nearest prototype")
print("  4. Evaluation: Check similarity for confidence")
print()
print("Performance tips:")
print("  - Use FractionalPowerEncoder for pixel values (continuous)")
print("  - Normalize images to [0, 1] or [0, 255]")
print("  - More training examples → better prototypes")
print("  - Consider ImageEncoder (17) for true 2D spatial structure")
print()
print("Next steps:")
print("  → Try with MNIST digits (28x28 = 784D)")
print("  → Experiment with different scalar encoders")
print("  → Use ImageEncoder (17) for full spatial encoding")
print("  → Combine with 25_app_integration_patterns.py for multimodal")
print()
print("=" * 70)
