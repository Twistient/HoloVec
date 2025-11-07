"""
Gesture Recognition from Motion Trajectories
============================================

Topics: Trajectory encoding, motion classification, time series, HCI
Time: 15 minutes
Prerequisites: 15_encoders_trajectory.py, 10_encoders_scalar.py
Related: 20_app_text_classification.py, 21_app_image_recognition.py

This example demonstrates practical gesture recognition using trajectory
encoding and hyperdimensional computing. Learn how to classify motion
patterns from continuous trajectories.

Key concepts:
- Trajectory encoding: Continuous paths in 2D/3D space
- Temporal patterns: Motion sequences over time
- Gesture classification: Similarity-based matching
- Real-time processing: Efficient online recognition

Gesture recognition with HDC is fast, memory-efficient, and works well
for real-time applications on edge devices (wearables, smartphones, etc.).
"""

import numpy as np
from holovec import VSA
from holovec.encoders import TrajectoryEncoder, FractionalPowerEncoder
from holovec.retrieval import ItemStore

print("=" * 70)
print("Gesture Recognition from Motion Trajectories")
print("=" * 70)
print()

# Create model and encoder
model = VSA.create('FHRR', dim=10000, seed=42)

# Trajectory encoder for 2D motion
# Use a single scalar encoder for all dimensions (x, y)
scalar_encoder = FractionalPowerEncoder(model, min_val=-1, max_val=1, bandwidth=0.1, seed=42)

trajectory_encoder = TrajectoryEncoder(
    model,
    scalar_encoder=scalar_encoder,
    n_dimensions=2,
    seed=44
)

print(f"Model: {model.model_name}, dimension={model.dimension}")
print(f"Trajectory encoder: 2D motion, 20 time steps")
print()

# ============================================================================
# Dataset: Simple 2D Gestures
# ============================================================================
print("=" * 70)
print("Dataset: Simple 2D Gesture Patterns")
print("=" * 70)

np.random.seed(42)

# Define gesture generators
def create_circle(noise=0.0):
    """Circular motion (clockwise)."""
    t = np.linspace(0, 2*np.pi, 20)
    x = 0.5 * np.cos(t) + np.random.randn(20) * noise
    y = 0.5 * np.sin(t) + np.random.randn(20) * noise
    return np.column_stack([x, y])

def create_line_horizontal(noise=0.0):
    """Horizontal line (left to right)."""
    t = np.linspace(-0.8, 0.8, 20)
    x = t + np.random.randn(20) * noise
    y = np.zeros(20) + np.random.randn(20) * noise
    return np.column_stack([x, y])

def create_line_vertical(noise=0.0):
    """Vertical line (bottom to top)."""
    t = np.linspace(-0.8, 0.8, 20)
    x = np.zeros(20) + np.random.randn(20) * noise
    y = t + np.random.randn(20) * noise
    return np.column_stack([x, y])

def create_zigzag(noise=0.0):
    """Zigzag pattern."""
    t = np.linspace(0, 4*np.pi, 20)
    x = np.linspace(-0.8, 0.8, 20) + np.random.randn(20) * noise
    y = 0.3 * np.sin(3*t) + np.random.randn(20) * noise
    return np.column_stack([x, y])

# Generate training examples
print("\nGenerating training gestures (4 classes, 5 examples each):")

training_data = []
noise_level = 0.05

# Circle gestures
for i in range(5):
    traj = create_circle(noise=noise_level)
    training_data.append((traj, "circle"))

# Horizontal lines
for i in range(5):
    traj = create_line_horizontal(noise=noise_level)
    training_data.append((traj, "horizontal"))

# Vertical lines
for i in range(5):
    traj = create_line_vertical(noise=noise_level)
    training_data.append((traj, "vertical"))

# Zigzags
for i in range(5):
    traj = create_zigzag(noise=noise_level)
    training_data.append((traj, "zigzag"))

print(f"  circle:     5 examples")
print(f"  horizontal: 5 examples")
print(f"  vertical:   5 examples")
print(f"  zigzag:     5 examples")
print(f"\nTotal: {len(training_data)} training gestures")

# ============================================================================
# Training: Build Gesture Prototypes
# ============================================================================
print("\n" + "=" * 70)
print("Training: Building Gesture Prototypes")
print("=" * 70)

# Group by class
gesture_classes = {}
for traj, label in training_data:
    if label not in gesture_classes:
        gesture_classes[label] = []
    gesture_classes[label].append(traj)

# Encode and bundle per class
gesture_prototypes = {}

print("\nEncoding gesture patterns:")
for label, trajectories in gesture_classes.items():
    encoded = [trajectory_encoder.encode(traj) for traj in trajectories]
    prototype = model.bundle(encoded)
    gesture_prototypes[label] = prototype
    print(f"  {label:12s}: {len(trajectories)} examples → prototype")

print(f"\nGesture prototypes created: {len(gesture_prototypes)}")

# ============================================================================
# Recognition: Test on New Gestures
# ============================================================================
print("\n" + "=" * 70)
print("Recognition: Testing on New Gestures")
print("=" * 70)

# Create test gestures with moderate noise
test_gestures = [
    (create_circle(noise=0.08), "circle"),
    (create_line_horizontal(noise=0.08), "horizontal"),
    (create_line_vertical(noise=0.08), "vertical"),
    (create_zigzag(noise=0.08), "zigzag"),
]

print("\nRecognizing test gestures:")
print()

correct = 0
for i, (traj, expected) in enumerate(test_gestures, 1):
    # Encode test gesture
    test_hv = trajectory_encoder.encode(traj)

    # Find most similar prototype
    best_label = None
    best_sim = float('-inf')

    for label, prototype in gesture_prototypes.items():
        sim = float(model.similarity(test_hv, prototype))
        if sim > best_sim:
            best_sim = sim
            best_label = label

    is_correct = (best_label == expected)
    correct += (1 if is_correct else 0)
    marker = "✓" if is_correct else "✗"

    print(f"{i}. Gesture: {expected:12s}")
    print(f"   Recognized: {best_label:12s} (similarity={best_sim:.3f}) {marker}")
    print()

accuracy = correct / len(test_gestures)
print(f"Accuracy: {correct}/{len(test_gestures)} = {accuracy:.1%}")

# ============================================================================
# Analysis: Gesture Similarity
# ============================================================================
print("\n" + "=" * 70)
print("Analysis: Gesture Confusion Matrix")
print("=" * 70)

print("\nSimilarity between gesture prototypes:")
labels = sorted(gesture_prototypes.keys())

# Print header
print(f"{'':12s}", end="")
for label in labels:
    print(f" {label:>10s}", end="")
print()

# Print matrix
for label1 in labels:
    print(f"{label1:12s}", end="")
    for label2 in labels:
        sim = float(model.similarity(gesture_prototypes[label1],
                                      gesture_prototypes[label2]))
        print(f" {sim:10.3f}", end="")
    print()

print("\nKey observation:")
print("  - Diagonal = 1.0 (self-similarity)")
print("  - Horizontal & vertical somewhat similar (both lines)")
print("  - Circle & zigzag clearly distinct")

# ============================================================================
# Real-Time Considerations
# ============================================================================
print("\n" + "=" * 70)
print("Real-Time Gesture Recognition")
print("=" * 70)

print("\n⚡ Real-time processing advantages:")
print("  - Fast encoding: ~1ms for 20-point trajectory")
print("  - Immediate classification: Single similarity computation")
print("  - Memory efficient: Fixed-size hypervectors")
print("  - Incremental: Can process partial gestures")
print("  - No GPU required: Runs on CPU, microcontrollers")
print()

print("Implementation tips:")
print("  - Sample trajectory at fixed rate (e.g., 20 points/sec)")
print("  - Normalize to [-1, 1] range before encoding")
print("  - Use sliding window for continuous recognition")
print("  - Threshold similarity for rejection (unknown gestures)")
print("  - Retrain prototypes with user-specific data")
print()

# ============================================================================
# Extension: Multi-User Recognition System
# ============================================================================
print("=" * 70)
print("Extension: Multi-User Gesture Library")
print("=" * 70)

# Build gesture library with ItemStore
gesture_library = ItemStore(model)
for label, prototype in gesture_prototypes.items():
    gesture_library.add(label, prototype)

print(f"\nGesture library built with {len(gesture_prototypes)} gestures")

# Test with ambiguous gesture (partial circle)
t = np.linspace(0, np.pi, 20)  # Half circle
partial_x = 0.5 * np.cos(t) + np.random.randn(20) * 0.05
partial_y = 0.5 * np.sin(t) + np.random.randn(20) * 0.05
partial_circle = np.column_stack([partial_x, partial_y])

test_hv = trajectory_encoder.encode(partial_circle)

# Query library
results = gesture_library.query(test_hv, k=4)

print("\nTest gesture: partial circle (first half only)")
print("\nTop matches:")
for i, (label, sim) in enumerate(results, 1):
    print(f"  {i}. {label:12s}: {sim:.3f}")

print("\nKey observation:")
print("  - Partial gestures still match similar patterns")
print("  - Can examine top-k for ambiguous cases")
print("  - Threshold similarity to reject uncertain gestures")

# ============================================================================
# Practical Considerations
# ============================================================================
print("\n" + "=" * 70)
print("Practical Considerations")
print("=" * 70)

print("\n✓ Advantages of HDC Gesture Recognition:")
print("  - Fast: Real-time processing on edge devices")
print("  - Efficient: Low memory and compute requirements")
print("  - Robust: Tolerant to noise and variations")
print("  - Adaptable: Easy to add new gesture classes")
print("  - Interpretable: Similarity scores show confidence")
print()

print("✗ Limitations:")
print("  - Fixed length: Requires normalizing to fixed points")
print("  - Simple patterns: Best for distinct gestures")
print("  - No learning: Doesn't adapt like neural networks")
print("  - Temporal detail: May miss fine-grained timing")
print()

print("Best use cases:")
print("  - Wearable devices (smartwatches, fitness trackers)")
print("  - Smartphone gesture controls")
print("  - Sign language recognition (simple gestures)")
print("  - Robot control (motion commands)")
print("  - VR/AR interaction (hand tracking)")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("Summary: Gesture Recognition with HDC")
print("=" * 70)
print()
print("Complete workflow:")
print("  1. Setup: Create model + TrajectoryEncoder")
print("  2. Training: Encode trajectories + bundle per gesture")
print("  3. Recognition: Encode test trajectory + find nearest prototype")
print("  4. Deployment: Real-time recognition with similarity threshold")
print()
print("Performance tips:")
print("  - Normalize trajectories to consistent range")
print("  - Use ~20-30 time steps for good temporal resolution")
print("  - More training examples → better noise tolerance")
print("  - Consider user-specific adaptation (personalization)")
print()
print("Next steps:")
print("  → Try with real accelerometer/gyroscope data")
print("  → Extend to 3D trajectories (x, y, z)")
print("  → Implement sliding window for continuous recognition")
print("  → Combine with 25_app_integration_patterns.py for multimodal")
print()
print("=" * 70)
