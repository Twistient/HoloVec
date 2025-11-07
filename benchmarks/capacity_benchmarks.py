"""Capacity benchmarks for VSA models.

Reproduces key experiments from Schlegel et al. (2022):
- Figure 3/4: Bundling capacity
- Figure 6: Approximate unbinding performance
- Figure 7/8: Unbinding of bundled pairs

These benchmarks validate that our implementations achieve the expected
capacity characteristics reported in the literature.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from holovec import VSA


def bundling_capacity_experiment(
    model_name: str,
    dimension: int,
    k_values: List[int],
    item_memory_size: int = 1000,
    trials: int = 10,
    accuracy_threshold: float = 0.99
) -> Dict[str, any]:
    """Measure bundling capacity: how many vectors can be bundled and retrieved.

    Reproduces Schlegel et al. (2022) Section 3.1, Figure 3-4.

    Procedure:
    1. Create item memory with N random vectors
    2. Randomly choose k vectors
    3. Bundle them
    4. Query: Find k most similar vectors in item memory
    5. Measure accuracy: ratio of correctly retrieved vectors

    Args:
        model_name: VSA model to test ('MAP', 'FHRR', 'HRR', etc.)
        dimension: Hypervector dimension
        k_values: List of k (number of vectors to bundle)
        item_memory_size: Size of item memory (default: 1000)
        trials: Number of repetitions per k value
        accuracy_threshold: Threshold for "success" (default: 0.99)

    Returns:
        Dictionary with results:
        - 'k_values': List of k values tested
        - 'accuracies': Mean accuracy for each k (shape: len(k_values))
        - 'std_devs': Standard deviation for each k
        - 'min_dimension': Minimum dimension needed for 99% accuracy
        - 'capacity_ratio': k/D ratio at 99% accuracy
    """
    print(f"\n{'='*60}")
    print(f"Bundling Capacity: {model_name}, D={dimension}")
    print(f"{'='*60}")

    accuracies = []
    std_devs = []

    for k in k_values:
        trial_accuracies = []

        for trial in range(trials):
            # Create item memory
            model = VSA.create(model_name, dim=dimension, seed=trial)
            item_memory = [model.random(seed=i) for i in range(item_memory_size)]

            # Randomly select k vectors to bundle
            np.random.seed(trial)
            selected_indices = np.random.choice(item_memory_size, size=k, replace=False)
            selected_vectors = [item_memory[i] for i in selected_indices]

            # Bundle the selected vectors
            bundle = model.bundle(selected_vectors)

            # Query: Find k most similar vectors in item memory
            similarities = [model.similarity(bundle, v) for v in item_memory]
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            # Calculate accuracy
            correct = len(set(top_k_indices) & set(selected_indices))
            accuracy = correct / k
            trial_accuracies.append(accuracy)

        # Compute statistics
        mean_accuracy = np.mean(trial_accuracies)
        std_accuracy = np.std(trial_accuracies)
        accuracies.append(mean_accuracy)
        std_devs.append(std_accuracy)

        print(f"k={k:2d}: accuracy={mean_accuracy:.3f} ± {std_accuracy:.3f}")

    # Find minimum dimension for 99% accuracy
    # (Approximation: we're testing one dimension, so report if achieved)
    achieves_threshold = [acc >= accuracy_threshold for acc in accuracies]
    max_k_at_threshold = max([k for k, achieved in zip(k_values, achieves_threshold)] + [0])

    results = {
        'model_name': model_name,
        'dimension': dimension,
        'k_values': k_values,
        'accuracies': accuracies,
        'std_devs': std_devs,
        'max_k_at_99': max_k_at_threshold,
        'capacity_ratio': max_k_at_threshold / dimension if max_k_at_threshold > 0 else 0.0,
        'item_memory_size': item_memory_size,
        'trials': trials
    }

    print(f"\nMax k at {accuracy_threshold*100:.0f}% accuracy: {max_k_at_threshold}")
    print(f"Capacity ratio (k/D): {results['capacity_ratio']:.3f}")

    return results


def dimension_sweep_for_k(
    model_name: str,
    k: int,
    dimension_range: List[int],
    item_memory_size: int = 1000,
    trials: int = 10,
    accuracy_threshold: float = 0.99
) -> Dict[str, any]:
    """Find minimum dimension needed to bundle k vectors with given accuracy.

    Reproduces Schlegel et al. (2022) Figure 4: dimension efficiency comparison.

    Args:
        model_name: VSA model to test
        k: Number of vectors to bundle
        dimension_range: List of dimensions to test
        item_memory_size: Size of item memory
        trials: Number of repetitions per dimension
        accuracy_threshold: Target accuracy threshold

    Returns:
        Dictionary with results including minimum dimension needed
    """
    print(f"\n{'='*60}")
    print(f"Dimension Sweep: {model_name}, k={k}")
    print(f"{'='*60}")

    for dim in dimension_range:
        result = bundling_capacity_experiment(
            model_name, dim, [k], item_memory_size, trials, accuracy_threshold
        )

        if result['accuracies'][0] >= accuracy_threshold:
            print(f"\n✓ Minimum dimension found: D={dim} for k={k}")
            return {
                'model_name': model_name,
                'k': k,
                'min_dimension': dim,
                'accuracy': result['accuracies'][0],
                'std_dev': result['std_devs'][0]
            }

    print(f"\n✗ No dimension in range achieved {accuracy_threshold*100:.0f}% accuracy")
    return {
        'model_name': model_name,
        'k': k,
        'min_dimension': None,
        'accuracy': None,
        'std_dev': None
    }


def approximate_unbinding_experiment(
    model_name: str,
    dimension: int = 1024,
    sequence_length: int = 40,
    trials: int = 10
) -> Dict[str, any]:
    """Measure quality of approximate unbinding over long sequences.

    Reproduces Schlegel et al. (2022) Section 3.2, Figure 6.

    Procedure:
    1. Start with random vector v
    2. Bind sequentially with n random vectors: S = (((v ⊗ r₁) ⊗ r₂) ... ⊗ rₙ)
    3. Unbind sequentially: v' = r₁ ⊘ ...(rₙ₋₁ ⊘ (rₙ ⊘ S))
    4. Measure similarity between v and v'

    This tests approximate inverse quality for models like MAP-C, HRR, VTB.

    Args:
        model_name: VSA model to test
        dimension: Hypervector dimension
        sequence_length: Number of bind/unbind operations
        trials: Number of repetitions

    Returns:
        Dictionary with similarity values over sequence
    """
    print(f"\n{'='*60}")
    print(f"Approximate Unbinding: {model_name}, D={dimension}, n={sequence_length}")
    print(f"{'='*60}")

    # Track similarity at each unbinding step
    similarities_over_sequence = []

    for trial in range(trials):
        model = VSA.create(model_name, dim=dimension, seed=trial)

        # Initial vector
        v = model.random(seed=trial * 1000)

        # Generate random binding vectors
        r_vectors = [model.random(seed=trial * 1000 + i + 1) for i in range(sequence_length)]

        # Sequential binding: v ⊗ r₁ ⊗ r₂ ⊗ ... ⊗ rₙ
        result = v
        for r in r_vectors:
            result = model.bind(result, r)

        # Sequential unbinding: ... ⊘ r₂ ⊘ r₁
        for i, r in enumerate(r_vectors[::-1]):
            result = model.unbind(result, r)

            # Measure similarity to original
            sim = model.similarity(v, result)
            if trial == 0:
                similarities_over_sequence.append([sim])
            else:
                similarities_over_sequence[i].append(sim)

    # Compute statistics
    mean_sims = [np.mean(sims) for sims in similarities_over_sequence]
    std_sims = [np.std(sims) for sims in similarities_over_sequence]

    # Final recovery quality
    final_similarity = mean_sims[-1]

    print(f"Similarity after {sequence_length} unbindings: {final_similarity:.4f} ± {std_sims[-1]:.4f}")

    results = {
        'model_name': model_name,
        'dimension': dimension,
        'sequence_length': sequence_length,
        'trials': trials,
        'mean_similarities': mean_sims,
        'std_similarities': std_sims,
        'final_similarity': final_similarity,
        'final_std': std_sims[-1]
    }

    return results


def bundled_pairs_experiment(
    model_name: str,
    dimension: int,
    k_values: List[int],
    item_memory_size: int = 1000,
    trials: int = 10,
    accuracy_threshold: float = 0.99
) -> Dict[str, any]:
    """Measure capacity for bundled role-filler pairs.

    Reproduces Schlegel et al. (2022) Section 3.3, Figure 7-8.

    Procedure:
    1. Create item memory with N random vectors
    2. Choose 2k vectors (k roles, k fillers)
    3. Bind pairs: role₁⊗filler₁, role₂⊗filler₂, ...
    4. Bundle pairs into single representation R
    5. Retrieve all 2k vectors by unbinding
    6. Measure accuracy

    Args:
        model_name: VSA model to test
        dimension: Hypervector dimension
        k_values: List of k (number of pairs to bundle)
        item_memory_size: Size of item memory
        trials: Number of repetitions per k
        accuracy_threshold: Threshold for success

    Returns:
        Dictionary with results similar to bundling_capacity_experiment
    """
    print(f"\n{'='*60}")
    print(f"Bundled Pairs Capacity: {model_name}, D={dimension}")
    print(f"{'='*60}")

    accuracies = []
    std_devs = []

    for k in k_values:
        trial_accuracies = []

        for trial in range(trials):
            # Create item memory
            model = VSA.create(model_name, dim=dimension, seed=trial)
            item_memory = [model.random(seed=i) for i in range(item_memory_size)]

            # Randomly select 2k vectors (k roles, k fillers)
            np.random.seed(trial)
            selected_indices = np.random.choice(item_memory_size, size=2*k, replace=False)
            role_indices = selected_indices[:k]
            filler_indices = selected_indices[k:]

            roles = [item_memory[i] for i in role_indices]
            fillers = [item_memory[i] for i in filler_indices]

            # Bind pairs
            pairs = [model.bind(r, f) for r, f in zip(roles, fillers)]

            # Bundle pairs into single representation
            R = model.bundle(pairs)

            # Try to retrieve all vectors
            correct_count = 0

            # Unbind with each role to retrieve fillers
            for i, role in enumerate(roles):
                retrieved = model.unbind(R, role)
                similarities = [model.similarity(retrieved, v) for v in item_memory]
                best_match = np.argmax(similarities)

                if best_match == filler_indices[i]:
                    correct_count += 1

            # Unbind with each filler to retrieve roles
            for i, filler in enumerate(fillers):
                retrieved = model.unbind(R, filler)
                similarities = [model.similarity(retrieved, v) for v in item_memory]
                best_match = np.argmax(similarities)

                if best_match == role_indices[i]:
                    correct_count += 1

            # Accuracy: ratio of correctly retrieved to total (2k) vectors
            accuracy = correct_count / (2 * k)
            trial_accuracies.append(accuracy)

        # Compute statistics
        mean_accuracy = np.mean(trial_accuracies)
        std_accuracy = np.std(trial_accuracies)
        accuracies.append(mean_accuracy)
        std_devs.append(std_accuracy)

        print(f"k={k:2d} pairs: accuracy={mean_accuracy:.3f} ± {std_accuracy:.3f}")

    # Find max k at threshold
    achieves_threshold = [acc >= accuracy_threshold for acc in accuracies]
    max_k_at_threshold = max([k for k, achieved in zip(k_values, achieves_threshold)] + [0])

    results = {
        'model_name': model_name,
        'dimension': dimension,
        'k_values': k_values,
        'accuracies': accuracies,
        'std_devs': std_devs,
        'max_k_at_99': max_k_at_threshold,
        'capacity_ratio': max_k_at_threshold / dimension if max_k_at_threshold > 0 else 0.0,
        'item_memory_size': item_memory_size,
        'trials': trials
    }

    print(f"\nMax k at {accuracy_threshold*100:.0f}% accuracy: {max_k_at_threshold}")
    print(f"Capacity ratio (k/D): {results['capacity_ratio']:.3f}")

    return results


def plate_2003_fhrr_capacity(dimension: int = 512, trials: int = 10) -> Dict[str, float]:
    """Reproduce Plate (2003) capacity result: ~0.35D items for FHRR.

    Plate (2003) showed FHRR can bundle approximately 0.35*D items
    with high accuracy. This validates our FHRR implementation.

    Args:
        dimension: FHRR dimension (default: 512 as in Plate 2003)
        trials: Number of trials to average

    Returns:
        Dictionary with capacity estimate and comparison to theory
    """
    print(f"\n{'='*60}")
    print(f"Plate (2003) FHRR Capacity Validation")
    print(f"{'='*60}")
    print(f"Expected capacity: ~{0.35 * dimension:.0f} items (0.35 × D)")

    # Test range around 0.35*D
    expected_capacity = int(0.35 * dimension)
    k_values = list(range(
        max(2, expected_capacity - 50),
        expected_capacity + 50,
        5
    ))

    result = bundling_capacity_experiment(
        'FHRR',
        dimension,
        k_values,
        item_memory_size=1000,
        trials=trials,
        accuracy_threshold=0.99
    )

    measured_capacity = result['max_k_at_99']
    theoretical_capacity = 0.35 * dimension
    relative_error = abs(measured_capacity - theoretical_capacity) / theoretical_capacity

    print(f"\nTheoretical capacity: {theoretical_capacity:.0f} items")
    print(f"Measured capacity: {measured_capacity} items")
    print(f"Relative error: {relative_error*100:.1f}%")

    return {
        'dimension': dimension,
        'theoretical_capacity': theoretical_capacity,
        'measured_capacity': measured_capacity,
        'relative_error': relative_error,
        'passes': relative_error < 0.15  # Within 15% is good
    }


def run_all_benchmarks(quick: bool = False):
    """Run all capacity benchmarks.

    Args:
        quick: If True, run reduced benchmarks for faster execution
    """
    print("\n" + "="*60)
    print("VSA CAPACITY BENCHMARKS")
    print("Reproducing Schlegel et al. (2022) experiments")
    print("="*60)

    if quick:
        print("\n[Quick mode: reduced parameters for faster execution]")
        dimension = 512
        k_values = [2, 5, 10, 15, 20]
        trials = 3
    else:
        dimension = 1000
        k_values = list(range(2, 51, 3))
        trials = 10

    results = {}

    # Test key models from Schlegel et al. (2022)
    models_to_test = ['FHRR', 'HRR', 'MAP', 'BSC']

    # 1. Bundling capacity
    print("\n" + "="*60)
    print("1. BUNDLING CAPACITY (Schlegel Fig. 3-4)")
    print("="*60)

    for model_name in models_to_test:
        try:
            result = bundling_capacity_experiment(
                model_name, dimension, k_values, trials=trials
            )
            results[f'{model_name}_bundling'] = result
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

    # 2. Approximate unbinding (only for approximate models)
    print("\n" + "="*60)
    print("2. APPROXIMATE UNBINDING (Schlegel Fig. 6)")
    print("="*60)

    approximate_models = ['HRR', 'VTB', 'MAP']
    seq_length = 20 if quick else 40

    for model_name in approximate_models:
        try:
            result = approximate_unbinding_experiment(
                model_name, dimension, seq_length, trials=trials
            )
            results[f'{model_name}_unbinding'] = result
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

    # 3. Bundled pairs (subset of models)
    print("\n" + "="*60)
    print("3. BUNDLED PAIRS CAPACITY (Schlegel Fig. 7-8)")
    print("="*60)

    pair_k_values = [2, 5, 10, 15] if quick else list(range(2, 31, 2))

    for model_name in ['FHRR', 'HRR']:
        try:
            result = bundled_pairs_experiment(
                model_name, dimension, pair_k_values, trials=trials
            )
            results[f'{model_name}_pairs'] = result
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

    # 4. Plate (2003) validation
    if not quick:
        print("\n" + "="*60)
        print("4. PLATE (2003) FHRR CAPACITY VALIDATION")
        print("="*60)

        try:
            result = plate_2003_fhrr_capacity(dimension=512, trials=trials)
            results['plate_2003_validation'] = result
        except Exception as e:
            print(f"Error in Plate validation: {e}")

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    for key, result in results.items():
        if 'bundling' in key or 'pairs' in key:
            print(f"{key}: max_k={result.get('max_k_at_99', 'N/A')}, "
                  f"ratio={result.get('capacity_ratio', 0):.3f}")
        elif 'unbinding' in key:
            print(f"{key}: final_sim={result.get('final_similarity', 0):.4f}")

    return results


if __name__ == '__main__':
    import sys

    # Allow running in quick mode for testing
    quick_mode = '--quick' in sys.argv

    results = run_all_benchmarks(quick=quick_mode)

    print("\n✓ All benchmarks completed!")
    print("\nResults saved to 'results' dictionary")
