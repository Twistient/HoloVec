# HoloVec Examples - Quick Reference Index

**Quick lookup table for all examples organized by topic, difficulty, and use case.**

For the full learning path guide, see [README.md](README.md).

---

## By Number

| # | File | Topics | Time | Level |
|---|------|--------|------|-------|
| 00 | [quickstart.py](00_quickstart.py) | First steps, basic workflow | 5 min | Beginner |
| 01 | [basic_operations.py](01_basic_operations.py) | Binding, bundling, permutation | 10 min | Beginner |
| 02 | [models_comparison.py](02_models_comparison.py) | MAP, FHRR, HRR, BSC models | 15 min | Beginner |
| 10 | [encoders_scalar.py](10_encoders_scalar.py) | FPE, Thermometer, Level | 20 min | Intermediate |
| 11 | [encoders_fractional_power.py](11_encoders_fractional_power.py) | FPE bandwidth tuning | 15 min | Intermediate |
| 12 | [encoders_thermometer_level.py](12_encoders_thermometer_level.py) | Ordinal & discrete | 10 min | Intermediate |
| 13 | [encoders_position_binding.py](13_encoders_position_binding.py) | Sequence encoding | 15 min | Intermediate |
| 14 | [encoders_ngram.py](14_encoders_ngram.py) | N-gram text patterns | 20 min | Intermediate |
| 15 | [encoders_trajectory.py](15_encoders_trajectory.py) | Continuous sequences | 25 min | Intermediate |
| 16 | [encoders_vector.py](16_encoders_vector.py) | Multivariate vectors | 15 min | Intermediate |
| 17 | [encoders_image.py](17_encoders_image.py) | 2D spatial data | 20 min | Intermediate |
| 18 | [encoders_graph.py](18_encoders_graph.py) | Graph structures | 20 min | Advanced |
| 20 | [app_text_classification.py](20_app_text_classification.py) | Document classification | 15 min | Intermediate |
| 21 | [app_image_recognition.py](21_app_image_recognition.py) | Pattern matching | 15 min | Intermediate |
| 22 | [app_gesture_recognition.py](22_app_gesture_recognition.py) | Motion classification | 15 min | Intermediate |
| 23 | [app_symbolic_reasoning.py](23_app_symbolic_reasoning.py) | Role-filler binding | 20 min | Advanced |
| 24 | [app_working_memory.py](24_app_working_memory.py) | Cognitive architecture | 20 min | Advanced |
| 25 | [app_integration_patterns.py](25_app_integration_patterns.py) | Multimodal fusion | 25 min | Advanced |
| 26 | [retrieval_basics.py](26_retrieval_basics.py) | Codebook, ItemStore | 10 min | Intermediate |
| 27 | [cleanup_strategies.py](27_cleanup_strategies.py) | BruteForce vs Resonator | 15 min | Advanced |
| 28 | [factorization_methods.py](28_factorization_methods.py) | Multi-factor unbinding | 15 min | Advanced |
| 30 | [theory_fpe_validation.py](30_theory_fpe_validation.py) | FPE theoretical validation | 20 min | Advanced |
| 31 | [performance_benchmarks.py](31_performance_benchmarks.py) | Speed, accuracy, backends | 15 min | Advanced |
| 32 | [distributed_representations.py](32_distributed_representations.py) | Capacity analysis | 15 min | Advanced |
| 33 | [error_handling_robustness.py](33_error_handling_robustness.py) | Noise tolerance | 15 min | Advanced |
| 40 | [model_hrr_correlation.py](40_model_hrr_correlation.py) | HRR circular correlation | 5 min | Advanced |
| 41 | [model_ghrr_diagonality.py](41_model_ghrr_diagonality.py) | GHRR parameter sweep | 10 min | Advanced |
| 42 | [model_bsdc_seg.py](42_model_bsdc_seg.py) | Segment-sparse codes | 10 min | Advanced |

---

## By Topic

### Core Concepts

- **Getting Started**: 00, 01, 02
- **VSA Operations**: 01 (binding, bundling, permutation)
- **Models**: 02, 40, 41, 42

### Encoders

- **Scalar Values**: 10 (FPE, Thermometer, Level), 11 (FPE deep dive), 12 (Thermometer/Level deep dive)
- **Sequences**: 13 (position), 14 (n-gram), 15 (trajectory)
- **Spatial Data**: 16 (vectors), 17 (images), 18 (graphs)

### Applications

- **Text/NLP**: 14, 20
- **Vision**: 17, 21
- **Time Series**: 15, 22
- **Symbolic AI**: 18, 23, 24
- **Multimodal**: 25

### Retrieval & Memory

- **Basic Retrieval**: 26
- **Cleanup Methods**: 27
- **Factorization**: 28
- **Working Memory**: 24

### Theory & Validation

- **FPE Theory**: 30
- **Performance**: 31 (benchmarks), 32 (capacity), 33 (robustness)
- **Model Specifics**: 40, 41, 42

---

## By Difficulty Level

### Beginner (Start Here!)

| File | Topics | Time |
|------|--------|------|
| 00_quickstart.py | First steps | 5 min |
| 01_basic_operations.py | Core operations | 10 min |
| 02_models_comparison.py | Model selection | 15 min |

### Intermediate

| File | Topics | Time |
|------|--------|------|
| 10_encoders_scalar.py | Continuous values | 20 min |
| 11_encoders_fractional_power.py | FPE bandwidth tuning | 15 min |
| 12_encoders_thermometer_level.py | Ordinal & discrete | 10 min |
| 13_encoders_position_binding.py | Sequences | 15 min |
| 14_encoders_ngram.py | Text patterns | 20 min |
| 15_encoders_trajectory.py | Motion paths | 25 min |
| 16_encoders_vector.py | Feature vectors | 15 min |
| 17_encoders_image.py | Images | 20 min |
| 20_app_text_classification.py | Text classification | 15 min |
| 21_app_image_recognition.py | Image recognition | 15 min |
| 22_app_gesture_recognition.py | Gesture classification | 15 min |
| 26_retrieval_basics.py | Codebook queries | 10 min |

### Advanced

| File | Topics | Time |
|------|--------|------|
| 18_encoders_graph.py | Knowledge graphs | 20 min |
| 23_app_symbolic_reasoning.py | Role-filler reasoning | 20 min |
| 24_app_working_memory.py | Cognitive architecture | 20 min |
| 25_app_integration_patterns.py | Multimodal systems | 25 min |
| 27_cleanup_strategies.py | Cleanup comparison | 15 min |
| 28_factorization_methods.py | Factorization | 15 min |
| 30_theory_fpe_validation.py | Theory validation | 20 min |
| 31_performance_benchmarks.py | Speed & accuracy | 15 min |
| 32_distributed_representations.py | Capacity analysis | 15 min |
| 33_error_handling_robustness.py | Noise tolerance | 15 min |
| 40_model_hrr_correlation.py | HRR specifics | 5 min |
| 41_model_ghrr_diagonality.py | GHRR parameters | 10 min |
| 42_model_bsdc_seg.py | Sparse codes | 10 min |

---

## By Use Case

### Natural Language Processing

- **Text Encoding**: 14_encoders_ngram.py
- **Classification**: 20_app_text_classification.py
- **Semantic Structures**: 23_app_symbolic_reasoning.py

### Computer Vision

- **Image Encoding**: 17_encoders_image.py
- **Pattern Recognition**: 21_app_image_recognition.py
- **Feature Vectors**: 16_encoders_vector.py

### Time Series & Motion

- **Trajectory Encoding**: 15_encoders_trajectory.py
- **Gesture Recognition**: 22_app_gesture_recognition.py
- **Continuous Sequences**: 15_encoders_trajectory.py

### Symbolic AI & Knowledge

- **Graph Encoding**: 18_encoders_graph.py
- **Symbolic Reasoning**: 23_app_symbolic_reasoning.py
- **Working Memory**: 24_app_working_memory.py
- **Knowledge Retrieval**: 26_retrieval_basics.py

### Robotics & Embodied AI

- **Motion Encoding**: 15_encoders_trajectory.py
- **Gesture Recognition**: 22_app_gesture_recognition.py
- **Sensor Fusion**: 25_app_integration_patterns.py

### Research & Validation

- **Theoretical Validation**: 30_theory_fpe_validation.py
- **Performance Benchmarks**: 31_performance_benchmarks.py
- **Capacity Analysis**: 32_distributed_representations.py
- **Robustness Testing**: 33_error_handling_robustness.py
- **Model Comparison**: 02_models_comparison.py
- **Cleanup Performance**: 27_cleanup_strategies.py

---

## By Required Prerequisites

### No Prerequisites (Start Here)

- 00_quickstart.py
- 01_basic_operations.py

### Requires Understanding of Basic Operations

- 02_models_comparison.py
- 10_encoders_scalar.py
- 13_encoders_position_binding.py
- 26_retrieval_basics.py

### Requires Understanding of Encoders

- 14_encoders_ngram.py
- 15_encoders_trajectory.py
- 16_encoders_vector.py
- 17_encoders_image.py
- 18_encoders_graph.py
- 20-22_app_*.py (applications)

### Requires Advanced Concepts

- 23_app_symbolic_reasoning.py (role-filler binding)
- 24_app_working_memory.py (cleanup strategies)
- 25_app_integration_patterns.py (multiple encoders)
- 27_cleanup_strategies.py (resonator networks)
- 28_factorization_methods.py (multi-factor unbinding)
- 30_theory_fpe_validation.py (FPE theory)

---

## Jupyter Notebooks

Interactive versions of key examples (in `notebooks/` directory):

| Notebook | Based On | Topics |
|----------|----------|--------|
| 00_quickstart.ipynb | 00_quickstart.py | Interactive intro |
| 01_basic_operations.ipynb | 01_basic_operations.py | Hands-on VSA |
| 10_encoders_scalar.ipynb | 10_encoders_scalar.py | Scalar encoding playground |
| 14_encoders_ngram.ipynb | 14_encoders_ngram.py | Text encoding workshop |
| 17_encoders_image.ipynb | 17_encoders_image.py | Image encoding lab |
| 25_app_integration_patterns.ipynb | 25_app_integration_patterns.py | Multimodal fusion |

---

## Recommended Learning Sequences

### Quick Tour (30 minutes)

1. 00_quickstart.py (5 min)
2. 01_basic_operations.py (10 min)
3. 10_encoders_scalar.py (15 min)

### Application Developer Path (2 hours)

1. 00-02: Fundamentals (30 min)
2. Choose your domain:
   - NLP: 14 → 20 (35 min)
   - Vision: 17 → 21 (35 min)
   - Time Series: 15 → 22 (40 min)
3. Integration: 25 (25 min)
4. Retrieval: 26, 27 (25 min)

### Researcher Path (5 hours)

1. 00-02: Fundamentals (30 min)
2. 10-18: All encoders (3 hours)
3. 27-28: Advanced retrieval (30 min)
4. 30-33: Theory & performance (65 min)
5. 40-42: Model specifics (25 min)

### Full Course (7 hours)

Work through all examples in numerical order (00 → 42)

---

## Support & Documentation

- **Full Guide**: [README.md](README.md)
- **Documentation**: <https://docs.holovecai.com>
- **Issues**: <https://github.com/twistient/holovec/issues>
- **Discussions**: <https://github.com/twistient/holovec/discussions>

---

*Last updated: 11-07-2025*
