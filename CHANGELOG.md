# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Retrieval module with Codebook and ItemStore classes
- Backend capability detection (supports_gpu, supports_jit, supports_complex, supports_sparse, supports_device)
- Phase 3C utilities complete:
  - Cleanup strategies (BruteForceCleanup, ResonatorCleanup)
  - Search utilities (nearest_neighbors, threshold_search, batch_similarity)
  - CPSE/CPSD utilities for context-preserving encoding
  - General operations (select_top_k, add_noise, similarity_matrix)

### Changed

- Improved backend availability detection to check actual dependencies
- Enhanced cross-backend test coverage

## [0.1.0] - 2025-11-06

### Added

#### Core Architecture

- **Backend Abstraction Layer**: Unified interface for NumPy, PyTorch, and JAX
  - NumPy backend (CPU-only, default, zero additional dependencies)
  - PyTorch backend (GPU support via CUDA/Metal, neural network integration)
  - JAX backend (JIT compilation, TPU support, automatic differentiation)
  - Runtime backend switching without code changes
  - Backend capability detection system for adaptive code

#### VSA Models (7 validated implementations)

- **MAP (Multiply-Add-Permute)**: Self-inverse binding, neuromorphic-friendly
- **FHRR (Fourier Holographic Reduced Representations)**: Complex-valued, exact inverses, best capacity
- **HRR (Holographic Reduced Representations)**: Classic circular convolution binding
- **BSC (Binary Spatter Codes)**: XOR-based binding for binary vectors
- **BSDC (Binary Sparse Distributed Codes)**: Sparse binary representations
- **BSDC-SEG (Segmented BSDC)**: Block-based sparse codes
- **GHRR (Generalized HRR)**: 2024 state-of-the-art model with matrix binding
- **VTB (Vector-derived Transformation Binding)**: Learned transformation matrices

All models validated against academic literature with property-based testing.

#### Vector Spaces

- **BipolarSpace**: {-1, +1} vectors for MAP, HRR
- **BinarySpace**: {0, 1} vectors for BSC
- **RealSpace**: Real-valued dense vectors
- **ComplexSpace**: Complex-valued vectors for FHRR
- **SparseSpace**: Sparse vector representations for BSDC
- **MatrixSpace**: Matrix-based hypervectors for GHRR, VTB

#### Encoders (8 production-ready implementations)

**Scalar Encoders:**

- **FractionalPowerEncoder**: Continuous values using fractional binding (Frady et al. 2021)
  - Preserves metric structure (similar values → similar vectors)
  - Configurable exponent for capacity/precision tradeoff
  - Efficient decoding via correlation
- **ThermometerEncoder**: Thermometer-style encoding (Kanerva 2009)
  - Smooth transitions between adjacent values
  - Natural for ordinal data
- **LevelEncoder**: Discrete level encoding
  - Direct mapping for categorical values
  - Clean separation between levels

**Sequence Encoders:**

- **PositionBindingEncoder**: Order-sensitive sequence encoding
  - Binds position vectors with content
  - Supports variable-length sequences
  - Query by position or content
- **NGramEncoder**: N-gram based sequence encoding
  - Captures local context windows
  - Configurable n-gram size
- **TrajectoryEncoder**: Temporal trajectory encoding
  - Sequential binding with positional information
  - Suitable for time-series and paths

**Spatial Encoder:**

- **ImageEncoder**: 2D grid/image encoding
  - Position-aware pixel encoding
  - Preserves spatial relationships
  - Scalable to different resolutions

**Structured Encoder:**

- **VectorEncoder**: Multi-dimensional vector composition
  - Role-filler binding for dimensions
  - Composable with scalar encoders
  - Supports high-dimensional feature vectors

#### Cleanup and Retrieval

- **BruteForceCleanup**: Exhaustive codebook search
  - Guaranteed optimal match
  - Suitable for small codebooks (<1000 items)
- **ResonatorCleanup**: Iterative resonator network (Kymn et al. 2024)
  - 10-100x faster than brute force
  - Handles composite vectors (bound products)
  - Configurable iterations for accuracy/speed tradeoff
  - Hard and soft resonator variants
- **Codebook**: Label-to-vector mapping with persistence
  - Store and retrieve named vectors
  - Similarity-based lookup
  - JSON serialization
- **ItemStore**: High-level retrieval interface
  - Automatic cleanup strategy selection
  - Batch operations
  - Statistics and diagnostics

#### Search and Utilities

- **K-Nearest Neighbors**: Find k most similar vectors
- **Threshold Search**: Find all vectors above similarity threshold
- **Batch Similarity**: Efficient pairwise similarity computation
- **Similarity Matrix**: Compute full similarity matrices
- **Top-K Selection**: Select k vectors with highest values
- **Noise Injection**: Add controlled noise for testing robustness

#### CPSE/CPSD (Context-Preserving Encoding)

- Implementation of Malits & Mendelson 2025 algorithms
- Context-preserving SDR encoding/decoding
- Metadata-aware encoding for structured data
- Compositional encoding with role preservation

#### High-Level API

- **VSA.create()**: Simple model creation with sensible defaults
- **VSA.backend_info()**: Query available backends and capabilities
- Unified interface across all models and backends
- Automatic backend selection based on hardware

### Testing

- **480+ test functions** across all modules
- **90-98% code coverage** (lines)
- **Property-based testing** with Hypothesis
  - Algebraic properties (commutativity, associativity, distributivity)
  - Inverse properties (bind/unbind recovery)
  - Similarity invariants
  - Capacity bounds
- **Cross-backend consistency tests**
  - Ensures identical behavior across NumPy, PyTorch, JAX
  - Validates numerical stability
- **Encoder validation tests**
  - Locality preservation
  - Decoding accuracy
  - Edge case handling

### Documentation

- **Comprehensive README** (605 lines)
  - Installation for all backends
  - Quick start guide
  - Core concepts explanation
  - 6 detailed examples
  - Model comparison table
  - Architecture overview
- **Theory Documentation**
  - VSA model mathematics
  - Encoder theory and validation
  - Capacity analysis
  - Backend comparison
- **API Documentation**
  - NumPy-style docstrings for all public APIs
  - Type hints throughout
  - Usage examples in docstrings
- **Validation Reports**
  - Model validation against literature
  - Encoder accuracy measurements
  - Performance benchmarks
- **10+ Working Examples**
  - Basic operations
  - Analogical reasoning
  - Sequence encoding
  - Role-filler binding
  - Continuous value encoding
  - Multi-dimensional vectors

### Infrastructure

- **Development Tools**
  - Black code formatting (100 char line length)
  - Ruff linting and import sorting
  - Mypy type checking with strict mode
  - Pre-commit hooks for code quality
- **Project Configuration**
  - Modern pyproject.toml setup
  - Setuptools build system
  - Optional dependencies (torch, jax, dev, docs)
  - Python 3.9+ support
- **Code Quality**
  - Type-safe with comprehensive type hints
  - Well-documented with NumPy-style docstrings
  - Clean architecture with clear separation of concerns
  - Minimal dependencies (only NumPy for base install)

### Technical Details

- **9,725 lines** of production Python code
- **7,044 lines** of test code
- **Python 3.9+** support (3.9, 3.10, 3.11, 3.12 tested)
- **Apache License 2.0**
- **Backend-agnostic** design for maximum portability
- **Type-safe** with comprehensive type hints
- **Well-documented** with NumPy-style docstrings
- **Zero runtime dependencies** beyond NumPy for base install

### Performance

- Efficient backend implementations optimized for each framework
- JAX JIT compilation for 10-100x speedup on compute-intensive operations
- PyTorch GPU acceleration for large-scale operations
- Sparse representations for memory efficiency (BSDC, BSDC-SEG)
- Vectorized operations throughout for NumPy performance

### Research Foundation

Based on comprehensive academic research:

- Kanerva (1993, 2009) - SDM and hyperdimensional computing foundations
- Plate (2003) - HRR model
- Kanerva (2009) - VSA introduction
- Frady et al. (2021) - Fractional power encoding
- Schlegel et al. (2022) - VSA model comparison
- Kleyko et al. (2023) - Comprehensive HDC/VSA survey
- Kymn et al. (2024) - Resonator cleanup networks
- Malits & Mendelson (2025) - CPSE/CPSD algorithms

[Unreleased]: https://github.com/Twistient/HoloVec/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Twistient/HoloVec/releases/tag/v0.1.0
