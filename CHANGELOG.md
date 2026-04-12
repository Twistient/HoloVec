# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a model-aware benchmark runner at `python -m benchmarks.run` with JSON/CSV output and
  suites for primitives, bundle capacity, approximate unbinding, cleanup factorization, order
  sensitivity, and sparse retrieval
- Added benchmark methodology documentation and CLI smoke tests

### Changed

- BREAKING: `VSA.create()` now validates model/backend kwargs and raises on unsupported arguments instead of silently ignoring them
- BREAKING: `AssocStore.query_value()` is now intentionally top-1 only and no longer accepts the ignored `top` parameter
- Threshold-based retrieval now validates against the active model's similarity range, allowing negative thresholds for continuous models

### Fixed

- Restored correct factory forwarding for BSDC (`sparsity`, `binding_mode`), GHRR (`matrix_size`, `diagonality`), and VTB (`n_bases`, `shifts`, `temperature`) configuration
- Tightened retrieval fallback behavior so fast-path query failures only fall back on expected capability/shape issues
- Reworked `SparseSegmentSpace`, `BSDC-SEG` bundling, BSDC CDT binding, and GHRR diagonality checks to avoid unnecessary full-vector NumPy round-trips on optional backends
- `MatrixSpace.random()` now generates unitary batches through backend-native SVD paths instead of constructing the entire hypervector in NumPy first

### Security

- `Codebook` persistence now uses a safe versioned `.npz` format by default and refuses legacy pickle-backed archives unless `allow_unsafe_legacy=True` is passed explicitly for migration
- Raised lockfile and extra dependency floors to patched releases for `black`, `Pygments`, `requests`, `urllib3`, and `filelock` to clear the current GitHub dependency alerts

### Testing

- Added regression coverage for factory kwarg forwarding/rejection, negative thresholds on continuous models, and legacy persistence migration paths
- Added backend coverage for integer RNG, dtype casting, identity-matrix creation, backend-native segmented sparse helpers, BSDC CDT binding, and matrix-space random generation
- Added smoke execution of the canonical release-facing example scripts to the pytest suite

### Documentation

- Updated retrieval and performance docs to match the supported factory surface and new persistence/migration behavior
- Added agent/contributor workflow guidance for changelog maintenance, release-note discipline, and PR protocol
- Rewrote the README, quick start, model comparison, patterns, and examples index around the
  smoke-tested canonical example set
- Added migration guidance for the tightened factory, retrieval, and persistence behavior
- Added benchmark methodology guidance tied to the maintained benchmark runner and literature map

## [0.3.2] - 2026-04-11

### Fixed

- Updated the README citation block to match the current package version
- Replaced README self-anchor links with absolute documentation URLs so the PyPI project page renders the top navigation cleanly

## [0.3.1] - 2026-04-11

### Changed

- Added tag-driven PyPI publishing via GitHub Actions trusted publishing
- Improved PyPI-facing package metadata, including SPDX license metadata and absolute README asset URLs
- Simplified the repository security policy and reduced over-specific claims in project copy

### Fixed

- Stabilized docs CI by building documentation with the locked `uv` docs environment
- Updated GitHub Actions workflow dependencies to current supported releases

### Documentation

- Simplified security reporting guidance to rely on GitHub private vulnerability reporting
- Cleaned up README installation guidance and adjusted model/backend wording to be less aspirational

## [0.3.0] - 2026-04-11

### Added

- **ResidueEncoder**: Residue Hyperdimensional Computing encoder (Kymn et al. 2024)
  - Encodes integers using residue number system with co-prime moduli
  - Enables addition on encoded integers: z(x₁ + x₂) = z(x₁) ⊙ z(x₂)
  - Enables subtraction: z(x₁ - x₂) = z(x₁) ⊙ z(x₂)*
  - Logarithmic codebook scaling: range M = ∏mᵢ, codebook size = ∑mᵢ
  - Chinese Remainder Theorem for unique integer encoding
  - Example: moduli [97, 101, 103] → range ~1M, only 301 codebook vectors
- **AttentionResonatorCleanup**: Softmax attention-based resonator network (Yeung et al. 2024)
  - Works with FHRR models (unlike traditional ResonatorCleanup which requires self-inverse)
  - Leverages Modern Hopfield Network theory with exponential memory capacity
  - Configurable temperature parameter (β) for attention sharpness
  - Supports multi-factor decomposition with convergence detection
  - 10-100x faster than brute force for composite vector factorization
- **Capacity Analysis Module** (`holovec.analysis`):
  - `theoretical_capacity()`: Theoretical capacity metrics from literature (Schlegel et al. 2022, Kleyko et al. 2023)
  - `recommend_dimension()`: Dimension recommendations based on items, bindings, and target accuracy
  - `compare_models()`: Side-by-side comparison of all VSA models at given dimension
  - `empirical_capacity_test()`: Empirical validation of model capacity via progressive testing
- **Context-Dependent Thinning (CDT)** binding mode for BSDC (Rachkovskij 2001):
  - Alternative to XOR binding that preserves similarity to components
  - Preserves both unstructured similarity (result ~ components) and structured similarity (similar inputs → similar outputs)
  - Useful for analogical reasoning where component similarity matters
  - Enable with `VSA.create('BSDC', binding_mode='cdt')`
  - Multi-component binding via `model.context_dependent_thinning([a, b, c])`

### Changed

- Split cleanup strategies into `holovec.utils.cleanup` package modules while preserving public imports
- Added dedicated engine CI for tests, Ruff, and source-only mypy
- Updated developer commands and README examples to use `uv run` / `uv sync`
- Clarified backend support policy: NumPy is release-blocking, PyTorch and JAX are optional paths

### Fixed

- `holovec.__version__` now resolves from installed package metadata instead of a stale hardcoded value
- Sequence symbol auto-generation is now deterministic across Python processes and insertion order
- Periodic encoder helper functions no longer raise `NameError` on `np.pi`
- Source typing now passes `mypy holovec`

### Documentation

- Updated installation, testing, and contribution guidance to reflect the enforced engine workflow

### Testing

- Added regression tests for runtime/package version alignment
- Added regression tests for periodic helper APIs
- Added regression tests for deterministic sequence symbol generation across processes and insertion order
- Added 48 tests for ResidueEncoder (encoding, addition, subtraction, decoding, CRT)
- Added 27 tests for AttentionResonatorCleanup (initialization, factorization, edge cases)
- Added 24 tests for capacity analysis module
- Added 17 tests for BSDC CDT binding mode
- Total tests: 805 → 921 passing

## [0.2.0] - 2025-12-23

### Added

- **Codebook dict-like interface**: `__getitem__`, `__contains__`, `__len__`, `__iter__`, `items()`, `keys()`, `values()`, `get()` methods for intuitive codebook access
- **'bsdc-seg' model alias**: `VSA.create('bsdc-seg')` now works alongside 'bsdc_seg' to match `model_name` output
- Pre-commit configuration for automated code quality checks
- HoloVec favicon for documentation site

### Changed

- **BREAKING: Minimum Python version raised to 3.11** (from 3.9)
- Modernized type annotations using Python 3.11+ syntax (`X | None` instead of `Optional[X]`)
- Migrated documentation from Sphinx to MkDocs with shadcn theme
- Updated ruff configuration to non-deprecated `[tool.ruff.lint]` format
- Cleaned up unused imports and encapsulated numpy usage in backend modules

### Fixed

- PyTorch backend: Fixed complex dot product double-conjugation bug causing FHRR similarity to return near-zero instead of 1.0
- HRR docstring: Corrected unbind accuracy claims (actual ~0.71, not ~0.99)
- pytest configuration: Removed coverage from addopts to fix `--no-cov` conflict
- Backend RNG test: Rewritten for same-backend consistency instead of cross-backend comparison
- BSDC-SEG documentation: Fixed model name references

### Testing

- Improved overall test coverage from 79% to 84%
- GHRR model coverage: 46% → 100%
- ResonatorCleanup coverage: 56% → 98%
- AssocStore coverage: 36% → 98%
- Codebook coverage: 97% → 100%
- ItemStore coverage: 70% → 95%
- Total tests: 761 → 805 passing

### Documentation

- Major README cleanup: removed aspirational content, updated metrics (tests 525→805, coverage 70→84%)
- Replaced "Project Status" phases with accurate "Current Status" table
- Simplified "Technical Positioning" into concise "Feature Summary"
- Fixed URLs and dates in examples/INDEX.md
- Fixed CITATION.cff version and release date

## [0.1.1] - 2025-12-21

### Added

- Retrieval module with Codebook, ItemStore, and AssocStore classes
- Backend capability detection (supports_gpu, supports_jit, supports_complex, supports_sparse, supports_device)
- Cleanup strategies: BruteForceCleanup and ResonatorCleanup (Kymn et al. 2024)
- Search utilities: nearest_neighbors, threshold_search, batch_similarity
- CPSE/CPSD utilities for context-preserving encoding (Malits & Mendelson 2025)
- General operations: select_top_k, add_noise, similarity_matrix
- hypothesis to dev dependencies for property-based testing

### Changed

- Improved backend availability detection to check actual dependencies
- Enhanced cross-backend test coverage (525+ tests passing)

### Fixed

- HRR model: Replaced Wiener deconvolution with classic circular correlation for unbinding
- HRR model: Removed incorrect normalization from bind() and bundle() operations
- HRR model: Corrected docstring to reflect actual ~0.65-0.75 recovery similarity (was incorrectly stated as ~0.99)
- VSA.create(): Now accepts Backend instances in addition to string names
- VTB model: Clarified unbind() docstring to explain non-commutative recovery semantics
- Test assertions: Updated regex patterns and HRR threshold for property tests

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
- **70% overall coverage** (core modules 90%+)
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

[Unreleased]: https://github.com/Twistient/HoloVec/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/Twistient/HoloVec/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Twistient/HoloVec/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Twistient/HoloVec/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Twistient/HoloVec/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Twistient/HoloVec/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Twistient/HoloVec/releases/tag/v0.1.0
