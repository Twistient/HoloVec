"""Utility functions for VSA operations.

This module provides utility functions for:
- CPSE/CPSD context-preserving encoding
- Cleanup and nearest-neighbor search
- Search utilities and codebook operations
- General utility operations

Sub-modules:
    cpse: CPSE/CPSD utilities for compositional encoding
    cleanup: Cleanup strategies including Resonator Networks
    search: Search utilities for codebook operations
    operations: General utility operations
"""

from .cpse import (
    CPSEMetadata,
    generate_permutation_patterns,
    validate_cpse_convergence,
)

from .cleanup import (
    CleanupStrategy,
    BruteForceCleanup,
    ResonatorCleanup,
)

from .search import (
    nearest_neighbors,
    threshold_search,
    batch_similarity,
)

from .operations import (
    select_top_k,
    add_noise,
    similarity_matrix,
)

__all__ = [
    # CPSE utilities
    'CPSEMetadata',
    'generate_permutation_patterns',
    'validate_cpse_convergence',
    # Cleanup strategies
    'CleanupStrategy',
    'BruteForceCleanup',
    'ResonatorCleanup',
    # Search utilities
    'nearest_neighbors',
    'threshold_search',
    'batch_similarity',
    # General operations
    'select_top_k',
    'add_noise',
    'similarity_matrix',
]
