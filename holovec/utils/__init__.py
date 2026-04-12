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

from .cleanup import (
    AttentionResonatorCleanup,
    BruteForceCleanup,
    CleanupStrategy,
    ResonatorCleanup,
)
from .cpse import (
    CPSEMetadata,
    generate_permutation_patterns,
    validate_cpse_convergence,
)
from .operations import (
    add_noise,
    select_top_k,
    similarity_matrix,
)
from .search import (
    batch_similarity,
    nearest_neighbors,
    threshold_search,
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
    'AttentionResonatorCleanup',
    # Search utilities
    'nearest_neighbors',
    'threshold_search',
    'batch_similarity',
    # General operations
    'select_top_k',
    'add_noise',
    'similarity_matrix',
]
