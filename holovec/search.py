"""Public search facade.

Re-exports user-facing search helpers. Internally backed by utils.search.
"""

from .utils.search import (
    nearest_neighbors,
    threshold_search,
    batch_similarity,
    segment_pattern,
    find_by_segment_pattern,
)

__all__ = [
    "nearest_neighbors",
    "threshold_search",
    "batch_similarity",
    "segment_pattern",
    "find_by_segment_pattern",
]

