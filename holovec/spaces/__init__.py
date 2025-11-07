"""Vector spaces for hyperdimensional computing.

This module provides different vector spaces used by various VSA models.
Each space defines how random vectors are generated and what similarity
metric is appropriate.
"""

from .base import ContinuousSpace, DiscreteSpace, VectorSpace
from .spaces import (
    BinarySpace,
    BipolarSpace,
    ComplexSpace,
    MatrixSpace,
    RealSpace,
    SparseSpace,
    SparseSegmentSpace,
    create_space,
)

__all__ = [
    'VectorSpace',
    'DiscreteSpace',
    'ContinuousSpace',
    'BipolarSpace',
    'BinarySpace',
    'RealSpace',
    'ComplexSpace',
    'SparseSpace',
    'SparseSegmentSpace',
    'MatrixSpace',
    'create_space',
]
