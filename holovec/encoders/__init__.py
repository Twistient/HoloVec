"""
Encoders for transforming various data types into hypervectors.

This module provides encoders that map different data types (scalars,
sequences, structured data) into hypervector representations compatible
with VSA models.
"""

from holovec.encoders.base import (
    Encoder,
    ScalarEncoder,
    SequenceEncoder,
    StructuredEncoder,
)
from holovec.encoders.scalar import (
    FractionalPowerEncoder,
    ThermometerEncoder,
    LevelEncoder,
)
from holovec.encoders.vector import (
    VectorFPE,
)
from holovec.encoders.periodic import (
    PeriodicAngleEncoder,
    encode_day_of_week,
    encode_time_of_day,
)
from holovec.encoders.sequence import (
    PositionBindingEncoder,
    NGramEncoder,
    TrajectoryEncoder,
)
from holovec.encoders.structured import (
    VectorEncoder,
)
from holovec.encoders.spatial import (
    ImageEncoder,
)

__all__ = [
    # Base classes
    "Encoder",
    "ScalarEncoder",
    "SequenceEncoder",
    "StructuredEncoder",
    # Scalar encoders
    "FractionalPowerEncoder",
    "ThermometerEncoder",
    "LevelEncoder",
    "VectorFPE",
    "PeriodicAngleEncoder",
    "encode_day_of_week",
    "encode_time_of_day",
    # Sequence encoders
    "PositionBindingEncoder",
    "NGramEncoder",
    "TrajectoryEncoder",
    # Structured encoders
    "VectorEncoder",
    # Spatial encoders
    "ImageEncoder",
]
