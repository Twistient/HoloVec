"""VSA (Vector Symbolic Architecture) models for holovec.

This module provides different VSA models with various algebraic properties:
- MAP: Multiply-Add-Permute (self-inverse, simple)
- FHRR: Fourier HRR (exact inverse, best capacity)
- HRR: Holographic Reduced Representations (approximate inverse)
- BSC: Binary Spatter Codes (self-inverse, binary)
- GHRR: Generalized HRR (non-commutative, compositional structures)
- VTB: Vector-derived Transformation Binding (non-commutative)
- BSDC: Binary Sparse Distributed Codes (sparse, memory efficient)
"""

from .base import VSAModel
from .bsc import BSCModel
from .bsdc import BSDCModel
from .fhrr import FHRRModel
from .ghrr import GHRRModel
from .hrr import HRRModel
from .map import MAPModel
from .vtb import VTBModel

__all__ = [
    'VSAModel',
    'MAPModel',
    'FHRRModel',
    'HRRModel',
    'BSCModel',
    'BSDCModel',
    'GHRRModel',
    'VTBModel',
]
