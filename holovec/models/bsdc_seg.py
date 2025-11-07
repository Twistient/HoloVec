"""BSDC-SEG (Segmented Sparse Binary Codes) VSA model.

Implements a segment-sparse binary code model where the space ensures exactly
one active bit per segment. Binding uses XOR (self-inverse) as in BSDC; bundling
is segment-wise majority with deterministic tie-breaking.

Similarity is provided by the space (fraction of matching segments).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..backends import Backend
from ..backends.base import Array
from ..spaces import SparseSegmentSpace, VectorSpace
from .base import VSAModel


class BSDCSEGModel(VSAModel):
    """Segment-sparse binary VSA model (BSDC-SEG).

    Binding: XOR (element-wise, self-inverse)
    Unbinding: XOR (self-inverse)
    Bundling: segment-wise majority (exactly 1 per segment)
    Permutation: circular shift

    Uses SparseSegmentSpace with S segments (D % S == 0).
    """

    def __init__(
        self,
        dimension: int,
        segments: int,
        space: Optional[VectorSpace] = None,
        backend: Optional[Backend] = None,
        seed: Optional[int] = None,
    ):
        if space is None:
            from ..backends import get_backend
            backend = backend if backend is not None else get_backend()
            space = SparseSegmentSpace(dimension, segments=segments, backend=backend, seed=seed)

        super().__init__(space, backend)
        assert isinstance(space, SparseSegmentSpace)
        self.segments = space.segments
        self.segment_length = space.segment_length

    @property
    def model_name(self) -> str:
        return "BSDC-SEG"

    @property
    def is_self_inverse(self) -> bool:
        return True

    @property
    def is_commutative(self) -> bool:
        return True

    @property
    def is_exact_inverse(self) -> bool:
        return True

    def bind(self, a: Array, b: Array) -> Array:
        """Bind using XOR (self-inverse)."""
        return self.backend.xor(a, b)

    def unbind(self, a: Array, b: Array) -> Array:
        """Unbind using XOR (self-inverse)."""
        return self.bind(a, b)

    def bundle(self, vectors: Sequence[Array]) -> Array:
        """Segment-wise majority with exactly 1 winner per segment.

        Counts votes per index within each segment and selects the index with
        maximum count (deterministic tie-break: lowest index).
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")
        import numpy as _np
        # Normalize each to a valid segment pattern first
        seg_norm = [self.space.normalize(v) for v in vectors]
        arrs = [_np.array(self.backend.to_numpy(v)) for v in seg_norm]
        # Accumulate counts per segment position
        counts = _np.zeros((self.dimension,), dtype=_np.int32)
        for a in arrs:
            counts += a
        out = _np.zeros_like(counts, dtype=_np.int32)
        L = self.segment_length
        for s in range(self.segments):
            start = s * L
            end = start + L
            seg_counts = counts[start:end]
            idx = int(_np.argmax(seg_counts))  # deterministic tie-breaker
            out[start + idx] = 1
        return self.backend.from_numpy(out)

    def permute(self, vec: Array, k: int = 1) -> Array:
        return self.backend.roll(vec, shift=k, axis=0)

