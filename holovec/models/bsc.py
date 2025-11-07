"""BSC (Binary Spatter Codes) VSA model.

BSC uses XOR for binding on binary vectors {0, 1}. It's mathematically
equivalent to MAP with bipolar vectors via the transformation x → 2x-1.

Properties:
- Binding: XOR
- Unbinding: XOR (self-inverse)
- Commutative: Yes
- Exact inverse: Yes
- Complexity: O(D) via bitwise XOR
- Simple hardware implementation

References:
- Kanerva (1996): "Binary Spatter-Coding of Ordered K-Tuples"
- Kanerva (2009): "Hyperdimensional Computing"
- Schlegel et al. (2021): Comparison showing equivalence to MAP
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..backends import Backend
from ..backends.base import Array
from ..spaces import BinarySpace, VectorSpace
from .base import VSAModel


class BSCModel(VSAModel):
    """BSC (Binary Spatter Codes) model.

    Binding: XOR
    Unbinding: XOR (self-inverse)
    Bundling: element-wise addition + majority vote
    Permutation: circular shift

    Uses BinarySpace with values in {0, 1}.
    """

    def __init__(
        self,
        dimension: int = 10000,
        space: Optional[VectorSpace] = None,
        backend: Optional[Backend] = None,
        seed: Optional[int] = None
    ):
        """Initialize BSC model.

        Args:
            dimension: Dimensionality of hypervectors
            space: Vector space (defaults to BinarySpace)
            backend: Computational backend
            seed: Random seed for space
        """
        if space is None:
            from ..backends import get_backend
            backend = backend if backend is not None else get_backend()
            space = BinarySpace(dimension, backend=backend, seed=seed)

        super().__init__(space, backend)

    @property
    def model_name(self) -> str:
        return "BSC"

    @property
    def is_self_inverse(self) -> bool:
        return True  # XOR is self-inverse

    @property
    def is_commutative(self) -> bool:
        return True  # XOR is commutative

    @property
    def is_exact_inverse(self) -> bool:
        return True  # XOR provides exact inverse

    def bind(self, a: Array, b: Array) -> Array:
        """Bind using XOR.

        For binary vectors: a XOR b
        Property: a XOR b XOR b = a (self-inverse)

        Args:
            a: First vector (binary {0, 1})
            b: Second vector (binary {0, 1})

        Returns:
            Bound vector c = a XOR b
        """
        return self.backend.xor(a, b)

    def unbind(self, a: Array, b: Array) -> Array:
        """Unbind using XOR (self-inverse).

        Since XOR is self-inverse: unbind(c, b) = c XOR b

        Args:
            a: Bound vector (or first operand)
            b: Second operand

        Returns:
            Unbound vector (exact recovery)
        """
        # For BSC, binding = unbinding (self-inverse)
        return self.bind(a, b)

    def bundle(self, vectors: Sequence[Array]) -> Array:
        """Bundle using element-wise addition + majority vote.

        Sum all binary vectors element-wise, then threshold at n/2
        where n is the number of vectors.

        Args:
            vectors: Sequence of vectors to bundle

        Returns:
            Bundled vector (binary {0, 1})

        Raises:
            ValueError: If vectors is empty
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")

        vectors = list(vectors)
        n = len(vectors)

        # Sum all vectors (each element is 0 or 1)
        summed = self.backend.sum(self.backend.stack(vectors, axis=0), axis=0)

        # Majority vote: threshold at n/2
        threshold = n / 2.0
        result = self.backend.threshold(summed, threshold=threshold, above=1.0, below=0.0)

        # Ensure binary dtype
        return result.astype(self.space.dtype) if hasattr(result, 'astype') else result

    def permute(self, vec: Array, k: int = 1) -> Array:
        """Permute using circular shift.

        Shifts vector elements by k positions to the right.
        Negative k shifts left.

        Args:
            vec: Vector to permute
            k: Number of positions to shift

        Returns:
            Permuted vector
        """
        return self.backend.roll(vec, shift=k)

    def to_bipolar(self, vec: Array) -> Array:
        """Convert binary {0, 1} to bipolar {-1, +1}.

        Transformation: x → 2x - 1

        Args:
            vec: Binary vector

        Returns:
            Bipolar vector
        """
        return 2 * vec - 1

    def from_bipolar(self, vec: Array) -> Array:
        """Convert bipolar {-1, +1} to binary {0, 1}.

        Transformation: x → (x + 1) / 2

        Args:
            vec: Bipolar vector

        Returns:
            Binary vector
        """
        return (vec + 1) / 2

    def __repr__(self) -> str:
        return (f"BSCModel(dimension={self.dimension}, "
                f"space={self.space.space_name}, "
                f"backend={self.backend.name})")
