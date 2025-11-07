from __future__ import annotations

from typing import Iterable, List, Optional
import math

from holovec.encoders.base import ScalarEncoder
from holovec.models.base import VSAModel
from holovec.backends.base import Array


class PeriodicAngleEncoder(ScalarEncoder):
    """Encode angles on a circle with correct periodicity.

    Encodes θ (in radians by default) using harmonic features:
        φ(θ) = exp(i * k_j * θ)  (complex) or cos(k_j * θ) (real)
    where k_j are integer harmonics assigned per embedding dimension.

    Parameters
    ----------
    model : VSAModel
        Destination model (complex preferred)
    harmonics : List[int] | int
        Either explicit list of harmonics or a maximum K to sample from 1..K
    radians : bool
        If False, inputs are degrees and are converted to radians.
    seed : int | None
        RNG seed
    """

    def __init__(
        self,
        model: VSAModel,
        harmonics: List[int] | int = 3,
        radians: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(model, min_val=0.0, max_val=2 * math.pi)
        self.is_complex = self.model.space.space_name in ("complex",) or "matrix" in self.model.space.space_name
        self.radians = bool(radians)
        self.seed = seed

        if isinstance(harmonics, int):
            H = int(harmonics)
            if H < 1:
                raise ValueError("harmonics must be >= 1")
            self.harmonics = list(range(1, H + 1))
        else:
            self.harmonics = [int(h) for h in harmonics]
            if not self.harmonics:
                raise ValueError("harmonics list must be non-empty")

        self._assign_harmonics()

    def _assign_harmonics(self) -> None:
        """Assign harmonics to each dimension using deterministic random sampling.

        Uses local numpy import for random number generation, then converts to backend.
        """
        import numpy as _np

        rng = _np.random.default_rng(self.seed)
        D = self.dimension
        H = len(self.harmonics)
        # Assign each dimension a harmonic
        idx = rng.integers(low=0, high=H, size=(D,))
        k_numpy = _np.array([self.harmonics[i] for i in idx], dtype=_np.float32)
        # Convert to backend array
        self.k_arr = self.backend.from_numpy(k_numpy)

    def encode(self, value: float) -> Array:
        """Encode angle using harmonic basis functions.

        Parameters
        ----------
        value : float
            Angle in radians (default) or degrees (if radians=False).

        Returns
        -------
        Array
            Hypervector encoding of the angle with correct periodicity.
        """
        theta = float(value)
        if not self.radians:
            # Convert degrees to radians: θ_rad = θ_deg × π/180
            theta = math.radians(theta)
        # Wrap to [0, 2π) for numerical stability
        theta = theta % (2 * math.pi)

        # angle per dimension: k_j * θ
        angles = self.backend.multiply_scalar(self.k_arr, theta)
        if self.is_complex:
            # Complex exponential: e^(i*k_j*θ)
            return self.backend.exp(1j * angles)
        else:
            # Real part only: cos(k_j*θ)
            return self.backend.real(self.backend.exp(1j * angles))

    def decode(self, hv: Array, resolution: int = 720) -> float:
        """Approximate decode by grid search over [0, 2π).

        Parameters
        ----------
        hv : Array
            Encoded angle hypervector.
        resolution : int, optional
            Number of grid points to search. Default 720 (0.5° resolution).

        Returns
        -------
        float
            Decoded angle in radians (default) or degrees (if radians=False).

        Notes
        -----
        Uses brute-force grid search over the angle space. Resolution of 720
        provides 0.5° precision, which is sufficient for most applications.
        """
        # Generate grid of candidate angles
        import numpy as _np
        grid = _np.linspace(0.0, 2 * math.pi, num=resolution, endpoint=False)

        best_t = 0.0
        best_s = -1e9
        for t in grid:
            # Encode candidate angle
            enc = self.encode(t if self.radians else math.degrees(t))
            sim = float(self.model.similarity(hv, enc))
            if sim > best_s:
                best_s = sim
                best_t = t

        # Convert back to degrees if needed
        return best_t if self.radians else math.degrees(best_t)

    @property
    def is_reversible(self) -> bool:
        return True

    @property
    def compatible_models(self) -> list[str]:
        return ["FHRR", "HRR"]

    @property
    def input_type(self) -> str:
        return "scalar angle (radians by default)"


# Convenience thin wrappers for common periodic domains
def encode_day_of_week(model: VSAModel, day_index: int, harmonics: int = 3, seed: Optional[int] = None) -> Array:
    """Encode day of week as periodic angle on 7-cycle (0..6)."""
    enc = PeriodicAngleEncoder(model, harmonics=harmonics, radians=True, seed=seed)
    theta = 2 * np.pi * (int(day_index) % 7) / 7.0
    return enc.encode(theta)


def encode_time_of_day(model: VSAModel, hour: float, harmonics: int = 3, seed: Optional[int] = None) -> Array:
    """Encode time of day (0..24) as periodic angle."""
    enc = PeriodicAngleEncoder(model, harmonics=harmonics, radians=True, seed=seed)
    theta = 2 * np.pi * (float(hour) % 24.0) / 24.0
    return enc.encode(theta)
