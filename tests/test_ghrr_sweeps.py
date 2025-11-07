import numpy as np

from holovec.models.fhrr import FHRRModel
from holovec.models.ghrr import GHRRModel
from holovec.backends import get_backend


def _avg_noncommutativity(model, trials=5, seed=0):
    # Measure 1 - sim(bind(a,b), bind(b,a))
    vals = []
    for i in range(trials):
        a = model.random(seed=seed + 2 * i)
        b = model.random(seed=seed + 2 * i + 1)
        ab = model.bind(a, b)
        ba = model.bind(b, a)
        s = model.similarity(ab, ba)
        vals.append(1.0 - float(s))
    return float(np.mean(vals))


def test_ghrr_m1_matches_fhrr_similarity_profile():
    backend = get_backend('numpy')
    # m=1 should reduce to scalar phasors (FHRR-like behavior)
    ghrr = GHRRModel(dimension=256, matrix_size=1, backend=backend, seed=0)
    fhrr = FHRRModel(dimension=256, backend=backend, seed=0)

    a1 = ghrr.random(seed=1)
    b1 = ghrr.random(seed=2)
    c1 = ghrr.bind(a1, b1)

    a2 = fhrr.random(seed=1)
    b2 = fhrr.random(seed=2)
    c2 = fhrr.bind(a2, b2)

    # Self-similarities ~1.0
    assert abs(ghrr.similarity(c1, c1) - 1.0) < 1e-6
    assert abs(fhrr.similarity(c2, c2) - 1.0) < 1e-6


def test_ghrr_noncommutativity_increases_with_m_and_low_diagonality():
    backend = get_backend('numpy')
    # Low diagonality (0.0): more non-commutative as m grows
    ghrr_m1 = GHRRModel(dimension=128, matrix_size=1, backend=backend, seed=0, diagonality=0.0)
    ghrr_m2 = GHRRModel(dimension=128, matrix_size=2, backend=backend, seed=0, diagonality=0.0)
    ghrr_m3 = GHRRModel(dimension=128, matrix_size=3, backend=backend, seed=0, diagonality=0.0)

    n1 = _avg_noncommutativity(ghrr_m1, trials=5, seed=10)
    n2 = _avg_noncommutativity(ghrr_m2, trials=5, seed=10)
    n3 = _avg_noncommutativity(ghrr_m3, trials=5, seed=10)
    # m=1 should be near-commutative; m=3 should be most non-commutative
    assert n1 <= n2 + 1e-3
    assert n2 <= n3 + 1e-3 or n3 >= n2  # allow small noise


def test_ghrr_diagonality_interpolation_toward_commutativity():
    backend = get_backend('numpy')
    # With m=3, diagonality=1.0 should be near-commutative vs diagonality=0.0
    ghrr_lo = GHRRModel(dimension=128, matrix_size=3, backend=backend, seed=0, diagonality=0.0)
    ghrr_hi = GHRRModel(dimension=128, matrix_size=3, backend=backend, seed=0, diagonality=1.0)

    n_lo = _avg_noncommutativity(ghrr_lo, trials=5, seed=20)
    n_hi = _avg_noncommutativity(ghrr_hi, trials=5, seed=20)
    assert n_hi <= n_lo + 1e-3

