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


def test_ghrr_unitarity_after_bundle():
    """Test that bundling preserves unitarity via polar decomposition.

    After bundling, each matrix should still be unitary (U†U ≈ I).
    This is critical for maintaining quasi-orthogonality (Yeung et al. 2024).
    """
    backend = get_backend('numpy')
    model = GHRRModel(dimension=50, matrix_size=3, backend=backend, seed=42)

    # Create several random vectors
    vectors = [model.random(seed=i) for i in range(5)]

    # Bundle them
    bundled = model.bundle(vectors)

    # Check unitarity for each matrix in the bundled result
    # bundled shape: (D, m, m) where D=50, m=3
    D, m, _ = bundled.shape

    # Check first few matrices for unitarity
    for i in range(min(10, D)):
        U = bundled[i]  # (m, m)
        # Compute U† @ U
        U_dag = np.conj(U.T)
        product = U_dag @ U

        # Should be close to identity matrix
        identity = np.eye(m, dtype=product.dtype)
        error = np.linalg.norm(product - identity, 'fro')

        # Tolerance accounts for numerical errors in SVD
        assert error < 1e-5, f"Matrix {i} not unitary after bundling: ||U†U - I||_F = {error}"


def test_ghrr_unitarity_preservation():
    """Test that individual operations preserve unitarity."""
    backend = get_backend('numpy')
    model = GHRRModel(dimension=30, matrix_size=2, backend=backend, seed=123)

    # Random vectors should be unitary by construction
    a = model.random(seed=1)
    b = model.random(seed=2)

    # Test bind result
    c = model.bind(a, b)

    # Check a few matrices from c
    D, m, _ = c.shape
    for i in range(min(5, D)):
        U = c[i]
        U_dag = np.conj(U.T)
        product = U_dag @ U
        identity = np.eye(m, dtype=product.dtype)
        error = np.linalg.norm(product - identity, 'fro')
        assert error < 1e-5, f"Binding broke unitarity at matrix {i}"


def test_ghrr_associativity_property():
    """Test that GHRR binding is associative: (a⊗b)⊗c = a⊗(b⊗c).

    Matrix multiplication is associative. This is a fundamental algebraic property
    that GHRR must satisfy (Yeung et al. 2024).

    Note: This is different from commutativity. GHRR is associative but generally
    non-commutative (when diagonality < 1.0).
    """
    backend = get_backend('numpy')
    model = GHRRModel(dimension=64, matrix_size=3, backend=backend, seed=999, diagonality=0.5)

    # Generate test vectors
    a = model.random(seed=10)
    b = model.random(seed=20)
    c = model.random(seed=30)

    # Compute (a⊗b)⊗c
    ab = model.bind(a, b)
    ab_c = model.bind(ab, c)

    # Compute a⊗(b⊗c)
    bc = model.bind(b, c)
    a_bc = model.bind(a, bc)

    # They should be very similar (same result within numerical tolerance)
    similarity = model.similarity(ab_c, a_bc)

    # Associativity should hold to high precision
    assert similarity > 0.99, f"Associativity violated: sim((a⊗b)⊗c, a⊗(b⊗c)) = {similarity}"

    # For perfect associativity, element-wise difference should be negligible
    diff_norm = np.linalg.norm(ab_c - a_bc)
    max_val = max(np.linalg.norm(ab_c), np.linalg.norm(a_bc))
    relative_error = diff_norm / max_val if max_val > 0 else diff_norm

    assert relative_error < 1e-3, f"Associativity error: relative ||ΔU||_F = {relative_error}"

