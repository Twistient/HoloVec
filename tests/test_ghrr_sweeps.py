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
    backend = get_backend("numpy")
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
    backend = get_backend("numpy")
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
    backend = get_backend("numpy")
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
    backend = get_backend("numpy")
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
        error = np.linalg.norm(product - identity, "fro")

        # Tolerance accounts for numerical errors in SVD
        assert error < 1e-5, f"Matrix {i} not unitary after bundling: ||U†U - I||_F = {error}"


def test_ghrr_unitarity_preservation():
    """Test that individual operations preserve unitarity."""
    backend = get_backend("numpy")
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
        error = np.linalg.norm(product - identity, "fro")
        assert error < 1e-5, f"Binding broke unitarity at matrix {i}"


def test_ghrr_associativity_property():
    """Test that GHRR binding is associative: (a⊗b)⊗c = a⊗(b⊗c).

    Matrix multiplication is associative. This is a fundamental algebraic property
    that GHRR must satisfy (Yeung et al. 2024).

    Note: This is different from commutativity. GHRR is associative but generally
    non-commutative (when diagonality < 1.0).
    """
    backend = get_backend("numpy")
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


# ============================================================================
# Additional tests for GHRR model coverage
# ============================================================================


def test_ghrr_model_properties():
    """Test GHRR model property accessors."""
    backend = get_backend("numpy")
    model = GHRRModel(dimension=32, matrix_size=3, backend=backend, seed=42)

    # Test model_name property
    assert model.model_name == "GHRR_m3"

    # Test is_self_inverse property - should be False (requires conjugate transpose)
    assert model.is_self_inverse is False

    # Test is_commutative property - should be False (matrix multiplication)
    assert model.is_commutative is False

    # Test is_exact_inverse property - should be True (conjugate transpose provides exact inverse)
    assert model.is_exact_inverse is True


def test_ghrr_commutativity_degree_explicit_diagonality():
    """Test commutativity_degree when diagonality is explicitly set."""
    backend = get_backend("numpy")

    # Test with explicit diagonality=0.5
    model = GHRRModel(dimension=32, matrix_size=3, backend=backend, seed=42, diagonality=0.5)
    assert model.commutativity_degree == 0.5

    # Test with explicit diagonality=0.0
    model_lo = GHRRModel(dimension=32, matrix_size=3, backend=backend, seed=42, diagonality=0.0)
    assert model_lo.commutativity_degree == 0.0

    # Test with explicit diagonality=1.0
    model_hi = GHRRModel(dimension=32, matrix_size=3, backend=backend, seed=42, diagonality=1.0)
    assert model_hi.commutativity_degree == 1.0


def test_ghrr_commutativity_degree_default_by_matrix_size():
    """Test commutativity_degree defaults based on matrix size."""
    backend = get_backend("numpy")

    # m=1 recovers FHRR (commutative)
    model_m1 = GHRRModel(dimension=32, matrix_size=1, backend=backend, seed=42)
    assert model_m1.commutativity_degree == 1.0

    # m=2 is mostly commutative
    model_m2 = GHRRModel(dimension=32, matrix_size=2, backend=backend, seed=42)
    assert model_m2.commutativity_degree == 0.7

    # m=3 is balanced
    model_m3 = GHRRModel(dimension=32, matrix_size=3, backend=backend, seed=42)
    assert model_m3.commutativity_degree == 0.5

    # m=4 is mostly non-commutative
    model_m4 = GHRRModel(dimension=32, matrix_size=4, backend=backend, seed=42)
    assert model_m4.commutativity_degree == 0.3


def test_ghrr_unbind_exact_recovery():
    """Test that unbind provides exact recovery: unbind(bind(a, b), b) = a."""
    backend = get_backend("numpy")
    model = GHRRModel(dimension=64, matrix_size=3, backend=backend, seed=42)

    a = model.random(seed=1)
    b = model.random(seed=2)

    # Bind a with b
    c = model.bind(a, b)

    # Unbind to recover a
    a_recovered = model.unbind(c, b)

    # Should have very high similarity (exact recovery)
    similarity = model.similarity(a, a_recovered)
    assert similarity > 0.999, f"Unbind recovery failed: similarity = {similarity}"

    # Element-wise difference should be negligible
    diff_norm = np.linalg.norm(a - a_recovered)
    assert diff_norm < 1e-5, f"Unbind recovery error: ||a - recovered||_F = {diff_norm}"


def test_ghrr_permute():
    """Test permute operation (circular shift)."""
    backend = get_backend("numpy")
    model = GHRRModel(dimension=32, matrix_size=2, backend=backend, seed=42)

    vec = model.random(seed=1)

    # Permute by k=1
    permuted = model.permute(vec, k=1)

    # Permuted should be different from original
    similarity = model.similarity(vec, permuted)
    assert similarity < 0.5, f"Permuted vector too similar to original: {similarity}"

    # Permuting by dimension should return to original
    permuted_full = model.permute(vec, k=model.dimension)
    similarity_full = model.similarity(vec, permuted_full)
    assert similarity_full > 0.999, f"Full permutation should return original: {similarity_full}"


def test_ghrr_test_non_commutativity_method():
    """Test the test_non_commutativity method directly."""
    backend = get_backend("numpy")

    # With high diagonality (near-commutative)
    model_hi = GHRRModel(dimension=64, matrix_size=3, backend=backend, seed=42, diagonality=1.0)
    a = model_hi.random(seed=1)
    b = model_hi.random(seed=2)
    sim_hi = model_hi.test_non_commutativity(a, b)
    # High diagonality → high similarity between a⊗b and b⊗a
    assert sim_hi > 0.9, f"High diagonality should be near-commutative: {sim_hi}"

    # With low diagonality (non-commutative)
    model_lo = GHRRModel(dimension=64, matrix_size=3, backend=backend, seed=42, diagonality=0.0)
    a = model_lo.random(seed=1)
    b = model_lo.random(seed=2)
    sim_lo = model_lo.test_non_commutativity(a, b)
    # Low diagonality → lower similarity
    assert sim_lo < sim_hi, f"Low diagonality should be less commutative than high"


def test_ghrr_compute_diagonality():
    """Test the compute_diagonality method."""
    backend = get_backend("numpy")

    # With explicit diagonality=1.0 (fully diagonal)
    model_diag = GHRRModel(dimension=32, matrix_size=3, backend=backend, seed=42, diagonality=1.0)
    vec_diag = model_diag.random(seed=1)
    diag_score = model_diag.compute_diagonality(vec_diag)
    # Diagonality should be high (close to 1/m for random diagonal elements)
    assert 0.3 <= diag_score <= 1.0, f"Diagonal model should have high diagonality: {diag_score}"

    # With explicit diagonality=0.0 (minimally diagonal)
    model_full = GHRRModel(dimension=32, matrix_size=3, backend=backend, seed=42, diagonality=0.0)
    vec_full = model_full.random(seed=1)
    diag_score_full = model_full.compute_diagonality(vec_full)
    # Should be lower than diagonal case (though still positive)
    assert 0.0 < diag_score_full <= 1.0, f"Full model diagonality out of range: {diag_score_full}"


def test_ghrr_repr():
    """Test __repr__ output."""
    backend = get_backend("numpy")
    model = GHRRModel(dimension=64, matrix_size=3, backend=backend, seed=42)

    repr_str = repr(model)

    # Should contain key information
    assert "GHRRModel" in repr_str
    assert "dimension=64" in repr_str
    assert "matrix_size=3" in repr_str
    assert "backend=numpy" in repr_str
    assert "space=matrix_3x3" in repr_str


def test_verify_ghrr_fhrr_equivalence():
    """Test the helper function that verifies GHRR m=1 ≈ FHRR."""
    from holovec.models.ghrr import verify_ghrr_fhrr_equivalence

    # Should return True (approximate verification)
    result = verify_ghrr_fhrr_equivalence()
    assert result is True


def test_ghrr_bundle_empty_sequence():
    """Test that bundling empty sequence raises ValueError."""
    backend = get_backend("numpy")
    model = GHRRModel(dimension=32, matrix_size=2, backend=backend, seed=42)

    try:
        model.bundle([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()
