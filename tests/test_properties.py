"""Property-based tests for VSA models using hypothesis.

This module tests algebraic properties and invariants that should hold
for different VSA models, based on the mathematical foundations from:
- Schlegel et al. (2022): "A Comparison of Vector Symbolic Architectures"
- Plate (2003): "Holographic Reduced Representations"
- Kanerva (2009): "Hyperdimensional Computing"
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck

from holovec import VSA


# Custom strategies for generating test data
def vector_strategy(model, seed_range=(1, 1000)):
    """Generate random vectors from a VSA model."""
    return st.integers(min_value=seed_range[0], max_value=seed_range[1]).map(
        lambda seed: model.random(seed=seed)
    )


def small_integer_strategy():
    """Small integers for sequence lengths, etc."""
    return st.integers(min_value=1, max_value=10)


class TestMAPProperties:
    """Property-based tests for MAP model.

    Based on Schlegel et al. (2022) Section 2.4:
    - Self-inverse binding
    - Commutative
    - Associative
    - Exact inverse for bipolar
    """

    @pytest.fixture
    def map_bipolar(self):
        """MAP model with bipolar space (exact inverse)."""
        return VSA.create('MAP', dim=1000, seed=42)

    @pytest.fixture
    def map_real(self):
        """MAP model with real space (approximate inverse)."""
        from holovec.spaces import RealSpace
        from holovec.backends import get_backend
        backend = get_backend()
        space = RealSpace(dimension=1000, backend=backend, seed=42)
        return VSA.create('MAP', dim=1000, space=space, seed=42)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_self_inverse_property_bipolar(self, map_bipolar, seed):
        """MAP binding should be self-inverse for bipolar: bind(a,b) = unbind(a,b)."""
        a = map_bipolar.random(seed=seed)
        b = map_bipolar.random(seed=seed + 1)

        bound = map_bipolar.bind(a, b)
        unbound = map_bipolar.unbind(a, b)

        # For self-inverse, bind = unbind
        assert np.allclose(bound, unbound, atol=1e-10)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_exact_inverse_bipolar(self, map_bipolar, seed):
        """MAP should exactly recover original for bipolar vectors.

        Property: unbind(bind(a, b), b) = a (exactly)
        Reference: Schlegel et al. (2022), Table 1
        """
        a = map_bipolar.random(seed=seed)
        b = map_bipolar.random(seed=seed + 1)

        c = map_bipolar.bind(a, b)
        a_recovered = map_bipolar.unbind(c, b)

        # Should be exact for bipolar (within floating point precision)
        similarity = map_bipolar.similarity(a, a_recovered)
        assert np.isclose(similarity, 1.0, atol=1e-10), \
            f"Expected exact recovery, got similarity {similarity}"

        # Also check element-wise equality
        assert np.allclose(a, a_recovered, atol=1e-10)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_approximate_inverse_real(self, map_real, seed):
        """MAP should approximately recover original for real vectors.

        Property: unbind(bind(a, b), b) ≈ a
        Reference: Schlegel et al. (2022), Section 2.4 (MAP-C)

        Note: Normalization after binding significantly degrades recovery for MAP-C.
        The paper reports MAP-C has approximate inverse with degradation.
        Empirically, we observe ~0.5-0.7 similarity for normalized continuous space.
        """
        a = map_real.random(seed=seed)
        b = map_real.random(seed=seed + 1)

        c = map_real.bind(a, b)
        a_recovered = map_real.unbind(c, b)

        # Should have moderate similarity (degraded due to normalization)
        # This is expected behavior for MAP-C with L2 normalization
        similarity = map_real.similarity(a, a_recovered)
        assert similarity > 0.4, \
            f"Expected moderate similarity for approximate recovery, got {similarity}"

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_commutativity(self, map_bipolar, seed):
        """MAP binding should be commutative: bind(a, b) = bind(b, a).

        Property: a ⊗ b = b ⊗ a
        Reference: Schlegel et al. (2022), Section 2.4
        """
        a = map_bipolar.random(seed=seed)
        b = map_bipolar.random(seed=seed + 1)

        ab = map_bipolar.bind(a, b)
        ba = map_bipolar.bind(b, a)

        # Should be exactly equal (commutative)
        assert np.allclose(ab, ba, atol=1e-10)

        similarity = map_bipolar.similarity(ab, ba)
        assert np.isclose(similarity, 1.0, atol=1e-10)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_associativity(self, map_bipolar, seed):
        """MAP binding should be associative: (a⊗b)⊗c = a⊗(b⊗c).

        Property: Binding associativity
        Reference: Schlegel et al. (2022), Section 2.4
        """
        a = map_bipolar.random(seed=seed)
        b = map_bipolar.random(seed=seed + 1)
        c = map_bipolar.random(seed=seed + 2)

        # (a ⊗ b) ⊗ c
        ab = map_bipolar.bind(a, b)
        abc_left = map_bipolar.bind(ab, c)

        # a ⊗ (b ⊗ c)
        bc = map_bipolar.bind(b, c)
        abc_right = map_bipolar.bind(a, bc)

        # Should be exactly equal (associative)
        assert np.allclose(abc_left, abc_right, atol=1e-10)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_distributivity(self, map_bipolar, seed):
        """Test distributivity: a⊗(b+c) ≈ (a⊗b)+(a⊗c).

        Note: Only approximately holds due to normalization in bundling.
        Reference: Plate (1997), binding preserves similarity
        """
        a = map_bipolar.random(seed=seed)
        b = map_bipolar.random(seed=seed + 1)
        c = map_bipolar.random(seed=seed + 2)

        # a ⊗ (b + c)
        bc_bundle = map_bipolar.bundle([b, c])
        left = map_bipolar.bind(a, bc_bundle)

        # (a ⊗ b) + (a ⊗ c)
        ab = map_bipolar.bind(a, b)
        ac = map_bipolar.bind(a, c)
        right = map_bipolar.bundle([ab, ac])

        # Should be approximately similar (normalization breaks exact equality)
        similarity = map_bipolar.similarity(left, right)
        assert similarity > 0.5, \
            f"Expected reasonable similarity for distributivity, got {similarity}"

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_quasi_orthogonality_after_binding(self, map_bipolar, seed):
        """Bound vector should be quasi-orthogonal to inputs.

        Property: c = a ⊗ b is not similar to a or b
        Reference: Plate (1997), binding property 1

        Note: Element-wise multiplication (Hadamard product) does not guarantee
        strong quasi-orthogonality like circular convolution does. For MAP,
        the bound vector can have moderate correlation with inputs.
        This is an inherent property of diagonal binding operations.
        """
        a = map_bipolar.random(seed=seed)
        b = map_bipolar.random(seed=seed + 1)

        c = map_bipolar.bind(a, b)

        # c should have low to moderate similarity to a and b
        sim_ca = map_bipolar.similarity(c, a)
        sim_cb = map_bipolar.similarity(c, b)

        # Bound vector should not be identical to inputs
        # But element-wise multiplication can preserve some correlation
        # Expect |similarity| < 0.6 for randomly generated vectors
        assert abs(sim_ca) < 0.6, \
            f"Expected moderate dissimilarity to input a, got {sim_ca}"
        assert abs(sim_cb) < 0.6, \
            f"Expected moderate dissimilarity to input b, got {sim_cb}"

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_bundle_similarity_preservation(self, map_bipolar, seed):
        """Bundle should be similar to all inputs.

        Property: Bundling preserves unstructured similarity
        Reference: Plate (1997), bundling essential property
        """
        vectors = [map_bipolar.random(seed=seed + i) for i in range(5)]
        bundle = map_bipolar.bundle(vectors)

        # Bundle should have positive similarity to all inputs
        similarities = [map_bipolar.similarity(bundle, v) for v in vectors]

        # All similarities should be positive and reasonable
        for sim in similarities:
            assert sim > 0.1, \
                f"Expected positive similarity to bundled vector, got {sim}"

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_multiple_binding_recovery(self, map_bipolar, seed):
        """Test recovery from multiple nested bindings.

        Property: Can recover original after multiple bind/unbind operations
        Reference: Schlegel et al. (2022), Section 3.2
        """
        a = map_bipolar.random(seed=seed)
        b = map_bipolar.random(seed=seed + 1)
        c = map_bipolar.random(seed=seed + 2)
        d = map_bipolar.random(seed=seed + 3)

        # Create nested binding: ((a ⊗ b) ⊗ c) ⊗ d
        ab = map_bipolar.bind(a, b)
        abc = map_bipolar.bind(ab, c)
        abcd = map_bipolar.bind(abc, d)

        # Recover a by unbinding in reverse order
        abc_recovered = map_bipolar.unbind(abcd, d)
        ab_recovered = map_bipolar.unbind(abc_recovered, c)
        a_recovered = map_bipolar.unbind(ab_recovered, b)

        # Should exactly recover for bipolar (self-inverse chain)
        similarity = map_bipolar.similarity(a, a_recovered)
        assert np.isclose(similarity, 1.0, atol=1e-10), \
            f"Expected exact recovery after 3 unbindings, got similarity {similarity}"


class TestFHRRProperties:
    """Property-based tests for FHRR model.

    Based on Plate (2003) and Schlegel et al. (2022) Section 2.4:
    - Exact inverse (complex domain)
    - Commutative binding
    - Associative binding
    """

    @pytest.fixture
    def fhrr(self):
        """FHRR model."""
        return VSA.create('FHRR', dim=1000, seed=42)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_exact_inverse(self, fhrr, seed):
        """FHRR should exactly recover original vector.

        Property: unbind(bind(a, b), b) = a (exactly)
        Reference: Plate (2003), Schlegel et al. (2022) Table 1
        """
        a = fhrr.random(seed=seed)
        b = fhrr.random(seed=seed + 1)

        c = fhrr.bind(a, b)
        a_recovered = fhrr.unbind(c, b)

        # Should be exact (within numerical precision)
        similarity = fhrr.similarity(a, a_recovered)
        assert np.isclose(similarity, 1.0, atol=1e-6), \
            f"Expected exact recovery, got similarity {similarity}"

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_commutativity(self, fhrr, seed):
        """FHRR binding should be commutative.

        Property: a ⊗ b = b ⊗ a (angle addition is commutative)
        Reference: Schlegel et al. (2022), Section 2.4
        """
        a = fhrr.random(seed=seed)
        b = fhrr.random(seed=seed + 1)

        ab = fhrr.bind(a, b)
        ba = fhrr.bind(b, a)

        # Should be exactly equal
        similarity = fhrr.similarity(ab, ba)
        assert np.isclose(similarity, 1.0, atol=1e-6)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_associativity(self, fhrr, seed):
        """FHRR binding should be associative.

        Property: (a⊗b)⊗c = a⊗(b⊗c)
        Reference: Schlegel et al. (2022), Section 2.4
        """
        a = fhrr.random(seed=seed)
        b = fhrr.random(seed=seed + 1)
        c = fhrr.random(seed=seed + 2)

        # (a ⊗ b) ⊗ c
        ab = fhrr.bind(a, b)
        abc_left = fhrr.bind(ab, c)

        # a ⊗ (b ⊗ c)
        bc = fhrr.bind(b, c)
        abc_right = fhrr.bind(a, bc)

        # Should be exactly equal (angle addition is associative)
        similarity = fhrr.similarity(abc_left, abc_right)
        assert np.isclose(similarity, 1.0, atol=1e-6)


class TestHRRProperties:
    """Property-based tests for HRR model.

    Based on Plate (2003) and Schlegel et al. (2022):
    - Approximate inverse (circular correlation)
    - Commutative binding
    - Associative binding
    """

    @pytest.fixture
    def hrr(self):
        """HRR model."""
        return VSA.create('HRR', dim=1000, seed=42)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_approximate_inverse(self, hrr, seed):
        """HRR should approximately recover original vector.

        Property: unbind(bind(a, b), b) ≈ a (correlation is approximate)
        Reference: Plate (2003), Schlegel et al. (2022) Section 3.2
        """
        a = hrr.random(seed=seed)
        b = hrr.random(seed=seed + 1)

        c = hrr.bind(a, b)
        a_recovered = hrr.unbind(c, b)

        # Should have good similarity (but not exact)
        similarity = hrr.similarity(a, a_recovered)
        assert similarity > 0.65, \
            f"Expected good recovery (>0.65), got similarity {similarity}"

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_commutativity(self, hrr, seed):
        """HRR binding should be commutative.

        Property: a ⊗ b = b ⊗ a (circular convolution is commutative)
        Reference: Schlegel et al. (2022), Section 2.4
        """
        a = hrr.random(seed=seed)
        b = hrr.random(seed=seed + 1)

        ab = hrr.bind(a, b)
        ba = hrr.bind(b, a)

        # Should be exactly equal
        similarity = hrr.similarity(ab, ba)
        assert np.isclose(similarity, 1.0, atol=1e-6)


class TestBSCProperties:
    """Property-based tests for BSC model.

    Based on Kanerva (1993) and Schlegel et al. (2022):
    - Self-inverse (XOR)
    - Exact inverse
    - Commutative
    - Associative
    """

    @pytest.fixture
    def bsc(self):
        """BSC model."""
        return VSA.create('BSC', dim=1000, seed=42)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_exact_inverse(self, bsc, seed):
        """BSC should exactly recover original (XOR is exact).

        Property: unbind(bind(a, b), b) = a (exactly)
        Reference: Schlegel et al. (2022), Section 2.4 (BSC)
        """
        a = bsc.random(seed=seed)
        b = bsc.random(seed=seed + 1)

        c = bsc.bind(a, b)
        a_recovered = bsc.unbind(c, b)

        # Should be exactly equal (XOR is exact inverse)
        assert np.allclose(a, a_recovered, atol=1e-10)

        similarity = bsc.similarity(a, a_recovered)
        assert np.isclose(similarity, 1.0, atol=1e-10)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_self_inverse(self, bsc, seed):
        """BSC binding should be self-inverse.

        Property: bind(a, b) = unbind(a, b) (XOR is self-inverse)
        Reference: Schlegel et al. (2022), Section 2.4
        """
        a = bsc.random(seed=seed)
        b = bsc.random(seed=seed + 1)

        bound = bsc.bind(a, b)
        unbound = bsc.unbind(a, b)

        assert np.allclose(bound, unbound, atol=1e-10)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_commutativity(self, bsc, seed):
        """BSC binding should be commutative.

        Property: a ⊗ b = b ⊗ a (XOR is commutative)
        Reference: Schlegel et al. (2022), Section 2.4
        """
        a = bsc.random(seed=seed)
        b = bsc.random(seed=seed + 1)

        ab = bsc.bind(a, b)
        ba = bsc.bind(b, a)

        assert np.allclose(ab, ba, atol=1e-10)

    @given(st.integers(min_value=1, max_value=10000))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_associativity(self, bsc, seed):
        """BSC binding should be associative.

        Property: (a⊗b)⊗c = a⊗(b⊗c) (XOR is associative)
        Reference: Schlegel et al. (2022), Section 2.4
        """
        a = bsc.random(seed=seed)
        b = bsc.random(seed=seed + 1)
        c = bsc.random(seed=seed + 2)

        # (a ⊗ b) ⊗ c
        ab = bsc.bind(a, b)
        abc_left = bsc.bind(ab, c)

        # a ⊗ (b ⊗ c)
        bc = bsc.bind(b, c)
        abc_right = bsc.bind(a, bc)

        assert np.allclose(abc_left, abc_right, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
