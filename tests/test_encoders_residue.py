"""Tests for ResidueEncoder - Residue Hyperdimensional Computing.

Tests the implementation of Kymn et al. 2024 "Computing With Residue Numbers
in High-Dimensional Representation".
"""

import math

import numpy as np
import pytest

from holovec import VSA
from holovec.encoders.residue import ResidueEncoder


class TestResidueEncoderBasics:
    """Basic tests for ResidueEncoder initialization and properties."""

    def test_basic_creation(self):
        """Test basic encoder creation with defaults."""
        encoder = ResidueEncoder(dim=1000)
        assert encoder.dim == 1000
        assert encoder.moduli == [3, 5, 7]
        assert encoder.range == 105
        assert encoder.codebook_size == 15

    def test_custom_moduli(self):
        """Test encoder creation with custom moduli."""
        encoder = ResidueEncoder(dim=1000, moduli=[7, 11, 13])
        assert encoder.moduli == [7, 11, 13]
        assert encoder.range == 7 * 11 * 13
        assert encoder.codebook_size == 7 + 11 + 13

    def test_large_moduli(self):
        """Test encoder with large co-prime moduli."""
        encoder = ResidueEncoder(dim=2000, moduli=[97, 101, 103])
        assert encoder.range == 97 * 101 * 103  # ~1 million
        assert encoder.codebook_size == 97 + 101 + 103  # 301

    def test_two_moduli(self):
        """Test encoder with just two moduli."""
        encoder = ResidueEncoder(dim=1000, moduli=[5, 7])
        assert encoder.range == 35
        assert encoder.codebook_size == 12
        assert encoder.K == 2

    def test_repr(self):
        """Test string representation."""
        encoder = ResidueEncoder(dim=1000, moduli=[3, 5, 7])
        repr_str = repr(encoder)
        assert "ResidueEncoder" in repr_str
        assert "dim=1000" in repr_str
        assert "moduli=[3, 5, 7]" in repr_str
        assert "range=105" in repr_str
        assert "codebook_size=15" in repr_str


class TestResidueEncoderValidation:
    """Tests for input validation."""

    def test_non_coprime_moduli_raises(self):
        """Test that non-coprime moduli raise an error."""
        with pytest.raises(ValueError, match="co-prime"):
            ResidueEncoder(dim=1000, moduli=[4, 6, 9])  # 4 and 6 share factor 2

    def test_non_coprime_pair_raises(self):
        """Test specific non-coprime pair detection."""
        with pytest.raises(ValueError, match="gcd\\(6, 9\\)"):
            ResidueEncoder(dim=1000, moduli=[5, 6, 9])  # 6 and 9 share factor 3

    def test_modulus_less_than_2_raises(self):
        """Test that moduli must be at least 2."""
        with pytest.raises(ValueError, match="Moduli must be >= 2"):
            ResidueEncoder(dim=1000, moduli=[1, 3, 5])

    def test_encode_negative_raises(self):
        """Test that negative values raise an error."""
        encoder = ResidueEncoder(dim=1000)
        with pytest.raises(ValueError, match="must be in"):
            encoder.encode(-1)

    def test_encode_out_of_range_raises(self):
        """Test that values >= M raise an error."""
        encoder = ResidueEncoder(dim=1000, moduli=[3, 5, 7])  # range 105
        with pytest.raises(ValueError, match="must be in"):
            encoder.encode(105)
        with pytest.raises(ValueError, match="must be in"):
            encoder.encode(200)


class TestResidueEncoderEncoding:
    """Tests for encoding functionality."""

    def test_encode_zero(self):
        """Test encoding of zero."""
        encoder = ResidueEncoder(dim=2000, seed=42)
        z_0 = encoder.encode(0)
        assert z_0 is not None
        assert z_0.shape == (2000,)

    def test_encode_identity(self):
        """Test that encoded vector is similar to itself."""
        encoder = ResidueEncoder(dim=2000, seed=42)
        z_20 = encoder.encode(20)
        sim = encoder.model.similarity(z_20, z_20)
        assert float(sim) > 0.99

    def test_encode_different_values_orthogonal(self):
        """Test that different values encode to near-orthogonal vectors."""
        encoder = ResidueEncoder(dim=10000, seed=42)
        z_20 = encoder.encode(20)
        z_50 = encoder.encode(50)
        sim = encoder.model.similarity(z_20, z_50)
        # Different values should have low similarity
        assert abs(float(sim)) < 0.3

    def test_encode_deterministic_with_seed(self):
        """Test that encoding is deterministic with same seed."""
        enc1 = ResidueEncoder(dim=1000, moduli=[3, 5, 7], seed=42)
        enc2 = ResidueEncoder(dim=1000, moduli=[3, 5, 7], seed=42)

        z1 = enc1.encode(20)
        z2 = enc2.encode(20)

        sim = enc1.model.similarity(z1, z2)
        assert float(sim) > 0.99

    def test_encode_with_residues(self):
        """Test encode_with_residues returns correct residues."""
        encoder = ResidueEncoder(dim=1000, moduli=[3, 5, 7], seed=42)

        z, residues = encoder.encode_with_residues(20)
        assert residues == {3: 2, 5: 0, 7: 6}  # 20 = 2 mod 3, 0 mod 5, 6 mod 7

        z, residues = encoder.encode_with_residues(0)
        assert residues == {3: 0, 5: 0, 7: 0}

        z, residues = encoder.encode_with_residues(104)
        assert residues == {3: 2, 5: 4, 7: 6}


class TestResidueEncoderAddition:
    """Tests for additive binding."""

    def test_addition_basic(self):
        """Test basic addition: z(20) + z(5) ≈ z(25)."""
        encoder = ResidueEncoder(dim=10000, moduli=[3, 5, 7], seed=42)
        z_20 = encoder.encode(20)
        z_5 = encoder.encode(5)
        z_sum = encoder.add(z_20, z_5)
        z_25 = encoder.encode(25)

        sim = encoder.model.similarity(z_sum, z_25)
        assert float(sim) > 0.9, f"Addition failed: similarity {float(sim):.3f}"

    def test_addition_commutative(self):
        """Test that addition is commutative."""
        encoder = ResidueEncoder(dim=10000, seed=42)
        z_a = encoder.encode(30)
        z_b = encoder.encode(40)

        z_ab = encoder.add(z_a, z_b)
        z_ba = encoder.add(z_b, z_a)

        sim = encoder.model.similarity(z_ab, z_ba)
        assert float(sim) > 0.99

    def test_addition_associative(self):
        """Test that addition is associative."""
        encoder = ResidueEncoder(dim=10000, seed=42)
        z_a = encoder.encode(10)
        z_b = encoder.encode(20)
        z_c = encoder.encode(30)

        # (a + b) + c
        z_ab = encoder.add(z_a, z_b)
        z_abc_1 = encoder.add(z_ab, z_c)

        # a + (b + c)
        z_bc = encoder.add(z_b, z_c)
        z_abc_2 = encoder.add(z_a, z_bc)

        sim = encoder.model.similarity(z_abc_1, z_abc_2)
        assert float(sim) > 0.99

    def test_addition_wraps_around(self):
        """Test that addition wraps around modulo M."""
        encoder = ResidueEncoder(dim=10000, moduli=[3, 5, 7], seed=42)  # range 105
        z_100 = encoder.encode(100)
        z_10 = encoder.encode(10)
        z_sum = encoder.add(z_100, z_10)  # 110 mod 105 = 5
        z_5 = encoder.encode(5)

        sim = encoder.model.similarity(z_sum, z_5)
        assert float(sim) > 0.9, f"Wraparound failed: similarity {float(sim):.3f}"

    def test_addition_with_zero(self):
        """Test that adding zero is identity."""
        encoder = ResidueEncoder(dim=10000, seed=42)
        z_20 = encoder.encode(20)
        z_0 = encoder.encode(0)
        z_sum = encoder.add(z_20, z_0)

        sim = encoder.model.similarity(z_sum, z_20)
        assert float(sim) > 0.99


class TestResidueEncoderSubtraction:
    """Tests for subtractive unbinding."""

    def test_subtraction_basic(self):
        """Test basic subtraction: z(25) - z(5) ≈ z(20)."""
        encoder = ResidueEncoder(dim=10000, moduli=[3, 5, 7], seed=42)
        z_25 = encoder.encode(25)
        z_5 = encoder.encode(5)
        z_diff = encoder.subtract(z_25, z_5)
        z_20 = encoder.encode(20)

        sim = encoder.model.similarity(z_diff, z_20)
        assert float(sim) > 0.9, f"Subtraction failed: similarity {float(sim):.3f}"

    def test_subtraction_inverse_of_addition(self):
        """Test that subtraction reverses addition."""
        encoder = ResidueEncoder(dim=10000, seed=42)
        z_a = encoder.encode(30)
        z_b = encoder.encode(40)

        z_sum = encoder.add(z_a, z_b)
        z_recovered = encoder.subtract(z_sum, z_b)

        sim = encoder.model.similarity(z_recovered, z_a)
        assert float(sim) > 0.99

    def test_subtraction_self_gives_zero(self):
        """Test that x - x = 0."""
        encoder = ResidueEncoder(dim=10000, seed=42)
        z_20 = encoder.encode(20)
        z_diff = encoder.subtract(z_20, z_20)
        z_0 = encoder.encode(0)

        sim = encoder.model.similarity(z_diff, z_0)
        assert float(sim) > 0.99

    def test_subtraction_wraps_around(self):
        """Test that subtraction wraps around for negative results."""
        encoder = ResidueEncoder(dim=10000, moduli=[3, 5, 7], seed=42)  # range 105
        z_5 = encoder.encode(5)
        z_10 = encoder.encode(10)
        z_diff = encoder.subtract(z_5, z_10)  # 5 - 10 = -5 mod 105 = 100
        z_100 = encoder.encode(100)

        sim = encoder.model.similarity(z_diff, z_100)
        assert float(sim) > 0.9, f"Wraparound failed: similarity {float(sim):.3f}"


class TestResidueEncoderNegation:
    """Tests for negation."""

    def test_negate_basic(self):
        """Test that -x + x = 0."""
        encoder = ResidueEncoder(dim=10000, seed=42)
        z_20 = encoder.encode(20)
        z_neg_20 = encoder.negate(z_20)
        z_sum = encoder.add(z_20, z_neg_20)
        z_0 = encoder.encode(0)

        sim = encoder.model.similarity(z_sum, z_0)
        assert float(sim) > 0.99

    def test_double_negation_identity(self):
        """Test that --x = x."""
        encoder = ResidueEncoder(dim=10000, seed=42)
        z_20 = encoder.encode(20)
        z_double_neg = encoder.negate(encoder.negate(z_20))

        sim = encoder.model.similarity(z_double_neg, z_20)
        assert float(sim) > 0.99


class TestResidueEncoderMultiplication:
    """Tests for multiplication (from values)."""

    def test_multiply_from_values(self):
        """Test multiplication from known values."""
        encoder = ResidueEncoder(dim=10000, moduli=[3, 5, 7], seed=42)
        z_product = encoder.multiply_from_values(5, 7)  # 35
        z_35 = encoder.encode(35)

        sim = encoder.model.similarity(z_product, z_35)
        assert float(sim) > 0.99

    def test_multiply_wraps_around(self):
        """Test that multiplication wraps around modulo M."""
        encoder = ResidueEncoder(dim=10000, moduli=[3, 5, 7], seed=42)  # range 105
        z_product = encoder.multiply_from_values(20, 10)  # 200 mod 105 = 95
        z_95 = encoder.encode(95)

        sim = encoder.model.similarity(z_product, z_95)
        assert float(sim) > 0.99


class TestResidueEncoderDecoding:
    """Tests for decoding functionality."""

    def test_decode_roundtrip_single(self):
        """Test single value encode-decode roundtrip."""
        encoder = ResidueEncoder(dim=5000, moduli=[3, 5, 7], seed=42)
        x = 20
        z = encoder.encode(x)
        decoded = encoder.decode(z)
        assert decoded == x

    def test_decode_roundtrip_all_values(self):
        """Test encode-decode roundtrip for all values in small range."""
        encoder = ResidueEncoder(dim=5000, moduli=[3, 5, 7], seed=42)  # range 105

        for x in range(encoder.range):
            z = encoder.encode(x)
            decoded = encoder.decode(z)
            assert decoded == x, f"Decode failed for {x}: got {decoded}"

    def test_decode_roundtrip_sample_values(self):
        """Test encode-decode roundtrip for specific values."""
        encoder = ResidueEncoder(dim=5000, moduli=[3, 5, 7], seed=42)

        test_values = [0, 1, 20, 50, 104]
        for x in test_values:
            z = encoder.encode(x)
            decoded = encoder.decode(z)
            assert decoded == x, f"Decode failed for {x}: got {decoded}"

    def test_decode_zero(self):
        """Test decoding of zero."""
        encoder = ResidueEncoder(dim=5000, seed=42)
        z = encoder.encode(0)
        decoded = encoder.decode(z)
        assert decoded == 0

    def test_decode_max_value(self):
        """Test decoding of maximum value (M-1)."""
        encoder = ResidueEncoder(dim=5000, moduli=[3, 5, 7], seed=42)
        x = encoder.range - 1  # 104
        z = encoder.encode(x)
        decoded = encoder.decode(z)
        assert decoded == x

    def test_decode_after_addition(self):
        """Test decoding after addition."""
        encoder = ResidueEncoder(dim=10000, moduli=[3, 5, 7], seed=42)
        z_20 = encoder.encode(20)
        z_5 = encoder.encode(5)
        z_sum = encoder.add(z_20, z_5)
        decoded = encoder.decode(z_sum)
        assert decoded == 25

    def test_decode_after_subtraction(self):
        """Test decoding after subtraction."""
        encoder = ResidueEncoder(dim=10000, moduli=[3, 5, 7], seed=42)
        z_25 = encoder.encode(25)
        z_5 = encoder.encode(5)
        z_diff = encoder.subtract(z_25, z_5)
        decoded = encoder.decode(z_diff)
        assert decoded == 20


class TestChineseRemainderTheorem:
    """Tests for Chinese Remainder Theorem implementation."""

    def test_crt_basic(self):
        """Test CRT reconstruction."""
        encoder = ResidueEncoder(dim=1000, moduli=[3, 5, 7])

        # 20 = [2 mod 3, 0 mod 5, 6 mod 7]
        remainders = {3: 2, 5: 0, 7: 6}
        recovered = encoder._chinese_remainder_theorem(remainders)
        assert recovered == 20

    def test_crt_zero(self):
        """Test CRT for zero."""
        encoder = ResidueEncoder(dim=1000, moduli=[3, 5, 7])
        remainders = {3: 0, 5: 0, 7: 0}
        recovered = encoder._chinese_remainder_theorem(remainders)
        assert recovered == 0

    def test_crt_one(self):
        """Test CRT for one."""
        encoder = ResidueEncoder(dim=1000, moduli=[3, 5, 7])
        remainders = {3: 1, 5: 1, 7: 1}
        recovered = encoder._chinese_remainder_theorem(remainders)
        assert recovered == 1

    def test_crt_all_values(self):
        """Test CRT for all values in range."""
        encoder = ResidueEncoder(dim=1000, moduli=[3, 5, 7])  # range 105

        for x in range(encoder.range):
            remainders = {m: x % m for m in encoder.moduli}
            recovered = encoder._chinese_remainder_theorem(remainders)
            assert recovered == x, f"CRT failed for {x}: got {recovered}"


class TestLogarithmicScaling:
    """Tests for logarithmic codebook scaling property."""

    def test_scaling_basic(self):
        """Test that codebook size scales logarithmically with range."""
        # Small range
        small = ResidueEncoder(dim=1000, moduli=[3, 5, 7])
        assert small.range == 105
        assert small.codebook_size == 15  # 3+5+7

        # Large range
        large = ResidueEncoder(dim=1000, moduli=[97, 101, 103])
        assert large.range == 97 * 101 * 103  # ~1 million
        assert large.codebook_size == 301  # 97+101+103

        # Codebook grows logarithmically with range
        ratio = large.range / small.range
        codebook_ratio = large.codebook_size / small.codebook_size
        # Codebook ratio should be much smaller than sqrt of range ratio
        assert codebook_ratio < ratio**0.5

    def test_scaling_comparison(self):
        """Compare logarithmic scaling to linear."""
        # With linear scaling, a codebook for range 1,000,000 would need
        # 1,000,000 entries. With residue encoding:
        encoder = ResidueEncoder(dim=1000, moduli=[97, 101, 103])

        linear_size = encoder.range  # 1,010,291
        residue_size = encoder.codebook_size  # 301

        # Over 3000x smaller!
        assert linear_size / residue_size > 3000


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_results(self):
        """Test that same seed gives same results."""
        enc1 = ResidueEncoder(dim=1000, moduli=[3, 5, 7], seed=42)
        enc2 = ResidueEncoder(dim=1000, moduli=[3, 5, 7], seed=42)

        for x in [0, 10, 50, 100]:
            z1 = enc1.encode(x)
            z2 = enc2.encode(x)
            sim = enc1.model.similarity(z1, z2)
            assert float(sim) > 0.99

    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        enc1 = ResidueEncoder(dim=1000, moduli=[3, 5, 7], seed=42)
        enc2 = ResidueEncoder(dim=1000, moduli=[3, 5, 7], seed=123)

        z1 = enc1.encode(20)
        z2 = enc2.encode(20)

        # Different random base vectors, so different encodings
        # But both should decode correctly
        d1 = enc1.decode(z1)
        d2 = enc2.decode(z2)
        assert d1 == 20
        assert d2 == 20


class TestWithCustomModel:
    """Tests with custom FHRR model."""

    def test_with_provided_model(self):
        """Test encoder with externally provided model."""
        model = VSA.create("FHRR", dim=2000, seed=42)
        encoder = ResidueEncoder(dim=2000, moduli=[3, 5, 7], model=model)

        z = encoder.encode(20)
        decoded = encoder.decode(z)
        assert decoded == 20
        assert encoder.model is model

    def test_dimension_matches_model(self):
        """Test that encoder dimension matches model dimension."""
        model = VSA.create("FHRR", dim=3000)
        encoder = ResidueEncoder(dim=3000, model=model)
        assert encoder.dimension == 3000
        assert encoder.model.dimension == 3000


class TestEdgeCases:
    """Edge case tests."""

    def test_single_modulus(self):
        """Test with single modulus (degenerate case, still co-prime)."""
        encoder = ResidueEncoder(dim=1000, moduli=[7])
        assert encoder.range == 7
        assert encoder.codebook_size == 7

        z = encoder.encode(3)
        decoded = encoder.decode(z)
        assert decoded == 3

    def test_prime_moduli(self):
        """Test with first few primes."""
        encoder = ResidueEncoder(dim=5000, moduli=[2, 3, 5, 7, 11], seed=42)
        assert encoder.range == 2 * 3 * 5 * 7 * 11  # 2310
        assert encoder.codebook_size == 28

        # Test encode-decode for a few values
        for x in [0, 1, 100, 1000, 2309]:
            z = encoder.encode(x)
            decoded = encoder.decode(z)
            assert decoded == x

    def test_large_dimension(self):
        """Test with larger dimension for better accuracy."""
        encoder = ResidueEncoder(dim=20000, moduli=[3, 5, 7], seed=42)

        # Should have very high accuracy
        for x in [20, 50, 100]:
            z = encoder.encode(x)
            decoded = encoder.decode(z)
            assert decoded == x
