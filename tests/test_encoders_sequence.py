"""
Tests for sequence encoders.

Tests the PositionBindingEncoder and future sequence encoders.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from holovec import VSA
from holovec.encoders import PositionBindingEncoder, NGramEncoder, TrajectoryEncoder
from holovec.encoders import FractionalPowerEncoder, ThermometerEncoder


class TestPositionBindingEncoderInitialization:
    """Test initialization of PositionBindingEncoder."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = PositionBindingEncoder(model)

        assert encoder.model == model
        assert encoder.backend == model.backend
        assert encoder.dimension == 1000
        assert encoder.auto_generate is True
        assert encoder.max_length is None
        assert len(encoder.codebook) == 0

    def test_init_with_codebook(self):
        """Test initialization with pre-defined codebook."""
        model = VSA.create('MAP', dim=1000, seed=42)

        # Create codebook
        codebook = {
            'a': model.random(seed=1),
            'b': model.random(seed=2),
            'c': model.random(seed=3)
        }

        encoder = PositionBindingEncoder(model, codebook=codebook)
        assert encoder.get_codebook_size() == 3
        assert 'a' in encoder.codebook
        assert 'b' in encoder.codebook
        assert 'c' in encoder.codebook

    def test_init_with_max_length(self):
        """Test initialization with max_length constraint."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = PositionBindingEncoder(model, max_length=10)

        assert encoder.max_length == 10

    def test_init_with_seed(self):
        """Test initialization with seed for reproducibility."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder1 = PositionBindingEncoder(model, seed=123)
        encoder2 = PositionBindingEncoder(model, seed=123)

        # Same seed should produce same symbol vectors
        encoder1.add_symbol('test')
        encoder2.add_symbol('test')

        vec1 = encoder1.codebook['test']
        vec2 = encoder2.codebook['test']

        similarity = float(model.similarity(vec1, vec2))
        assert similarity > 0.99  # Should be nearly identical


class TestPositionBindingEncoderEncoding:
    """Test encoding functionality."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('MAP', dim=1000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        """Create test encoder."""
        return PositionBindingEncoder(model, seed=42)

    def test_encode_simple_sequence(self, encoder):
        """Test encoding a simple sequence."""
        sequence = ['a', 'b', 'c']
        hv = encoder.encode(sequence)

        assert hv.shape == (encoder.dimension,)
        assert encoder.get_codebook_size() == 3

    def test_encode_empty_sequence_fails(self, encoder):
        """Test that encoding empty sequence raises error."""
        with pytest.raises(ValueError, match="Cannot encode empty sequence"):
            encoder.encode([])

    def test_encode_exceeds_max_length_fails(self, model):
        """Test that exceeding max_length raises error."""
        encoder = PositionBindingEncoder(model, max_length=3)

        with pytest.raises(ValueError, match="exceeds max_length"):
            encoder.encode(['a', 'b', 'c', 'd'])

    def test_auto_generation_of_symbols(self, encoder):
        """Test automatic generation of unknown symbols."""
        sequence = ['x', 'y', 'z']
        encoder.encode(sequence)

        # All symbols should be in codebook
        assert 'x' in encoder.codebook
        assert 'y' in encoder.codebook
        assert 'z' in encoder.codebook

    def test_auto_generate_false_fails_on_unknown(self, model):
        """Test error when auto_generate=False and symbol unknown."""
        encoder = PositionBindingEncoder(model, auto_generate=False)

        with pytest.raises(ValueError, match="not in codebook"):
            encoder.encode(['unknown'])

    def test_different_sequences_differ(self, encoder):
        """Test that different sequences produce different encodings."""
        hv1 = encoder.encode(['a', 'b', 'c'])
        hv2 = encoder.encode(['d', 'e', 'f'])

        similarity = float(encoder.model.similarity(hv1, hv2))
        assert similarity < 0.6  # Should be different (relaxed threshold for dim=1000)

    def test_integer_symbols(self, encoder):
        """Test encoding sequences of integers."""
        sequence = [1, 2, 3, 4]
        hv = encoder.encode(sequence)

        assert hv.shape == (encoder.dimension,)
        assert encoder.get_codebook_size() == 4


class TestPositionBindingEncoderOrderSensitivity:
    """Test that encoder is order-sensitive."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('MAP', dim=1000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        """Create test encoder."""
        return PositionBindingEncoder(model, seed=42)

    def test_same_symbols_different_order(self, encoder):
        """Test that same symbols in different order produce different encodings."""
        hv1 = encoder.encode(['a', 'b', 'c'])
        hv2 = encoder.encode(['c', 'b', 'a'])  # Reversed

        similarity = float(encoder.model.similarity(hv1, hv2))
        # Should be different (not orthogonal but distinct)
        assert similarity < 0.7

    def test_permuted_sequence_differs(self, encoder):
        """Test that permuted sequence has different encoding."""
        hv1 = encoder.encode(['a', 'b', 'c', 'd'])
        hv2 = encoder.encode(['b', 'c', 'd', 'a'])  # Rotated

        similarity = float(encoder.model.similarity(hv1, hv2))
        assert similarity < 0.7


class TestPositionBindingEncoderSimilarity:
    """Test similarity properties of encodings."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('MAP', dim=5000, seed=42)  # Higher dim for better convergence

    @pytest.fixture
    def encoder(self, model):
        """Create test encoder."""
        return PositionBindingEncoder(model, seed=42)

    def test_shared_prefix_increases_similarity(self, encoder):
        """Test that shared prefix increases similarity."""
        hv1 = encoder.encode(['a', 'b', 'c'])
        hv2 = encoder.encode(['a', 'b', 'd'])  # Shared prefix 'a', 'b'
        hv3 = encoder.encode(['x', 'y', 'z'])  # No shared prefix

        sim_shared = float(encoder.model.similarity(hv1, hv2))
        sim_different = float(encoder.model.similarity(hv1, hv3))

        # Shared prefix should have higher similarity
        assert sim_shared > sim_different

    def test_longer_shared_subsequence_higher_similarity(self, encoder):
        """Test that longer shared subsequence → higher similarity."""
        reference = ['a', 'b', 'c', 'd', 'e']

        hv_ref = encoder.encode(reference)
        hv_1char = encoder.encode(['a', 'x', 'x', 'x', 'x'])  # 1 char shared
        hv_3char = encoder.encode(['a', 'b', 'c', 'x', 'x'])  # 3 char shared
        hv_5char = encoder.encode(['a', 'b', 'c', 'd', 'e'])  # All shared

        sim_1 = float(encoder.model.similarity(hv_ref, hv_1char))
        sim_3 = float(encoder.model.similarity(hv_ref, hv_3char))
        sim_5 = float(encoder.model.similarity(hv_ref, hv_5char))

        # More shared characters → higher similarity
        assert sim_3 > sim_1
        assert sim_5 > sim_3

    def test_identical_sequences_high_similarity(self, encoder):
        """Test that identical sequences have similarity ~1.0."""
        hv1 = encoder.encode(['a', 'b', 'c'])
        hv2 = encoder.encode(['a', 'b', 'c'])

        similarity = float(encoder.model.similarity(hv1, hv2))
        assert similarity > 0.99  # Should be very close to 1.0


class TestPositionBindingEncoderDecoding:
    """Test decoding functionality."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('MAP', dim=5000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        """Create test encoder with pre-defined codebook."""
        codebook = {
            'a': model.random(seed=1),
            'b': model.random(seed=2),
            'c': model.random(seed=3),
            'd': model.random(seed=4)
        }
        return PositionBindingEncoder(model, codebook=codebook)

    def test_decode_simple_sequence(self, encoder):
        """Test decoding a simple sequence."""
        original = ['a', 'b', 'c']
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded, max_positions=5)

        # Should recover at least the first few symbols
        assert len(decoded) > 0
        # First symbol should match
        assert decoded[0] == original[0]

    def test_decode_with_threshold(self, encoder):
        """Test decoding with different similarity thresholds."""
        original = ['a', 'b']
        encoded = encoder.encode(original)

        # Lower threshold → more lenient
        decoded_low = encoder.decode(encoded, max_positions=5, threshold=0.1)
        # Higher threshold → more strict
        decoded_high = encoder.decode(encoded, max_positions=5, threshold=0.5)

        # Lower threshold should decode more positions (or same)
        assert len(decoded_low) >= len(decoded_high)

    def test_decode_empty_codebook_fails(self, model):
        """Test that decoding with empty codebook fails."""
        encoder = PositionBindingEncoder(model)
        # Create a random hypervector
        hv = model.random(seed=123)

        with pytest.raises(RuntimeError, match="codebook is empty"):
            encoder.decode(hv)

    def test_decode_max_positions(self, encoder):
        """Test that max_positions limits decoded length."""
        original = ['a', 'b', 'c', 'd']
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded, max_positions=2, threshold=0.0)

        # Should decode at most 2 positions
        assert len(decoded) <= 2


class TestPositionBindingEncoderCodebook:
    """Test codebook management."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('MAP', dim=1000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        """Create test encoder."""
        return PositionBindingEncoder(model, seed=42)

    def test_add_symbol_auto_generate(self, encoder):
        """Test adding symbol with auto-generated vector."""
        encoder.add_symbol('new_symbol')

        assert 'new_symbol' in encoder.codebook
        assert encoder.get_codebook_size() == 1

    def test_add_symbol_custom_vector(self, encoder, model):
        """Test adding symbol with custom vector."""
        custom_vec = model.random(seed=999)
        encoder.add_symbol('custom', custom_vec)

        assert 'custom' in encoder.codebook
        # Should be the exact vector we provided
        similarity = float(model.similarity(encoder.codebook['custom'], custom_vec))
        assert similarity > 0.99

    def test_get_codebook_size(self, encoder):
        """Test get_codebook_size method."""
        assert encoder.get_codebook_size() == 0

        encoder.encode(['a', 'b', 'c'])
        assert encoder.get_codebook_size() == 3

    def test_symbol_vector_consistency(self, encoder):
        """Test that same symbol always gets same vector."""
        encoder.encode(['test'])
        vec1 = encoder.codebook['test']

        # Re-encode (should not regenerate if already exists)
        encoder.encode(['test', 'other'])
        vec2 = encoder.codebook['test']

        # Should be identical
        similarity = float(encoder.model.similarity(vec1, vec2))
        assert similarity > 0.99


class TestPositionBindingEncoderProperties:
    """Property-based tests using Hypothesis."""

    @given(
        sequence_length=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_encode_decode_roundtrip_length(self, sequence_length, seed):
        """Test that decoded sequences have reasonable length."""
        model = VSA.create('MAP', dim=5000, seed=seed)
        encoder = PositionBindingEncoder(model, seed=seed)

        # Create random sequence
        sequence = [f"sym_{i}" for i in range(sequence_length)]

        encoded = encoder.encode(sequence)
        decoded = encoder.decode(encoded, max_positions=sequence_length * 2, threshold=0.2)

        # Decoded length should be reasonable
        assert len(decoded) <= sequence_length * 2
        # Should decode at least something
        assert len(decoded) > 0

    @given(
        symbols=st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=5), min_size=2, max_size=5)
    )
    def test_different_order_different_encoding(self, symbols):
        """Test that different orders produce different encodings."""
        if len(set(symbols)) < 2:  # Skip if not enough unique symbols
            return

        # Skip palindromes (same forwards and backwards)
        if symbols == list(reversed(symbols)):
            return

        model = VSA.create('MAP', dim=2000, seed=42)
        encoder = PositionBindingEncoder(model, seed=42)

        # Encode original sequence
        hv1 = encoder.encode(symbols)

        # Encode reversed sequence
        hv2 = encoder.encode(list(reversed(symbols)))

        # Should be different
        similarity = float(model.similarity(hv1, hv2))
        assert similarity < 0.9


class TestPositionBindingEncoderRepresentation:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = PositionBindingEncoder(model, max_length=10, auto_generate=True)

        repr_str = repr(encoder)

        assert 'PositionBindingEncoder' in repr_str
        assert 'MAP' in repr_str
        assert 'max_length=10' in repr_str
        assert 'auto_generate=True' in repr_str


class TestPositionBindingEncoderCompatibility:
    """Test compatibility with different VSA models."""

    @pytest.mark.parametrize('model_name', ['MAP', 'FHRR', 'HRR', 'BSC'])
    def test_works_with_multiple_models(self, model_name):
        """Test that encoder works with multiple VSA models."""
        model = VSA.create(model_name, dim=1000, seed=42)
        encoder = PositionBindingEncoder(model, seed=42)

        sequence = ['a', 'b', 'c']
        hv = encoder.encode(sequence)

        assert hv.shape == (1000,)
        assert encoder.get_codebook_size() == 3

    def test_is_reversible_property(self):
        """Test is_reversible property."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = PositionBindingEncoder(model)

        assert encoder.is_reversible is True

    def test_compatible_models_property(self):
        """Test compatible_models property."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = PositionBindingEncoder(model)

        compatible = encoder.compatible_models
        assert 'MAP' in compatible
        assert 'FHRR' in compatible
        assert 'HRR' in compatible
        assert len(compatible) > 0


# ============================================================================
# NGramEncoder Tests
# ============================================================================


class TestNGramEncoderInitialization:
    """Test initialization of NGramEncoder."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model)

        assert encoder.model == model
        assert encoder.n == 2  # Default bigrams
        assert encoder.stride == 1  # Default overlapping
        assert encoder.mode == 'bundling'  # Default mode
        assert encoder.get_codebook_size() == 0

    def test_init_with_custom_n(self):
        """Test initialization with custom n-gram size."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model, n=3, seed=42)

        assert encoder.n == 3

    def test_init_with_custom_stride(self):
        """Test initialization with custom stride."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model, n=2, stride=2, seed=42)

        assert encoder.stride == 2

    def test_init_with_bundling_mode(self):
        """Test initialization with bundling mode."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model, mode='bundling', seed=42)

        assert encoder.mode == 'bundling'
        assert encoder.is_reversible is False

    def test_init_with_chaining_mode(self):
        """Test initialization with chaining mode."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model, mode='chaining', seed=42)

        assert encoder.mode == 'chaining'
        assert encoder.is_reversible is True

    def test_init_invalid_n_fails(self):
        """Test that invalid n raises error."""
        model = VSA.create('MAP', dim=1000, seed=42)

        with pytest.raises(ValueError, match="n must be >= 1"):
            NGramEncoder(model, n=0)

        with pytest.raises(ValueError, match="n must be >= 1"):
            NGramEncoder(model, n=-1)

    def test_init_invalid_stride_fails(self):
        """Test that invalid stride raises error."""
        model = VSA.create('MAP', dim=1000, seed=42)

        with pytest.raises(ValueError, match="stride must be >= 1"):
            NGramEncoder(model, stride=0)

    def test_init_invalid_mode_fails(self):
        """Test that invalid mode raises error."""
        model = VSA.create('MAP', dim=1000, seed=42)

        with pytest.raises(ValueError, match="mode must be"):
            NGramEncoder(model, mode='invalid')


class TestNGramEncoderEncoding:
    """Test basic encoding functionality."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return VSA.create('MAP', dim=1000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        """Create test encoder with default parameters."""
        return NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)

    def test_encode_simple_sequence(self, encoder):
        """Test encoding a simple sequence."""
        sequence = ['A', 'B', 'C', 'D']
        hv = encoder.encode(sequence)

        assert hv.shape == (encoder.dimension,)
        # Should have 4 unique symbols in codebook
        assert encoder.get_codebook_size() == 4

    def test_encode_too_short_sequence_fails(self, encoder):
        """Test that sequence shorter than n fails."""
        with pytest.raises(ValueError, match="is less than n="):
            encoder.encode(['A'])  # n=2 but sequence is length 1

    def test_encode_bigrams_stride_1(self, encoder):
        """Test bigram extraction with stride=1 (overlapping)."""
        sequence = ['A', 'B', 'C', 'D']
        # Expected n-grams: AB, BC, CD (3 total)
        hv = encoder.encode(sequence)

        assert hv.shape == (1000,)

    def test_encode_bigrams_stride_2(self, model):
        """Test bigram extraction with stride=2 (non-overlapping)."""
        encoder = NGramEncoder(model, n=2, stride=2, seed=42)
        sequence = ['A', 'B', 'C', 'D']
        # Expected n-grams: AB, CD (2 total)
        hv = encoder.encode(sequence)

        assert hv.shape == (1000,)

    def test_encode_trigrams(self, model):
        """Test trigram encoding."""
        encoder = NGramEncoder(model, n=3, stride=1, seed=42)
        sequence = ['A', 'B', 'C', 'D', 'E']
        # Expected n-grams: ABC, BCD, CDE (3 total)
        hv = encoder.encode(sequence)

        assert hv.shape == (1000,)
        assert encoder.get_codebook_size() == 5

    def test_encode_integer_symbols(self, encoder):
        """Test encoding sequences of integers."""
        sequence = [1, 2, 3, 4]
        hv = encoder.encode(sequence)

        assert hv.shape == (encoder.dimension,)
        assert encoder.get_codebook_size() == 4


class TestNGramEncoderNGramExtraction:
    """Test n-gram extraction logic."""

    @pytest.fixture
    def model(self):
        return VSA.create('MAP', dim=1000, seed=42)

    def test_extract_unigrams(self, model):
        """Test unigram extraction (n=1)."""
        encoder = NGramEncoder(model, n=1, stride=1, seed=42)
        sequence = ['A', 'B', 'C']
        # N-grams: A, B, C
        hv = encoder.encode(sequence)
        assert hv.shape == (1000,)

    def test_extract_bigrams_overlapping(self, model):
        """Test overlapping bigrams."""
        encoder = NGramEncoder(model, n=2, stride=1, seed=42)
        sequence = ['A', 'B', 'C', 'D']
        # N-grams: AB, BC, CD (3 total)
        hv = encoder.encode(sequence)
        assert encoder.get_codebook_size() == 4

    def test_extract_bigrams_non_overlapping(self, model):
        """Test non-overlapping bigrams."""
        encoder = NGramEncoder(model, n=2, stride=2, seed=42)
        sequence = ['A', 'B', 'C', 'D']
        # N-grams: AB, CD (2 total)
        hv = encoder.encode(sequence)
        assert hv.shape == (1000,)

    def test_extract_trigrams_stride_2(self, model):
        """Test trigrams with stride=2 (partial overlap)."""
        encoder = NGramEncoder(model, n=3, stride=2, seed=42)
        sequence = ['A', 'B', 'C', 'D', 'E', 'F']
        # N-grams: ABC (pos 0), CDE (pos 2), EF? - no, only ABC, CDE (2 total)
        hv = encoder.encode(sequence)
        assert hv.shape == (1000,)


class TestNGramEncoderModes:
    """Test bundling vs chaining modes."""

    @pytest.fixture
    def model(self):
        return VSA.create('MAP', dim=5000, seed=42)

    def test_bundling_mode_order_invariance(self, model):
        """Test that bundling mode is less sensitive to n-gram order."""
        encoder = NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)

        # These sequences have same bigrams but different order
        # Actually not quite - let's use sequences with same n-gram set
        seq1 = ['A', 'B', 'C']  # N-grams: AB, BC
        seq2 = ['C', 'B', 'A']  # N-grams: CB, BA

        hv1 = encoder.encode(seq1)
        hv2 = encoder.encode(seq2)

        # Should be different (different n-grams)
        sim = float(model.similarity(hv1, hv2))
        assert sim < 0.7

    def test_chaining_mode_order_sensitive(self, model):
        """Test that chaining mode is order-sensitive."""
        encoder = NGramEncoder(model, n=2, stride=1, mode='chaining', seed=42)

        seq1 = ['A', 'B', 'C']  # N-grams at positions: AB (0), BC (1)
        seq2 = ['A', 'B', 'C']  # Same sequence
        seq3 = ['B', 'C', 'A']  # Different n-gram order

        hv1 = encoder.encode(seq1)
        hv2 = encoder.encode(seq2)
        hv3 = encoder.encode(seq3)

        # Same sequence should be identical
        sim_same = float(model.similarity(hv1, hv2))
        assert sim_same > 0.99

        # Different order should be different
        sim_diff = float(model.similarity(hv1, hv3))
        assert sim_diff < 0.7

    def test_bundling_mode_similarity(self, model):
        """Test similarity in bundling mode."""
        encoder = NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)

        seq1 = ['A', 'B', 'C', 'D']  # N-grams: AB, BC, CD
        seq2 = ['A', 'B', 'C', 'E']  # N-grams: AB, BC, CE (shares 2/3)

        hv1 = encoder.encode(seq1)
        hv2 = encoder.encode(seq2)

        sim = float(model.similarity(hv1, hv2))
        # Should have moderate similarity (shared n-grams)
        assert sim > 0.3


class TestNGramEncoderDecoding:
    """Test decoding functionality."""

    @pytest.fixture
    def model(self):
        return VSA.create('MAP', dim=5000, seed=42)

    def test_decode_chaining_mode(self, model):
        """Test decoding in chaining mode."""
        encoder = NGramEncoder(model, n=2, stride=1, mode='chaining', seed=42)

        sequence = ['A', 'B', 'C']
        hv = encoder.encode(sequence)

        # Decode
        decoded_ngrams = encoder.decode(hv, max_ngrams=3, threshold=0.2)

        # Should recover at least first n-gram
        assert len(decoded_ngrams) >= 1
        # First n-gram should be close to ['A', 'B']
        # (approximate due to bundling noise)

    def test_decode_bundling_mode_fails(self, model):
        """Test that decoding fails in bundling mode."""
        encoder = NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)

        sequence = ['A', 'B', 'C']
        hv = encoder.encode(sequence)

        with pytest.raises(NotImplementedError, match="only supported for 'chaining' mode"):
            encoder.decode(hv)

    def test_decode_empty_codebook_fails(self, model):
        """Test that decoding with empty codebook fails."""
        encoder = NGramEncoder(model, n=2, stride=1, mode='chaining', seed=42)

        # Create arbitrary hypervector without encoding
        hv = model.random()

        with pytest.raises(RuntimeError, match="codebook is empty"):
            encoder.decode(hv)


class TestNGramEncoderCodebook:
    """Test codebook management."""

    @pytest.fixture
    def model(self):
        return VSA.create('MAP', dim=1000, seed=42)

    def test_get_codebook(self, model):
        """Test get_codebook method."""
        encoder = NGramEncoder(model, n=2, seed=42)
        encoder.encode(['A', 'B', 'C'])

        codebook = encoder.get_codebook()
        assert 'A' in codebook
        assert 'B' in codebook
        assert 'C' in codebook

    def test_get_codebook_size(self, model):
        """Test get_codebook_size method."""
        encoder = NGramEncoder(model, n=2, seed=42)
        assert encoder.get_codebook_size() == 0

        encoder.encode(['A', 'B', 'C'])
        assert encoder.get_codebook_size() == 3

    def test_codebook_shared_across_ngrams(self, model):
        """Test that codebook is shared across all n-grams."""
        encoder = NGramEncoder(model, n=2, seed=42)

        # Encode sequence with repeated symbols
        encoder.encode(['A', 'B', 'A', 'B'])  # N-grams: AB, BA, AB

        # Should only have 2 unique symbols
        assert encoder.get_codebook_size() == 2


class TestNGramEncoderProperties:
    """Test encoder properties."""

    @pytest.mark.parametrize('model_name', ['MAP', 'FHRR', 'HRR', 'BSC'])
    def test_works_with_multiple_models(self, model_name):
        """Test that encoder works with multiple VSA models."""
        model = VSA.create(model_name, dim=1000, seed=42)
        encoder = NGramEncoder(model, n=2, seed=42)

        sequence = ['A', 'B', 'C']
        hv = encoder.encode(sequence)

        assert hv.shape == (1000,)

    def test_is_reversible_bundling(self):
        """Test is_reversible property for bundling mode."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model, mode='bundling')

        assert encoder.is_reversible is False

    def test_is_reversible_chaining(self):
        """Test is_reversible property for chaining mode."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model, mode='chaining')

        assert encoder.is_reversible is True

    def test_compatible_models_property(self):
        """Test compatible_models property."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model, n=2)

        compatible = encoder.compatible_models
        assert 'MAP' in compatible
        assert 'FHRR' in compatible
        assert 'HRR' in compatible
        assert len(compatible) > 0

    def test_repr(self):
        """Test string representation."""
        model = VSA.create('MAP', dim=1000, seed=42)
        encoder = NGramEncoder(model, n=3, stride=2, mode='chaining', seed=42)

        repr_str = repr(encoder)
        assert 'NGramEncoder' in repr_str
        assert 'n=3' in repr_str
        assert 'stride=2' in repr_str
        assert "mode='chaining'" in repr_str


# ============================================================================
# TrajectoryEncoder Tests
# ============================================================================


class TestTrajectoryEncoderInitialization:
    """Test initialization of TrajectoryEncoder."""

    @pytest.fixture
    def model(self):
        return VSA.create('FHRR', dim=1000, seed=42)

    @pytest.fixture
    def scalar_encoder(self, model):
        return FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)

    def test_init_default_parameters(self, model, scalar_encoder):
        """Test initialization with default parameters."""
        encoder = TrajectoryEncoder(model, scalar_encoder)

        assert encoder.model == model
        assert encoder.scalar_encoder == scalar_encoder
        assert encoder.n_dimensions == 1
        assert encoder.time_range is None

    def test_init_1d(self, model, scalar_encoder):
        """Test initialization for 1D time series."""
        encoder = TrajectoryEncoder(model, scalar_encoder, n_dimensions=1, seed=42)

        assert encoder.n_dimensions == 1
        assert len(encoder.dim_vectors) == 1
        assert encoder.input_type == "1D time series"

    def test_init_2d(self, model, scalar_encoder):
        """Test initialization for 2D paths."""
        encoder = TrajectoryEncoder(model, scalar_encoder, n_dimensions=2, seed=42)

        assert encoder.n_dimensions == 2
        assert len(encoder.dim_vectors) == 2
        assert encoder.input_type == "2D path"

    def test_init_3d(self, model, scalar_encoder):
        """Test initialization for 3D trajectories."""
        encoder = TrajectoryEncoder(model, scalar_encoder, n_dimensions=3, seed=42)

        assert encoder.n_dimensions == 3
        assert len(encoder.dim_vectors) == 3
        assert encoder.input_type == "3D trajectory"

    def test_init_with_time_range(self, model, scalar_encoder):
        """Test initialization with time range."""
        encoder = TrajectoryEncoder(
            model, scalar_encoder, n_dimensions=1, time_range=(0.0, 10.0)
        )

        assert encoder.time_range == (0.0, 10.0)

    def test_init_invalid_dimensions_fails(self, model, scalar_encoder):
        """Test that invalid n_dimensions raises error."""
        with pytest.raises(ValueError, match="n_dimensions must be"):
            TrajectoryEncoder(model, scalar_encoder, n_dimensions=0)

        with pytest.raises(ValueError, match="n_dimensions must be"):
            TrajectoryEncoder(model, scalar_encoder, n_dimensions=4)

    def test_init_wrong_scalar_encoder_fails(self, model):
        """Test that non-ScalarEncoder raises error."""
        with pytest.raises(TypeError, match="must be a ScalarEncoder"):
            TrajectoryEncoder(model, "not_an_encoder")

    def test_init_mismatched_model_fails(self, model, scalar_encoder):
        """Test that mismatched models raise error."""
        model2 = VSA.create('MAP', dim=1000, seed=43)

        with pytest.raises(ValueError, match="must use the same VSA model"):
            TrajectoryEncoder(model2, scalar_encoder)


class TestTrajectoryEncoder1D:
    """Test 1D time series encoding."""

    @pytest.fixture
    def model(self):
        return VSA.create('FHRR', dim=5000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)
        return TrajectoryEncoder(model, scalar_enc, n_dimensions=1, seed=42)

    def test_encode_1d_time_series(self, encoder):
        """Test encoding a 1D time series."""
        ts = [10.0, 20.0, 30.0, 40.0]
        hv = encoder.encode(ts)

        assert hv.shape == (encoder.dimension,)

    def test_encode_1d_with_integers(self, encoder):
        """Test encoding 1D with integer values."""
        ts = [10, 20, 30, 40]
        hv = encoder.encode(ts)

        assert hv.shape == (encoder.dimension,)

    def test_encode_empty_fails(self, encoder):
        """Test that empty trajectory fails."""
        with pytest.raises(ValueError, match="Cannot encode empty trajectory"):
            encoder.encode([])

    def test_similar_series_have_high_similarity(self, encoder, model):
        """Test that similar time series have high similarity."""
        ts1 = [10.0, 20.0, 30.0, 40.0]
        ts2 = [10.0, 20.0, 30.0, 41.0]  # Slightly different

        hv1 = encoder.encode(ts1)
        hv2 = encoder.encode(ts2)

        sim = float(model.similarity(hv1, hv2))
        assert sim > 0.8  # Should be quite similar

    def test_different_series_have_low_similarity(self, encoder, model):
        """Test that different time series have low similarity."""
        ts1 = [10.0, 20.0, 30.0, 40.0]
        ts2 = [50.0, 60.0, 70.0, 80.0]  # Completely different

        hv1 = encoder.encode(ts1)
        hv2 = encoder.encode(ts2)

        sim = float(model.similarity(hv1, hv2))
        assert sim < 0.7  # Should be different


class TestTrajectoryEncoder2D:
    """Test 2D path encoding."""

    @pytest.fixture
    def model(self):
        return VSA.create('FHRR', dim=5000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)
        return TrajectoryEncoder(model, scalar_enc, n_dimensions=2, seed=42)

    def test_encode_2d_path(self, encoder):
        """Test encoding a 2D path."""
        path = [(0, 0), (10, 10), (20, 20), (30, 30)]
        hv = encoder.encode(path)

        assert hv.shape == (encoder.dimension,)

    def test_encode_2d_with_lists(self, encoder):
        """Test encoding 2D path as lists instead of tuples."""
        path = [[0, 0], [10, 10], [20, 20]]
        hv = encoder.encode(path)

        assert hv.shape == (encoder.dimension,)

    def test_wrong_dimensionality_fails(self, encoder):
        """Test that wrong dimensionality raises error."""
        # 1D points when expecting 2D
        with pytest.raises(ValueError, match="Expected 2D point"):
            encoder.encode([(0,), (10,), (20,)])

        # 3D points when expecting 2D
        with pytest.raises(ValueError, match="Expected 2D point"):
            encoder.encode([(0, 0, 0), (10, 10, 10)])

    def test_similar_paths_have_high_similarity(self, encoder, model):
        """Test that similar paths have high similarity."""
        path1 = [(0, 0), (10, 10), (20, 20), (30, 30)]
        path2 = [(0, 0), (10, 10), (20, 20), (30, 35)]  # Slightly different end

        hv1 = encoder.encode(path1)
        hv2 = encoder.encode(path2)

        sim = float(model.similarity(hv1, hv2))
        assert sim > 0.8  # Should be quite similar


class TestTrajectoryEncoder3D:
    """Test 3D trajectory encoding."""

    @pytest.fixture
    def model(self):
        return VSA.create('FHRR', dim=5000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)
        return TrajectoryEncoder(model, scalar_enc, n_dimensions=3, seed=42)

    def test_encode_3d_trajectory(self, encoder):
        """Test encoding a 3D trajectory."""
        traj = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
        hv = encoder.encode(traj)

        assert hv.shape == (encoder.dimension,)

    def test_similar_trajectories_have_high_similarity(self, encoder, model):
        """Test that similar 3D trajectories have high similarity."""
        traj1 = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        traj2 = [(0, 0, 0), (1, 1, 1), (2, 2, 3)]  # Slightly different

        hv1 = encoder.encode(traj1)
        hv2 = encoder.encode(traj2)

        sim = float(model.similarity(hv1, hv2))
        assert sim > 0.8


class TestTrajectoryEncoderTimeRange:
    """Test time range normalization."""

    @pytest.fixture
    def model(self):
        return VSA.create('FHRR', dim=5000, seed=42)

    def test_with_time_range(self, model):
        """Test encoding with explicit time range."""
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)
        encoder = TrajectoryEncoder(
            model, scalar_enc, n_dimensions=1, time_range=(0.0, 10.0), seed=42
        )

        ts = [10.0, 20.0, 30.0, 40.0]
        hv = encoder.encode(ts)

        assert hv.shape == (5000,)

    def test_without_time_range(self, model):
        """Test encoding without time range (uses indices)."""
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)
        encoder = TrajectoryEncoder(model, scalar_enc, n_dimensions=1, seed=42)

        ts = [10.0, 20.0, 30.0, 40.0]
        hv = encoder.encode(ts)

        assert hv.shape == (5000,)


class TestTrajectoryEncoderDecoding:
    """Test decoding functionality."""

    @pytest.fixture
    def model(self):
        return VSA.create('FHRR', dim=5000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)
        return TrajectoryEncoder(model, scalar_enc, n_dimensions=2, seed=42)

    def test_decode_not_implemented(self, encoder):
        """Test that decoding raises NotImplementedError."""
        path = [(0, 0), (10, 10), (20, 20)]
        hv = encoder.encode(path)

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            encoder.decode(hv)


class TestTrajectoryEncoderProperties:
    """Test encoder properties."""

    @pytest.fixture
    def model(self):
        return VSA.create('FHRR', dim=1000, seed=42)

    @pytest.fixture
    def encoder(self, model):
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)
        return TrajectoryEncoder(model, scalar_enc, n_dimensions=2, seed=42)

    @pytest.mark.parametrize('model_name', ['MAP', 'FHRR', 'HRR', 'BSC'])
    def test_works_with_multiple_models(self, model_name):
        """Test that encoder works with multiple VSA models."""
        model = VSA.create(model_name, dim=1000, seed=42)

        # Use compatible scalar encoder based on model type
        if model_name in ['FHRR', 'HRR']:
            scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)
        else:  # MAP, BSC, etc.
            scalar_enc = ThermometerEncoder(model, min_val=0, max_val=100, n_bins=100, seed=42)

        encoder = TrajectoryEncoder(model, scalar_enc, n_dimensions=2, seed=42)

        path = [(0, 0), (10, 10), (20, 20)]
        hv = encoder.encode(path)

        assert hv.shape == (1000,)

    def test_is_reversible_property(self, encoder):
        """Test is_reversible property."""
        # Decoding not implemented yet
        assert encoder.is_reversible is False

    def test_compatible_models_property(self, encoder):
        """Test compatible_models property."""
        compatible = encoder.compatible_models
        assert 'MAP' in compatible
        assert 'FHRR' in compatible
        assert 'HRR' in compatible
        assert len(compatible) > 0

    def test_input_type_property(self, model):
        """Test input_type property for different dimensions."""
        scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100, seed=42)

        encoder_1d = TrajectoryEncoder(model, scalar_enc, n_dimensions=1)
        assert encoder_1d.input_type == "1D time series"

        encoder_2d = TrajectoryEncoder(model, scalar_enc, n_dimensions=2)
        assert encoder_2d.input_type == "2D path"

        encoder_3d = TrajectoryEncoder(model, scalar_enc, n_dimensions=3)
        assert encoder_3d.input_type == "3D trajectory"

    def test_repr(self, encoder):
        """Test string representation."""
        repr_str = repr(encoder)
        assert 'TrajectoryEncoder' in repr_str
        assert 'n_dimensions=2' in repr_str
        assert 'FractionalPowerEncoder' in repr_str
