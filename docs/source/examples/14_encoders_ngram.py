"""
Demonstration of N-gram Encoder for local sequence pattern encoding.
====================================================================

This demo showcases the NGramEncoder, which captures local patterns in
sequences using sliding windows (n-grams). This is particularly useful for:

- Text analysis (character/word n-grams)
- Pattern matching in sequences
- Similarity detection based on local context
- NLP applications

The encoder supports:
- Multiple n-gram sizes (unigrams, bigrams, trigrams, etc.)
- Overlapping and non-overlapping windows (stride parameter)
- Two modes: bundling (bag-of-ngrams) and chaining (ordered n-grams)
"""

from holovec import VSA
from holovec.encoders import NGramEncoder


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print('=' * 70)


def demo_basic_ngram_encoding():
    """Demonstrate basic n-gram encoding."""
    print_section("Demo 1: Basic N-gram Encoding")

    model = VSA.create('MAP', dim=5000, seed=42)
    encoder = NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)

    print(f"\nEncoder: {encoder}")
    print(f"Configuration: n={encoder.n}, stride={encoder.stride}, mode='{encoder.mode}'")

    # Encode a sequence
    sequence = ['the', 'quick', 'brown', 'fox']
    hv = encoder.encode(sequence)

    print(f"\nInput sequence: {sequence}")
    print(f"Bigrams (n=2, stride=1):")
    print("  - ['the', 'quick']")
    print("  - ['quick', 'brown']")
    print("  - ['brown', 'fox']")
    print(f"\nEncoded hypervector shape: {hv.shape}")
    print(f"Codebook size: {encoder.get_codebook_size()} unique symbols")


def demo_different_n_values():
    """Demonstrate different n-gram sizes."""
    print_section("Demo 2: Different N-gram Sizes")

    model = VSA.create('MAP', dim=5000, seed=42)
    sequence = ['A', 'B', 'C', 'D', 'E']

    print(f"\nInput sequence: {sequence}\n")

    for n in [1, 2, 3, 4]:
        encoder = NGramEncoder(model, n=n, stride=1, mode='bundling', seed=42)
        hv = encoder.encode(sequence)

        # Calculate number of n-grams
        num_ngrams = len(sequence) - n + 1

        name = {1: "Unigrams", 2: "Bigrams", 3: "Trigrams", 4: "4-grams"}[n]
        print(f"{name} (n={n}):")
        print(f"  Number of n-grams: {num_ngrams}")
        print(f"  Hypervector shape: {hv.shape}")


def demo_stride_parameter():
    """Demonstrate stride (overlapping vs non-overlapping)."""
    print_section("Demo 3: Stride Parameter (Overlapping vs Non-overlapping)")

    model = VSA.create('MAP', dim=5000, seed=42)
    sequence = ['A', 'B', 'C', 'D', 'E', 'F']

    print(f"\nInput sequence: {sequence}\n")

    # Overlapping bigrams (stride=1)
    encoder1 = NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)
    hv1 = encoder1.encode(sequence)

    print("Overlapping bigrams (stride=1):")
    print("  N-grams: ['A','B'], ['B','C'], ['C','D'], ['D','E'], ['E','F']")
    print(f"  Count: 5 n-grams")

    # Non-overlapping bigrams (stride=2)
    encoder2 = NGramEncoder(model, n=2, stride=2, mode='bundling', seed=42)
    hv2 = encoder2.encode(sequence)

    print("\nNon-overlapping bigrams (stride=2):")
    print("  N-grams: ['A','B'], ['C','D'], ['E','F']")
    print(f"  Count: 3 n-grams")

    # Partial overlap (stride=2 with trigrams)
    encoder3 = NGramEncoder(model, n=3, stride=2, mode='bundling', seed=42)
    hv3 = encoder3.encode(sequence)

    print("\nPartial overlap trigrams (n=3, stride=2):")
    print("  N-grams: ['A','B','C'], ['C','D','E']")
    print(f"  Count: 2 n-grams")


def demo_text_similarity():
    """Demonstrate text similarity using n-grams."""
    print_section("Demo 4: Text Similarity with N-grams")

    model = VSA.create('MAP', dim=10000, seed=42)
    encoder = NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)

    # Encode sentences as word bigrams
    sent1 = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    sent2 = ['the', 'cat', 'sat', 'on', 'the', 'hat']  # Similar (1 word diff)
    sent3 = ['a', 'dog', 'ran', 'in', 'the', 'park']   # Different

    hv1 = encoder.encode(sent1)
    hv2 = encoder.encode(sent2)
    hv3 = encoder.encode(sent3)

    print("\nSentence 1:", ' '.join(sent1))
    print("Sentence 2:", ' '.join(sent2), "(differs by 1 word)")
    print("Sentence 3:", ' '.join(sent3), "(completely different)\n")

    sim_1_2 = float(model.similarity(hv1, hv2))
    sim_1_3 = float(model.similarity(hv1, hv3))

    print(f"Similarity (sent1 vs sent2): {sim_1_2:.3f}")
    print(f"Similarity (sent1 vs sent3): {sim_1_3:.3f}")

    print("\nKey insight:")
    print("  Sentences sharing more bigrams have higher similarity.")
    print("  Bigrams shared by sent1 and sent2:")
    print("    ['the','cat'], ['cat','sat'], ['sat','on'], ['on','the']")


def demo_bundling_vs_chaining():
    """Demonstrate bundling mode vs chaining mode."""
    print_section("Demo 5: Bundling vs Chaining Modes")

    model = VSA.create('MAP', dim=10000, seed=42)

    sequence = ['A', 'B', 'C']

    # Bundling mode: order-invariant across n-grams
    encoder_bundle = NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)
    hv_bundle = encoder_bundle.encode(sequence)

    print("Bundling Mode (bag-of-ngrams):")
    print(f"  Sequence: {sequence}")
    print(f"  N-grams: ['A','B'], ['B','C']")
    print(f"  Encoding: bundle(encode(['A','B']), encode(['B','C']))")
    print(f"  Order-sensitive: No (n-grams bundled)")
    print(f"  Reversible: {encoder_bundle.is_reversible}")

    # Chaining mode: order-sensitive
    encoder_chain = NGramEncoder(model, n=2, stride=1, mode='chaining', seed=42)
    hv_chain = encoder_chain.encode(sequence)

    print("\nChaining Mode (ordered n-grams):")
    print(f"  Sequence: {sequence}")
    print(f"  N-grams: ['A','B'] at position 0, ['B','C'] at position 1")
    print(f"  Encoding: bundle(permute(encode(['A','B']), 0), permute(encode(['B','C']), 1))")
    print(f"  Order-sensitive: Yes (positions encoded)")
    print(f"  Reversible: {encoder_chain.is_reversible}")

    # Test decoding in chaining mode
    if encoder_chain.is_reversible:
        decoded = encoder_chain.decode(hv_chain, max_ngrams=3, threshold=0.2)
        print(f"\nDecoded n-grams (approximate): {decoded}")


def demo_character_ngrams():
    """Demonstrate character-level n-grams."""
    print_section("Demo 6: Character-Level N-grams")

    model = VSA.create('MAP', dim=10000, seed=42)
    encoder = NGramEncoder(model, n=3, stride=1, mode='bundling', seed=42)

    # Encode words as character trigrams
    word1 = list("pattern")   # ['p','a','t','t','e','r','n']
    word2 = list("patter")    # ['p','a','t','t','e','r']
    word3 = list("matter")    # ['m','a','t','t','e','r']

    hv1 = encoder.encode(word1)
    hv2 = encoder.encode(word2)
    hv3 = encoder.encode(word3)

    print("\nWord 1: 'pattern'")
    print("  Trigrams: pat, att, tte, ter, ern")

    print("\nWord 2: 'patter'")
    print("  Trigrams: pat, att, tte, ter")
    print("  (shares 4/5 with 'pattern')")

    print("\nWord 3: 'matter'")
    print("  Trigrams: mat, att, tte, ter")
    print("  (shares 3/5 with 'pattern')")

    sim_1_2 = float(model.similarity(hv1, hv2))
    sim_1_3 = float(model.similarity(hv1, hv3))

    print(f"\nSimilarity 'pattern' vs 'patter': {sim_1_2:.3f}")
    print(f"Similarity 'pattern' vs 'matter': {sim_1_3:.3f}")

    print("\nKey insight:")
    print("  Character n-grams capture sub-word similarity")
    print("  Useful for misspelling detection and fuzzy matching")


def demo_application_text_classification():
    """Demonstrate application: text classification."""
    print_section("Demo 7: Application - Text Classification")

    model = VSA.create('MAP', dim=10000, seed=42)
    encoder = NGramEncoder(model, n=2, stride=1, mode='bundling', seed=42)

    print("\nScenario: Classify sentences as positive or negative\n")

    # Training examples
    positive_examples = [
        ['i', 'love', 'this', 'product'],
        ['great', 'quality', 'and', 'service'],
        ['highly', 'recommend', 'this', 'item']
    ]

    negative_examples = [
        ['terrible', 'quality', 'and', 'service'],
        ['do', 'not', 'recommend', 'this'],
        ['very', 'poor', 'experience']
    ]

    # Create class prototypes by bundling examples
    positive_hvs = [encoder.encode(ex) for ex in positive_examples]
    negative_hvs = [encoder.encode(ex) for ex in negative_examples]

    positive_prototype = model.bundle(positive_hvs)
    negative_prototype = model.bundle(negative_hvs)

    print("Training Data:")
    print("  Positive examples: 3")
    print("  Negative examples: 3")

    # Test examples
    test_sentences = [
        (['i', 'recommend', 'this', 'product'], "Positive"),
        (['poor', 'quality', 'item'], "Negative"),
        (['great', 'experience'], "Positive"),
    ]

    print("\nTest Results:")
    for sentence, true_label in test_sentences:
        hv = encoder.encode(sentence)

        sim_positive = float(model.similarity(hv, positive_prototype))
        sim_negative = float(model.similarity(hv, negative_prototype))

        predicted = "Positive" if sim_positive > sim_negative else "Negative"

        print(f"\n  Sentence: {' '.join(sentence)}")
        print(f"  Sim to positive: {sim_positive:.3f}")
        print(f"  Sim to negative: {sim_negative:.3f}")
        print(f"  Predicted: {predicted}, True: {true_label} "
              f"{'✓' if predicted == true_label else '✗'}")


def main():
    """Run all demos."""
    print("=" * 70)
    print("N-gram Encoder - Comprehensive Demonstration")
    print("=" * 70)
    print("\nThe NGramEncoder captures local patterns in sequences using")
    print("sliding windows. This is essential for:")
    print("  - Text analysis (NLP)")
    print("  - Pattern recognition")
    print("  - Sequence similarity")
    print("  - Classification based on local features")

    demo_basic_ngram_encoding()
    demo_different_n_values()
    demo_stride_parameter()
    demo_text_similarity()
    demo_bundling_vs_chaining()
    demo_character_ngrams()
    demo_application_text_classification()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - See docs/theory/encoders.md for mathematical details")
    print("  - Run tests: pytest tests/test_encoders_sequence.py -k NGram")
    print("  - Try with different VSA models (FHRR, HRR, BSC)")


if __name__ == '__main__':
    main()
