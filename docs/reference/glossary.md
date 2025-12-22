Key terms in hyperdimensional computing and HoloVec.

## A

**Associative Memory**
: Memory system where data is retrieved by content similarity rather than address. VSA naturally implements associative memory through similarity-based lookup.

## B

**Backend**
: Computational engine that executes numerical operations. HoloVec supports NumPy, PyTorch, and JAX backends.

**Binding (⊗)**
: Operation that associates two hypervectors, producing a result dissimilar to both inputs. Used for key-value pairs and role-filler structures.

**Bipolar**
: Vector values restricted to {-1, +1}. Used by MAP and HRR models.

**BSC (Binary Spatter Codes)**
: VSA model using binary {0,1} vectors with XOR binding.

**BSDC (Binary Sparse Distributed Codes)**
: Sparse binary VSA model where only ~1% of bits are active.

**Bundling (+)**
: Operation that combines multiple hypervectors into a superposition. The result is similar to all inputs.

## C

**Capacity**
: Maximum number of items that can be reliably stored and retrieved. Depends on dimension and model.

**Cleanup**
: Process of projecting a noisy hypervector to the nearest known item in a codebook.

**Codebook**
: Collection of labeled hypervectors serving as a vocabulary or symbol table.

**Commutativity**
: Property where operation order doesn't matter: a ⊗ b = b ⊗ a. Most VSA models are commutative.

**Complex Space**
: Vector space using unit phasors (e^iθ). Used by FHRR model.

## D

**Dimension (D)**
: Number of components in a hypervector. Typically 1,000-10,000.

**Distributed Representation**
: Information encoding where meaning is spread across many dimensions rather than localized.

## E

**Encoder**
: Module that transforms real-world data (numbers, text, images) into hypervectors.

**Exact Inverse**
: Property where unbinding perfectly recovers the original: unbind(bind(a, b), b) = a. FHRR and GHRR have exact inverse.

## F

**FHRR (Fourier Holographic Reduced Representations)**
: VSA model using complex phasors with element-wise multiplication. Best capacity, exact inverse.

**FPE (Fractional Power Encoder)**
: Encoder that preserves similarity for continuous values using fractional exponentiation.

## G

**GHRR (Generalized HRR)**
: Extension of FHRR using matrices instead of scalars. Non-commutative, state-of-the-art (2024).

## H

**HDC (Hyperdimensional Computing)**
: Computing paradigm using high-dimensional vectors for representation and computation. Synonymous with VSA.

**Holographic**
: Property where information is distributed across the entire representation (like a hologram).

**HRR (Holographic Reduced Representations)**
: Classic VSA model using circular convolution for binding. Has approximate inverse.

**Hypervector**
: High-dimensional vector (~1,000-10,000 dimensions) used to represent data in VSA/HDC.

## I

**ItemStore**
: HoloVec component for high-level retrieval operations.

## J

**JIT (Just-In-Time)**
: Compilation technique used by JAX for performance optimization.

## L

**Locality Preservation**
: Property where similar inputs produce similar hypervectors.

## M

**MAP (Multiply-Add-Permute)**
: Simple VSA model using element-wise multiplication. Self-inverse, hardware-friendly.

**Matrix Space**
: Vector space using unitary matrices. Used by GHRR and VTB models.

## N

**N-gram**
: Sequence of n consecutive items. NGramEncoder captures local context.

**Non-commutative**
: Property where operation order matters: a ⊗ b ≠ b ⊗ a. GHRR and VTB are non-commutative.

**Normalization**
: Scaling a vector to unit magnitude or specific constraints.

## O

**Orthogonality**
: Property where random hypervectors are nearly perpendicular (similarity ≈ 0).

## P

**Permutation (ρ)**
: Operation that shifts vector coordinates, encoding position or order.

**Phasor**
: Complex number on the unit circle: e^iθ. Used in FHRR.

**Prototype**
: Representative hypervector for a class, typically created by bundling examples.

## R

**Resonator Network**
: Iterative algorithm for cleanup and factorization. More efficient than brute force for large codebooks.

**Role-Filler Binding**
: Pattern where roles (slots) are bound to fillers (values) to create structured representations.

## S

**Self-Inverse**
: Property where the same operation serves as both bind and unbind. MAP and BSC are self-inverse.

**Similarity**
: Measure of relatedness between hypervectors (0 = orthogonal, 1 = identical).

**Space**
: Type of vector values and associated similarity measure. Examples: Bipolar, Complex, Binary.

**Sparse**
: Representation where most values are zero. Memory-efficient for high dimensions.

**Superposition**
: State where a single hypervector represents multiple items simultaneously (via bundling).

## T

**Thermometer Encoding**
: Ordinal encoder where similarity accumulates with adjacent levels.

## U

**Unbinding (⊗⁻¹)**
: Operation that recovers one operand given a bound pair and the other operand.

## V

**VSA (Vector Symbolic Architecture)**
: Framework for compositional representation using hypervectors and algebraic operations.

**VTB (Vector-derived Transformation Binding)**
: Non-commutative VSA model using vector-derived transformations.

## See Also

- [Core Concepts](../getting-started/core-concepts.md) — Detailed explanations
- [References](../reference/bibliography.md) — Academic sources
