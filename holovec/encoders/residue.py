"""Residue Hyperdimensional Computing encoder.

This module implements Residue HDC based on Kymn et al. 2024 "Computing With
Residue Numbers in High-Dimensional Representation", enabling both addition
AND multiplication on encoded integers with logarithmic resource scaling.

Key Features:
    - Additive binding: z(x₁ + x₂) = z(x₁) ⊙ z(x₂) (Hadamard product)
    - Multiplicative binding: z(x₁ × x₂) via phase exponentiation (Phase 2)
    - Logarithmic codebook scaling: range M = ∏mᵢ, codebook size = ∑mᵢ
    - Chinese Remainder Theorem for unique integer encoding
    - Integration with FHRR's complex phasor operations

Mathematical Foundation:
    Given K pairwise co-prime moduli m₁, m₂, ..., mₖ:
    - Range M = m₁ × m₂ × ... × mₖ (Chinese Remainder Theorem)
    - Codebook size = m₁ + m₂ + ... + mₖ (logarithmic scaling)
    - Each modulus has base vector with phases constrained to mth roots of unity
    - Encoding: z(x) = z_m₁(x mod m₁) ⊙ z_m₂(x mod m₂) ⊙ ... ⊙ z_mₖ(x mod mₖ)

Example:
    >>> from holovec.encoders import ResidueEncoder
    >>> encoder = ResidueEncoder(dim=2000, moduli=[3, 5, 7])
    >>> print(f"Range: {encoder.range}, Codebook: {encoder.codebook_size}")
    Range: 105, Codebook: 15
    >>> z_20 = encoder.encode(20)
    >>> z_5 = encoder.encode(5)
    >>> z_25 = encoder.add(z_20, z_5)  # Represents 20 + 5 = 25
    >>> decoded = encoder.decode(z_25)
    >>> print(decoded)  # 25

References:
    Kymn et al. (2024): "Computing With Residue Numbers in High-Dimensional
        Representation" - RHC algorithm and theory
    Chinese Remainder Theorem: Unique encoding guarantee for co-prime moduli
"""

from __future__ import annotations

import math

import numpy as np

from holovec.backends.base import Array
from holovec.models.fhrr import FHRRModel


class ResidueEncoder:
    """Residue Hyperdimensional Computing encoder.

    Encodes integers using residue number system with multiple co-prime moduli.
    Enables both additive (Hadamard) and multiplicative (star) binding.

    Based on Kymn et al. 2024 "Computing With Residue Numbers in HD".

    Properties:
        - Range: M = ∏ᵢ mᵢ (product of moduli via Chinese Remainder Theorem)
        - Codebook size: ∑ᵢ mᵢ (sum of moduli - logarithmic scaling)
        - Addition: z(x₁+x₂) = z(x₁) ⊙ z(x₂) (Hadamard product)
        - Subtraction: z(x₁-x₂) = z(x₁) ⊙ z(x₂)* (Hadamard with conjugate)

    Args:
        dim: Hypervector dimension (default 1000)
        moduli: List of co-prime moduli (e.g., [3, 5, 7] for range 105)
        model: FHRR model instance (optional, creates one if None)
        seed: Random seed for reproducibility

    Raises:
        ValueError: If moduli are not pairwise co-prime

    Example:
        >>> encoder = ResidueEncoder(dim=2000, moduli=[3, 5, 7])
        >>> z_20 = encoder.encode(20)  # 20 = [2 mod 3, 0 mod 5, 6 mod 7]
        >>> z_5 = encoder.encode(5)
        >>> z_25 = encoder.add(z_20, z_5)      # Represents 25
        >>> decoded = encoder.decode(z_25)      # Returns 25

    Attributes:
        dim: Hypervector dimension
        moduli: List of co-prime moduli
        M: Total range (product of moduli)
        K: Number of moduli
        model: FHRR model for complex phasor operations
        backend: Computational backend
    """

    def __init__(
        self,
        dim: int = 1000,
        moduli: list[int] | None = None,
        model: FHRRModel | None = None,
        seed: int | None = None,
    ):
        """Initialize ResidueEncoder.

        Args:
            dim: Hypervector dimension (default 1000)
            moduli: List of co-prime moduli (default [3, 5, 7] for range 105)
            model: FHRR model instance (optional, creates one if None)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.moduli = moduli or [3, 5, 7]  # Default: range 105
        self._validate_coprime(self.moduli)

        self.M = math.prod(self.moduli)  # Total range
        self.K = len(self.moduli)  # Number of moduli

        # Create FHRR model if not provided
        if model is None:
            model = FHRRModel(dimension=dim, seed=seed)
        self.model = model
        self.backend = model.backend
        self._seed = seed

        # Generate base vectors for each modulus (mth roots of unity)
        self._base_vectors = self._generate_base_vectors()

        # Pre-compute codebooks for efficient decoding
        self._codebooks = self._generate_codebooks()

    def _validate_coprime(self, moduli: list[int]) -> None:
        """Verify all moduli are pairwise co-prime.

        Args:
            moduli: List of moduli to validate

        Raises:
            ValueError: If any pair of moduli share a common factor > 1
        """
        for i, m1 in enumerate(moduli):
            if m1 < 2:
                raise ValueError(f"Moduli must be >= 2, got {m1}")
            for m2 in moduli[i + 1 :]:
                if math.gcd(m1, m2) != 1:
                    raise ValueError(
                        f"Moduli must be co-prime: gcd({m1}, {m2}) = {math.gcd(m1, m2)} != 1"
                    )

    def _generate_base_vectors(self) -> dict[int, Array]:
        """Generate base phasor for each modulus using mth roots of unity.

        For modulus m, phases are restricted to {2πk/m | k=0,...,m-1}.
        This ensures z_m(x + m) = z_m(x) (periodicity).

        The key insight from Kymn et al. is that constraining phases to
        mth roots of unity enables exact periodic arithmetic modulo m.

        Returns:
            Dictionary mapping modulus m to its base phasor vector.
        """
        base_vectors = {}
        rng = np.random.default_rng(self._seed)

        for m in self.moduli:
            # Sample random integers k ∈ {0, 1, ..., m-1} for each dimension
            # Each dimension gets a random frequency from the allowed set
            k_values = rng.integers(0, m, size=self.dim)

            # Convert to phases: θ = 2πk/m (restricts to mth roots of unity)
            phases = 2 * np.pi * k_values / m

            # Create phasor: z = exp(iθ)
            phasor = np.exp(1j * phases).astype(np.complex64)
            base_vectors[m] = self.backend.from_numpy(phasor)

        return base_vectors

    def _generate_codebooks(self) -> dict[int, list[Array]]:
        """Pre-compute codebook vectors for each modulus.

        For modulus m, generates vectors z_m(0), z_m(1), ..., z_m(m-1)
        where z_m(i) = base^i (element-wise exponentiation).

        Returns:
            Dictionary mapping modulus m to list of m codebook vectors.
        """
        codebooks: dict[int, list[Array]] = {}

        for m in self.moduli:
            base = self._base_vectors[m]
            base_np = self.backend.to_numpy(base)

            # Generate z_m(0), z_m(1), ..., z_m(m-1)
            codebook = []
            for i in range(m):
                # z_m(i) = base^i (element-wise exponentiation)
                # For phasor e^(iθ), raising to power i gives e^(i·i·θ)
                phasor_i = np.power(base_np, i).astype(np.complex64)
                codebook.append(self.backend.from_numpy(phasor_i))
            codebooks[m] = codebook

        return codebooks

    def encode(self, x: int) -> Array:
        """Encode integer x into hypervector.

        Uses residue number system encoding:
            z(x) = z_m₁(x mod m₁) ⊙ z_m₂(x mod m₂) ⊙ ... ⊙ z_mₖ(x mod mₖ)

        Where each z_mₖ(rₖ) = base_mₖ^rₖ is the rₖth codebook vector for modulus mₖ.

        Args:
            x: Integer to encode (0 ≤ x < M)

        Returns:
            Complex hypervector encoding x

        Raises:
            ValueError: If x is outside valid range [0, M)
        """
        if not (0 <= x < self.M):
            raise ValueError(f"x must be in [0, {self.M}), got {x}")

        # Start with first modulus's residue encoding
        first_m = self.moduli[0]
        result = self._codebooks[first_m][x % first_m]

        # Hadamard product with remaining moduli
        for m in self.moduli[1:]:
            remainder = x % m
            z_m = self._codebooks[m][remainder]
            result = self.backend.multiply(result, z_m)

        return result

    def add(self, z1: Array, z2: Array) -> Array:
        """Additive binding: result represents x₁ + x₂ (mod M).

        z(x₁ + x₂) = z(x₁) ⊙ z(x₂) (Hadamard product)

        This works because for each modulus m:
            z_m(r₁) ⊙ z_m(r₂) = base^r₁ ⊙ base^r₂ = base^(r₁+r₂)
                              = z_m((r₁ + r₂) mod m)

        Args:
            z1: Encoded hypervector for x₁
            z2: Encoded hypervector for x₂

        Returns:
            Encoded hypervector representing x₁ + x₂ (mod M)
        """
        return self.backend.multiply(z1, z2)

    def subtract(self, z1: Array, z2: Array) -> Array:
        """Subtractive unbinding: result represents x₁ - x₂ (mod M).

        z(x₁ - x₂) = z(x₁) ⊙ z(x₂)* (Hadamard with conjugate)

        This works because conjugate inverts the phase:
            z_m(r)* = base^(-r) = base^(m-r) = z_m(-r mod m)

        Args:
            z1: Encoded hypervector for x₁
            z2: Encoded hypervector for x₂

        Returns:
            Encoded hypervector representing x₁ - x₂ (mod M)
        """
        z2_conj = self.backend.conjugate(z2)
        return self.backend.multiply(z1, z2_conj)

    def negate(self, z: Array) -> Array:
        """Negate: result represents -x (mod M).

        z(-x) = z(x)* (conjugate)

        Args:
            z: Encoded hypervector for x

        Returns:
            Encoded hypervector representing -x (mod M)
        """
        return self.backend.conjugate(z)

    def multiply_from_values(self, x1: int, x2: int) -> Array:
        """Multiply two known integer values.

        z(x₁ × x₂) computed directly from values.

        Note: Full multiply() on encoded vectors requires factorization and
        is deferred to Phase 2. Use this method when you know the values.

        Args:
            x1: First integer
            x2: Second integer

        Returns:
            Encoded hypervector representing x₁ × x₂ (mod M)
        """
        product = (x1 * x2) % self.M
        return self.encode(product)

    def decode(self, z: Array, method: str = "auto") -> int:
        """Decode hypervector back to integer.

        Args:
            z: Encoded hypervector
            method: Decoding method:
                - "auto": Use brute force for range <= 10000, else iterative
                - "brute_force": Try all possible values (exact but slow)
                - "iterative": Use iterative unbinding (fast but approximate)

        Returns:
            Decoded integer value in [0, M)
        """
        if method == "auto":
            method = "brute_force" if self.M <= 10000 else "iterative"

        if method == "brute_force":
            return self._decode_brute_force(z)
        else:
            remainders = self._decode_iterative(z)
            return self._chinese_remainder_theorem(remainders)

    def _decode_brute_force(self, z: Array) -> int:
        """Decode by trying all possible values.

        For each integer x in [0, M), compute encode(x) and compare with z.
        Return the x with highest similarity.

        This is exact but O(M * D) complexity.

        Args:
            z: Encoded hypervector

        Returns:
            Decoded integer with highest similarity
        """
        z_np = self.backend.to_numpy(z)
        dim = self.dim

        best_sim = -float("inf")
        best_x = 0

        for x in range(self.M):
            encoded = self.encode(x)
            encoded_np = self.backend.to_numpy(encoded)
            similarity = float(np.real(np.vdot(encoded_np, z_np))) / dim

            if similarity > best_sim:
                best_sim = similarity
                best_x = x

        return best_x

    def _decode_iterative(
        self, z: Array, max_iterations: int = 10
    ) -> dict[int, int]:
        """Decode using iterative unbinding with multiple starting points.

        Algorithm:
            1. Try different initializations for the first modulus
            2. For each initialization, run iterative refinement:
               - For each modulus m: unbind others, find best match
               - Repeat until convergence
            3. Pick the result with highest reconstruction similarity

        This handles the cold start problem by trying multiple starting
        points and selecting the one that reconstructs best.

        Args:
            z: Encoded hypervector
            max_iterations: Maximum iterations per starting point

        Returns:
            Dictionary mapping modulus to decoded remainder
        """
        dim = self.dim
        z_np = self.backend.to_numpy(z)

        # Try different starting points for the first (smallest) modulus
        first_m = self.moduli[0]
        best_remainders = dict.fromkeys(self.moduli, 0)
        best_reconstruction_sim = -float("inf")

        for start_i in range(first_m):
            # Initialize with this starting point
            remainders = dict.fromkeys(self.moduli, 0)
            remainders[first_m] = start_i

            # Iterative refinement
            for _iteration in range(max_iterations):
                changed = False

                for m in self.moduli:
                    # Compute product of other moduli's current estimates
                    other_product = self._compute_other_product(remainders, m)

                    # Unbind other moduli from z
                    unbound = self.backend.multiply(
                        z, self.backend.conjugate(other_product)
                    )
                    unbound_np = self.backend.to_numpy(unbound)

                    # Find best matching codebook entry for this modulus
                    best_sim = -float("inf")
                    best_match = remainders[m]

                    for i in range(m):
                        codebook_np = self.backend.to_numpy(self._codebooks[m][i])
                        similarity = float(np.real(np.vdot(codebook_np, unbound_np))) / dim

                        if similarity > best_sim:
                            best_sim = similarity
                            best_match = i

                    if best_match != remainders[m]:
                        remainders[m] = best_match
                        changed = True

                if not changed:
                    break

            # Evaluate reconstruction quality
            reconstruction = self._reconstruct(remainders)
            reconstruction_np = self.backend.to_numpy(reconstruction)
            reconstruction_sim = float(np.real(np.vdot(reconstruction_np, z_np))) / dim

            if reconstruction_sim > best_reconstruction_sim:
                best_reconstruction_sim = reconstruction_sim
                best_remainders = remainders.copy()

        return best_remainders

    def _reconstruct(self, remainders: dict[int, int]) -> Array:
        """Reconstruct encoded vector from remainders.

        Args:
            remainders: Dictionary mapping modulus to remainder

        Returns:
            Reconstructed hypervector
        """
        result = self._codebooks[self.moduli[0]][remainders[self.moduli[0]]]
        for m in self.moduli[1:]:
            result = self.backend.multiply(result, self._codebooks[m][remainders[m]])
        return result

    def _compute_other_product(
        self, remainders: dict[int, int], exclude_m: int
    ) -> Array:
        """Compute product of all moduli's encodings except one.

        Args:
            remainders: Current residue estimates for all moduli
            exclude_m: Modulus to exclude from product

        Returns:
            Hadamard product of codebook entries for other moduli
        """
        result = None

        for m in self.moduli:
            if m == exclude_m:
                continue
            z_m = self._codebooks[m][remainders[m]]
            if result is None:
                result = z_m
            else:
                result = self.backend.multiply(result, z_m)

        if result is None:
            # Only one modulus - return ones vector (identity)
            ones = np.ones(self.dim, dtype=np.complex64)
            return self.backend.from_numpy(ones)

        return result

    def _chinese_remainder_theorem(self, remainders: dict[int, int]) -> int:
        """Combine remainders to recover original integer via CRT.

        Given x ≡ rᵢ (mod mᵢ) for all i, finds unique x in [0, M).

        The Chinese Remainder Theorem guarantees a unique solution exists
        when the moduli are pairwise co-prime.

        Algorithm:
            x = Σᵢ rᵢ · Mᵢ · yᵢ (mod M)

            where:
            - Mᵢ = M / mᵢ (product of all other moduli)
            - yᵢ = Mᵢ⁻¹ (mod mᵢ) (modular multiplicative inverse)

        Args:
            remainders: Dictionary mapping modulus to remainder

        Returns:
            Recovered integer x in [0, M)
        """
        x = 0
        for m in self.moduli:
            r = remainders[m]
            # M_i = M / m_i
            M_i = self.M // m
            # y_i such that M_i * y_i ≡ 1 (mod m_i)
            y_i = pow(M_i, -1, m)
            x += r * M_i * y_i

        return x % self.M

    def encode_with_residues(self, x: int) -> tuple[Array, dict[int, int]]:
        """Encode integer and return both hypervector and residues.

        Useful for debugging or when you need access to intermediate values.

        Args:
            x: Integer to encode (0 ≤ x < M)

        Returns:
            Tuple of (encoded hypervector, dict of modulus → remainder)
        """
        residues = {m: x % m for m in self.moduli}
        z = self.encode(x)
        return z, residues

    @property
    def range(self) -> int:
        """Maximum encodable value (exclusive)."""
        return self.M

    @property
    def codebook_size(self) -> int:
        """Total number of codebook vectors needed (sum of moduli)."""
        return sum(self.moduli)

    @property
    def dimension(self) -> int:
        """Hypervector dimension."""
        return self.dim

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ResidueEncoder(dim={self.dim}, moduli={self.moduli}, "
            f"range={self.M}, codebook_size={self.codebook_size})"
        )
