"""
Structured data encoders for mapping vectors and structured data to hypervectors.

This module implements encoders that transform multi-dimensional vectors
and structured data into hypervector representations by composing scalar
encoders with dimension binding.
"""

from typing import List, Optional
from holovec.encoders.base import StructuredEncoder, ScalarEncoder
from holovec.models.base import VSAModel
from holovec.backends.base import Array


class VectorEncoder(StructuredEncoder):
    """
    Vector encoder for multi-dimensional numeric data using role-filler binding.

    Encodes vectors by binding each dimension with its scalar-encoded value:

        encode([v₁, v₂, ..., vₐ]) = Σᵢ bind(Dᵢ, scalar_encode(vᵢ))

    where:
    - Dᵢ is a random hypervector for dimension i
    - scalar_encode(vᵢ) encodes the scalar value using FPE/Thermometer/Level
    - bind() creates a role-filler binding
    - Σ bundles all dimension-value pairs

    This creates a compositional encoding where:
    - Each dimension has explicit representation (Dᵢ)
    - Similar values in corresponding dimensions → higher similarity
    - Supports partial matching across dimensions
    - Enables approximate decoding via unbinding

    Attributes:
        scalar_encoder: Encoder for individual scalar values
        n_dimensions: Number of dimensions in input vectors
        dim_vectors: List of dimension hypervectors (Dᵢ)
        normalize_input: Whether to normalize input vectors

    Example:
        >>> from holovec import VSA
        >>> from holovec.encoders import FractionalPowerEncoder, VectorEncoder
        >>>
        >>> model = VSA.create('FHRR', dim=10000)
        >>> scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=1)
        >>> encoder = VectorEncoder(model, scalar_encoder=scalar_enc, n_dims=128)
        >>>
        >>> # Encode a feature vector (list or any backend array)
        >>> features = [0.5] * 128  # Can also use numpy/torch/jax arrays
        >>> hv = encoder.encode(features)
        >>>
        >>> # Similar vectors have high similarity
        >>> features2 = [0.51] * 128  # Slightly different
        >>> hv2 = encoder.encode(features2)
        >>> model.similarity(hv, hv2)  # High similarity
        >>>
        >>> # Decode to recover approximate values
        >>> recovered = encoder.decode(hv)
        >>> # Verify approximate recovery via similarity
        >>> model.similarity(encoder.encode(recovered), hv) > 0.9
    """

    def __init__(
        self,
        model: VSAModel,
        scalar_encoder: ScalarEncoder,
        n_dimensions: int,
        normalize_input: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize vector encoder.

        Args:
            model: VSA model instance
            scalar_encoder: Encoder for individual scalar values
            n_dimensions: Number of dimensions in input vectors
            normalize_input: Whether to normalize input vectors to unit length
            seed: Random seed for dimension vector generation

        Raises:
            ValueError: If n_dimensions < 1
            TypeError: If scalar_encoder is not a ScalarEncoder
        """
        super().__init__(model)

        if n_dimensions < 1:
            raise ValueError(f"n_dimensions must be >= 1, got {n_dimensions}")

        if not isinstance(scalar_encoder, ScalarEncoder):
            raise TypeError(
                f"scalar_encoder must be a ScalarEncoder, got {type(scalar_encoder)}"
            )

        # Check model compatibility
        if model != scalar_encoder.model:
            raise ValueError(
                "scalar_encoder must use the same VSA model as VectorEncoder"
            )

        self.scalar_encoder = scalar_encoder
        self.n_dimensions = n_dimensions
        self.normalize_input = normalize_input
        self.seed = seed

        # Generate dimension hypervectors (one per dimension)
        # These are the "roles" in role-filler binding
        self.dim_vectors: List[Array] = []
        for i in range(n_dimensions):
            # Use deterministic seeding for reproducibility
            if seed is not None:
                dim_seed = seed + i
            else:
                dim_seed = i + 1000  # Offset to avoid collision with symbol seeds

            self.dim_vectors.append(model.random(seed=dim_seed))

    def encode(self, vector: Array) -> Array:
        """
        Encode a vector using dimension binding.

        Each element is bound with its corresponding dimension vector:

            result = Σᵢ bind(Dᵢ, scalar_encode(vector[i]))

        Args:
            vector: Input vector to encode, shape (n_dimensions,)

        Returns:
            Hypervector representing the vector

        Raises:
            ValueError: If vector shape doesn't match n_dimensions

        Example:
            >>> encoder = VectorEncoder(model, scalar_enc, n_dims=3)
            >>> vector = [1.0, 2.0, 3.0]  # Can also be numpy/torch/jax array
            >>> hv = encoder.encode(vector)
        """
        # Convert to backend array if needed
        vector = self.backend.array(vector)

        if vector.shape != (self.n_dimensions,):
            raise ValueError(
                f"Expected vector of shape ({self.n_dimensions},), "
                f"got {vector.shape}"
            )

        # Optional: normalize to unit length
        if self.normalize_input:
            vector = self.backend.normalize(vector)

        # Bind each dimension with its scalar-encoded value
        bound_dims = []
        for i, value in enumerate(vector):
            # Encode scalar value as hypervector
            value_hv = self.scalar_encoder.encode(float(value))

            # Bind dimension role with value filler
            dim_hv = self.dim_vectors[i]
            bound = self.model.bind(dim_hv, value_hv)

            bound_dims.append(bound)

        # Bundle all dimension-value bindings
        vector_hv = self.model.bundle(bound_dims)

        return vector_hv

    def decode(self, hypervector: Array) -> Array:
        """
        Decode vector hypervector to recover approximate values.

        For each dimension i:
        1. Unbind dimension: value_hv = unbind(hypervector, Dᵢ)
        2. Decode scalar: value ≈ scalar_encoder.decode(value_hv)

        Args:
            hypervector: Vector hypervector to decode, shape (dimension,)

        Returns:
            Decoded vector, shape (n_dimensions,) (backend array type)

        Raises:
            NotImplementedError: If scalar_encoder doesn't support decoding

        Note:
            Decoding is approximate and quality depends on:
            - VSA model (exact vs. approximate binding)
            - Scalar encoder precision
            - Number of dimensions (more dims → more noise)

        Example:
            >>> original = [1.0, 2.0, 3.0]
            >>> encoded = encoder.encode(original)
            >>> decoded = encoder.decode(encoded)
            >>> # Check approximate recovery (using backend operations)
            >>> model.similarity(encoder.encode(decoded), encoded) > 0.9
        """
        if not self.scalar_encoder.is_reversible:
            raise NotImplementedError(
                f"Cannot decode: scalar_encoder {type(self.scalar_encoder).__name__} "
                "does not support decoding"
            )

        decoded_values = []

        for i in range(self.n_dimensions):
            # Unbind dimension to recover value hypervector
            dim_hv = self.dim_vectors[i]
            value_hv = self.model.unbind(hypervector, dim_hv)

            # Decode scalar value
            value = self.scalar_encoder.decode(value_hv)
            decoded_values.append(value)

        return self.backend.array(decoded_values)

    @property
    def is_reversible(self) -> bool:
        """
        VectorEncoder supports approximate decoding if scalar_encoder does.

        Returns:
            True if scalar_encoder supports decoding, False otherwise
        """
        return self.scalar_encoder.is_reversible

    @property
    def compatible_models(self) -> List[str]:
        """
        Works with all VSA models.

        Decoding quality varies:
        - Exact models (FHRR, MAP): High accuracy
        - Approximate models (HRR, BSC): Moderate accuracy

        Returns:
            List of all model names
        """
        return ["MAP", "FHRR", "HRR", "BSC", "GHRR", "VTB", "BSDC"]

    @property
    def input_type(self) -> str:
        """Input type description."""
        return f"{self.n_dimensions}-dimensional vector"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VectorEncoder("
            f"model={self.model.model_name}, "
            f"scalar_encoder={type(self.scalar_encoder).__name__}, "
            f"n_dimensions={self.n_dimensions}, "
            f"normalize_input={self.normalize_input})"
        )
