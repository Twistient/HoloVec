"""
Base classes and interfaces for encoders.

Encoders transform various data types (scalars, sequences, structured data)
into hypervectors for processing with VSA models.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from holovec.models.base import VSAModel
from holovec.backends.base import Array


class Encoder(ABC):
    """
    Abstract base class for all encoders.

    Encoders transform data into hypervectors compatible with VSA models.
    They follow the principle of locality preservation: similar inputs
    should map to similar hypervectors.

    Attributes:
        model: VSA model instance used for vector operations
        backend: Backend instance (inherited from model)
        dimension: Dimensionality of hypervectors (inherited from model)
    """

    def __init__(self, model: VSAModel):
        """
        Initialize encoder with a VSA model.

        Args:
            model: VSA model instance to use for operations

        Raises:
            ValueError: If model is not compatible with this encoder
        """
        # Validate model compatibility
        if not self._is_model_compatible(model):
            compatible = ", ".join(self.compatible_models)
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with {model.model_name}. "
                f"Compatible models: {compatible}"
            )

        self.model = model
        self.backend = model.backend
        self.dimension = model.dimension

    @abstractmethod
    def encode(self, data: Any) -> Array:
        """
        Encode data into hypervector.

        Args:
            data: Input data (type depends on encoder)

        Returns:
            Hypervector representation of shape (dimension,)

        Raises:
            ValueError: If data is invalid for this encoder
        """
        pass

    def encode_batch(self, data_list: List[Any]) -> List[Array]:
        """
        Encode multiple data points.

        Default implementation encodes each item individually.
        Subclasses may override for more efficient batch encoding.

        Args:
            data_list: List of data items to encode

        Returns:
            List of encoded hypervectors
        """
        return [self.encode(data) for data in data_list]

    @abstractmethod
    def decode(self, hypervector: Array) -> Any:
        """
        Decode hypervector back to data (if possible).

        Args:
            hypervector: Hypervector to decode, shape (dimension,)

        Returns:
            Decoded data, or None if encoder is not reversible

        Raises:
            NotImplementedError: If encoder does not support decoding
        """
        pass

    @property
    @abstractmethod
    def is_reversible(self) -> bool:
        """
        Whether this encoder supports decoding.

        Returns:
            True if decode() is implemented and functional, False otherwise
        """
        pass

    @property
    @abstractmethod
    def compatible_models(self) -> List[str]:
        """
        List of compatible VSA model names.

        Returns:
            List of model names (e.g., ['FHRR', 'HRR'])
        """
        pass

    @property
    @abstractmethod
    def input_type(self) -> str:
        """
        Description of expected input type.

        Returns:
            Human-readable string describing input type
            (e.g., "scalar float", "sequence of symbols", "2D array")
        """
        pass

    def _is_model_compatible(self, model: VSAModel) -> bool:
        """
        Check if given model is compatible with this encoder.

        Args:
            model: VSA model to check

        Returns:
            True if compatible, False otherwise
        """
        return model.model_name in self.compatible_models

    def __repr__(self) -> str:
        """String representation of encoder."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.model_name}, "
            f"dimension={self.dimension}, "
            f"reversible={self.is_reversible})"
        )


class ScalarEncoder(Encoder):
    """
    Abstract base class for encoders that map scalar values to hypervectors.

    Scalar encoders should preserve ordering: similar scalars should map
    to similar hypervectors. Most scalar encoders support a value range
    [min_val, max_val] that is normalized internally.
    """

    def __init__(
        self,
        model: VSAModel,
        min_val: float,
        max_val: float
    ):
        """
        Initialize scalar encoder.

        Args:
            model: VSA model instance
            min_val: Minimum value of encoding range
            max_val: Maximum value of encoding range

        Raises:
            ValueError: If min_val >= max_val
        """
        super().__init__(model)

        if min_val >= max_val:
            raise ValueError(
                f"min_val must be less than max_val, got {min_val} >= {max_val}"
            )

        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def normalize(self, value: float) -> float:
        """
        Normalize value to [0, 1] range.

        Args:
            value: Value to normalize

        Returns:
            Normalized value in [0, 1]

        Note:
            Values outside [min_val, max_val] are clipped.
        """
        # Clip to valid range
        value = max(self.min_val, min(self.max_val, value))

        # Normalize to [0, 1]
        return (value - self.min_val) / self.range

    def denormalize(self, normalized_value: float) -> float:
        """
        Denormalize value from [0, 1] to [min_val, max_val].

        Args:
            normalized_value: Value in [0, 1] range

        Returns:
            Denormalized value in [min_val, max_val]
        """
        return self.min_val + normalized_value * self.range

    @property
    def input_type(self) -> str:
        """Input type description."""
        return f"scalar float in [{self.min_val}, {self.max_val}]"


class SequenceEncoder(Encoder):
    """
    Abstract base class for encoders that map sequences to hypervectors.

    Sequence encoders typically require a codebook mapping elements to
    hypervectors, and encode sequences by combining element representations
    with position information.
    """

    def __init__(
        self,
        model: VSAModel,
        max_length: Optional[int] = None
    ):
        """
        Initialize sequence encoder.

        Args:
            model: VSA model instance
            max_length: Maximum sequence length (None for unlimited)
        """
        super().__init__(model)
        self.max_length = max_length

    @property
    def input_type(self) -> str:
        """Input type description."""
        if self.max_length is not None:
            return f"sequence of length <= {self.max_length}"
        return "sequence of variable length"


class StructuredEncoder(Encoder):
    """
    Abstract base class for encoders that map structured data to hypervectors.

    Structured encoders handle multi-dimensional data like vectors, images,
    or graphs. They typically compose scalar or symbol encoders with
    structural binding operations.
    """

    @property
    def input_type(self) -> str:
        """Input type description."""
        return "structured data"
