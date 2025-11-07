"""holovec: Hyperdimensional Computing / Vector Symbolic Architectures

A comprehensive library for hyperdimensional computing with support for
multiple VSA models, backends (NumPy, PyTorch, JAX), and encoders.

Quick Start:
    >>> from holovec import VSA
    >>>
    >>> # Create a VSA model
    >>> model = VSA.create('FHRR', dim=512)
    >>>
    >>> # Generate random vectors
    >>> a = model.random()
    >>> b = model.random()
    >>>
    >>> # Bind vectors (association)
    >>> c = model.bind(a, b)
    >>>
    >>> # Unbind (recovery)
    >>> a_recovered = model.unbind(c, b)
    >>>
    >>> # Check similarity
    >>> similarity = model.similarity(a, a_recovered)
    >>> print(f"Similarity: {similarity:.3f}")  # Should be close to 1.0

Models:
    - MAP: Multiply-Add-Permute (self-inverse, simple hardware)
    - FHRR: Fourier HRR (exact inverse, best capacity)
    - HRR: Holographic Reduced Representations (circular convolution)
    - BSC: Binary Spatter Codes (self-inverse, binary)

Backends:
    - NumPy: Default, always available
    - PyTorch: GPU acceleration, neural network integration
    - JAX: JIT compilation, automatic differentiation

Vector Spaces:
    - Bipolar: {-1, +1} for MAP
    - Binary: {0, 1} for BSC
    - Real: Gaussian for HRR
    - Complex: Unit phasors for FHRR
    - Sparse: Sparse binary for BSDC
"""

from __future__ import annotations

from typing import Optional

from . import backends, models, spaces
from .backends import Backend, get_backend
from .models import BSCModel, BSDCModel, FHRRModel, GHRRModel, HRRModel, MAPModel, VSAModel, VTBModel
from .spaces import VectorSpace, create_space

__version__ = "0.1.0"


class VSA:
    """High-level factory interface for creating VSA models.

    This class provides a simple, unified API for creating and using
    different VSA models. It's the recommended entry point for most users.

    Examples:
        >>> # Create a MAP model with default settings
        >>> model = VSA.create('MAP')
        >>>
        >>> # Create FHRR with specific dimension and backend
        >>> model = VSA.create('FHRR', dim=512, backend='torch', device='cuda')
        >>>
        >>> # Use the model
        >>> a, b = model.random(), model.random()
        >>> c = model.bind(a, b)
        >>> similarity = model.similarity(a, model.unbind(c, b))
    """

    # Model registry
    _MODELS = {
        'map': MAPModel,
        'fhrr': FHRRModel,
        'hrr': HRRModel,
        'bsc': BSCModel,
        'bsdc': BSDCModel,
        'ghrr': GHRRModel,
        'vtb': VTBModel,
    }

    # Default vector spaces for each model
    _DEFAULT_SPACES = {
        'map': 'bipolar',
        'fhrr': 'complex',
        'hrr': 'real',
        'bsc': 'binary',
        'bsdc': 'sparse',
        'ghrr': 'matrix',
        'vtb': 'real',
    }

    @classmethod
    def create(
        cls,
        model_type: str,
        dim: int = 10000,
        backend: Optional[str] = None,
        space: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> VSAModel:
        """Create a VSA model with the specified configuration.

        Args:
            model_type: Model name ('MAP', 'FHRR', 'HRR', 'BSC', etc.)
            dim: Dimensionality of hypervectors
            backend: Backend name ('numpy', 'torch', 'jax') or None for default
            space: Vector space name or None for model's default
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to backend (e.g., device='cuda')

        Returns:
            VSA model instance

        Raises:
            ValueError: If model_type is not recognized

        Examples:
            >>> model = VSA.create('MAP', dim=10000)
            >>> model = VSA.create('FHRR', dim=512, backend='torch', device='cuda')
            >>> model = VSA.create('MAP', space='real')  # Use real-valued MAP
        """
        # Normalize model type
        model_type_lower = model_type.lower()

        if model_type_lower not in cls._MODELS:
            available = list(cls._MODELS.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available models: {available}"
            )

        # Get model class
        model_class = cls._MODELS[model_type_lower]

        # Create backend
        backend_kwargs = {k: v for k, v in kwargs.items() if k in ['device']}
        backend_instance = get_backend(backend, **backend_kwargs) if backend else None

        # Determine space type
        if space is None:
            space = cls._DEFAULT_SPACES.get(model_type_lower)

        # Create space if string provided
        if isinstance(space, str):
            space_instance = create_space(
                space,
                dimension=dim,
                backend=backend_instance,
                seed=seed
            )
        else:
            space_instance = space

        # Create model
        model = model_class(
            dimension=dim,
            space=space_instance,
            backend=backend_instance,
            seed=seed
        )

        return model

    @classmethod
    def available_models(cls) -> list[str]:
        """Return list of available model names.

        Returns:
            List of model names that can be used with create()
        """
        return list(cls._MODELS.keys())

    @classmethod
    def model_info(cls, model_type: str) -> dict:
        """Get information about a specific model.

        Args:
            model_type: Model name

        Returns:
            Dictionary with model properties

        Example:
            >>> info = VSA.model_info('FHRR')
            >>> print(info['is_exact_inverse'])  # True
        """
        model_type_lower = model_type.lower()
        if model_type_lower not in cls._MODELS:
            raise ValueError(f"Unknown model type '{model_type}'")

        # Create a temporary instance to query properties
        model = cls.create(model_type, dim=100)

        return {
            'name': model.model_name,
            'is_self_inverse': model.is_self_inverse,
            'is_commutative': model.is_commutative,
            'is_exact_inverse': model.is_exact_inverse,
            'default_space': cls._DEFAULT_SPACES.get(model_type_lower),
            'class': model.__class__.__name__,
        }


# Convenience functions
def create_model(model_type: str, **kwargs) -> VSAModel:
    """Convenience function to create a VSA model.

    This is an alias for VSA.create() for users who prefer functional style.

    Args:
        model_type: Model name ('MAP', 'FHRR', etc.)
        **kwargs: Arguments passed to VSA.create()

    Returns:
        VSA model instance

    Example:
        >>> model = create_model('FHRR', dim=512)
    """
    return VSA.create(model_type, **kwargs)


def backend_info() -> dict:
    """Get information about available backends.

    Returns:
        Dictionary with backend availability and capabilities

    Example:
        >>> info = backend_info()
        >>> print(info['available_backends'])
        ['numpy', 'torch', 'jax']
    """
    return backends.backend_info()


__all__ = [
    # Main API
    'VSA',
    'create_model',
    'backend_info',
    # Models
    'VSAModel',
    'MAPModel',
    'FHRRModel',
    'HRRModel',
    'BSCModel',
    'BSDCModel',
    'GHRRModel',
    'VTBModel',
    # Spaces
    'VectorSpace',
    'create_space',
    # Backends
    'Backend',
    'get_backend',
    # Submodules
    'models',
    'spaces',
    'backends',
]
