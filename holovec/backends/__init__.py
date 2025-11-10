"""Backend management for holovec.

This module provides automatic backend detection and a unified interface
for accessing different computational backends (NumPy, PyTorch, JAX).
"""

from __future__ import annotations

from typing import Dict, Optional, Type, Union

from .base import Backend, BackendError, BackendNotAvailableError
from .numpy_backend import NumPyBackend

# Try to import optional backends
try:
    from .torch_backend import TorchBackend, TORCH_AVAILABLE
except ImportError:
    TorchBackend = None
    TORCH_AVAILABLE = False

try:
    from .jax_backend import JAXBackend, JAX_AVAILABLE
except ImportError:
    JAXBackend = None
    JAX_AVAILABLE = False


# Backend registry
_BACKENDS: Dict[str, Type[Backend]] = {
    'numpy': NumPyBackend,
}

# Only register backends that are actually available (i.e., their dependencies are installed)
if TORCH_AVAILABLE and TorchBackend is not None:
    _BACKENDS['torch'] = TorchBackend
    _BACKENDS['pytorch'] = TorchBackend  # Alias

if JAX_AVAILABLE and JAXBackend is not None:
    _BACKENDS['jax'] = JAXBackend


# Default backend
_DEFAULT_BACKEND: Optional[Backend] = None


def get_available_backends() -> list[str]:
    """Return a list of available backend names.

    Returns:
        List of backend names that can be used
    """
    return list(_BACKENDS.keys())


def is_backend_available(name: str) -> bool:
    """Check if a specific backend is available.

    Args:
        name: Backend name ('numpy', 'torch', 'jax')

    Returns:
        True if backend is available, False otherwise
    """
    return name.lower() in _BACKENDS


def get_backend(name: Union[str, Backend, None] = None, **kwargs) -> Backend:
    """Get a backend instance by name.

    Args:
        name: Backend name ('numpy', 'torch', 'jax'), a Backend instance, or None.
              If a Backend instance is passed, it is returned as-is.
              If None, returns default backend.
        **kwargs: Backend-specific arguments (e.g., device='cuda' for torch)

    Returns:
        Backend instance

    Raises:
        BackendNotAvailableError: If requested backend is not available
        ValueError: If backend name is not recognized

    Examples:
        >>> backend = get_backend('numpy')
        >>> backend = get_backend('torch', device='cuda')
        >>> backend = get_backend()  # Returns default
        >>> backend = get_backend(existing_backend)  # Returns existing_backend
    """
    global _DEFAULT_BACKEND

    # If already a Backend instance, return it directly
    if isinstance(name, Backend):
        return name

    # If no name specified, return or create default
    if name is None:
        if _DEFAULT_BACKEND is None:
            _DEFAULT_BACKEND = _create_default_backend()
        return _DEFAULT_BACKEND

    # Normalize name
    name = name.lower()

    # Check if backend is available
    if name not in _BACKENDS:
        available = get_available_backends()
        raise ValueError(f"Unknown backend '{name}'. Available backends: {available}")

    # Create backend instance
    backend_class = _BACKENDS[name]
    try:
        return backend_class(**kwargs)
    except Exception as e:
        raise BackendError(f"Failed to initialize {name} backend: {e}")


def set_default_backend(name: str, **kwargs) -> None:
    """Set the default backend.

    Args:
        name: Backend name ('numpy', 'torch', 'jax')
        **kwargs: Backend-specific arguments

    Raises:
        BackendNotAvailableError: If requested backend is not available

    Examples:
        >>> set_default_backend('torch', device='cuda')
        >>> set_default_backend('numpy')
    """
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = get_backend(name, **kwargs)


def _create_default_backend() -> Backend:
    """Create the default backend based on availability.

    Priority order:
    1. NumPy (always available)
    2. PyTorch (if available)
    3. JAX (if available)

    Returns:
        Default backend instance
    """
    # NumPy is always the default fallback
    return NumPyBackend()


def auto_detect_backend() -> str:
    """Automatically detect the best available backend.

    Priority order:
    1. JAX (best for research/JIT)
    2. PyTorch (best for GPU/neural)
    3. NumPy (always available)

    Returns:
        Name of the best available backend
    """
    if JAX_AVAILABLE:
        return 'jax'
    elif TORCH_AVAILABLE:
        return 'torch'
    else:
        return 'numpy'


def backend_info() -> dict:
    """Get information about available backends.

    Returns:
        Dictionary with backend availability and capabilities
    """
    return {
        'available_backends': get_available_backends(),
        'default_backend': _DEFAULT_BACKEND.name if _DEFAULT_BACKEND else None,
        'recommended_backend': auto_detect_backend(),
        'numpy': True,
        'torch': TORCH_AVAILABLE,
        'jax': JAX_AVAILABLE,
    }


__all__ = [
    'Backend',
    'BackendError',
    'BackendNotAvailableError',
    'NumPyBackend',
    'TorchBackend',
    'JAXBackend',
    'get_backend',
    'set_default_backend',
    'get_available_backends',
    'is_backend_available',
    'auto_detect_backend',
    'backend_info',
]
