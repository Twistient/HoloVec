## Requirements

- Python 3.11 or higher
- NumPy (installed automatically)

## Basic Installation

### From PyPI

```bash
pip install holovec
```

### From Source

```bash
git clone https://github.com/Twistient/HoloVec.git
cd HoloVec
uv pip install -e .
```

## Optional Backends

HoloVec supports three computational backends. NumPy is always available; PyTorch and JAX are optional.

### PyTorch (GPU Support)

```bash
pip install holovec[torch]
# or
pip install torch
```

Enables CUDA (NVIDIA) and MPS (Apple Silicon) GPU acceleration.

### JAX (JIT Compilation)

```bash
pip install holovec[jax]
# or
pip install jax jaxlib
```

Enables JIT compilation for 10-100x speedup on repeated operations.

### All Backends

```bash
pip install holovec[all]
```

## Development Installation

For contributing or running tests:

```bash
git clone https://github.com/Twistient/HoloVec.git
cd HoloVec
uv sync --extra dev
```

This includes:
- pytest for testing
- black for code formatting
- ruff for linting
- mypy for type checking
- hypothesis for property-based testing

## Verifying Installation

```python
from holovec import VSA, backend_info

# Check available backends
print(backend_info())

# Create a model and test basic operation
model = VSA.create('FHRR', dim=1024)
a, b = model.random(), model.random()
c = model.bind(a, b)
print(f"Binding works: {model.similarity(model.unbind(c, b), a) > 0.99}")
```

Expected output:
```
{'numpy': True, 'torch': True/False, 'jax': True/False}
Binding works: True
```

## Backend Selection

HoloVec automatically selects NumPy as the default backend. To use a specific backend:

```python
# NumPy (default, always available)
model = VSA.create('FHRR', dim=2048)

# PyTorch with CPU
model = VSA.create('FHRR', dim=2048, backend='torch')

# PyTorch with CUDA GPU
model = VSA.create('FHRR', dim=2048, backend='torch', device='cuda')

# PyTorch with Apple Silicon GPU
model = VSA.create('FHRR', dim=2048, backend='torch', device='mps')

# JAX with JIT compilation
model = VSA.create('FHRR', dim=2048, backend='jax')
```

## Troubleshooting

### PyTorch not detected

Ensure PyTorch is installed:
```bash
python -c "import torch; print(torch.__version__)"
```

### JAX not detected

Ensure JAX and jaxlib are installed:
```bash
python -c "import jax; print(jax.__version__)"
```

### CUDA not available

Check CUDA installation:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
```

### Apple Silicon MPS issues

MPS support requires macOS 12.3+ and PyTorch 1.12+:
```python
import torch
print(torch.backends.mps.is_available())
```

!!! note
    Some complex number operations may have limited MPS support. FHRR works correctly; fall back to CPU if issues arise.
