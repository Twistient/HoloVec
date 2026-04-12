<div align="center">
  <img src="https://raw.githubusercontent.com/Twistient/HoloVec/master/docs/assets/logo.svg" alt="HoloVec Logo" width="400">
</div>
<div align="center">

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Vector Symbolic Architectures for compositional, high-dimensional computing.**

[Documentation](https://twistient.github.io/HoloVec/) |
[Installation](https://twistient.github.io/HoloVec/getting-started/installation/) |
[Quick Start](https://twistient.github.io/HoloVec/getting-started/quick-start/) |
[Examples](https://github.com/Twistient/HoloVec/tree/master/examples)

</div>

---

## What HoloVec Is

HoloVec is a Python library for hyperdimensional computing (HDC) and vector symbolic
architectures (VSA). It gives you a consistent API for:

- binding and unbinding structured representations
- bundling sets, prototypes, and memories
- permutation-based order encoding
- scalar, sequence, spatial, and structured encoders
- associative retrieval and cleanup memories

The library currently treats NumPy as the release-blocking backend. PyTorch and JAX are
supported as optional backends when their environment-specific tests are enabled.

---

## Installation

```bash
pip install holovec
```

Optional extras:

```bash
pip install "holovec[torch]"  # PyTorch backend
pip install "holovec[jax]"    # JAX backend
pip install "holovec[all]"    # torch + jax + dev + docs extras
```

For source work:

```bash
git clone https://github.com/Twistient/HoloVec.git
cd HoloVec
uv sync --extra dev
```

Use `uv pip install -e .[dev]` only when you explicitly want an editable install outside the
normal `uv sync` workflow.

---

## Quick Start

```python
from holovec import VSA
from holovec.encoders import FractionalPowerEncoder
from holovec.retrieval import Codebook, ItemStore

model = VSA.create("FHRR", dim=4096, seed=7)

role_color = model.random(seed=1)
role_temp = model.random(seed=2)
red = model.random(seed=10)
apple = model.random(seed=11)

temp_encoder = FractionalPowerEncoder(
    model,
    min_val=0.0,
    max_val=100.0,
    bandwidth=1.5,
    seed=3,
)

record = model.bundle(
    [
        model.bind(role_color, red),
        model.bind(role_temp, temp_encoder.encode(24.0)),
    ]
)

recovered_color = model.unbind(record, role_color)
store = ItemStore(model).fit(
    Codebook({"red": red, "apple": apple}, backend=model.backend)
)

print(store.query(recovered_color, k=1))
```

If you want a runnable version of that workflow, start with
[examples/00_quickstart.py](examples/00_quickstart.py).

---

## Choosing a Model

| Model | Space | Inverse Style | Order Sensitivity | Typical Use |
|-------|-------|---------------|-------------------|-------------|
| `FHRR` | Complex | Exact | No | General-purpose default, continuous encoders |
| `GHRR` | Matrix | Exact | Yes | Order-sensitive and nested structures |
| `MAP` | Bipolar | Self-inverse | No | Fast algebra, hardware-friendly workflows |
| `HRR` | Real/Bipolar | Approximate | No | Classic baseline and literature reproduction |
| `VTB` | Matrix/Real | Approximate | Yes | Directional bindings and asymmetric relations |
| `BSC` | Binary | Self-inverse | No | Binary and FPGA-style workflows |
| `BSDC` | Sparse | Approximate | No | Sparse memory-efficient representations |
| `BSDC-SEG` | Sparse segment | Self-inverse | No | Segment-sparse retrieval and fast pattern search |

Start with `FHRR` unless you have a clear reason to prefer order-sensitive matrix models,
self-inverse discrete models, or sparse segment-based storage.

The factory now validates model-specific kwargs instead of ignoring them. Supported examples:

```python
VSA.create("BSDC", dim=20000, sparsity=0.01, binding_mode="cdt")
VSA.create("BSDC-SEG", dim=400, segments=20)
VSA.create("GHRR", dim=96, matrix_size=3, diagonality=0.4)
VSA.create("VTB", dim=512, n_bases=4, temperature=50.0)
VSA.create("FHRR", dim=4096, backend="torch", device="cuda")
```

See the full comparison in
[docs/models/index.md](https://twistient.github.io/HoloVec/models/index/).

---

## Release-Facing Examples

These examples are the audited path through the library and are smoke-tested in pytest:

- [examples/00_quickstart.py](examples/00_quickstart.py): create a model, encode data, bind,
  retrieve
- [examples/02_models_comparison.py](examples/02_models_comparison.py): compare model families
- [examples/10_encoders_scalar.py](examples/10_encoders_scalar.py): scalar encoders and decoding
- [examples/13_encoders_position_binding.py](examples/13_encoders_position_binding.py): sequence
  encoding and decoding
- [examples/26_retrieval_basics.py](examples/26_retrieval_basics.py): codebooks, item stores,
  threshold retrieval, persistence
- [examples/27_cleanup_strategies.py](examples/27_cleanup_strategies.py): brute-force vs
  resonator cleanup
- [examples/41_model_ghrr_diagonality.py](examples/41_model_ghrr_diagonality.py): GHRR order
  sensitivity
- [examples/42_model_bsdc_seg.py](examples/42_model_bsdc_seg.py): BSDC-SEG segment patterns

Additional examples remain in the repository, but the list above is the maintained `v1` learning
path.

---

## Documentation Highlights

- [Quick Start](https://twistient.github.io/HoloVec/getting-started/quick-start/)
- [Model Comparison](https://twistient.github.io/HoloVec/models/index/)
- [Design Patterns](https://twistient.github.io/HoloVec/guides/patterns/)
- [Performance Guidance](https://twistient.github.io/HoloVec/guides/performance/)
- [Migration Notes](https://twistient.github.io/HoloVec/guides/migration/)

---

## Testing

```bash
uv run --extra dev pytest --no-cov
uv run --extra dev ruff check holovec tests
uv run --extra dev mypy holovec
uv sync --extra docs && uv run --extra docs mkdocs build --strict
```

The release gate also includes smoke execution of the canonical example scripts.

---

## References

HoloVec draws on the core HDC/VSA literature, including Kanerva (1988), Plate (2003), Gayler
(2003), Frady et al. (2021), Schlegel et al. (2022), Kleyko et al. (2023), and Kymn et al.
(2024).

---

## Citation

```bibtex
@software{HoloVec2025,
  author       = {Brodie Schroeder},
  title        = {HoloVec: Vector Symbolic Architectures for Python},
  year         = {2025},
  version      = {0.3.2},
  url          = {https://github.com/Twistient/HoloVec},
  license      = {Apache-2.0}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Key areas: new models, encoders, documentation, and performance.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Contact

[GitHub Issues](https://github.com/Twistient/HoloVec/issues) • [Discussions](https://github.com/Twistient/HoloVec/discussions) • brodie@twistient.com
