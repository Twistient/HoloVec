Welcome to the HoloVec documentation.

**[Overview](overview.md)** — What is HoloVec and why use it?

---

## Getting Started

- [Installation](getting-started/installation.md) — Install HoloVec with pip or uv
- [Quick Start](getting-started/quick-start.md) — Your first hypervectors in 5 minutes
- [Core Concepts](getting-started/core-concepts.md) — Binding, bundling, permutation explained

---

## Architecture

- [Architecture Overview](architecture/index.md) — Layered design and module structure
- [Backends](architecture/backends.md) — NumPy, PyTorch, JAX computational engines
- [Spaces](architecture/spaces.md) — Vector types: Bipolar, Complex, Binary, Sparse

---

## Models

- [Models](models/index.md) — Comparison table and selection guide

| Model | Page | Best For |
|-------|------|----------|
| FHRR | [FHRR](models/fhrr.md) | General use, best capacity |
| GHRR | [GHRR](models/ghrr.md) | Non-commutative relations |
| MAP | [MAP](models/map.md) | Hardware, neuromorphic |
| HRR | [HRR](models/hrr.md) | Classic baseline |
| VTB | [VTB](models/vtb.md) | Directional binding |
| BSC | [BSC](models/bsc.md) | Binary, FPGA |
| BSDC | [BSDC](models/bsdc.md) | Memory efficient |
| BSDC-SEG | [BSDC-SEG](models/bsdc-seg.md) | Fast sparse search |

---

## Encoders

- [Encoders Overview](encoders/index.md) — Which encoder for which data type?
- [Fractional Power](encoders/fractional-power.md) — Continuous values with similarity preservation
- [Thermometer & Level](encoders/thermometer-level.md) — Ordinal and categorical data
- [Sequences](encoders/sequence.md) — Position binding, N-grams, trajectories
- [Spatial](encoders/spatial.md) — Images and multi-dimensional vectors

---

## Retrieval & Memory

- [Retrieval Overview](retrieval/index.md) — Codebook, ItemStore, AssocStore
- [Cleanup Strategies](retrieval/cleanup.md) — Brute force vs Resonator networks

---

## Guides

- [Patterns](guides/patterns.md) — Common VSA patterns and recipes
- [Performance](guides/performance.md) — Backend selection and optimization
- [Troubleshooting](guides/troubleshooting.md) — Common issues and solutions

---

## Reference

- [API Reference](reference/api.md) — Quick reference for all classes and methods
- [Glossary](reference/glossary.md) — HDC/VSA terminology
- [References](reference/bibliography.md) — Academic papers and citations

---

**Links:** [GitHub](https://github.com/Twistient/HoloVec) | [PyPI](https://pypi.org/project/holovec/) | [Issues](https://github.com/Twistient/HoloVec/issues)
