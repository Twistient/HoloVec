**Vector Symbolic Architectures for compositional, high-dimensional computing.**

HoloVec is a Python library for hyperdimensional computing (HDC) and Vector Symbolic Architectures (VSA). It represents data as high-dimensional vectors (~1,000-10,000 dimensions) that can be composed using algebraic operations.

## Why HoloVec?

- **One-shot learning** — No gradient descent, encode patterns directly
- **Noise-tolerant** — Graceful degradation under corruption
- **Transparent** — Symbolic reasoning without black-box models
- **Compositional** — Build complex structures from simple operations

## What Can You Do?

```python
from holovec import VSA

# Create a model
model = VSA.create('FHRR', dim=2048)

# Generate and combine hypervectors
a, b = model.random(), model.random()
c = model.bind(a, b)          # Association (key-value)
d = model.bundle([a, b, c])   # Superposition (set)
e = model.permute(a, k=1)     # Sequence encoding

# Query and recover
a_recovered = model.unbind(c, b)
print(model.similarity(a, a_recovered))  # 1.0
```

## Use Cases

- **Semantic memory** — Store and query knowledge structures
- **Classification** — One-shot text, image, gesture recognition
- **Symbolic AI** — Role-filler binding, analogical reasoning
- **Sensor fusion** — Combine multimodal data streams
- **Edge deployment** — Hardware-friendly binary representations

## Models at a Glance

| Model | Binding | Inverse | Best For |
|-------|---------|---------|----------|
| [FHRR](models/fhrr.md) | Complex multiply | Exact | General use, best capacity |
| [GHRR](models/ghrr.md) | Matrix product | Exact | Non-commutative relations |
| [MAP](models/map.md) | Element multiply | Self | Hardware, neuromorphic |
| [HRR](models/hrr.md) | Circular convolution | Approx | Classic baseline |
| [VTB](models/vtb.md) | Matrix transform | Approx | Directional binding |
| [BSC](models/bsc.md) | XOR | Self | Binary, FPGA |
| [BSDC](models/bsdc.md) | Sparse XOR | Approx | Memory efficient |
| [BSDC-SEG](models/bsdc-seg.md) | Segment XOR | Self | Fast sparse search |

## Project Links

- [GitHub Repository](https://github.com/Twistient/HoloVec)
- [PyPI Package](https://pypi.org/project/holovec/)
- [Examples](https://github.com/Twistient/HoloVec/tree/master/examples)
- [Issue Tracker](https://github.com/Twistient/HoloVec/issues)
