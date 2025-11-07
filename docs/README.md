# holovec Documentation

This directory contains comprehensive documentation for the holovec library, organized hierarchically following professional library best practices.

## Structure

```
docs/
├── theory/          Mathematical foundations and algorithm explanations
│   ├── vsa_models.md    - Theory of 7 VSA models (MAP, FHRR, HRR, BSC, GHRR, VTB, BSDC)
│   └── encoders.md      - Encoder theory (FractionalPower, Thermometer, Level)
│
└── validation/      Empirical validation reports
    └── phase2_models.md - Validation of VSA models against academic literature
```

## Theory Documents

**Purpose**: Explain the mathematical foundations, algorithms, and theoretical properties

- **`theory/vsa_models.md`** (1000+ lines)
  - Mathematical foundations of hyperdimensional computing
  - Detailed explanations of all 7 VSA models
  - Operations: bind, unbind, bundle, permute
  - Model comparison and capacity analysis
  - Academic references

- **`theory/encoders.md`** (584 lines)
  - Fractional Power Encoding (Frady et al. 2021)
  - Thermometer and Level encoders
  - Mathematical formulas and convergence properties
  - Practical usage guidelines
  - Comparison of encoder approaches

## Validation Documents

**Purpose**: Empirical validation of implementations against academic literature

- **`validation/phase2_models.md`** (480 lines)
  - Validation of all 7 VSA models
  - Property-based testing results
  - Capacity benchmarks (Schlegel et al. 2022)
  - Comparison with published results

## Design Principle

**Implementation code stays clean** - The `holovec/` package contains only Python code (`.py` files). All documentation lives here in `docs/`, separated by purpose:

- **Theory**: What the algorithms do and why they work
- **Validation**: Proof that implementations match theory
- **User guides** (future): How to use the library

This separation follows patterns from major libraries (PyTorch, scikit-learn, NumPy) and ensures:

- Clean, professional code organization
- Easy navigation for different audiences
- Scalability as documentation grows

## Related Documentation

- **README.md** (root) - Quick start and overview
- **examples/** - Working code demonstrations
- **IMPLEMENTATION_PLAN.md** - Development roadmap
- **PHASE_*_COMPLETE.md** - Completion reports for each phase
