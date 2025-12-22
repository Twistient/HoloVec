# API Reference

Complete API documentation auto-generated from source code docstrings.

## VSA Factory

The main entry point for creating VSA models.

::: holovec.VSA
    options:
      show_root_heading: true
      members:
        - create
        - backend_info

---

## Models

All VSA model implementations.

### Base Model

::: holovec.models.base.VSAModel
    options:
      show_root_heading: true
      members:
        - bind
        - unbind
        - bundle
        - permute
        - unpermute
        - similarity
        - normalize
        - random

### FHRR

::: holovec.models.fhrr.FHRRModel
    options:
      show_root_heading: true

### GHRR

::: holovec.models.ghrr.GHRRModel
    options:
      show_root_heading: true

### MAP

::: holovec.models.map.MAPModel
    options:
      show_root_heading: true

### HRR

::: holovec.models.hrr.HRRModel
    options:
      show_root_heading: true

### VTB

::: holovec.models.vtb.VTBModel
    options:
      show_root_heading: true

### BSC

::: holovec.models.bsc.BSCModel
    options:
      show_root_heading: true

### BSDC

::: holovec.models.bsdc.BSDCModel
    options:
      show_root_heading: true

### BSDC-SEG

::: holovec.models.bsdc_seg.BSDCSEGModel
    options:
      show_root_heading: true

---

## Encoders

### Scalar Encoders

::: holovec.encoders.scalar.FractionalPowerEncoder
    options:
      show_root_heading: true

::: holovec.encoders.scalar.ThermometerEncoder
    options:
      show_root_heading: true

::: holovec.encoders.scalar.LevelEncoder
    options:
      show_root_heading: true

### Sequence Encoders

::: holovec.encoders.sequence.PositionBindingEncoder
    options:
      show_root_heading: true

::: holovec.encoders.sequence.NGramEncoder
    options:
      show_root_heading: true

::: holovec.encoders.sequence.TrajectoryEncoder
    options:
      show_root_heading: true

### Spatial Encoders

::: holovec.encoders.spatial.ImageEncoder
    options:
      show_root_heading: true

::: holovec.encoders.structured.VectorEncoder
    options:
      show_root_heading: true

---

## Retrieval

### Codebook

::: holovec.retrieval.codebook.Codebook
    options:
      show_root_heading: true

### ItemStore

::: holovec.retrieval.itemstore.ItemStore
    options:
      show_root_heading: true

### AssocStore

::: holovec.retrieval.assocstore.AssocStore
    options:
      show_root_heading: true

---

## Cleanup

::: holovec.utils.cleanup.BruteForceCleanup
    options:
      show_root_heading: true

::: holovec.utils.cleanup.ResonatorCleanup
    options:
      show_root_heading: true

---

## Backends

::: holovec.backends.base.Backend
    options:
      show_root_heading: true
      members:
        - zeros
        - ones
        - random
        - random_normal
        - add
        - multiply
        - dot
        - fft
        - ifft
        - norm
        - normalize
        - to_numpy
        - from_numpy
        - supports_gpu
        - supports_complex
        - supports_sparse
