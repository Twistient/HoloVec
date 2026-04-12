================
HoloVec Examples
================

This directory contains two kinds of material:

1. **Canonical examples**: the maintained, smoke-tested learning path for the public API.
2. **Exploratory examples**: useful scripts that are not currently treated as release-gated docs
   artifacts.

Canonical Examples
==================

These are the scripts to read first and the ones the test suite executes with ``--smoke``:

- ``00_quickstart.py`` - first end-to-end workflow
- ``02_models_comparison.py`` - choose a model family
- ``10_encoders_scalar.py`` - scalar encoding patterns
- ``13_encoders_position_binding.py`` - order-sensitive sequence encoding
- ``26_retrieval_basics.py`` - codebooks, item stores, retrieval, persistence
- ``27_cleanup_strategies.py`` - cleanup and factorization
- ``41_model_ghrr_diagonality.py`` - GHRR structure controls
- ``42_model_bsdc_seg.py`` - BSDC-SEG segment-pattern workflows

Suggested Path
==============

For most users:

1. ``00_quickstart.py``
2. ``02_models_comparison.py``
3. ``10_encoders_scalar.py``
4. ``13_encoders_position_binding.py``
5. ``26_retrieval_basics.py``
6. ``27_cleanup_strategies.py``

Then pick the advanced model example that matches your workload:

- ``41_model_ghrr_diagonality.py`` for order-sensitive matrix binding
- ``42_model_bsdc_seg.py`` for segment-sparse retrieval

Exploratory Scripts
===================

The rest of the directory remains valuable, but those scripts are better thought of as extended
notes, application sketches, or research probes than as release-facing tutorials.

Examples include:

- encoder and modality deep dives: ``11_*`` through ``18_*``
- application sketches: ``20_*`` through ``25_*``
- research and analysis scripts: ``28_*`` through ``33_*`` and ``40_*``

Notebooks
=========

Interactive notebooks live in ``examples/notebooks/``. They are convenient for exploration, but
the canonical ``.py`` scripts are the authoritative source for the maintained example workflows.

Documentation
=============

- Documentation: https://twistient.github.io/HoloVec/
- Quick start: https://twistient.github.io/HoloVec/getting-started/quick-start/
- Migration notes: https://twistient.github.io/HoloVec/guides/migration/
