# HoloVec Examples Index

This index separates the maintained release-facing examples from the broader exploratory set.

## Canonical Examples

These scripts are the public learning path and are smoke-tested in pytest with `--smoke`.

| File | Focus | Why it matters |
|------|-------|----------------|
| [00_quickstart.py](00_quickstart.py) | First model, encoder, retrieval pass | shortest end-to-end introduction |
| [02_models_comparison.py](02_models_comparison.py) | Model-family tradeoffs | choose an algebra before you build |
| [10_encoders_scalar.py](10_encoders_scalar.py) | Scalar encoders | continuous, ordinal, and level encoding |
| [13_encoders_position_binding.py](13_encoders_position_binding.py) | Sequence encoding | order-sensitive composition and decoding |
| [26_retrieval_basics.py](26_retrieval_basics.py) | Codebook and `ItemStore` workflows | cleanup retrieval and persistence |
| [27_cleanup_strategies.py](27_cleanup_strategies.py) | Brute force vs resonator cleanup | factorization and cleanup behavior |
| [41_model_ghrr_diagonality.py](41_model_ghrr_diagonality.py) | GHRR structure knobs | diagonality and non-commutativity |
| [42_model_bsdc_seg.py](42_model_bsdc_seg.py) | BSDC-SEG | segment-pattern retrieval and sparse structure |

## Suggested Learning Path

1. `00_quickstart.py`
2. `02_models_comparison.py`
3. `10_encoders_scalar.py`
4. `13_encoders_position_binding.py`
5. `26_retrieval_basics.py`
6. `27_cleanup_strategies.py`
7. `41_model_ghrr_diagonality.py` or `42_model_bsdc_seg.py` if those model families matter

## Exploratory Examples

The scripts below remain useful, but they are not currently treated as release-gated examples:

### Core and encoder deep dives

- `01_basic_operations.py`
- `11_encoders_fractional_power.py`
- `12_encoders_thermometer_level.py`
- `14_encoders_ngram.py`
- `15_encoders_trajectory.py`
- `16_encoders_vector.py`
- `17_encoders_image.py`
- `18_encoders_graph.py`

### Application sketches

- `20_app_text_classification.py`
- `21_app_image_recognition.py`
- `22_app_gesture_recognition.py`
- `23_app_symbolic_reasoning.py`
- `24_app_working_memory.py`
- `25_app_integration_patterns.py`

### Research and analysis scripts

- `28_factorization_methods.py`
- `30_theory_fpe_validation.py`
- `31_performance_benchmarks.py`
- `32_distributed_representations.py`
- `33_error_handling_robustness.py`
- `40_model_hrr_correlation.py`

## Notebooks

The notebooks in `examples/notebooks/` are convenient for exploration, but the `.py` scripts above
are the authoritative source for release-facing examples.

## Documentation

- Docs landing page: <https://twistient.github.io/HoloVec/>
- Quick start: <https://twistient.github.io/HoloVec/getting-started/quick-start/>
- Migration notes: <https://twistient.github.io/HoloVec/guides/migration/>

## Recommended Learning Sequences

### Quick Tour (30 minutes)

1. 00_quickstart.py (5 min)
2. 01_basic_operations.py (10 min)
3. 10_encoders_scalar.py (15 min)

### Application Developer Path (2 hours)

1. 00-02: Fundamentals (30 min)
2. Choose your domain:
   - NLP: 14 → 20 (35 min)
   - Vision: 17 → 21 (35 min)
   - Time Series: 15 → 22 (40 min)
3. Integration: 25 (25 min)
4. Retrieval: 26, 27 (25 min)

### Researcher Path (5 hours)

1. 00-02: Fundamentals (30 min)
2. 10-18: All encoders (3 hours)
3. 27-28: Advanced retrieval (30 min)
4. 30-33: Theory & performance (65 min)
5. 40-42: Model specifics (25 min)

### Full Course (7 hours)

Work through all examples in numerical order (00 → 42)

---

## Support & Documentation

- **Full Guide**: [README.md](README.md)
- **Documentation**: <https://holovec.readthedocs.io>
- **Issues**: <https://github.com/Twistient/HoloVec/issues>
- **Discussions**: <https://github.com/Twistient/HoloVec/discussions>

---

*Last updated: 2025-12-21*
