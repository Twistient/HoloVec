Migration notes for the current pre-`v1.0` API tightening.

These changes are intentional and improve API trustworthiness, but they may require a small update
if you are coming from older `0.3.x` examples or local scripts.

## `VSA.create()` Now Rejects Unsupported Kwargs

Older versions could silently ignore unsupported factory kwargs. The factory now validates the
documented surface and raises `TypeError` on unsupported keys.

Examples of supported calls:

```python
from holovec import VSA

VSA.create("BSDC", dim=20000, sparsity=0.01, binding_mode="cdt")
VSA.create("BSDC-SEG", dim=400, segments=20)
VSA.create("GHRR", dim=96, matrix_size=3, diagonality=0.4)
VSA.create("VTB", dim=512, n_bases=4, temperature=50.0)
VSA.create("FHRR", dim=4096, backend="torch", device="cuda")
```

If you were relying on undocumented passthrough kwargs such as arbitrary dtype or backend internals,
move that logic into your own backend-specific setup code.

## `AssocStore.query_value()` Is Top-1 Only

The `top` parameter was previously accepted but ignored. The method is now intentionally top-1 only.

Before:

```python
value = assoc_store.query_value(key, top=3)
```

After:

```python
value = assoc_store.query_value(key)
```

If you need ranked retrieval, use APIs that explicitly support top-k label queries instead of
assuming `query_value()` will expand to multiple outputs.

## Safe Codebook Loading Is Now the Default

`Codebook.save()` now writes a versioned safe `.npz` format. Loading old pickle-backed archives now
fails closed unless you explicitly opt in for migration.

Migration path:

```python
from holovec.retrieval import Codebook

legacy = Codebook.load("old-codebook.npz", allow_unsafe_legacy=True)
legacy.save("old-codebook-migrated.npz")
```

The intended workflow is:

1. load the legacy archive once with `allow_unsafe_legacy=True`
2. immediately re-save it
3. use the migrated archive from then on

## Threshold Retrieval on Continuous Models

Continuous similarity spaces now accept negative thresholds when that matches the model's similarity
range.

```python
from holovec.utils.search import threshold_search

labels, sims = threshold_search(query, codebook, model, threshold=-0.2)
```

That is valid for continuous cosine-like similarities and remains invalid for discrete overlap or
Hamming-style spaces where the range is `[0.0, 1.0]`.

## Example and Docs Updates

The release-facing examples are now the audited, smoke-tested path through the library:

- `00_quickstart.py`
- `02_models_comparison.py`
- `10_encoders_scalar.py`
- `13_encoders_position_binding.py`
- `26_retrieval_basics.py`
- `27_cleanup_strategies.py`
- `41_model_ghrr_diagonality.py`
- `42_model_bsdc_seg.py`

If you have older notes or internal snippets, prefer updating them to match those scripts.
