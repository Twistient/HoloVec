import numpy as np

from holovec.backends import get_backend
from holovec.models.fhrr import FHRRModel
from holovec.utils.decode import decode_nearest, decode_threshold, decode_multilabel
from holovec.retrieval import Codebook, ItemStore


def test_decode_helpers_nearest_and_threshold():
    be = get_backend("numpy")
    model = FHRRModel(dimension=256, backend=be, seed=0)
    items = {f"item{i}": model.random(seed=10 + i) for i in range(6)}

    # nearest should return exact match for identical vec
    lbls = decode_nearest(items["item3"], items, model, k=1)
    assert lbls[0][0] == "item3"

    # threshold with high bar keeps only exact
    hits = decode_threshold(items["item4"], items, model, threshold=0.99)
    assert hits and hits[0][0] == "item4"

    # multilabel: top-k returns k items
    res = decode_multilabel(items["item5"], items, model, method="topk", k=3)
    assert len(res) == 3


def test_itemstore_batched_query_matches_scalar():
    be = get_backend("numpy")
    model = FHRRModel(dimension=512, backend=be, seed=0)
    items = {f"item{i}": model.random(seed=100 + i) for i in range(20)}
    cb = Codebook(items, backend=be)
    store = ItemStore(model).fit(cb)

    q = items["item7"]
    fast = store.query(q, k=5, fast=True)
    slow = store.query(q, k=5, fast=False)

    # Compare label sets (order may differ on ties, relax to sets)
    assert {l for l, _ in fast} == {l for l, _ in slow}


class TestCodebookDictInterface:
    """Tests for dict-like interface on Codebook."""

    def test_getitem(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        vec = model.random(seed=1)
        cb = Codebook({"test": vec}, backend=be)

        # __getitem__ should return the vector
        result = cb["test"]
        assert np.allclose(be.to_numpy(result), be.to_numpy(vec))

    def test_getitem_keyerror(self):
        be = get_backend("numpy")
        cb = Codebook({}, backend=be)

        try:
            _ = cb["nonexistent"]
            assert False, "Should raise KeyError"
        except KeyError:
            pass

    def test_contains(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        cb = Codebook({"a": model.random(seed=1)}, backend=be)

        assert "a" in cb
        assert "b" not in cb

    def test_len(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        cb = Codebook({}, backend=be)
        assert len(cb) == 0

        cb.add("x", model.random(seed=1))
        assert len(cb) == 1

        cb.add("y", model.random(seed=2))
        assert len(cb) == 2

    def test_iter(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        cb = Codebook(
            {
                "a": model.random(seed=1),
                "b": model.random(seed=2),
                "c": model.random(seed=3),
            },
            backend=be,
        )

        labels = list(cb)
        assert labels == ["a", "b", "c"]

    def test_items(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        vec_a = model.random(seed=1)
        vec_b = model.random(seed=2)
        cb = Codebook({"a": vec_a, "b": vec_b}, backend=be)

        items = list(cb.items())
        assert len(items) == 2
        assert items[0][0] == "a"
        assert items[1][0] == "b"

    def test_keys(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        cb = Codebook(
            {
                "x": model.random(seed=1),
                "y": model.random(seed=2),
            },
            backend=be,
        )

        assert list(cb.keys()) == ["x", "y"]

    def test_values(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        cb = Codebook(
            {
                "x": model.random(seed=1),
                "y": model.random(seed=2),
            },
            backend=be,
        )

        values = list(cb.values())
        assert len(values) == 2

    def test_get_existing(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        vec = model.random(seed=1)
        cb = Codebook({"test": vec}, backend=be)

        result = cb.get("test")
        assert np.allclose(be.to_numpy(result), be.to_numpy(vec))

    def test_get_missing_default(self):
        be = get_backend("numpy")
        cb = Codebook({}, backend=be)

        assert cb.get("missing") is None
        assert cb.get("missing", "default") == "default"
