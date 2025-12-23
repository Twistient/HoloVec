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


class TestAssocStore:
    """Tests for AssocStore heteroassociative memory."""

    def test_basic_fit_and_query(self):
        """Test basic fit and query operations."""
        from holovec.retrieval import AssocStore

        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        # Create key-value pairs
        keys = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
            "c": model.random(seed=3),
        }
        values = {
            "a": model.random(seed=10),
            "b": model.random(seed=20),
            "c": model.random(seed=30),
        }

        store = AssocStore(model).fit(keys, values)

        # Query with exact key should return correct label
        result = store.query_label(keys["b"], k=1)
        assert result[0][0] == "b"
        assert result[0][1] > 0.99  # High similarity for exact match

    def test_query_value_returns_correct_vector(self):
        """Test that query_value returns the associated value vector."""
        from holovec.retrieval import AssocStore

        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        keys = {"x": model.random(seed=1)}
        values = {"x": model.random(seed=100)}

        store = AssocStore(model).fit(keys, values)

        label, value_vec = store.query_value(keys["x"])
        assert label == "x"
        # Value should match what we stored
        assert np.allclose(be.to_numpy(value_vec), be.to_numpy(values["x"]))

    def test_add_method(self):
        """Test adding individual key-value pairs."""
        from holovec.retrieval import AssocStore

        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        store = AssocStore(model)
        store.add("item1", model.random(seed=1), model.random(seed=10))
        store.add("item2", model.random(seed=2), model.random(seed=20))

        assert len(store.keys) == 2
        assert len(store.values) == 2
        assert "item1" in store.keys
        assert "item2" in store.keys

    def test_fit_with_partial_overlap(self):
        """Test fit when keys and values have partial label overlap."""
        from holovec.retrieval import AssocStore

        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        # Keys has 'a', 'b', 'c'; values has 'b', 'c', 'd'
        # Only 'b' and 'c' should be in the store
        keys = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
            "c": model.random(seed=3),
        }
        values = {
            "b": model.random(seed=20),
            "c": model.random(seed=30),
            "d": model.random(seed=40),
        }

        store = AssocStore(model).fit(keys, values)

        assert len(store.keys) == 2
        assert "b" in store.keys
        assert "c" in store.keys
        assert "a" not in store.keys
        assert "d" not in store.keys

    def test_query_empty_store_raises(self):
        """Test that querying empty store raises ValueError."""
        from holovec.retrieval import AssocStore
        import pytest

        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        store = AssocStore(model)
        query_vec = model.random(seed=1)

        with pytest.raises(ValueError):
            store.query_value(query_vec)

    def test_save_and_load(self, tmp_path):
        """Test saving and loading an AssocStore."""
        from holovec.retrieval import AssocStore

        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        keys = {"a": model.random(seed=1), "b": model.random(seed=2)}
        values = {"a": model.random(seed=10), "b": model.random(seed=20)}

        store = AssocStore(model).fit(keys, values)

        keys_path = str(tmp_path / "keys.npz")
        values_path = str(tmp_path / "values.npz")
        store.save(keys_path, values_path)

        # Load into new store
        loaded = AssocStore.load(model, keys_path, values_path)

        assert len(loaded.keys) == 2
        assert "a" in loaded.keys
        assert "b" in loaded.keys

        # Verify vectors match
        for label in ["a", "b"]:
            orig_key = be.to_numpy(store.keys[label])
            load_key = be.to_numpy(loaded.keys[label])
            assert np.allclose(orig_key, load_key)


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
