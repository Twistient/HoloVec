import numpy as np
import pytest

from holovec import VSA
from holovec.backends import get_backend
from holovec.models.fhrr import FHRRModel
from holovec.retrieval import AssocStore, Codebook, ItemStore
from holovec.retrieval import assocstore as assocstore_module
from holovec.retrieval import itemstore as itemstore_module
from holovec.retrieval.rust_search import rust_search_library_path
from holovec.utils.decode import decode_multilabel, decode_nearest, decode_threshold
from holovec.utils.search import (
    nearest_neighbors,
    prepare_search_index,
    prepared_similarity_scores,
    query_prepared_index,
)

RUST_LIBRARY_AVAILABLE = rust_search_library_path().exists()


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
    assert {label for label, _ in fast} == {label for label, _ in slow}


@pytest.mark.parametrize(
    ("model_name", "dim", "kwargs"),
    [
        ("MAP", 512, {}),
        ("BSC", 512, {}),
        ("BSDC", 5000, {"sparsity": 0.02}),
        ("BSDC-SEG", 240, {"segments": 12}),
        ("FHRR", 512, {}),
        ("HRR", 512, {}),
        ("GHRR", 12, {"matrix_size": 2}),
    ],
)
def test_prepared_search_scores_match_scalar_similarity(
    model_name: str,
    dim: int,
    kwargs: dict[str, object],
) -> None:
    model = VSA.create(model_name, dim=dim, backend="numpy", seed=0, **kwargs)
    items = {f"item{i}": model.random(seed=100 + i) for i in range(16)}
    prepared = prepare_search_index(items, model)
    query = model.bundle([items["item1"], items["item5"], items["item9"]])

    prepared_scores = prepared_similarity_scores(query, prepared, model)
    scalar_scores = np.array(
        [float(model.similarity(query, items[label])) for label in prepared.labels],
        dtype=np.float64,
    )

    assert np.allclose(prepared_scores, scalar_scores, atol=1e-6)


@pytest.mark.parametrize(
    ("model_name", "dim", "kwargs"),
    [
        ("MAP", 512, {}),
        ("BSC", 512, {}),
        ("BSDC", 5000, {"sparsity": 0.02}),
        ("BSDC-SEG", 240, {"segments": 12}),
        ("FHRR", 512, {}),
        ("HRR", 512, {}),
        ("GHRR", 12, {"matrix_size": 2}),
    ],
)
def test_prepared_query_topk_matches_scalar_ordering(
    model_name: str,
    dim: int,
    kwargs: dict[str, object],
) -> None:
    model = VSA.create(model_name, dim=dim, backend="numpy", seed=0, **kwargs)
    items = {f"item{i}": model.random(seed=100 + i) for i in range(16)}
    prepared = prepare_search_index(items, model)
    query = model.bundle([items["item2"], items["item7"], items["item11"]])

    fast_labels, fast_sims = query_prepared_index(
        query, prepared, model, k=5, return_similarities=True
    )
    slow_labels, slow_sims = nearest_neighbors(query, items, model, k=5, return_similarities=True)

    assert fast_labels == slow_labels
    assert np.allclose(np.asarray(fast_sims), np.asarray(slow_sims), atol=1e-6)


def test_itemstore_rebuilds_prepared_index_when_shared_codebook_changes() -> None:
    model = VSA.create("MAP", dim=512, backend="numpy", seed=0)
    codebook = Codebook({"a": model.random(seed=1)}, backend=model.backend)
    store = ItemStore(model).fit(codebook)
    query = model.random(seed=999)

    store.query(query, k=1, fast=True)
    first_index_id = id(store._prepared_index)

    codebook.add("query", query)
    updated = store.query(query, k=1, fast=True)

    assert updated[0][0] == "query"
    assert id(store._prepared_index) != first_index_id


def test_itemstore_rebuilds_rust_index_when_shared_codebook_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VSA.create("MAP", dim=256, backend="numpy", seed=0)
    codebook = Codebook({"a": model.random(seed=1)}, backend=model.backend)
    store = ItemStore(model, search_backend="rust").fit(codebook)
    query = model.random(seed=999)
    build_labels: list[list[str]] = []

    class FakeRustIndex:
        def __init__(self, labels: list[str]) -> None:
            self.labels = labels

    def fake_prepare_rust_search(prepared_index, _model):
        build_labels.append(prepared_index.labels.copy())
        return FakeRustIndex(prepared_index.labels.copy())

    def fake_query_rust_search(query_vec, index, *, k, return_similarities):
        del query_vec, k
        labels = [index.labels[-1]]
        scores = [1.0] if return_similarities else None
        return labels, scores

    monkeypatch.setattr(
        itemstore_module,
        "prepare_rust_search_from_index",
        fake_prepare_rust_search,
    )
    monkeypatch.setattr(
        itemstore_module,
        "query_rust_prepared_index",
        fake_query_rust_search,
    )

    store.query(query, k=1, fast=True)
    first_rust_index_id = id(store._rust_index)

    codebook.add("query", query)
    updated = store.query(query, k=1, fast=True)

    assert updated[0][0] == "query"
    assert id(store._rust_index) != first_rust_index_id
    assert build_labels == [["a"], ["a", "query"]]


def test_itemstore_rust_backend_falls_back_to_prepared_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VSA.create("MAP", dim=256, backend="numpy", seed=0)
    items = {f"item{i}": model.random(seed=i) for i in range(8)}
    store = ItemStore(model, search_backend="rust").fit(items)

    def fail_prepare_rust_search(*_args, **_kwargs):
        raise FileNotFoundError("missing test rust library")

    monkeypatch.setattr(
        itemstore_module,
        "prepare_rust_search_from_index",
        fail_prepare_rust_search,
    )

    results = store.query(items["item3"], k=3, fast=True)

    assert results[0][0] == "item3"
    assert store._prepared_index is not None
    assert store._rust_index is None


@pytest.mark.parametrize(
    ("model_name", "dim", "kwargs"),
    [
        ("MAP", 512, {}),
        ("BSC", 512, {}),
        ("BSDC", 5000, {"sparsity": 0.02}),
        ("BSDC-SEG", 240, {"segments": 12}),
    ],
)
def test_itemstore_rust_backend_matches_numpy_on_supported_modes(
    model_name: str,
    dim: int,
    kwargs: dict[str, object],
) -> None:
    if not RUST_LIBRARY_AVAILABLE:
        pytest.skip("Rust retrieval library is not built")

    model = VSA.create(model_name, dim=dim, backend="numpy", seed=0, **kwargs)
    items = {f"item{i}": model.random(seed=100 + i) for i in range(32)}
    query = model.bundle([items["item2"], items["item7"], items["item11"]])

    numpy_store = ItemStore(model, search_backend="numpy").fit(items)
    rust_store = ItemStore(model, search_backend="rust").fit(items)

    rust_results = rust_store.query(query, k=5, fast=True)
    numpy_results = numpy_store.query(query, k=5, fast=True)

    assert [label for label, _ in rust_results] == [label for label, _ in numpy_results]
    assert np.allclose(
        [score for _, score in rust_results],
        [score for _, score in numpy_results],
        atol=1e-6,
    )


class TestAssocStore:
    """Tests for AssocStore heteroassociative memory."""

    def test_query_label_with_map_model(self):
        """Test exact key lookup with a discrete-space model."""
        model = VSA.create("MAP", dim=512, backend="numpy", seed=0)
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

        result = store.query_label(keys["b"], k=1)
        assert result[0][0] == "b"
        assert result[0][1] > 0.99

    def test_basic_fit_and_query(self):
        """Test basic fit and query operations."""
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
        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        store = AssocStore(model)
        query_vec = model.random(seed=1)

        with pytest.raises(ValueError):
            store.query_value(query_vec)

    def test_save_and_load(self, tmp_path):
        """Test saving and loading an AssocStore."""
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

    def test_invalid_search_backend_raises(self):
        model = VSA.create("MAP", dim=256, backend="numpy", seed=0)

        with pytest.raises(ValueError, match="search_backend"):
            AssocStore(model, search_backend="bogus")

    def test_rust_backend_falls_back_to_prepared_numpy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = VSA.create("MAP", dim=256, backend="numpy", seed=0)
        keys = {"x": model.random(seed=1), "y": model.random(seed=2)}
        values = {"x": model.random(seed=10), "y": model.random(seed=20)}
        store = AssocStore(model, search_backend="rust").fit(keys, values)

        def fail_prepare_rust_search(*_args, **_kwargs):
            raise FileNotFoundError("missing test rust library")

        monkeypatch.setattr(
            assocstore_module,
            "prepare_rust_search_from_index",
            fail_prepare_rust_search,
        )

        result = store.query_label(keys["y"], k=1)

        assert result[0][0] == "y"
        assert store._prepared_key_index is not None
        assert store._rust_key_index is None

    def test_rust_backend_rebuilds_key_index_when_codebook_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = VSA.create("MAP", dim=256, backend="numpy", seed=0)
        store = AssocStore(model, search_backend="rust")
        store.add("a", model.random(seed=1), model.random(seed=10))
        build_labels: list[list[str]] = []

        class FakeRustIndex:
            def __init__(self, labels: list[str]) -> None:
                self.labels = labels

        def fake_prepare_rust_search(prepared_index, _model):
            build_labels.append(prepared_index.labels.copy())
            return FakeRustIndex(prepared_index.labels.copy())

        def fake_query_rust_search(query_vec, index, *, k, return_similarities):
            del query_vec, k
            labels = [index.labels[-1]]
            scores = [1.0] if return_similarities else None
            return labels, scores

        monkeypatch.setattr(
            assocstore_module,
            "prepare_rust_search_from_index",
            fake_prepare_rust_search,
        )
        monkeypatch.setattr(
            assocstore_module,
            "query_rust_prepared_index",
            fake_query_rust_search,
        )

        store.query_label(model.random(seed=99), k=1)
        first_rust_index_id = id(store._rust_key_index)

        query = model.random(seed=123)
        store.add("query", query, model.random(seed=456))
        updated = store.query_label(query, k=1)

        assert updated[0][0] == "query"
        assert id(store._rust_key_index) != first_rust_index_id
        assert build_labels == [["a"], ["a", "query"]]

    def test_rust_backend_matches_numpy_on_supported_modes(self) -> None:
        if not RUST_LIBRARY_AVAILABLE:
            pytest.skip("Rust retrieval library is not built")

        model = VSA.create("MAP", dim=512, backend="numpy", seed=0)
        keys = {f"item{i}": model.random(seed=100 + i) for i in range(16)}
        values = {label: model.random(seed=1_000 + i) for i, label in enumerate(keys)}
        query = model.bundle([keys["item2"], keys["item7"], keys["item11"]])

        numpy_store = AssocStore(model, search_backend="numpy").fit(keys, values)
        rust_store = AssocStore(model, search_backend="rust").fit(keys, values)

        rust_results = rust_store.query_label(query, k=5)
        numpy_results = numpy_store.query_label(query, k=5)

        assert [label for label, _ in rust_results] == [label for label, _ in numpy_results]
        assert np.allclose(
            [score for _, score in rust_results],
            [score for _, score in numpy_results],
            atol=1e-6,
        )


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

        with pytest.raises(KeyError):
            _ = cb["nonexistent"]

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

    def test_as_list(self):
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        vec_a = model.random(seed=1)
        vec_b = model.random(seed=2)
        cb = Codebook({"a": vec_a, "b": vec_b}, backend=be)

        result = cb.as_list()
        assert len(result) == 2
        assert result[0][0] == "a"
        assert result[1][0] == "b"
        assert np.allclose(be.to_numpy(result[0][1]), be.to_numpy(vec_a))

    def test_as_matrix_empty(self):
        """Test as_matrix with empty codebook."""
        be = get_backend("numpy")
        cb = Codebook({}, backend=be)

        labels, matrix = cb.as_matrix(be)
        assert labels == []
        assert matrix.shape == (0,)

    def test_save_and_load_roundtrip_uses_safe_format(self, tmp_path):
        """Test Codebook persistence round-trip with versioned safe format."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        cb = Codebook({"a": model.random(seed=1), "b": model.random(seed=2)}, backend=be)

        path = tmp_path / "codebook.npz"
        cb.save(str(path))

        with np.load(path, allow_pickle=False) as data:
            assert int(np.asarray(data["format_version"]).item()) == Codebook.FORMAT_VERSION
            assert data["labels"].dtype.kind in {"U", "S"}

        loaded = Codebook.load(str(path), backend=be)
        assert list(loaded.keys()) == ["a", "b"]

    def test_load_legacy_codebook_requires_explicit_unsafe_flag(self, tmp_path):
        """Test legacy pickle-backed codebooks fail closed by default."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        matrix = np.stack(
            [be.to_numpy(model.random(seed=1)), be.to_numpy(model.random(seed=2))],
            axis=0,
        )
        path = tmp_path / "legacy_codebook.npz"
        np.savez(path, labels=np.array(["a", "b"], dtype=object), matrix=matrix)

        with pytest.raises(ValueError, match="allow_unsafe_legacy=True"):
            Codebook.load(str(path), backend=be)

        loaded = Codebook.load(str(path), backend=be, allow_unsafe_legacy=True)
        assert list(loaded.keys()) == ["a", "b"]


class TestItemStoreExtended:
    """Additional tests for ItemStore coverage."""

    def test_fit_with_dict(self):
        """Test fit() with a dict instead of Codebook."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        items = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
        }

        store = ItemStore(model)
        result = store.fit(items)

        # Should return self for chaining
        assert result is store
        assert store.codebook.size == 2
        assert "a" in store.codebook
        assert "b" in store.codebook

    def test_invalid_search_backend_raises(self):
        model = VSA.create("MAP", dim=256, backend="numpy", seed=0)

        with pytest.raises(ValueError, match="search_backend"):
            ItemStore(model, search_backend="bogus")

    def test_add_method(self):
        """Test add() method on ItemStore."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        store = ItemStore(model)
        store.add("item1", model.random(seed=1))
        store.add("item2", model.random(seed=2))

        assert store.codebook.size == 2
        assert "item1" in store.codebook
        assert "item2" in store.codebook

    def test_extend_method(self):
        """Test extend() method on ItemStore."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        store = ItemStore(model)
        store.extend(
            {
                "a": model.random(seed=1),
                "b": model.random(seed=2),
                "c": model.random(seed=3),
            }
        )

        assert store.codebook.size == 3

    def test_query_return_similarities_false(self):
        """Test query with return_similarities=False."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        items = {f"item{i}": model.random(seed=i) for i in range(5)}
        store = ItemStore(model).fit(items)

        # Query without similarities
        results = store.query(items["item2"], k=3, return_similarities=False)

        assert len(results) == 3
        # All similarities should be 0.0 when not returned
        for label, sim in results:
            assert sim == 0.0
            assert isinstance(label, str)

    def test_query_with_map_model(self):
        """Test query with non-complex space (MAP uses real vectors)."""
        from holovec import VSA

        model = VSA.create("MAP", dim=256, seed=42)

        items = {f"item{i}": model.random(seed=i) for i in range(10)}
        store = ItemStore(model).fit(items)

        # Query should use cosine similarity path
        results = store.query(items["item5"], k=3, fast=True)

        assert len(results) == 3
        assert results[0][0] == "item5"  # Exact match should be first
        assert results[0][1] > 0.99  # High similarity

    def test_query_partial_sort(self):
        """Test query with k < codebook size triggers partial sort."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        # Large codebook
        items = {f"item{i}": model.random(seed=i) for i in range(100)}
        store = ItemStore(model).fit(items)

        # Query with small k should trigger partial sort
        results = store.query(items["item50"], k=5, fast=True)

        assert len(results) == 5
        assert results[0][0] == "item50"

    def test_factorize(self):
        """Test factorize method on ItemStore."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=1000, backend=be, seed=0)

        items = {
            "a": model.random(seed=1),
            "b": model.random(seed=2),
            "c": model.random(seed=3),
        }
        store = ItemStore(model).fit(items)

        # Create a composition to factorize
        composition = model.bind(items["a"], items["b"])

        labels, similarities = store.factorize(composition, n_factors=2)

        assert len(labels) == 2
        assert len(similarities) == 2
        assert all(isinstance(s, float) for s in similarities)

    def test_save_and_load(self, tmp_path):
        """Test save and load round-trip for ItemStore."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        items = {
            "alpha": model.random(seed=1),
            "beta": model.random(seed=2),
            "gamma": model.random(seed=3),
        }
        store = ItemStore(model).fit(items)

        # Save
        save_path = str(tmp_path / "itemstore.npz")
        store.save(save_path)

        # Load into new store
        loaded = ItemStore.load(model, save_path)

        # Verify
        assert loaded.codebook.size == 3
        assert "alpha" in loaded.codebook
        assert "beta" in loaded.codebook
        assert "gamma" in loaded.codebook

        # Vectors should match
        for label in ["alpha", "beta", "gamma"]:
            orig = be.to_numpy(store.codebook[label])
            load = be.to_numpy(loaded.codebook[label])
            assert np.allclose(orig, load)

    def test_load_with_custom_cleanup(self, tmp_path):
        """Test load with custom cleanup strategy."""
        from holovec.utils.cleanup import ResonatorCleanup

        be = get_backend("numpy")
        model = FHRRModel(dimension=256, backend=be, seed=0)

        items = {"a": model.random(seed=1)}
        store = ItemStore(model).fit(items)

        save_path = str(tmp_path / "itemstore.npz")
        store.save(save_path)

        # Load with custom cleanup
        loaded = ItemStore.load(model, save_path, cleanup=ResonatorCleanup())

        assert isinstance(loaded.cleanup, ResonatorCleanup)
        assert loaded.codebook.size == 1

    def test_load_legacy_codebook_requires_explicit_unsafe_flag(self, tmp_path):
        """Test ItemStore exposes the legacy codebook migration flag."""
        be = get_backend("numpy")
        model = FHRRModel(dimension=64, backend=be, seed=0)
        matrix = np.stack(
            [be.to_numpy(model.random(seed=1)), be.to_numpy(model.random(seed=2))],
            axis=0,
        )
        path = tmp_path / "legacy_itemstore.npz"
        np.savez(path, labels=np.array(["a", "b"], dtype=object), matrix=matrix)

        with pytest.raises(ValueError, match="allow_unsafe_legacy=True"):
            ItemStore.load(model, str(path))

        loaded = ItemStore.load(model, str(path), allow_unsafe_legacy=True)
        assert loaded.codebook.size == 2
