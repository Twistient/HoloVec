import numpy as np

from holovec.backends import get_backend
from holovec.models.fhrr import FHRRModel
from holovec.utils.decode import decode_nearest, decode_threshold, decode_multilabel
from holovec.retrieval import Codebook, ItemStore


def test_decode_helpers_nearest_and_threshold():
    be = get_backend('numpy')
    model = FHRRModel(dimension=256, backend=be, seed=0)
    items = {f"item{i}": model.random(seed=10 + i) for i in range(6)}

    # nearest should return exact match for identical vec
    lbls = decode_nearest(items['item3'], items, model, k=1)
    assert lbls[0][0] == 'item3'

    # threshold with high bar keeps only exact
    hits = decode_threshold(items['item4'], items, model, threshold=0.99)
    assert hits and hits[0][0] == 'item4'

    # multilabel: top-k returns k items
    res = decode_multilabel(items['item5'], items, model, method='topk', k=3)
    assert len(res) == 3


def test_itemstore_batched_query_matches_scalar():
    be = get_backend('numpy')
    model = FHRRModel(dimension=512, backend=be, seed=0)
    items = {f"item{i}": model.random(seed=100 + i) for i in range(20)}
    cb = Codebook(items, backend=be)
    store = ItemStore(model).fit(cb)

    q = items['item7']
    fast = store.query(q, k=5, fast=True)
    slow = store.query(q, k=5, fast=False)

    # Compare label sets (order may differ on ties, relax to sets)
    assert {l for l, _ in fast} == {l for l, _ in slow}

