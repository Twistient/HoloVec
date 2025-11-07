"""
Retrieval demo using ItemStore + Codebook
==========================================

Run (optional):
  python -m examples.retrieval_demo
"""

from holovec.backends import get_backend
from holovec.models.fhrr import FHRRModel
from holovec.retrieval import Codebook, ItemStore
from holovec.utils.cleanup import ResonatorCleanup


def main():
    backend = get_backend('numpy')
    model = FHRRModel(dimension=512, backend=backend, seed=0)

    # build a small codebook
    items = {f"item{i}": model.random(seed=10 + i) for i in range(8)}
    cb = Codebook(items, backend=backend)

    # store and query
    store = ItemStore(model, cleanup=ResonatorCleanup()).fit(cb)

    q = items['item3']
    print("Top-3 nearest:")
    for lbl, sim in store.query(q, k=3):
        print(f"  {lbl}: {sim:.3f}")

    # factorize a composition of 3 items
    comp = model.bind_multiple([items['item1'], items['item5'], items['item6']])
    labels, sims = store.factorize(comp, n_factors=3, mode='soft', top_k=4, temperature=10.0)
    print("Factorized labels:", labels)
    print("Similarities:", [f"{s:.3f}" for s in sims])


if __name__ == "__main__":
    main()

