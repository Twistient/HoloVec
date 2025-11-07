"""Microbenchmarks for Resonator vs Brute Force cleanup.

Run:
  python -m benchmarks.bench_resonator

Generates synthetic codebooks and composite queries, compares runtime and
convergence behavior for 3–5 factors.
"""

import time
from typing import Dict, List, Tuple

from holovec.backends import get_backend
from holovec.models.fhrr import FHRRModel
from holovec.utils.cleanup import BruteForceCleanup, ResonatorCleanup


def make_codebook(model: FHRRModel, n_items: int, seed: int = 0) -> Dict[str, object]:
    cb = {}
    for i in range(n_items):
        cb[f"X{i}"] = model.random(seed=seed + i)
    return cb


def synth_query(model: FHRRModel, codebook: Dict[str, object], labels: List[str]) -> object:
    vecs = [codebook[l] for l in labels]
    comp = model.bind_multiple(vecs)
    return comp


def bench(n_factors: int = 3, n_items: int = 256, dim: int = 2048, reps: int = 5) -> Tuple[float, float]:
    backend = get_backend('numpy')
    model = FHRRModel(dimension=dim, backend=backend, seed=123)
    codebook = make_codebook(model, n_items=n_items, seed=1000)
    brute = BruteForceCleanup()
    reso = ResonatorCleanup()

    # Choose true labels
    labels = [f"X{i}" for i in range(n_factors)]
    query = synth_query(model, codebook, labels)

    # Brute force baseline (sequential unbinding)
    t0 = time.perf_counter()
    for _ in range(reps):
        current = query
        for _k in range(n_factors):
            lbl, _ = brute.cleanup(current, codebook, model)
            current = model.unbind(current, codebook[lbl])
    t1 = time.perf_counter()
    brute_t = (t1 - t0) / reps

    # Resonator
    t0 = time.perf_counter()
    for _ in range(reps):
        reso.factorize(query, codebook, model, n_factors=n_factors, mode='soft', top_k=5, temperature=15.0)
    t1 = time.perf_counter()
    reso_t = (t1 - t0) / reps
    return brute_t, reso_t


def main():
    for nf in (3, 4, 5):
        brute_t, reso_t = bench(n_factors=nf)
        print(f"n_factors={nf}: brute={brute_t*1000:.2f} ms  resonator={reso_t*1000:.2f} ms  speedup={brute_t/reso_t:.1f}x")


if __name__ == "__main__":
    main()

