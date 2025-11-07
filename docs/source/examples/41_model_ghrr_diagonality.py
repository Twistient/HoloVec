"""
GHRR usage example: diagonality/m sweeps and non-commutativity trends
======================================================================

Run (optional):
  python -m examples.ghrr_diagonality_sweep
"""

import numpy as np

from holovec.backends import get_backend
from holovec.models.ghrr import GHRRModel


def noncommutativity_score(model: GHRRModel, trials=10, seed=0) -> float:
    vals = []
    for i in range(trials):
        a = model.random(seed=seed + 2 * i)
        b = model.random(seed=seed + 2 * i + 1)
        ab = model.bind(a, b)
        ba = model.bind(b, a)
        s = model.similarity(ab, ba)
        vals.append(1.0 - float(s))
    return float(np.mean(vals))


def main():
    backend = get_backend('numpy')
    D = 256
    ms = [1, 2, 3]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]  # diagonality

    print("Non-commutativity (1 - sim(ab,ba)) across m and diagonality:")
    table = []
    for m in ms:
        row = []
        for a in alphas:
            model = GHRRModel(dimension=D, matrix_size=m, backend=backend, seed=0, diagonality=a)
            score = noncommutativity_score(model, trials=8, seed=10)
            row.append(score)
        table.append(row)

    header = "alpha:" + " ".join(f"{a:6.2f}" for a in alphas)
    print(header)
    for m, row in zip(ms, table):
        print(f"m={m} :" + " ".join(f"{v:6.3f}" for v in row))

    # Optional plotting
    try:
        import matplotlib.pyplot as plt

        for m, row in zip(ms, table):
            plt.plot(alphas, row, marker='o', label=f"m={m}")
        plt.xlabel("Diagonality")
        plt.ylabel("Non-commutativity (1 - sim)")
        plt.title("GHRR: Non-commutativity vs Diagonality")
        plt.legend()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

