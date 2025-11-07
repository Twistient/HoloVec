"""
HRR correlation vs convolution demo
====================================

Shows how circular correlation approximates unbinding for HRR.
"""

import numpy as np

from holovec.backends import get_backend
from holovec.models.hrr import HRRModel


def main():
    be = get_backend('numpy')
    model = HRRModel(dimension=512, backend=be, seed=0)
    a = model.random(seed=1)
    b = model.random(seed=2)
    c = model.bind(a, b)  # circular convolution

    # Unbinding via circular correlation
    b_hat = model.unbind(c, a)
    sim = model.similarity(b, b_hat)
    print(f"corr-unbind similarity: {sim:.3f}")


if __name__ == "__main__":
    main()

