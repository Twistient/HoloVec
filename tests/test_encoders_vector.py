import numpy as np

from holovec.backends import get_backend
from holovec.models.fhrr import FHRRModel
from holovec.encoders.vector import VectorFPE


def test_vector_fpe_similarity_monotonic():
    backend = get_backend('numpy')
    model = FHRRModel(dimension=512, backend=backend, seed=0)
    enc = VectorFPE(model, input_dim=3, bandwidth=0.5, phase_dist='gaussian', seed=1)

    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    z = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    hx = enc.encode(x)
    hy = enc.encode(y)
    hz = enc.encode(z)

    s_xy = model.similarity(hx, hy)
    s_xz = model.similarity(hx, hz)

    assert s_xy > s_xz  # further point should be less similar on average

