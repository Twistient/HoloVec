import numpy as np

from holovec.backends import get_backend
from holovec.models.fhrr import FHRRModel
from holovec.encoders.periodic import PeriodicAngleEncoder


def test_periodic_angle_wraparound():
    backend = get_backend('numpy')
    model = FHRRModel(dimension=512, backend=backend, seed=0)
    enc = PeriodicAngleEncoder(model, harmonics=3, radians=True, seed=1)

    a = 0.05
    b = 2 * np.pi - 0.05  # close to wrap-around
    ha = enc.encode(a)
    hb = enc.encode(b)

    sim = model.similarity(ha, hb)
    # Should be fairly high due to periodicity and proximity
    assert sim > 0.5

