import numpy as np

from holovec.backends import get_backend
from holovec.encoders.periodic import (
    PeriodicAngleEncoder,
    encode_day_of_week,
    encode_time_of_day,
)
from holovec.models.fhrr import FHRRModel


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


def test_periodic_helper_functions_return_vectors():
    backend = get_backend("numpy")
    model = FHRRModel(dimension=512, backend=backend, seed=0)

    day_vec = encode_day_of_week(model, day_index=3, harmonics=3, seed=1)
    time_vec = encode_time_of_day(model, hour=13.5, harmonics=3, seed=2)

    assert day_vec.shape == (model.dimension,)
    assert time_vec.shape == (model.dimension,)
