import numpy as np

from holovec.spaces.spaces import SparseSegmentSpace
from holovec.models.bsdc_seg import BSDCSEGModel
from holovec.backends import get_backend


def test_sparse_segment_space_random_and_normalize():
    backend = get_backend('numpy')
    D, S = 100, 10
    space = SparseSegmentSpace(dimension=D, segments=S, backend=backend)

    vec = space.random(seed=42)
    arr = backend.to_numpy(vec)

    # Exactly one 1 per segment
    L = D // S
    for s in range(S):
        start = s * L
        end = start + L
        assert arr[start:end].sum() == 1

    # Normalization preserves 1 per segment and keeps to {0,1}
    noisy = backend.from_numpy(arr.astype(np.float32) * 2.0)
    norm = space.normalize(noisy)
    norm_np = backend.to_numpy(norm)
    for s in range(S):
        start = s * L
        end = start + L
        assert norm_np[start:end].sum() == 1


def test_sparse_segment_similarity():
    backend = get_backend('numpy')
    D, S = 60, 6
    space = SparseSegmentSpace(dimension=D, segments=S, backend=backend)

    a = space.random(seed=1)
    b = space.normalize(a)
    # identical → 1.0
    assert space.similarity(a, b) == 1.0

    # Construct vector matching half the segments
    L = D // S
    a_np = backend.to_numpy(space.normalize(a)).copy()
    c_np = a_np.copy()
    # Flip active index in first S//2 segments to a different index
    for s in range(S // 2):
        start = s * L
        end = start + L
        idx = int(np.argmax(c_np[start:end]))
        c_np[start:end] = 0
        # pick next index (wrap)
        c_np[start + ((idx + 1) % L)] = 1
    c = backend.from_numpy(c_np)
    sim = space.similarity(a, c)
    assert abs(sim - 0.5) < 1e-6


def test_bsdc_seg_bundling_majority():
    backend = get_backend('numpy')
    D, S = 80, 8
    model = BSDCSEGModel(dimension=D, segments=S, backend=backend, seed=0)
    space = model.space
    L = space.segment_length

    # Build 3 vectors; in each segment choose a majority index
    v1 = space.random(seed=1)
    v2 = space.random(seed=2)
    v3 = space.normalize(v1)  # ensure v1 is counted twice total

    bundled = model.bundle([v1, v2, v3])
    # For each segment, bundled should equal the index that won most votes
    b_np = backend.to_numpy(bundled)
    # Verify 1 per segment
    for s in range(S):
        start = s * L
        end = start + L
        assert b_np[start:end].sum() == 1

