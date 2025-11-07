"""
BSDC-SEG demo: segment-sparse codes, bundling, and segment-wise search
=======================================================================

Run (optional):
  python -m examples.bsdc_seg_demo
"""

from holovec.backends import get_backend
from holovec.models.bsdc_seg import BSDCSEGModel
from holovec.utils.search import segment_pattern, find_by_segment_pattern


def main():
    backend = get_backend('numpy')
    D, S = 80, 8
    model = BSDCSEGModel(dimension=D, segments=S, backend=backend, seed=0)
    space = model.space

    # Build a small codebook
    codebook = {
        f"item{i}": space.random(seed=10 + i) for i in range(10)
    }

    # Bundle a couple of items
    bundle = model.bundle([codebook['item1'], codebook['item2'], codebook['item3']])
    print("Bundled vector pattern (first 3 segments):", segment_pattern(bundle, space)[:3])

    # Query by segment pattern (wildcards allowed via None)
    # e.g., find items where segments 0 and 1 match a target vector
    target = codebook['item1']
    pat = segment_pattern(target, space)
    query_pattern = [pat[0], pat[1]] + [None] * (S - 2)

    exact = find_by_segment_pattern(codebook, space, query_pattern, match_mode='exact')
    print("Exact matches on first 2 segments:", [lbl for lbl, _ in exact])

    frac = find_by_segment_pattern(codebook, space, query_pattern, match_mode='fraction', min_fraction=0.5)
    print("Fraction >= 0.5 on specified segments:", frac)


if __name__ == "__main__":
    main()

