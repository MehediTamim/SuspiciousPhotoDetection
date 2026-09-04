"""Tests for pure math functions in app/pipeline_multiregion.py.

No model load, no I/O — runs in milliseconds.

    python -m pytest test_pipeline.py -v
    python test_pipeline.py          # standalone, no pytest needed
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
from pipeline_multiregion import geometric_median, modified_zscores, MAD_CUTOFF, _make_crop_fn


# --- geometric_median ---

def test_geometric_median_single_point():
    X = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(geometric_median(X), [1.0, 2.0, 3.0])


def test_geometric_median_symmetric_cluster():
    # Symmetric points — median must land at origin
    X = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert np.allclose(geometric_median(X), [0.0, 0.0], atol=1e-4)


def test_geometric_median_not_pulled_by_outlier():
    # One extreme outlier; geometric median should stay near the cluster
    cluster = np.random.default_rng(0).normal(loc=[0.0, 0.0], scale=0.01, size=(20, 2))
    X = np.vstack([cluster, [100.0, 100.0]])
    m = geometric_median(X)
    assert np.linalg.norm(m - [0.0, 0.0]) < 1.0  # outlier did not drag the median far


# --- modified_zscores ---

def test_zscores_uniform_cluster_nothing_flagged():
    dist = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    assert np.all(modified_zscores(dist) < MAD_CUTOFF)


def test_zscores_clear_outlier_flagged():
    dist = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
    z = modified_zscores(dist)
    assert z[-1] > MAD_CUTOFF           # outlier is flagged
    assert np.all(z[:-1] < MAD_CUTOFF)  # cluster members are not


def test_zscores_mad_zero_fallback_identical_distances():
    # All distances equal → MAD == 0; fallback must return zeros (nothing flagged)
    dist = np.full(8, 0.5)
    assert np.all(modified_zscores(dist) == 0.0)


def test_zscores_mad_zero_fallback_std_nonzero():
    # MAD == 0 but std != 0 (two distinct values, each repeated) → scores differ
    dist = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    z = modified_zscores(dist)
    assert not np.all(z == 0.0)


# --- _make_crop_fn ---

def test_region_crop_adapts_to_image_height():
    """Crop rows must derive from the actual image height, not hard-coded constants."""
    from PIL import Image as PILImage

    crop_top = _make_crop_fn(0.0, 0.5)
    crop_bot = _make_crop_fn(0.5, 1.0)

    for w, h in [(960, 480), (960, 720), (1280, 960), (960, 1280)]:
        img = PILImage.new("RGB", (w, h))
        assert crop_top(img).size == (w, h // 2), f"top half wrong for h={h}"
        assert crop_bot(img).size == (w, h - h // 2), f"bottom half wrong for h={h}"


# --- standalone runner ---

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}  — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}  — {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
