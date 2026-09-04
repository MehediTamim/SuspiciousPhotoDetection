import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from image_utils import detect_image_size
from pipeline_multiregion import (
    MAD_CUTOFF,
    analyze_outlet,
    geometric_median,
    modified_zscores,
    suspicion_from_modz,
)


def test_geometric_median_resists_outlier():
    rng = np.random.default_rng(0)
    cluster = rng.normal(0.0, 0.01, size=(9, 8))
    outlier = np.full((1, 8), 5.0)
    X = np.vstack([cluster, outlier])
    center = geometric_median(X)
    # median stays near the cluster; the mean would be dragged toward the outlier
    assert np.linalg.norm(center - cluster.mean(axis=0)) < 0.1
    assert np.linalg.norm(center - X.mean(axis=0)) > 0.1


def test_modified_zscores_flags_outlier():
    dist = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 4.0])
    z = modified_zscores(dist)
    assert z[-1] > MAD_CUTOFF
    assert all(z[:-1] < MAD_CUTOFF)


def test_modified_zscores_mad_zero_fallback():
    z = modified_zscores(np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.allclose(z, 0.0)


def test_suspicion_is_half_at_cutoff():
    assert suspicion_from_modz(np.array([MAD_CUTOFF]))[0] == pytest.approx(0.5)


def test_analyze_outlet_flags_planted_fake():
    # The pipeline detects angular difference (L2 normalization removes magnitude),
    # so the fake must point in a different direction, not just be larger.
    rng = np.random.default_rng(1)
    base = np.zeros(16)
    base[0] = 1.0
    real = base + rng.normal(0.0, 0.01, size=(9, 16))
    fake_dir = np.zeros(16)
    fake_dir[1] = 1.0
    fake = fake_dir + rng.normal(0.0, 0.01, size=(1, 16))
    emb = np.vstack([real, fake])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    names = [f"image_{i:04d}.jpg" for i in range(10)]

    record, rows = analyze_outlet("outlet_test", names, emb)

    assert record["total_images"] == 10
    assert len(record["ranking"]) == 10
    flagged_names = [f["file_name"] for f in record["flagged_images"]]
    assert flagged_names == ["image_0009.jpg"]
    for f in record["flagged_images"]:
        assert 0.0 <= f["suspicion_score"] <= 1.0
        assert set(f) == {"file_name", "suspicion_score", "reason"}
    assert len(rows) == 10


def test_analyze_outlet_clean_folder_returns_empty():
    rng = np.random.default_rng(2)
    emb = rng.normal(0.0, 0.01, size=(8, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    names = [f"image_{i:04d}.jpg" for i in range(8)]
    record, _ = analyze_outlet("outlet_clean", names, emb)
    assert record["flagged_images"] == []


def test_detect_image_size(tmp_path):
    outlet = tmp_path / "outlet_0001"
    outlet.mkdir()
    Image.new("RGB", (960, 1280)).save(outlet / "image_0001.jpg")
    Image.new("RGB", (960, 1280)).save(outlet / "image_0002.jpg")
    assert detect_image_size(tmp_path) == (960, 1280)


def test_detect_image_size_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_image_size(tmp_path)
