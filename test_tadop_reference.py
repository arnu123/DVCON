"""
Tests for the TADOP two-stage scorer.

Most tests are pure-math: they verify score_detections() and
extract_roi_features() with synthetic inputs, no model, no images. This
keeps them fast and runnable anywhere.

End-to-end tests check that running tadop_reference.py on the existing
baseline images produces well-formed output files.

Run:
    python -m pytest test_tadop_reference.py -v
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from tadop_reference import (
    extract_roi_features,
    score_detections,
    make_W,
    load_priors,
    MASK_THRESHOLD,
    NUM_CLASSES, NUM_TASKS, EMBED_DIM, STRIDE8_CHANNELS, STRIDE8_GRID,
    TASK_NAMES,
)


# ---------- Fixtures ----------

@pytest.fixture
def synthetic_lut():
    """80x14 LUT: most entries 0, with planted high-value entries we can target."""
    lut = np.zeros((NUM_CLASSES, NUM_TASKS), dtype=np.float32)
    # Plant a strong wine_glass (class 40) prior for serve_wine (task idx 9)
    lut[40, 9] = 0.91
    lut[41, 9] = 0.69  # cup
    # And a noise floor item below threshold
    lut[0, 9] = 0.05   # person, below MASK_THRESHOLD
    return lut


@pytest.fixture
def synthetic_task_embs():
    rng = np.random.default_rng(0)
    e = rng.standard_normal((NUM_TASKS, EMBED_DIM)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


@pytest.fixture
def synthetic_feature_map():
    """64-channel 80x80 feature map with planted signal."""
    rng = np.random.default_rng(42)
    fm = rng.standard_normal((STRIDE8_CHANNELS, STRIDE8_GRID, STRIDE8_GRID)).astype(np.float32)
    return fm


# ---------- W projection ----------

def test_W_shape_and_determinism():
    W1 = make_W(seed=1729)
    W2 = make_W(seed=1729)
    assert W1.shape == (STRIDE8_CHANNELS, EMBED_DIM)
    assert W1.dtype == np.float32
    np.testing.assert_array_equal(W1, W2)   # same seed -> same W

def test_W_different_seeds_differ():
    assert not np.array_equal(make_W(seed=1), make_W(seed=2))


# ---------- RoI extraction ----------

def test_roi_empty_input_returns_empty(synthetic_feature_map):
    rois = extract_roi_features(synthetic_feature_map, np.zeros((0, 4)), 480, 640)
    assert rois.shape == (0, STRIDE8_CHANNELS)


def test_roi_full_image_box_recovers_global_mean(synthetic_feature_map):
    """A box covering the whole image should recover the GAP of the whole feature."""
    fm = synthetic_feature_map
    boxes = np.array([[0, 0, 640, 480]], dtype=np.float32)
    rois = extract_roi_features(fm, boxes, 480, 640)
    expected = fm.mean(axis=(1, 2))   # (64,)
    # Allow a tiny rounding/letterbox padding discrepancy
    assert rois.shape == (1, STRIDE8_CHANNELS)
    np.testing.assert_allclose(rois[0], expected, atol=0.05)


def test_roi_small_box_uses_local_features(synthetic_feature_map):
    """Two different boxes should give two different RoI vectors."""
    fm = synthetic_feature_map
    box_a = np.array([[0, 0, 100, 100]], dtype=np.float32)
    box_b = np.array([[400, 300, 500, 400]], dtype=np.float32)
    roi_a = extract_roi_features(fm, box_a, 480, 640)
    roi_b = extract_roi_features(fm, box_b, 480, 640)
    assert not np.allclose(roi_a, roi_b)


def test_roi_handles_degenerate_box(synthetic_feature_map):
    """Zero-area box should not crash and should return a finite vector."""
    fm = synthetic_feature_map
    box = np.array([[100, 100, 100, 100]], dtype=np.float32)
    roi = extract_roi_features(fm, box, 480, 640)
    assert roi.shape == (1, STRIDE8_CHANNELS)
    assert np.isfinite(roi).all()


# ---------- Scorer logic ----------

def test_masked_class_score_is_zero(synthetic_lut, synthetic_task_embs):
    dets = [{"class_id": 0, "confidence": 0.9,    # person, below threshold
             "class_name": "person", "bbox_xyxy": [0, 0, 10, 10]}]
    rois = np.ones((1, STRIDE8_CHANNELS), dtype=np.float32)
    W = make_W()
    out = score_detections(dets, rois, task_id=9,
                           lut=synthetic_lut, task_embs=synthetic_task_embs, W=W)
    assert out[0]["masked_out"]
    assert out[0]["tadop_score"] == 0.0


def test_unmasked_class_gets_real_score(synthetic_lut, synthetic_task_embs):
    dets = [{"class_id": 40, "confidence": 0.9,   # wine_glass, high LUT
             "class_name": "wine glass", "bbox_xyxy": [0, 0, 10, 10]}]
    rois = np.ones((1, STRIDE8_CHANNELS), dtype=np.float32)
    W = make_W()
    out = score_detections(dets, rois, task_id=9,
                           lut=synthetic_lut, task_embs=synthetic_task_embs, W=W)
    assert not out[0]["masked_out"]
    # score = conf * LUT * max(0, cos) = 0.9 * 0.91 * cos. cos in [-1,1].
    # All factors are non-negative when cos > 0, score should be > 0.
    if out[0]["cos_sim"] > 0:
        assert out[0]["tadop_score"] > 0
        assert out[0]["tadop_score"] <= 0.9 * 0.91 * 1.0  # upper bound


def test_score_combines_all_three_factors(synthetic_lut, synthetic_task_embs):
    """ Sanity: score = conf * LUT * max(0, cos_sim). """
    dets = [{"class_id": 40, "confidence": 0.5,
             "class_name": "wine glass", "bbox_xyxy": [0, 0, 10, 10]}]
    rois = np.ones((1, STRIDE8_CHANNELS), dtype=np.float32)
    W = make_W()
    out = score_detections(dets, rois, task_id=9,
                           lut=synthetic_lut, task_embs=synthetic_task_embs, W=W)
    expected = 0.5 * 0.91 * max(0, out[0]["cos_sim"])
    assert abs(out[0]["tadop_score"] - expected) < 1e-6


def test_ranking_is_correct(synthetic_lut, synthetic_task_embs):
    """ High-LUT class should rank above low-LUT class with same conf+roi. """
    dets = [
        {"class_id": 41, "confidence": 0.9, "class_name": "cup",
         "bbox_xyxy": [0, 0, 10, 10]},        # LUT=0.69
        {"class_id": 40, "confidence": 0.9, "class_name": "wine glass",
         "bbox_xyxy": [0, 0, 10, 10]},        # LUT=0.91
    ]
    rois = np.ones((2, STRIDE8_CHANNELS), dtype=np.float32)
    W = make_W()
    out = score_detections(dets, rois, task_id=9,
                           lut=synthetic_lut, task_embs=synthetic_task_embs, W=W)
    # If both have the same cos_sim (same RoI), wine_glass should rank first
    assert out[0]["class_id"] == 40


def test_empty_detections(synthetic_lut, synthetic_task_embs):
    out = score_detections([], np.zeros((0, STRIDE8_CHANNELS)), task_id=9,
                           lut=synthetic_lut, task_embs=synthetic_task_embs, W=make_W())
    assert out == []


def test_negative_cosine_floored_to_zero(synthetic_lut, synthetic_task_embs):
    """ ReLU on cosine: if proj points away from task embedding, score=0."""
    # Construct a RoI whose projection lies opposite the task embedding
    W = make_W()                                    # (64, 384)
    task_emb = synthetic_task_embs[9]               # (384,)
    # We want roi (64,) such that (roi @ W) is anti-parallel to task_emb.
    # Solve: roi @ W = -task_emb   =>   roi = -task_emb @ pinv(W) = -task_emb @ W^+ ,
    # where W^+ has shape (384, 64).
    W_pinv = np.linalg.pinv(W)                      # (384, 64)
    roi = -task_emb @ W_pinv                        # (64,)
    rois = roi[np.newaxis, :].astype(np.float32)
    dets = [{"class_id": 40, "confidence": 0.9,
             "class_name": "wine glass", "bbox_xyxy": [0, 0, 10, 10]}]
    out = score_detections(dets, rois, task_id=9,
                           lut=synthetic_lut, task_embs=synthetic_task_embs, W=W)
    assert out[0]["cos_sim"] < 0
    assert out[0]["tadop_score"] == 0.0


# ---------- End-to-end (optional) ----------

@pytest.fixture(scope="module")
def e2e_output(tmp_path_factory):
    """ Run the real script on one image if priors and image are present. """
    repo = Path(__file__).resolve().parent
    image = repo / "coco_images" / "val2017"
    # Just need any one image
    if not image.exists():
        pytest.skip("COCO images not downloaded")
    one_image = next(image.glob("*.jpg"), None)
    if one_image is None:
        pytest.skip("No images in coco_images/val2017")
    if not (repo / "affordance_lut_fp32.npy").exists():
        pytest.skip("LUT not built")
    if not (repo / "task_embeddings_fp32.npy").exists():
        pytest.skip("Task embeddings not built")

    out_dir = tmp_path_factory.mktemp("tadop_e2e")
    proc = subprocess.run(
        [sys.executable, str(repo / "tadop_reference.py"),
         "--image", str(one_image),
         "--task", "10",
         "--lut-dir", str(repo),
         "--embeddings-dir", str(repo),
         "--out-dir", str(out_dir)],
        capture_output=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"tadop_reference.py failed: {proc.stderr.decode()[:500]}")
    return out_dir


def test_e2e_produces_result_file(e2e_output):
    files = list(e2e_output.rglob("tadop_result.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    for k in ["image_id", "task_id_1based", "task_name",
              "num_raw_detections", "num_after_mask",
              "timing_ms", "ranked_detections"]:
        assert k in record


def test_e2e_ranked_detections_well_formed(e2e_output):
    rec = json.loads(list(e2e_output.rglob("tadop_result.json"))[0].read_text())
    for d in rec["ranked_detections"]:
        for k in ["class_id", "class_name", "confidence", "bbox_xyxy",
                  "lut_prior", "cos_sim", "tadop_score", "masked_out"]:
            assert k in d
        # Score sanity
        if d["masked_out"]:
            assert d["tadop_score"] == 0.0
