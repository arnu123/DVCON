"""
Tests for the YOLOv8n baseline outputs.

Validates that whatever the baseline runner produced has the shapes,
dtypes, and structure that downstream Phase 1.4 (scorer) will rely on.

Run after generating outputs:
    python yolov8n_baseline.py --image-dir ./coco_images/val2017 \\
                               --max-images 5 \\
                               --out-dir ./data/baseline
    python -m pytest test_yolov8n_baseline.py -v --baseline-dir ./data/baseline

The --baseline-dir option is registered in conftest.py (same folder).
"""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def baseline_dir(request):
    p = Path(request.config.getoption("--baseline-dir"))
    if not p.exists():
        pytest.skip(f"Baseline dir not found: {p}. Run yolov8n_baseline.py first.")
    return p


@pytest.fixture(scope="module")
def summary(baseline_dir):
    summary_path = baseline_dir / "summary.json"
    if not summary_path.exists():
        pytest.skip(f"summary.json not found in {baseline_dir}")
    return json.loads(summary_path.read_text())


@pytest.fixture(scope="module")
def image_dirs(baseline_dir):
    """All per-image subdirectories."""
    dirs = [d for d in baseline_dir.iterdir() if d.is_dir()]
    if not dirs:
        pytest.skip("No per-image directories found")
    return dirs


def test_summary_has_required_fields(summary):
    required = ["model", "input_size", "stride8_layer_name",
                "stride8_layer_output_shape", "num_images_processed",
                "latency_ms", "per_image"]
    for k in required:
        assert k in summary, f"summary missing {k!r}"


def test_stride8_shape_is_canonical(summary):
    """
    The hooked layer must produce (B, 64, 80, 80) for square 640x640 input.
    YOLOv8n (nano, width multiplier 0.25) has 64 channels at stride-8.
    """
    shape = summary["stride8_layer_output_shape"]
    assert len(shape) == 4, f"expected 4D, got {shape}"
    assert shape[1] == 64, f"expected 64 channels, got {shape[1]}"
    assert shape[2] == 80 and shape[3] == 80, \
        f"expected 80x80 spatial (rect=False), got {shape[2]}x{shape[3]}"


def test_each_image_has_required_artifacts(image_dirs):
    required = ["image_meta.json", "detections.json", "c2f_stride8.npy"]
    for d in image_dirs:
        for name in required:
            assert (d / name).exists(), f"{d.name} missing {name}"


def test_c2f_feature_shape_and_dtype(image_dirs):
    for d in image_dirs:
        feat = np.load(d / "c2f_stride8.npy")
        assert feat.shape == (64, 80, 80), \
            f"{d.name}: c2f_stride8 shape {feat.shape}"
        assert feat.dtype == np.float32


def test_c2f_features_are_finite(image_dirs):
    """ No NaN or Inf in the captured features. """
    for d in image_dirs:
        feat = np.load(d / "c2f_stride8.npy")
        assert np.isfinite(feat).all(), f"{d.name}: non-finite values in c2f"


def test_c2f_features_have_signal(image_dirs):
    """
    Captured features should have non-trivial variance — they're post-conv
    activations, so std > 0 and a range of values.
    """
    for d in image_dirs:
        feat = np.load(d / "c2f_stride8.npy")
        assert feat.std() > 1e-3, f"{d.name}: c2f feature looks dead (std={feat.std():.6f})"


def test_detections_format(image_dirs):
    for d in image_dirs:
        dets = json.loads((d / "detections.json").read_text())
        for det in dets:
            assert "bbox_xyxy" in det and len(det["bbox_xyxy"]) == 4
            assert "class_id" in det and 0 <= det["class_id"] < 80
            assert "confidence" in det and 0 <= det["confidence"] <= 1
            assert "class_name" in det


def test_image_meta_format(image_dirs):
    for d in image_dirs:
        meta = json.loads((d / "image_meta.json").read_text())
        for k in ["image_id", "orig_shape_hw", "model_input_size",
                  "scale_factor_h", "scale_factor_w", "elapsed_ms",
                  "num_detections"]:
            assert k in meta, f"{d.name}: meta missing {k!r}"
        assert meta["model_input_size"] == 640


def test_latency_is_recorded(summary):
    lat = summary["latency_ms"]
    assert lat["mean"] > 0
    assert lat["median"] > 0
    # Sanity: on any modern laptop YOLOv8n should be well under 1 second per image
    assert lat["mean"] < 2000, f"mean latency suspiciously high: {lat['mean']:.0f} ms"