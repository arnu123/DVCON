"""
Tests for the affordance LUT builder.

Run with:  python -m pytest sw/lut/test_affordance_lut.py -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER   = REPO_ROOT / "sw" / "lut" / "build_affordance_lut.py"


@pytest.fixture(scope="module")
def synthetic_outputs(tmp_path_factory):
    """Run the builder once in synthetic mode and reuse the outputs."""
    out_dir = tmp_path_factory.mktemp("lut_out")
    subprocess.run(
        [sys.executable, str(BUILDER), "--synthetic", "--out-dir", str(out_dir)],
        check=True, capture_output=True,
    )
    return out_dir


def test_all_artifacts_exist(synthetic_outputs):
    expected = [
        "affordance_lut_fp32.npy",
        "affordance_lut_int8.npy",
        "affordance_lut.coe",
        "affordance_lut.mem",
        "affordance_lut.json",
        "lut_stats.json",
    ]
    for name in expected:
        assert (synthetic_outputs / name).exists(), f"missing {name}"


def test_shape_and_dtype(synthetic_outputs):
    fp = np.load(synthetic_outputs / "affordance_lut_fp32.npy")
    q  = np.load(synthetic_outputs / "affordance_lut_int8.npy")
    assert fp.shape == (80, 14)
    assert q.shape  == (80, 14)
    assert fp.dtype == np.float32
    assert q.dtype  == np.int8


def test_value_range(synthetic_outputs):
    fp = np.load(synthetic_outputs / "affordance_lut_fp32.npy")
    q  = np.load(synthetic_outputs / "affordance_lut_int8.npy")
    assert fp.min() >= 0.0 and fp.max() <= 1.0
    # Symmetric int8 with positive-only payload
    assert q.min() >= 0 and q.max() <= 127


def test_quant_round_trip_within_ulp(synthetic_outputs):
    fp = np.load(synthetic_outputs / "affordance_lut_fp32.npy")
    q  = np.load(synthetic_outputs / "affordance_lut_int8.npy")
    recon = q.astype(np.float32) / 127.0
    assert np.abs(fp - recon).max() < (1.0 / 127.0)


def test_mem_file_line_count(synthetic_outputs):
    mem_lines = (synthetic_outputs / "affordance_lut.mem").read_text().splitlines()
    assert len(mem_lines) == 80 * 14, f"expected 1120 lines, got {len(mem_lines)}"
    # Every line is two hex chars
    for line in mem_lines:
        assert len(line) == 2 and all(c in "0123456789abcdef" for c in line)


def test_coe_header_and_terminator(synthetic_outputs):
    coe = (synthetic_outputs / "affordance_lut.coe").read_text()
    assert coe.startswith("memory_initialization_radix=16;")
    assert "memory_initialization_vector=" in coe
    assert coe.rstrip().endswith(";")


def test_known_synthetic_priors(synthetic_outputs):
    """
    Lock in a few well-known affordances that should always survive
    any future refactor of the synthetic generator.
    """
    fp = np.load(synthetic_outputs / "affordance_lut_fp32.npy")

    # Indices match COCO80_NAMES ordering (verified once)
    WINE_GLASS = 40
    KNIFE      = 43
    SCISSORS   = 76
    BASEBALL_BAT = 34

    SERVE_WINE   = 9   # task 10 -> idx 9
    OPEN_PARCEL  = 8   # task 9  -> idx 8
    SMEAR_BUTTER = 11
    POUND_CARPET = 13

    assert fp[WINE_GLASS][SERVE_WINE]    > 0.85
    assert fp[SCISSORS][OPEN_PARCEL]     > 0.85
    assert fp[KNIFE][SMEAR_BUTTER]       > 0.85
    assert fp[BASEBALL_BAT][POUND_CARPET]> 0.85


def test_human_table_well_formed(synthetic_outputs):
    rows = json.loads((synthetic_outputs / "affordance_lut.json").read_text())
    assert len(rows) == 80
    for row in rows:
        assert "class" in row
        # 14 task columns in addition to class
        assert len(row) == 15
