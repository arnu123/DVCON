"""
Tests for the MiniLM task embedding cache builder.

Two layers:
  1. Standalone tests of the quantization + export logic using synthetic
     fake-MiniLM embeddings (no network required).
  2. Optional end-to-end test that actually calls the real model. Skipped
     automatically if sentence-transformers can't reach the network.

Run with:  python -m pytest sw/minilm/test_task_embeddings.py -v
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER   = REPO_ROOT / "sw" / "minilm" / "build_task_embeddings.py"

# Make the builder's helpers importable for unit-level testing
sys.path.insert(0, str(BUILDER.parent))
from build_task_embeddings import (
    quantize_per_row_int8,
    write_coe,
    write_mem,
    write_scales_mem,
    cosine_sanity_matrix,
    NUM_TASKS,
    EMBED_DIM,
    TASK_PROMPTS,
    TASK_NAMES,
)


# ---------- Synthetic fixture: fake MiniLM-like embeddings ----------

@pytest.fixture
def fake_embeddings():
    """
    14x384 random unit-norm float32 vectors. Same shape/dtype/normalization
    as real MiniLM output. Reproducible via fixed seed.
    """
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((NUM_TASKS, EMBED_DIM)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


# ---------- Quantization tests ----------

def test_quantize_shape_and_dtype(fake_embeddings):
    q, s = quantize_per_row_int8(fake_embeddings)
    assert q.shape == (NUM_TASKS, EMBED_DIM)
    assert s.shape == (NUM_TASKS,)
    assert q.dtype == np.int8
    assert s.dtype == np.float32


def test_quantize_uses_full_int8_range(fake_embeddings):
    """ Per-row quantization should saturate at ±127 in at least one entry. """
    q, _ = quantize_per_row_int8(fake_embeddings)
    for i in range(NUM_TASKS):
        assert np.max(np.abs(q[i])) == 127, \
            f"row {i} doesn't reach 127 — quantizer wastes dynamic range"


def test_reconstruction_error_bounded(fake_embeddings):
    """
    With per-row symmetric INT8, max reconstruction error per element
    should be < scale/2.
    """
    q, s = quantize_per_row_int8(fake_embeddings)
    recon = q.astype(np.float32) * s[:, None]
    for i in range(NUM_TASKS):
        err = np.max(np.abs(fake_embeddings[i] - recon[i]))
        assert err < s[i], f"row {i}: err={err:.5f} >= scale={s[i]:.5f}"


def test_cosine_preserved_after_quant(fake_embeddings):
    """
    Quantization should preserve cosine similarities to within a small
    tolerance. With per-row INT8 and 384-D, error should be well under 0.02
    even adversarially.
    """
    q, s = quantize_per_row_int8(fake_embeddings)
    recon = q.astype(np.float32) * s[:, None]
    recon /= np.linalg.norm(recon, axis=1, keepdims=True)

    cos_fp = cosine_sanity_matrix(fake_embeddings)
    cos_q  = recon @ recon.T

    assert np.max(np.abs(cos_fp - cos_q)) < 0.02


# ---------- Export-format tests ----------

def test_coe_format(tmp_path, fake_embeddings):
    q, _ = quantize_per_row_int8(fake_embeddings)
    p = tmp_path / "emb.coe"
    write_coe(q, p)
    txt = p.read_text()
    assert txt.startswith("memory_initialization_radix=16;")
    assert "memory_initialization_vector=" in txt
    assert txt.rstrip().endswith(";")
    # Count comma-separated entries
    n_entries = txt.count(",") + 1
    assert n_entries == NUM_TASKS * EMBED_DIM


def test_mem_format(tmp_path, fake_embeddings):
    q, _ = quantize_per_row_int8(fake_embeddings)
    p = tmp_path / "emb.mem"
    write_mem(q, p)
    lines = p.read_text().splitlines()
    assert len(lines) == NUM_TASKS * EMBED_DIM
    for line in lines:
        assert len(line) == 2
        assert all(c in "0123456789abcdef" for c in line)


def test_scales_mem_format(tmp_path, fake_embeddings):
    _, scales = quantize_per_row_int8(fake_embeddings)
    p = tmp_path / "scales.mem"
    write_scales_mem(scales, p)
    lines = p.read_text().splitlines()
    assert len(lines) == NUM_TASKS
    for i, line in enumerate(lines):
        # 4 bytes = 8 hex chars
        assert len(line) == 8
        raw = bytes.fromhex(line)
        decoded = struct.unpack("<f", raw)[0]
        assert abs(decoded - float(scales[i])) < 1e-7


def test_coe_uses_twos_complement(tmp_path):
    """ Negative INT8 values should round-trip through the .coe hex bytes. """
    q = np.array([[-128, -1, 0, 1, 127] + [0] * (EMBED_DIM - 5)] * NUM_TASKS,
                 dtype=np.int8)
    p = tmp_path / "emb.coe"
    write_coe(q, p)
    body = p.read_text().split("memory_initialization_vector=\n")[1].rstrip(";\n")
    bytes_hex = body.split(",\n")
    # -128 -> 0x80, -1 -> 0xff, 127 -> 0x7f
    assert bytes_hex[0] == "80"
    assert bytes_hex[1] == "ff"
    assert bytes_hex[2] == "00"
    assert bytes_hex[3] == "01"
    assert bytes_hex[4] == "7f"


# ---------- Prompt sanity ----------

def test_prompt_count_matches_tasks():
    assert len(TASK_PROMPTS) == NUM_TASKS
    assert len(TASK_NAMES) == NUM_TASKS


def test_prompts_are_natural_sentences():
    """
    Guard against accidentally regressing to underscore_separated labels.
    Each prompt should be a real sentence: starts with capital, has spaces,
    contains no underscores.
    """
    for i, p in enumerate(TASK_PROMPTS):
        assert "_" not in p, f"task {i}: prompt contains underscores: {p!r}"
        assert " " in p, f"task {i}: prompt has no spaces: {p!r}"
        assert p[0].isupper(), f"task {i}: prompt should start uppercase: {p!r}"
        assert len(p.split()) >= 4, f"task {i}: prompt too short: {p!r}"


# ---------- Optional end-to-end (requires network + ~80MB model) ----------

@pytest.fixture(scope="module")
def real_run_output(tmp_path_factory):
    """
    Runs the real builder. If sentence-transformers can't reach the network,
    the builder errors out — we mark the test skipped instead of failed.
    """
    out_dir = tmp_path_factory.mktemp("emb_out")
    proc = subprocess.run(
        [sys.executable, str(BUILDER), "--out-dir", str(out_dir)],
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="ignore")
        if "huggingface" in stderr.lower() or "connect" in stderr.lower():
            pytest.skip(f"Network unavailable for real model load: {stderr[:200]}")
        else:
            pytest.fail(f"builder failed: {stderr[:500]}")
    return out_dir


def test_e2e_artifacts_present(real_run_output):
    expected = [
        "task_embeddings_fp32.npy",
        "task_embeddings_int8.npy",
        "task_embedding_scales.npy",
        "task_embeddings.coe",
        "task_embeddings.mem",
        "task_scales.mem",
        "task_embeddings.json",
        "embedding_stats.json",
    ]
    for n in expected:
        assert (real_run_output / n).exists(), f"missing {n}"


def test_e2e_shapes(real_run_output):
    emb = np.load(real_run_output / "task_embeddings_fp32.npy")
    q   = np.load(real_run_output / "task_embeddings_int8.npy")
    s   = np.load(real_run_output / "task_embedding_scales.npy")
    assert emb.shape == (NUM_TASKS, EMBED_DIM)
    assert q.shape   == (NUM_TASKS, EMBED_DIM)
    assert s.shape   == (NUM_TASKS,)


def test_e2e_rows_are_unit_norm(real_run_output):
    """ Critical: cosine reduces to dot product only if rows are unit norm. """
    emb = np.load(real_run_output / "task_embeddings_fp32.npy")
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"norms = {norms}"


def test_e2e_no_confusable_task_pairs(real_run_output):
    """
    Off-diagonal cosines should be reasonably small. If two tasks have
    cosine > 0.85 by accident, a user query will likely confuse them.
    """
    emb = np.load(real_run_output / "task_embeddings_fp32.npy")
    cos = emb @ emb.T
    np.fill_diagonal(cos, 0.0)
    worst = cos.max()
    assert worst < 0.85, f"two task prompts are too similar: max off-diag cos = {worst:.3f}"
