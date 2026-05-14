"""
TADOP — Phase 1.2: MiniLM Task Embedding Cache Builder
=======================================================

Encodes the 14 COCO-Tasks task descriptions using the
sentence-transformers/all-MiniLM-L6-v2 model and saves the resulting
14x384 embedding matrix as a quantized cache for FPGA inference.

At inference time, user queries are encoded by MiniLM (on VEGA, with the
cached embeddings here serving as anchor points). Cosine similarity
between the user query embedding and each cached task embedding tells us
which task the user meant.

Output artifacts (in --out-dir):
    task_embeddings_fp32.npy   14x384 float32, L2-normalized
    task_embeddings_int8.npy   14x384 int8, per-row symmetric quantization
    task_embedding_scales.npy  14 float32 scales (one per task)
    task_embeddings.coe        Xilinx BRAM init (row-major flat bytes)
    task_embeddings.mem        Verilog $readmemh init (one byte per line)
    task_scales.mem            Per-row scales as fp32 hex (4 bytes each)
    task_embeddings.json       Human-readable: task_name -> first 8 dims
    embedding_stats.json       Norms, scales, max values, vocab info

Quantization scheme:
    Per-row symmetric INT8. For task t with FP32 row r_t (L2-normalized):
        scale_t = max(|r_t|) / 127
        q_t     = round(r_t / scale_t)  clipped to [-128, 127]
    Reconstruction:
        r_t_hat = q_t * scale_t
    Cosine similarity in INT32 accumulator:
        cos(q_user, q_t) = sum(q_user[i] * q_t[i]) * scale_user * scale_t
        (post-multiply by scales after accumulation)

Usage:
    python build_task_embeddings.py --out-dir ./data/embeddings
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


# 14 task descriptions, written as natural sentences that match how a user
# would phrase a query. Order matches COCO-Tasks task_id 1..14 (i.e. index
# i here corresponds to task_id = i + 1 in the annotations).
#
# These phrasings were chosen to be:
#   - First-person ("I want to ..."), matching likely user queries
#   - Concrete about the goal (a "container" for flowers, "a glass" for wine)
#   - Free of internal label tokens (no underscores or COCO jargon)
TASK_PROMPTS = [
    "I want to step on something to reach a higher place",        # 1
    "I want to sit down comfortably",                              # 2
    "I want to place flowers in a container",                      # 3
    "I need to get potatoes out of a fire",                        # 4
    "I want to water a plant",                                     # 5
    "I want to get a lemon slice out of my tea",                   # 6
    "I want to dig a hole in the ground",                          # 7
    "I want to open a bottle of beer",                             # 8
    "I want to open a parcel or package",                          # 9
    "I want to serve wine in a glass",                             # 10
    "I want to pour sugar into a cup",                             # 11
    "I want to smear butter on bread",                             # 12
    "I want to extinguish a fire",                                 # 13
    "I want to pound a carpet to clean it",                        # 14
]

TASK_NAMES = [
    "step_on_something", "sit_comfortably", "place_flowers",
    "get_potatoes_out_of_fire", "water_plant", "get_lemon_out_of_tea",
    "dig_hole", "open_bottle_of_beer", "open_parcel", "serve_wine",
    "pour_sugar", "smear_butter", "extinguish_fire", "pound_carpet",
]

EMBED_DIM = 384  # all-MiniLM-L6-v2 hidden size
NUM_TASKS = 14
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- Encoding ----------

def encode_prompts(prompts: list[str]) -> np.ndarray:
    """
    Run MiniLM on the 14 prompts. Returns 14x384 float32, L2-normalized
    (so cosine similarity reduces to a dot product).

    L2-normalization is critical: at FPGA inference time we will compute
    integer dot products and rely on the rows being unit norm to interpret
    the result as cosine similarity directly.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "    pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(MODEL_NAME)
    # normalize_embeddings=True applies L2 normalization on the output side.
    embs = model.encode(
        prompts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    assert embs.shape == (len(prompts), EMBED_DIM), \
        f"unexpected shape {embs.shape}, expected ({len(prompts)}, {EMBED_DIM})"
    return embs.astype(np.float32)


# ---------- Quantization ----------

def quantize_per_row_int8(emb_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row symmetric INT8. Each task gets its own scale, computed from
    its row's max absolute value so the largest entry maps to ±127.

    Returns:
        q:      (14, 384) int8
        scales: (14,)     float32   real_value = int8_value * scale
    """
    abs_max = np.max(np.abs(emb_fp32), axis=1)        # (14,)
    # Guard against any zero rows (shouldn't happen for MiniLM, but be safe)
    abs_max = np.where(abs_max > 0, abs_max, 1.0)
    scales = (abs_max / 127.0).astype(np.float32)

    q = np.round(emb_fp32 / scales[:, None])
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scales


# ---------- Export ----------

def write_coe(q_int8: np.ndarray, path: Path) -> None:
    """ Vivado .coe init file. Row-major, one byte per entry. """
    flat = q_int8.flatten().astype(np.uint8)  # int8 -> uint8 two's complement
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        hex_words = [f"{b:02x}" for b in flat]
        f.write(",\n".join(hex_words))
        f.write(";\n")


def write_mem(q_int8: np.ndarray, path: Path) -> None:
    """ Verilog $readmemh init, one byte per line. """
    flat = q_int8.flatten().astype(np.uint8)
    with open(path, "w") as f:
        for b in flat:
            f.write(f"{b:02x}\n")


def write_scales_mem(scales: np.ndarray, path: Path) -> None:
    """
    Per-row fp32 scales, dumped as 4 hex bytes per line.
    Little-endian, matching how VEGA will read them with a 4-byte load.
    """
    with open(path, "w") as f:
        for s in scales:
            raw = struct.pack("<f", float(s))   # 4 bytes, little-endian fp32
            f.write(raw.hex() + "\n")


def write_human_table(emb_fp32: np.ndarray, prompts: list[str], path: Path) -> None:
    """ Task name + prompt + first 8 dims, for eyeballing. """
    rows = []
    for i, (name, prompt) in enumerate(zip(TASK_NAMES, prompts)):
        rows.append({
            "task_id": i + 1,
            "name": name,
            "prompt": prompt,
            "norm": float(np.linalg.norm(emb_fp32[i])),
            "first_8_dims": [round(float(x), 4) for x in emb_fp32[i][:8]],
        })
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def cosine_sanity_matrix(emb_fp32: np.ndarray) -> np.ndarray:
    """
    14x14 cosine similarity matrix. The diagonal should be ~1.0
    (rows are unit-normalized). Off-diagonals show task confusability:
    if any off-diagonal pair is >0.7, those tasks may be hard to separate
    from a user query, and the prompts could be reworded.
    """
    # Rows are already L2-normalized, so dot = cos
    return emb_fp32 @ emb_fp32.T


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Encoding {NUM_TASKS} task prompts with {MODEL_NAME}...")
    emb_fp32 = encode_prompts(TASK_PROMPTS)
    print(f"  shape={emb_fp32.shape}, dtype={emb_fp32.dtype}")

    print("Quantizing per-row symmetric INT8...")
    q_int8, scales = quantize_per_row_int8(emb_fp32)

    # Reconstruction sanity check
    recon = q_int8.astype(np.float32) * scales[:, None]
    max_err = np.max(np.abs(emb_fp32 - recon))
    print(f"  max quantization error: {max_err:.5f}")

    # Cosine sanity: how well does quantization preserve similarity?
    cos_fp32 = cosine_sanity_matrix(emb_fp32)
    # Reconstructed cosine (post-quantization, post-multiply)
    recon_normed = recon / np.linalg.norm(recon, axis=1, keepdims=True)
    cos_q = recon_normed @ recon_normed.T
    cos_err = np.max(np.abs(cos_fp32 - cos_q))
    print(f"  max cosine-matrix error after quant: {cos_err:.5f}")

    # Save everything
    np.save(out_dir / "task_embeddings_fp32.npy", emb_fp32)
    np.save(out_dir / "task_embeddings_int8.npy", q_int8)
    np.save(out_dir / "task_embedding_scales.npy", scales)
    write_coe(q_int8, out_dir / "task_embeddings.coe")
    write_mem(q_int8, out_dir / "task_embeddings.mem")
    write_scales_mem(scales, out_dir / "task_scales.mem")
    write_human_table(emb_fp32, TASK_PROMPTS, out_dir / "task_embeddings.json")

    stats = {
        "model": MODEL_NAME,
        "num_tasks": NUM_TASKS,
        "embed_dim": EMBED_DIM,
        "bytes_embeddings": int(q_int8.size),
        "bytes_scales": int(scales.size * 4),
        "bytes_total": int(q_int8.size + scales.size * 4),
        "fp32_norms": [float(n) for n in np.linalg.norm(emb_fp32, axis=1)],
        "int8_scales": [float(s) for s in scales],
        "max_quant_error": float(max_err),
        "max_cosine_matrix_error": float(cos_err),
        "prompts": TASK_PROMPTS,
    }
    with open(out_dir / "embedding_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Pretty-print the cosine matrix for sanity inspection
    print("\nCosine similarity between cached tasks (off-diagonal max = confusability):")
    short_names = [n[:10] for n in TASK_NAMES]
    header = "          " + "".join(f"{n:>10}" for n in short_names)
    print(header)
    for i, n in enumerate(short_names):
        row = f"{n:<10}" + "".join(
            f"{cos_fp32[i][j]:>10.2f}" for j in range(NUM_TASKS)
        )
        print(row)

    # Flag any confusable pairs
    print("\nMost confusable task pairs (cosine > 0.55, off-diagonal):")
    flagged = []
    for i in range(NUM_TASKS):
        for j in range(i + 1, NUM_TASKS):
            if cos_fp32[i][j] > 0.55:
                flagged.append((cos_fp32[i][j], TASK_NAMES[i], TASK_NAMES[j]))
    flagged.sort(reverse=True)
    if not flagged:
        print("  (none — all task prompts are well-separated)")
    else:
        for c, a, b in flagged:
            print(f"  {c:.2f}  {a} <-> {b}")

    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
