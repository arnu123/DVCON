"""
TADOP — Phase 1.1: Affordance LUT Builder
==========================================

Builds the 80x14 affordance Look-Up Table from COCO-Tasks training annotations.

LUT[class_c][task_t] = count(images where class_c is preferred for task_t)
                      ────────────────────────────────────────────────────
                      count(images where class_c appears for task_t)

Output artifacts (in --out-dir):
    affordance_lut_fp32.npy   80x14 float32, range [0, 1]
    affordance_lut_int8.npy   80x14 int8,  scale = 1/127 (i.e. value*127 -> int8)
    affordance_lut.coe        Xilinx BRAM coefficient init file (radix=16)
    affordance_lut.mem        Verilog $readmemh init file
    affordance_lut.json       Human-readable: class_name x task_name table
    lut_stats.json            Sanity stats: nonzero entries, max/min, etc.

Input modes:
    --coco-tasks-dir <path>   Read real annotations (1.json .. 14.json)
                              from the official COCO-Tasks repo layout.
    --synthetic               Generate a plausible synthetic LUT from
                              hand-coded class/task affinities. For pipeline
                              bring-up only; replace with real data before
                              any reported results.

Usage:
    # Real data (you'll do this once you have the annotations)
    python build_affordance_lut.py --coco-tasks-dir ./data/coco_tasks/annotations \
                                   --out-dir ./data/lut

    # Synthetic (works right now, no downloads needed)
    python build_affordance_lut.py --synthetic --out-dir ./data/lut
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------- Constants ----------

NUM_CLASSES = 80   # COCO 80-class label space (the YOLO/contiguous one)
NUM_TASKS   = 14   # COCO-Tasks fixed task vocabulary

# COCO-Tasks task IDs are 1..14. Names from Sawatzky et al. (CVPR 2019).
TASK_NAMES = [
    "step_on_something",                # 1
    "sit_comfortably",                  # 2
    "place_flowers",                    # 3
    "get_potatoes_out_of_fire",         # 4
    "water_plant",                      # 5
    "get_lemon_out_of_tea",             # 6
    "dig_hole",                         # 7
    "open_bottle_of_beer",              # 8
    "open_parcel",                      # 9
    "serve_wine",                       # 10
    "pour_sugar",                       # 11
    "smear_butter",                     # 12
    "extinguish_fire",                  # 13
    "pound_carpet",                     # 14
]

# COCO 80-class names in the standard contiguous order used by YOLO/Ultralytics.
# Index here matches what YOLO outputs as class_id.
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# COCO uses non-contiguous category IDs 1..90 (with gaps). The 80-class
# contiguous label space is obtained by removing the 10 unused IDs.
# This mapping converts "COCO_category_id" (1..90, with gaps) -> 0..79.
COCO_CAT_ID_TO_CONTIG = {
    1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11,
    14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20,
    23:21, 24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29,
    35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38,
    44:39, 46:40, 47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47,
    54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56,
    63:57, 64:58, 65:59, 67:60, 70:61, 72:62, 73:63, 74:64, 75:65,
    76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74,
    86:75, 87:76, 88:77, 89:78, 90:79,
}


# ---------- Real-data path ----------

def build_lut_from_coco_tasks(annotations_dir: Path) -> tuple[np.ndarray, dict]:
    """
    Walks task_1_train.json .. task_14_train.json from the COCO-Tasks repo
    (cvpr2019 branch, annotations/ folder) and computes the LUT.

    We use train split only for the LUT — val is held out for evaluation.

    Each annotation is a single object instance. category_id ∈ {0,1} indicates
    whether *this instance* is preferred. We aggregate at the (image, class)
    level so an image counts once per class regardless of how many instances
    it has, matching the common reading of the affordance prior:

        numerator[c][t]   = # of (image, task=t) pairs where class c
                            has at least one preferred instance
        denominator[c][t] = # of (image, task=t) pairs where class c
                            appears at all

        LUT[c][t] = numerator / denominator   (zero where denom==0)
    """
    numer = np.zeros((NUM_CLASSES, NUM_TASKS), dtype=np.int64)
    denom = np.zeros((NUM_CLASSES, NUM_TASKS), dtype=np.int64)
    per_task_image_count = defaultdict(int)

    for task_id in range(1, NUM_TASKS + 1):
        # Support all known naming conventions:
        #   Kaggle mirror:    task_1.json
        #   Official repo:    task_1_train.json
        #   Numeric (legacy): 1.json
        candidates = [
            annotations_dir / f"task_{task_id}.json",
            annotations_dir / f"task_{task_id}_train.json",
            annotations_dir / f"{task_id}.json",
        ]
        ann_path = next((p for p in candidates if p.exists()), None)
        if ann_path is None:
            raise FileNotFoundError(
                f"Cannot find annotation for task {task_id} in {annotations_dir}.\n"
                f"Looked for: {[p.name for p in candidates]}\n"
                f"Download from: https://www.kaggle.com/datasets/sajjad006/coco-task-dataset\n"
                f"Then pass: --coco-tasks-dir ./coco-tasks/annotations"
            )
        with open(ann_path) as f:
            data = json.load(f)

        # COCO-style annotation list:
        anns = data["annotations"] if isinstance(data, dict) and "annotations" in data else data

        # Group by (image_id, contig_class) -> any preferred?
        per_image_class = defaultdict(lambda: {"appears": False, "preferred": False})
        seen_images = set()

        for a in anns:
            coco_cat = a.get("COCO_category_id") or a.get("category_id_coco")
            if coco_cat is None:
                # Some forks store the original COCO id under 'category_id'
                # and the preference flag under a different key. Skip if
                # it's clearly not in the 1..90 range.
                cat = a.get("category_id")
                if cat is not None and cat in COCO_CAT_ID_TO_CONTIG:
                    coco_cat = cat
                else:
                    continue
            if coco_cat not in COCO_CAT_ID_TO_CONTIG:
                continue
            contig = COCO_CAT_ID_TO_CONTIG[coco_cat]
            img_id = a["image_id"]
            seen_images.add(img_id)

            # In the COCO-Tasks schema the *task-preference* flag is the
            # field literally named 'category_id' (0 or 1). Guard the access:
            pref_flag = a.get("category_id", 0)
            # Some preprocessed forks rename it; fall back:
            if pref_flag not in (0, 1):
                pref_flag = a.get("preferred", 0)

            key = (img_id, contig)
            per_image_class[key]["appears"] = True
            if pref_flag == 1:
                per_image_class[key]["preferred"] = True

        per_task_image_count[task_id] = len(seen_images)

        t_idx = task_id - 1
        for (_img, contig), flags in per_image_class.items():
            if flags["appears"]:
                denom[contig][t_idx] += 1
            if flags["preferred"]:
                numer[contig][t_idx] += 1

    # Compute LUT with safe divide
    with np.errstate(divide="ignore", invalid="ignore"):
        lut = np.where(denom > 0, numer / denom, 0.0).astype(np.float32)

    # Support floor: classes with very few appearances for a task produce
    # untrustworthy ratios (e.g. 1/1 = 1.0). Inspection of COCO-Tasks train
    # shows only a handful of such entries. Zero them out — they would only
    # add noise to the class-mask + scorer pipeline.
    MIN_SUPPORT = 5
    low_support = (denom > 0) & (denom < MIN_SUPPORT)
    n_clamped = int(low_support.sum())
    lut[low_support] = 0.0

    stats = {
        "source": "coco_tasks_real",
        "images_per_task": {str(k): int(v) for k, v in per_task_image_count.items()},
        "total_appearances": int(denom.sum()),
        "total_preferences": int(numer.sum()),
        "nonzero_entries": int((lut > 0).sum()),
        "max_value": float(lut.max()),
        "min_support_threshold": MIN_SUPPORT,
        "entries_clamped_low_support": n_clamped,
    }
    return lut, stats


# ---------- Synthetic path ----------

def build_synthetic_lut(seed: int = 42) -> tuple[np.ndarray, dict]:
    """
    Hand-coded plausible affordances for pipeline bring-up.
    Loud warning: do NOT use for any reported numbers.
    """
    rng = np.random.default_rng(seed)
    lut = np.zeros((NUM_CLASSES, NUM_TASKS), dtype=np.float32)

    # (task_idx, class_name -> preference_score) — tuned by inspection of the
    # task names. Realistic order-of-magnitude only.
    affinities = {
        # 1 step_on_something
        0:  {"chair": 0.62, "couch": 0.45, "bench": 0.30, "bed": 0.20,
             "skateboard": 0.55, "suitcase": 0.40, "book": 0.15},
        # 2 sit_comfortably
        1:  {"couch": 0.91, "chair": 0.78, "bench": 0.55, "bed": 0.70,
             "toilet": 0.10},
        # 3 place_flowers
        2:  {"vase": 0.95, "cup": 0.45, "bowl": 0.30, "bottle": 0.25,
             "wine glass": 0.40},
        # 4 get_potatoes_out_of_fire
        3:  {"fork": 0.80, "spoon": 0.65, "knife": 0.40, "tongs": 0.0},
        # 5 water_plant
        4:  {"bottle": 0.72, "cup": 0.55, "bowl": 0.50, "wine glass": 0.30,
             "vase": 0.40},
        # 6 get_lemon_out_of_tea
        5:  {"spoon": 0.92, "fork": 0.55, "knife": 0.20},
        # 7 dig_hole
        6:  {"spoon": 0.60, "knife": 0.50, "fork": 0.35, "skateboard": 0.10},
        # 8 open_bottle_of_beer
        7:  {"knife": 0.55, "fork": 0.40, "spoon": 0.20, "scissors": 0.50},
        # 9 open_parcel
        8:  {"scissors": 0.93, "knife": 0.78, "fork": 0.30},
        # 10 serve_wine
        9:  {"wine glass": 0.94, "cup": 0.71, "bowl": 0.30, "bottle": 0.40,
             "vase": 0.10},
        # 11 pour_sugar
        10: {"spoon": 0.85, "cup": 0.55, "bowl": 0.50, "fork": 0.20},
        # 12 smear_butter
        11: {"knife": 0.92, "spoon": 0.65, "fork": 0.40},
        # 13 extinguish_fire
        12: {"bottle": 0.75, "cup": 0.55, "bowl": 0.50, "vase": 0.30},
        # 14 pound_carpet
        13: {"baseball bat": 0.88, "tennis racket": 0.70, "skateboard": 0.40,
             "umbrella": 0.50, "frisbee": 0.20, "book": 0.15},
    }

    name_to_idx = {n: i for i, n in enumerate(COCO80_NAMES)}
    for t_idx, prefs in affinities.items():
        for cname, score in prefs.items():
            if cname in name_to_idx:
                lut[name_to_idx[cname]][t_idx] = score

    # Add small noise floor on classes that *appear* in scenes for that task
    # but are rarely preferred. Random sparse floor.
    floor_mask = rng.random((NUM_CLASSES, NUM_TASKS)) < 0.15
    floor_vals = rng.uniform(0.005, 0.05, size=(NUM_CLASSES, NUM_TASKS)).astype(np.float32)
    lut = np.where((lut == 0) & floor_mask, floor_vals, lut)

    stats = {
        "source": "synthetic",
        "warning": "Hand-coded. Replace with real COCO-Tasks data before reporting.",
        "nonzero_entries": int((lut > 0).sum()),
        "max_value": float(lut.max()),
    }
    return lut, stats


# ---------- Quantization & export ----------

def quantize_int8(lut_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Symmetric quantization, range [0,1] -> [0,127] in int8.
    Scale: real_value = int8_value / 127.
    """
    scale = 1.0 / 127.0
    q = np.clip(np.round(lut_fp32 * 127.0), 0, 127).astype(np.int8)
    return q, scale


def write_coe_file(lut_int8: np.ndarray, path: Path) -> None:
    """
    Xilinx BRAM .coe file. Each entry is one byte; we lay them out
    row-major so address = class*14 + task. This matches the Verilog
    side using $readmemh-style flat addressing.
    """
    flat = lut_int8.flatten().astype(np.uint8)  # write as unsigned byte hex
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        hex_words = [f"{b:02x}" for b in flat]
        f.write(",\n".join(hex_words))
        f.write(";\n")


def write_mem_file(lut_int8: np.ndarray, path: Path) -> None:
    """ $readmemh-style flat dump, one byte per line. """
    flat = lut_int8.flatten().astype(np.uint8)
    with open(path, "w") as f:
        for b in flat:
            f.write(f"{b:02x}\n")


def write_human_table(lut_fp32: np.ndarray, path: Path) -> None:
    """ Class x Task table for humans to eyeball. """
    rows = []
    for c_idx, cname in enumerate(COCO80_NAMES):
        row = {"class": cname}
        row.update({TASK_NAMES[t]: round(float(lut_fp32[c_idx][t]), 3)
                    for t in range(NUM_TASKS)})
        rows.append(row)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


# ---------- Entry point ----------

def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--coco-tasks-dir", type=str,
                     help="Directory containing 1.json .. 14.json")
    src.add_argument("--synthetic", action="store_true",
                     help="Use synthetic affinities (bring-up only)")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        lut_fp32, stats = build_synthetic_lut()
    else:
        lut_fp32, stats = build_lut_from_coco_tasks(Path(args.coco_tasks_dir))

    lut_int8, scale = quantize_int8(lut_fp32)

    # Save artifacts
    np.save(out_dir / "affordance_lut_fp32.npy", lut_fp32)
    np.save(out_dir / "affordance_lut_int8.npy", lut_int8)
    write_coe_file(lut_int8, out_dir / "affordance_lut.coe")
    write_mem_file(lut_int8, out_dir / "affordance_lut.mem")
    write_human_table(lut_fp32, out_dir / "affordance_lut.json")

    stats["scale_int8_to_fp"] = scale
    stats["bytes_total"] = int(lut_int8.size)
    stats["shape"] = list(lut_int8.shape)
    with open(out_dir / "lut_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print a small visual summary
    print(f"\nLUT built: source={stats['source']}, "
          f"shape={lut_int8.shape}, bytes={lut_int8.size}")
    print(f"Saved to: {out_dir.resolve()}\n")

    # Show the strongest preferences per task (sanity check)
    print("Top class per task (sanity check):")
    for t_idx, tname in enumerate(TASK_NAMES):
        col = lut_fp32[:, t_idx]
        if col.max() == 0:
            print(f"  {tname:30s}  <empty>")
            continue
        top3 = np.argsort(col)[::-1][:3]
        line = ", ".join(f"{COCO80_NAMES[c]}={col[c]:.2f}" for c in top3)
        print(f"  {tname:30s}  {line}")


if __name__ == "__main__":
    main()