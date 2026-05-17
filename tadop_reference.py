"""
TADOP — Phase 1.4: Two-Stage Scorer (Golden Reference)
=======================================================

End-to-end software reference: takes an image and a task, returns the
ranked detections most suitable for that task.

This script ties together every Phase 1 artifact built so far:
  - Phase 1.1: 80x14 affordance LUT (data/lut/affordance_lut_fp32.npy)
  - Phase 1.2: 14x384 task embeddings (data/embeddings/task_embeddings_fp32.npy)
  - Phase 1.3: YOLOv8n FP32 + C2f stride-8 hook

For each detection, computes:

    score = conf × LUT[class, task] × max(0, cos_sim(W·roi, task_emb[task]))

then applies task-conditioned class masking (zeros out classes whose LUT
entry is below MASK_THRESHOLD) and ranks the survivors.

The W projection (64 → 384) is initialized as a "fixed random projection"
(seeded, deterministic). In Phase 1.5 we'll train it on COCO-Tasks. For
now the random W gives a stable, reproducible signal that exercises the
full pipeline end-to-end.

Usage:
    # Score a single image for a single task
    python tadop_reference.py \\
        --image ./coco_images/val2017/000000000139.jpg \\
        --task 10 \\
        --lut-dir . \\
        --embeddings-dir . \\
        --out-dir ./data/tadop_out

    # Or by task name
    python tadop_reference.py \\
        --image ./coco_images/val2017/000000000139.jpg \\
        --task-name serve_wine \\
        ...

    # Or batch over many images for one task
    python tadop_reference.py \\
        --image-dir ./coco_images/val2017 \\
        --max-images 10 \\
        --task 10 \\
        ...
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------- Constants matching the rest of Phase 1 ----------

YOLOV8N_INPUT_SIZE = 640
STRIDE8_DOWNSAMPLE = 8
STRIDE8_GRID       = YOLOV8N_INPUT_SIZE // STRIDE8_DOWNSAMPLE  # 80
STRIDE8_CHANNELS   = 64    # YOLOv8n nano
EMBED_DIM          = 384   # MiniLM-L6
NUM_CLASSES        = 80
NUM_TASKS          = 14

# Task-conditioned class mask threshold. From the Stage 1 design:
# any class with LUT[c, task] < MASK_THRESHOLD is suppressed for that task.
MASK_THRESHOLD = 0.10

TASK_NAMES = [
    "step_on_something", "sit_comfortably", "place_flowers",
    "get_potatoes_out_of_fire", "water_plant", "get_lemon_out_of_tea",
    "dig_hole", "open_bottle_of_beer", "open_parcel", "serve_wine",
    "pour_sugar", "smear_butter", "extinguish_fire", "pound_carpet",
]


# ---------- W projection (64 -> 384) ----------

def make_W(seed: int = 1729) -> np.ndarray:
    """
    Fixed random projection from RoI feature space (64-D) to task
    embedding space (384-D). Each output dim is a different random
    linear combination of the 64 input features. Seeded for
    reproducibility — running this script tomorrow gives the same W.

    Phase 1.5 will replace this with a trained W. The interface (shape,
    dtype, file location) stays identical, so swap-in is a one-line
    np.load() change.
    """
    rng = np.random.default_rng(seed)
    # Xavier-ish initialization: variance scaled so output variance
    # matches input variance.
    W = rng.standard_normal((64, EMBED_DIM)).astype(np.float32)
    W /= np.sqrt(64.0)
    return W


# ---------- Loading the priors ----------

def load_priors(lut_dir: Path, embeddings_dir: Path):
    """Load the LUT, task embeddings, and W projection."""
    lut_path = lut_dir / "affordance_lut_fp32.npy"
    emb_path = embeddings_dir / "task_embeddings_fp32.npy"

    if not lut_path.exists():
        raise FileNotFoundError(
            f"LUT not found at {lut_path}. Run build_affordance_lut.py first."
        )
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Task embeddings not found at {emb_path}. "
            f"Run build_task_embeddings.py first."
        )

    lut = np.load(lut_path)           # (80, 14) fp32
    task_embs = np.load(emb_path)     # (14, 384) fp32, L2-normalized

    assert lut.shape == (NUM_CLASSES, NUM_TASKS), f"bad LUT shape {lut.shape}"
    assert task_embs.shape == (NUM_TASKS, EMBED_DIM), \
        f"bad task embeddings shape {task_embs.shape}"

    W = make_W()
    return lut, task_embs, W


# ---------- YOLOv8n setup (reuses Phase 1.3 logic) ----------

def find_stride8_layer(model):
    """ Find the C2f layer producing the (1, 64, 80, 80) stride-8 feature. """
    import torch
    candidates = []

    def make_hook(name, module):
        def hook(_mod, _inp, out):
            try:
                candidates.append((name, module, tuple(out.shape)))
            except Exception:
                pass
        return hook

    handles = []
    for name, module in model.model.named_modules():
        cls_name = type(module).__name__
        if "C2f" in cls_name or "C3" in cls_name or "C2" in cls_name:
            handles.append(module.register_forward_hook(make_hook(name, module)))

    dummy = torch.zeros(1, 3, YOLOV8N_INPUT_SIZE, YOLOV8N_INPUT_SIZE)
    if next(model.model.parameters()).is_cuda:
        dummy = dummy.cuda()
    with torch.no_grad():
        model.model(dummy)
    for h in handles:
        h.remove()

    matches = [
        (n, m, s) for (n, m, s) in candidates
        if len(s) == 4 and s[1] == STRIDE8_CHANNELS
        and s[2] == STRIDE8_GRID and s[3] == STRIDE8_GRID
    ]
    if not matches:
        raise RuntimeError(
            f"No stride-8 candidate found. Got:\n"
            + "\n".join(f"  {n}: {s}" for n, _, s in candidates)
        )
    return matches[0]  # (name, module, shape)


class FeatureCapture:
    def __init__(self):
        self.feature = None
    def __call__(self, _m, _i, out):
        self.feature = out.detach().cpu().numpy()


# ---------- RoI feature extraction ----------

def extract_roi_features(feature_map: np.ndarray,
                         boxes_xyxy: np.ndarray,
                         orig_h: int,
                         orig_w: int) -> np.ndarray:
    """
    For each detection box, average-pool the (C, H, W) feature map over
    the box's spatial extent.

    Steps:
      1. Convert box from original image coords to letterboxed 640x640 coords
      2. Convert from 640x640 coords to feature-map coords (divide by 8)
      3. Mean over that region for each of the 64 channels

    Letterboxing detail: ultralytics rect=False scales the longer side
    to 640 and pads the shorter side symmetrically. We replicate the
    transform here so feature-space coordinates match the model's view.

    Returns (N, 64) float32.
    """
    C, H, W = feature_map.shape
    assert (H, W) == (STRIDE8_GRID, STRIDE8_GRID), \
        f"expected {STRIDE8_GRID}x{STRIDE8_GRID} feature, got {H}x{W}"

    # Letterbox math: scale so longer side fits in 640, pad the other
    scale = YOLOV8N_INPUT_SIZE / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    pad_h = (YOLOV8N_INPUT_SIZE - new_h) // 2
    pad_w = (YOLOV8N_INPUT_SIZE - new_w) // 2

    if len(boxes_xyxy) == 0:
        return np.zeros((0, C), dtype=np.float32)

    rois = np.zeros((len(boxes_xyxy), C), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        # original -> letterboxed 640x640
        x1_lb = x1 * scale + pad_w
        x2_lb = x2 * scale + pad_w
        y1_lb = y1 * scale + pad_h
        y2_lb = y2 * scale + pad_h
        # 640x640 -> feature grid (80x80)
        fx1 = max(0, int(np.floor(x1_lb / STRIDE8_DOWNSAMPLE)))
        fx2 = min(W, int(np.ceil(x2_lb / STRIDE8_DOWNSAMPLE)))
        fy1 = max(0, int(np.floor(y1_lb / STRIDE8_DOWNSAMPLE)))
        fy2 = min(H, int(np.ceil(y2_lb / STRIDE8_DOWNSAMPLE)))
        # Guard against degenerate boxes
        if fx2 <= fx1 or fy2 <= fy1:
            rois[i] = feature_map[:, fy1:fy1+1, fx1:fx1+1].mean(axis=(1, 2))
        else:
            rois[i] = feature_map[:, fy1:fy2, fx1:fx2].mean(axis=(1, 2))
    return rois


# ---------- The scorer itself ----------

def score_detections(
    detections: list,
    rois: np.ndarray,
    task_id: int,             # 0-indexed (i.e. task_id 0 = task 1 in COCO-Tasks)
    lut: np.ndarray,
    task_embs: np.ndarray,
    W: np.ndarray,
    mask_threshold: float = MASK_THRESHOLD,
) -> list:
    """
    Apply task-conditioned masking and two-stage scoring.

    Returns a list of dicts, sorted by score descending, each containing
    all input fields plus: lut_prior, cos_sim, tadop_score, masked_out.
    """
    task_emb = task_embs[task_id]                 # (384,)
    class_mask = lut[:, task_id] >= mask_threshold  # (80,) bool

    # Project all RoIs into task-embedding space, L2-normalize for cosine.
    # We L2-normalize the RoI vector first because raw C2f activations
    # (post-SiLU) can have very large magnitudes that overflow fp32 when
    # multiplied by W. Pre-normalizing is also closer to what the FPGA
    # will do: it operates on quantized features with a known scale.
    if len(rois) > 0:
        roi_norms = np.linalg.norm(rois, axis=1, keepdims=True)
        roi_norms = np.maximum(roi_norms, 1e-9)
        rois_normed = (rois / roi_norms).astype(np.float32)
        proj = rois_normed @ W                      # (N, 384)
        # Guard against any residual NaN/Inf from extreme edge cases
        proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        proj_normed = proj / norms
        cos_sims = (proj_normed @ task_emb).astype(np.float32)
    else:
        cos_sims = np.zeros(0, dtype=np.float32)

    scored = []
    for i, det in enumerate(detections):
        cls = int(det["class_id"])
        conf = float(det["confidence"])
        lut_prior = float(lut[cls, task_id])
        cos = float(cos_sims[i])
        # ReLU on cosine: negative similarity → 0 contribution.
        cos_pos = max(cos, 0.0)
        masked = not class_mask[cls]
        score = 0.0 if masked else (conf * lut_prior * cos_pos)
        scored.append({
            **det,
            "lut_prior": lut_prior,
            "cos_sim": cos,
            "tadop_score": score,
            "masked_out": masked,
        })

    scored.sort(key=lambda x: x["tadop_score"], reverse=True)
    return scored


# ---------- Main: per-image pipeline ----------

def process_image(model, hook: FeatureCapture, image_path: Path,
                  task_id: int, task_emb_idx: int,
                  lut: np.ndarray, task_embs: np.ndarray, W: np.ndarray,
                  out_dir: Path, conf_thresh: float = 0.25) -> dict:
    """Run the full pipeline for one (image, task) pair."""
    image_id = image_path.stem
    img_out = out_dir / f"{image_id}__task{task_id+1:02d}_{TASK_NAMES[task_id]}"
    img_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = model.predict(
        source=str(image_path),
        imgsz=YOLOV8N_INPUT_SIZE,
        rect=False,
        conf=conf_thresh,
        iou=0.45,
        verbose=False,
        save=False,
    )
    yolo_ms = (time.time() - t0) * 1000

    result = results[0]
    orig_h, orig_w = result.orig_shape

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        detections = [
            {"bbox_xyxy": b.tolist(), "class_id": int(c),
             "confidence": float(s), "class_name": model.names[int(c)]}
            for b, c, s in zip(xyxy, cls, conf)
        ]
    else:
        xyxy = np.zeros((0, 4), dtype=np.float32)
        detections = []

    # Capture C2f feature, extract RoIs
    if hook.feature is None:
        raise RuntimeError("C2f hook never fired")
    feat = hook.feature[0]   # (64, 80, 80)
    hook.feature = None

    t1 = time.time()
    rois = extract_roi_features(feat, xyxy, orig_h, orig_w)
    roi_ms = (time.time() - t1) * 1000

    t2 = time.time()
    scored = score_detections(detections, rois, task_id, lut, task_embs, W)
    score_ms = (time.time() - t2) * 1000

    # Save artifacts
    out_record = {
        "image_id": image_id,
        "image_path": str(image_path),
        "task_id_1based": task_id + 1,
        "task_name": TASK_NAMES[task_id],
        "orig_shape_hw": [int(orig_h), int(orig_w)],
        "num_raw_detections": len(detections),
        "num_after_mask": int(sum(1 for s in scored if not s["masked_out"])),
        "timing_ms": {
            "yolo_inference": yolo_ms,
            "roi_extraction": roi_ms,
            "scoring": score_ms,
            "tadop_overhead": roi_ms + score_ms,
            "total": yolo_ms + roi_ms + score_ms,
        },
        "ranked_detections": scored,
    }
    with open(img_out / "tadop_result.json", "w") as f:
        json.dump(out_record, f, indent=2)
    np.save(img_out / "c2f_stride8.npy", feat)
    np.save(img_out / "roi_features.npy", rois)

    return out_record


# ---------- Pretty printing ----------

def print_ranked(record: dict, top_k: int = 5):
    print(f"\n{'='*70}")
    print(f"Image: {record['image_id']}   "
          f"Task {record['task_id_1based']}: {record['task_name']}")
    print(f"Detections: {record['num_raw_detections']} raw -> "
          f"{record['num_after_mask']} after task mask")
    t = record["timing_ms"]
    print(f"Timing: YOLO {t['yolo_inference']:.1f}ms  "
          f"+ RoI {t['roi_extraction']:.2f}ms  "
          f"+ score {t['scoring']:.2f}ms  = {t['total']:.1f}ms")
    print(f"{'-'*70}")
    print(f"  {'rank':<5}{'class':<18}{'conf':>7}{'LUT':>7}"
          f"{'cos':>8}{'score':>10}  flags")
    survivors = [s for s in record["ranked_detections"] if not s["masked_out"]]
    if not survivors:
        print("  (no detections survived the task class mask)")
        return
    for i, s in enumerate(survivors[:top_k]):
        print(f"  {i+1:<5}{s['class_name']:<18}"
              f"{s['confidence']:>7.2f}{s['lut_prior']:>7.2f}"
              f"{s['cos_sim']:>+8.2f}{s['tadop_score']:>10.4f}")


# ---------- Entry ----------

def main():
    ap = argparse.ArgumentParser()
    img_group = ap.add_mutually_exclusive_group(required=True)
    img_group.add_argument("--image", type=str, help="Single image path")
    img_group.add_argument("--image-dir", type=str, help="Directory of .jpg files")

    task_group = ap.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", type=int, help="Task ID 1..14")
    task_group.add_argument("--task-name", type=str, help="e.g. serve_wine")

    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--lut-dir", type=str, default=".")
    ap.add_argument("--embeddings-dir", type=str, default=".")
    ap.add_argument("--out-dir", type=str, default="./data/tadop_out")
    ap.add_argument("--model-weights", type=str, default="yolov8n.pt")
    ap.add_argument("--conf-thresh", type=float, default=0.25)
    args = ap.parse_args()

    # Resolve task index (0-based internally)
    if args.task is not None:
        if not (1 <= args.task <= NUM_TASKS):
            raise SystemExit(f"--task must be 1..{NUM_TASKS}")
        task_id = args.task - 1
    else:
        try:
            task_id = TASK_NAMES.index(args.task_name)
        except ValueError:
            raise SystemExit(
                f"Unknown task name {args.task_name!r}. Valid names:\n  "
                + "\n  ".join(TASK_NAMES)
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load priors
    lut, task_embs, W = load_priors(Path(args.lut_dir), Path(args.embeddings_dir))
    print(f"Loaded LUT {lut.shape}, embeddings {task_embs.shape}, "
          f"W {W.shape}")
    print(f"Task: {task_id+1} ({TASK_NAMES[task_id]})")
    top_classes = np.argsort(lut[:, task_id])[::-1][:5]
    print(f"LUT top classes for this task:")
    for c in top_classes:
        if lut[c, task_id] > 0:
            from ultralytics import YOLO  # lazy import for class names
            tmp = YOLO(args.model_weights)
            cname = tmp.names[int(c)]
            print(f"  {cname:<18} {lut[c, task_id]:.3f}")
            del tmp
            break  # only need names once

    # Load model + hook
    from ultralytics import YOLO
    model = YOLO(args.model_weights)
    model.model.eval().float()
    target_name, target_module, target_shape = find_stride8_layer(model)
    print(f"\nHooking stride-8 layer: {target_name}  shape={target_shape}")
    hook = FeatureCapture()
    handle = target_module.register_forward_hook(hook)

    # Gather image paths
    if args.image is not None:
        image_paths = [Path(args.image)]
    else:
        image_paths = sorted(Path(args.image_dir).glob("*.jpg"))[:args.max_images]

    if not image_paths:
        raise SystemExit("No images to process")

    print(f"Processing {len(image_paths)} image(s)...")
    summaries = []
    for p in image_paths:
        rec = process_image(model, hook, p, task_id, task_id,
                            lut, task_embs, W, out_dir,
                            conf_thresh=args.conf_thresh)
        summaries.append(rec)
        print_ranked(rec, top_k=5)

    handle.remove()

    # Aggregate
    agg = {
        "task_id_1based": task_id + 1,
        "task_name": TASK_NAMES[task_id],
        "num_images": len(summaries),
        "mean_raw_detections": float(np.mean(
            [s["num_raw_detections"] for s in summaries])),
        "mean_after_mask": float(np.mean(
            [s["num_after_mask"] for s in summaries])),
        "mean_total_latency_ms": float(np.mean(
            [s["timing_ms"]["total"] for s in summaries])),
        "mean_tadop_overhead_ms": float(np.mean(
            [s["timing_ms"]["tadop_overhead"] for s in summaries])),
    }
    with open(out_dir / "tadop_summary.json", "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Aggregate over {agg['num_images']} images:")
    print(f"  Mean raw detections per image: {agg['mean_raw_detections']:.1f}")
    print(f"  Mean surviving task mask:      {agg['mean_after_mask']:.1f}  "
          f"({100*agg['mean_after_mask']/max(agg['mean_raw_detections'],1):.0f}%)")
    print(f"  Mean TADOP overhead:           "
          f"{agg['mean_tadop_overhead_ms']:.2f} ms")
    print(f"  Mean total latency:            "
          f"{agg['mean_total_latency_ms']:.1f} ms")
    print(f"\nOutputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
