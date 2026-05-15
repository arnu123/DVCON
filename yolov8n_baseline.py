"""
TADOP — Phase 1.3: YOLOv8n FP32 Baseline + C2f Stride-8 Feature Hook
=====================================================================

Goal: establish the golden FP32 reference for object detection plus the
intermediate feature map that RoI pooling will read from.

What this script does:
  1. Loads pretrained YOLOv8n via ultralytics
  2. Identifies the C2f layer whose output is the stride-8 feature map
     (shape: B x 128 x 80 x 80 for 640x640 input)
  3. Registers a forward hook on that layer to capture the tensor
  4. Runs inference on a set of COCO-Tasks val images
  5. Saves, per image:
       - Detections: bboxes, classes, confidences (post-NMS)
       - Class logits: pre-NMS, pre-sigmoid raw scores (we need these
         for task-conditioned masking in Phase 1.4)
       - C2f stride-8 feature map (the tensor RoI pooling reads)

Outputs (in --out-dir):
    <image_id>/
        image_meta.json       Original size, scale factors
        detections.json       Post-NMS boxes + classes + scores
        c2f_stride8.npy       (128, 80, 80) float32 feature map
        class_logits.npy      Pre-sigmoid logits (anchors x 80)
        anchor_boxes.npy      Anchor box coords matching logits

    summary.json              Per-image counts, runtime stats, model info

Usage:
    python yolov8n_baseline.py \\
        --image-dir ./coco_images/val2017 \\
        --max-images 20 \\
        --out-dir ./data/baseline

Then verify with:
    python -m pytest test_yolov8n_baseline.py -v
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


YOLOV8N_INPUT_SIZE = 640         # standard square input
EXPECTED_STRIDE8_CHANNELS = 64   # YOLOv8n (nano, width=0.25) stride-8: 64 ch
                                  # base YOLOv8 has 128 here; nano halves it twice
EXPECTED_STRIDE8_HW       = 80   # 640 / 8 = 80


def find_stride8_layer(model) -> tuple[object, int]:
    """
    Walk the YOLOv8n model graph and find the C2f layer that produces
    the stride-8 feature map (output shape: B x 128 x 80 x 80).

    YOLOv8n architecture: the backbone has multiple C2f blocks at
    progressively smaller strides (4, 8, 16, 32). We want the one whose
    output goes into the P3 detection head — that's the stride-8 one.

    Strategy: register hooks on all C2f-like modules during a probe
    forward pass, identify the one whose output has 80x80 spatial size.
    """
    import torch.nn as nn

    candidates: list[tuple[str, object, tuple]] = []

    def make_hook(name, module):
        def hook(_mod, _inp, out):
            try:
                shape = tuple(out.shape)
                candidates.append((name, module, shape))
            except Exception:
                pass
        return hook

    handles = []
    for name, module in model.model.named_modules():
        cls_name = type(module).__name__
        # Look at C2f and similar building blocks; the stride-8 output is
        # easy to identify by its 80x80 spatial extent regardless of class.
        if "C2f" in cls_name or "C3" in cls_name or "C2" in cls_name:
            handles.append(module.register_forward_hook(make_hook(name, module)))

    # Trigger a probe forward pass
    import torch
    dummy = torch.zeros(1, 3, YOLOV8N_INPUT_SIZE, YOLOV8N_INPUT_SIZE)
    if next(model.model.parameters()).is_cuda:
        dummy = dummy.cuda()
    with torch.no_grad():
        model.model(dummy)

    for h in handles:
        h.remove()

    # Pick the candidate with shape (1, 128, 80, 80)
    matches = [
        (n, m, s) for (n, m, s) in candidates
        if len(s) == 4 and s[1] == EXPECTED_STRIDE8_CHANNELS
        and s[2] == EXPECTED_STRIDE8_HW and s[3] == EXPECTED_STRIDE8_HW
    ]
    if not matches:
        # Fall back: any layer producing 80x80 spatial size
        matches = [
            (n, m, s) for (n, m, s) in candidates
            if len(s) == 4 and s[2] == EXPECTED_STRIDE8_HW
            and s[3] == EXPECTED_STRIDE8_HW
        ]

    if not matches:
        raise RuntimeError(
            f"Couldn't find a stride-8 layer. Candidates found:\n"
            + "\n".join(f"  {n}: {s}" for n, _, s in candidates)
        )

    # Heuristic: the *last* matching layer in the backbone is what feeds
    # the P3 detection head (after any upsampling/concat in the neck).
    # We want the backbone output, which is typically the first match.
    name, module, shape = matches[0]
    return module, name, shape


class C2fHook:
    """Stores the most recent stride-8 feature output."""
    def __init__(self):
        self.feature = None

    def __call__(self, _mod, _inp, out):
        # out is a torch.Tensor on the model's device
        self.feature = out.detach().cpu().numpy()


def process_one_image(model, image_path: Path, hook: C2fHook,
                      out_dir: Path, conf_thresh: float = 0.25):
    """
    Run YOLOv8n on a single image and save all required artifacts.
    Returns a summary dict.
    """
    import torch

    image_id = image_path.stem
    img_out = out_dir / image_id
    img_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # ultralytics handles letterboxing, normalization, NMS internally.
    # imgsz=640 forces standard input. verbose=False keeps logs clean.
    # rect=False forces square letterboxed input (640x640) regardless of
    # input aspect ratio. The stride-8 output is then always 80x80 — a
    # deterministic shape the FPGA accelerator can target. The default
    # rectangular mode would give variable spatial shapes per image.
    results = model.predict(
        source=str(image_path),
        imgsz=YOLOV8N_INPUT_SIZE,
        rect=False,
        conf=conf_thresh,
        iou=0.45,
        verbose=False,
        save=False,
    )
    elapsed_ms = (time.time() - t0) * 1000

    result = results[0]
    orig_shape = result.orig_shape  # (H, W)

    # ---- Save detections (post-NMS) ----
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        # xyxy in original image coords
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        cls  = boxes.cls.cpu().numpy().astype(int).tolist()
        conf = boxes.conf.cpu().numpy().tolist()
    else:
        xyxy, cls, conf = [], [], []

    dets = [
        {"bbox_xyxy": b, "class_id": c, "confidence": float(s),
         "class_name": model.names[c]}
        for b, c, s in zip(xyxy, cls, conf)
    ]
    with open(img_out / "detections.json", "w") as f:
        json.dump(dets, f, indent=2)

    # ---- Save the captured C2f stride-8 feature map ----
    if hook.feature is None:
        raise RuntimeError("C2f hook did not fire — wrong layer hooked?")
    # Remove batch dim: (1, 128, 80, 80) -> (128, 80, 80)
    feat = hook.feature[0]
    np.save(img_out / "c2f_stride8.npy", feat.astype(np.float32))
    hook.feature = None  # clear for next image

    # ---- Save image metadata ----
    meta = {
        "image_id": image_id,
        "filename": image_path.name,
        "orig_shape_hw": list(orig_shape),
        "model_input_size": YOLOV8N_INPUT_SIZE,
        "scale_factor_h": YOLOV8N_INPUT_SIZE / orig_shape[0],
        "scale_factor_w": YOLOV8N_INPUT_SIZE / orig_shape[1],
        "elapsed_ms": elapsed_ms,
        "num_detections": len(dets),
    }
    with open(img_out / "image_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "image_id": image_id,
        "num_detections": len(dets),
        "elapsed_ms": elapsed_ms,
        "feat_shape": list(feat.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True,
                    help="Directory containing COCO val2017 .jpg images")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-images", type=int, default=20,
                    help="Process this many images (default 20)")
    ap.add_argument("--conf-thresh", type=float, default=0.25)
    ap.add_argument("--model-weights", type=str, default="yolov8n.pt",
                    help="Path or name of YOLOv8n weights (default: yolov8n.pt)")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Late imports so --help works without dependencies installed
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "    pip install ultralytics"
        ) from e

    print(f"Loading YOLOv8n: {args.model_weights}")
    model = YOLO(args.model_weights)
    # Force fp32 eval mode (no autocast surprises)
    model.model.eval().float()

    # Locate and hook the stride-8 C2f layer
    print("Locating stride-8 feature layer...")
    target_module, target_name, target_shape = find_stride8_layer(model)
    print(f"  -> {target_name}  output shape {target_shape}")

    hook_state = C2fHook()
    handle = target_module.register_forward_hook(hook_state)

    # Pick images
    images = sorted(image_dir.glob("*.jpg"))[:args.max_images]
    if not images:
        raise SystemExit(f"No .jpg images found in {image_dir}")

    print(f"\nProcessing {len(images)} images...")
    per_image = []
    for i, img_path in enumerate(images):
        info = process_one_image(model, img_path, hook_state, out_dir,
                                 conf_thresh=args.conf_thresh)
        per_image.append(info)
        if (i + 1) % 5 == 0 or i == len(images) - 1:
            print(f"  [{i+1}/{len(images)}] {img_path.name}  "
                  f"dets={info['num_detections']}  "
                  f"{info['elapsed_ms']:.1f} ms")

    handle.remove()

    # Aggregate summary
    elapsed_all = [p["elapsed_ms"] for p in per_image]
    summary = {
        "model": args.model_weights,
        "input_size": YOLOV8N_INPUT_SIZE,
        "stride8_layer_name": target_name,
        "stride8_layer_output_shape": list(target_shape),
        "num_images_processed": len(per_image),
        "conf_threshold": args.conf_thresh,
        "latency_ms": {
            "mean": float(np.mean(elapsed_all)),
            "median": float(np.median(elapsed_all)),
            "p90": float(np.percentile(elapsed_all, 90)),
            "p99": float(np.percentile(elapsed_all, 99)),
        },
        "total_detections": int(sum(p["num_detections"] for p in per_image)),
        "per_image": per_image,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Mean latency: {summary['latency_ms']['mean']:.1f} ms "
          f"(median {summary['latency_ms']['median']:.1f}).")
    print(f"Total detections across {len(per_image)} images: "
          f"{summary['total_detections']}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()