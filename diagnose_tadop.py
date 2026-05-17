"""
Diagnose why no detections survived the task mask.

Prints, per image:
  - All raw detections with their class names
  - The LUT prior for each detection's class under the chosen task
  - Whether it would survive the mask
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tadop_reference import (
    MASK_THRESHOLD, TASK_NAMES, NUM_CLASSES, NUM_TASKS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--task", type=int, required=True, help="1..14")
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--lut-path", default="affordance_lut_fp32.npy")
    args = ap.parse_args()

    lut = np.load(args.lut_path)
    task_idx = args.task - 1
    task_name = TASK_NAMES[task_idx]

    print(f"=== LUT slice for task {args.task} ({task_name}) ===")
    print(f"Threshold: {MASK_THRESHOLD}")
    print(f"Classes with LUT >= threshold:")
    survivors = np.where(lut[:, task_idx] >= MASK_THRESHOLD)[0]
    for c in survivors:
        print(f"  class {c}: prior={lut[c, task_idx]:.3f}")
    print(f"Total: {len(survivors)} classes survive masking\n")

    # Now run yolo on the images and check
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    images = sorted(Path(args.image_dir).glob("*.jpg"))[:args.max_images]
    total_dets = 0
    total_would_survive = 0
    class_appearances = {}
    for img_path in images:
        res = model.predict(str(img_path), imgsz=640, rect=False,
                            conf=0.25, iou=0.45, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        print(f"--- {img_path.name} ---")
        for cid, conf in zip(cls_ids, confs):
            cname = model.names[int(cid)]
            prior = lut[cid, task_idx]
            survive = prior >= MASK_THRESHOLD
            class_appearances[cname] = class_appearances.get(cname, 0) + 1
            marker = "OK " if survive else "BLK"
            print(f"  {marker}  class={cname:<18}  conf={conf:.2f}  LUT={prior:.3f}")
            total_dets += 1
            if survive:
                total_would_survive += 1

    print(f"\n=== Summary ===")
    print(f"Total detections: {total_dets}")
    print(f"Would survive mask: {total_would_survive}")
    print(f"\nClass frequencies across these images:")
    for cname, count in sorted(class_appearances.items(), key=lambda x: -x[1])[:15]:
        prior = "?"
        for cid, n in model.names.items():
            if n == cname:
                prior = f"{lut[cid, task_idx]:.3f}"
                break
        print(f"  {cname:<18}  appearances={count}  task_prior={prior}")


if __name__ == "__main__":
    main()
