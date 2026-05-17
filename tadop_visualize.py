"""
TADOP — Visualizer
==================

Reads tadop_result.json + the original image, draws ranked boxes:
  - Top-1 detection (best for task): bright green, thick border, gold label
  - Top-2 to top-K: yellow
  - Surviving but lower-ranked: dim cyan
  - Masked-out (excluded by task class mask): faded red, dashed border

This is what we'll use to demo the system: feed it (image, task), get a
JPEG showing the system's reasoning.

Usage:
    python tadop_visualize.py \\
        --result ./data/tadop_out/<image_id>__task10_serve_wine/tadop_result.json \\
        --out-dir ./data/tadop_viz
"""

import argparse
import json
from pathlib import Path

import numpy as np


def draw_results(image_path: Path, record: dict, out_path: Path,
                 top_k: int = 5):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit("Install Pillow:  pip install Pillow")

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Font: try a real one, fall back to default
    try:
        font_big = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 22)
        font_small = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    detections = record["ranked_detections"]
    survivors = [d for d in detections if not d["masked_out"]]
    masked = [d for d in detections if d["masked_out"]]

    # Masked-out boxes first (so survivors draw on top)
    for d in masked:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=(180, 60, 60, 140), width=2)
        label = f"✗ {d['class_name']} (LUT={d['lut_prior']:.2f})"
        draw.text((x1 + 3, y1 + 3), label, fill=(180, 60, 60), font=font_small)

    # Surviving detections, ranked
    palette = [
        (60, 220, 90),    # rank 1: bright green
        (245, 200, 50),   # rank 2: gold
        (245, 200, 50),   # rank 3: gold
        (90, 200, 220),   # rank 4+: cyan
        (90, 200, 220),
    ]
    for rank, d in enumerate(survivors[:top_k]):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        color = palette[min(rank, len(palette) - 1)]
        thick = 5 if rank == 0 else 3
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thick)

        # Label
        label = (
            f"#{rank+1}  {d['class_name']}  "
            f"score={d['tadop_score']:.3f}\n"
            f"  conf={d['confidence']:.2f}  "
            f"LUT={d['lut_prior']:.2f}  cos={d['cos_sim']:+.2f}"
        )
        # Label background
        font = font_big if rank == 0 else font_small
        bbox_label = draw.multiline_textbbox(
            (x1 + 4, y1 + 4), label, font=font, spacing=2)
        draw.rectangle(
            [bbox_label[0] - 3, bbox_label[1] - 2,
             bbox_label[2] + 3, bbox_label[3] + 2],
            fill=(0, 0, 0, 200))
        draw.multiline_text(
            (x1 + 4, y1 + 4), label, fill=color,
            font=font, spacing=2)

    # Header strip with task info
    header = (
        f"Task {record['task_id_1based']}: {record['task_name']}    "
        f"({record['num_raw_detections']} dets → "
        f"{record['num_after_mask']} after mask)"
    )
    hbox = draw.textbbox((0, 0), header, font=font_big)
    draw.rectangle([0, 0, img.width, hbox[3] + 8],
                   fill=(0, 0, 0, 220))
    draw.text((10, 5), header, fill=(255, 255, 255), font=font_big)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--result", type=str,
                     help="A single tadop_result.json file")
    src.add_argument("--result-dir", type=str,
                     help="Directory containing per-image subfolders "
                          "with tadop_result.json files")
    ap.add_argument("--out-dir", type=str, default="./data/tadop_viz")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.result:
        results = [Path(args.result)]
    else:
        rd = Path(args.result_dir)
        results = sorted(rd.glob("*/tadop_result.json"))
        if not results:
            raise SystemExit(f"No tadop_result.json files found under {rd}")

    print(f"Rendering {len(results)} result(s)...")
    for rp in results:
        record = json.loads(rp.read_text())
        image_path = Path(record["image_path"])
        if not image_path.exists():
            print(f"  SKIP: image missing for {rp} ({image_path})")
            continue
        out_name = f"{record['image_id']}__task{record['task_id_1based']:02d}_{record['task_name']}.jpg"
        draw_results(image_path, record, out_dir / out_name, top_k=args.top_k)

    print(f"\nDone. Visualizations in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
