# TADOP Affordance LUT

Phase 1.1 of the Stage 2 build. Produces the 80×14 affordance Look-Up Table
that lives in BRAM on the FPGA and is read by VEGA at inference time.

## What this is

A precomputed matrix where `LUT[c][t]` is the empirical probability that
class `c` is the *preferred* object for task `t`, computed from COCO-Tasks
training annotations.

The Stage 1 design uses this as the class-level prior in the two-stage
scoring:

    score(instance) = conf × LUT[class] × cos_sim(W·roi_feat, task_emb)

## Files

    build_affordance_lut.py    Builder. Real-data and synthetic modes.
    test_affordance_lut.py     pytest suite. Locks in shape, dtype,
                               quantization round-trip, .coe/.mem format.
    README.md                  This file.

## Two modes

### Synthetic mode (works right now)

For pipeline bring-up before you have COCO-Tasks annotations on disk.
Hand-coded affinities calibrated to plausible values (wine_glass→serve_wine
near 0.94, etc.). **Do not use for any reported numbers.**

    python build_affordance_lut.py --synthetic --out-dir ../../data/lut

### Real mode (use this for the actual Stage 2 LUT)

Walks `1.json .. 14.json` from the official COCO-Tasks repo and computes
per-(class, task) preference fractions.

    # 1. Download COCO-Tasks annotations
    git clone https://github.com/coco-tasks/dataset.git
    # The annotation JSONs live in dataset/annotations/

    # 2. Build the LUT
    python build_affordance_lut.py \
        --coco-tasks-dir /path/to/dataset/annotations \
        --out-dir ../../data/lut

The aggregation rule is per-(image, class), not per-instance:

- denominator: # of (image, task) pairs in which the class appears at all
- numerator:   # of (image, task) pairs in which at least one instance
               of the class is flagged preferred

This matches the standard reading of "class-level affordance prior".

## Outputs (in --out-dir)

    affordance_lut_fp32.npy    80×14 float32 in [0,1]. Reference for the
                               PyTorch software pipeline.
    affordance_lut_int8.npy    80×14 int8 in [0,127]. Bit-exact match for
                               what VEGA will read back from BRAM.
                               scale = 1/127  (real = int8/127).
    affordance_lut.coe         Vivado BRAM init file. Use as the .coe in
                               the Block Memory Generator IP.
    affordance_lut.mem         Verilog $readmemh init for sim/Yosys flows.
    affordance_lut.json        Class×task table for human inspection.
    lut_stats.json             Aggregation stats. Source flag, image counts.

Address layout in flat memory: `addr = class_idx * 14 + task_idx` (row major).
The BRAM stores 1120 bytes — fits in a single BRAM36 with room to spare.

## Tests

    python -m pytest test_affordance_lut.py -v

Should report 8 passed. The tests are not just smoke checks — they enforce
the output contract that downstream RTL/VEGA code depends on. If you
change the int8 scale, the row-major layout, or the file formats, those
tests should be updated together with whatever consumes them.

## Known caveats

- Task name strings in `TASK_NAMES` are paraphrases of the official COCO-Tasks
  task descriptions. The MiniLM encoder will work with any phrasing — what
  matters for the LUT itself is the *task index*, not the string.
- Some forks of COCO-Tasks store the per-instance preference flag under
  `category_id` (0/1) and the original COCO class under `COCO_category_id`.
  The builder handles both schemas. If you hit a `KeyError`, dump one
  annotation and check which key holds the 0/1 flag.
- The synthetic generator's noise floor uses a fixed seed (42) so
  reruns are bit-identical. If you want randomized synthetic LUTs, expose
  the seed.

## Next steps in the pipeline

After this LUT is built, the remaining Phase 1 software work is:

1. **MiniLM cache** — encode the 14 task strings, save as 14×384 INT8.
2. **YOLOv8n FP32 baseline** — run pretrained model, hook the C2f stride-8
   layer for RoI features.
3. **Two-stage scorer** — combine LUT prior with RoI-task cosine similarity.
4. **Train W** (256→384 projection) on COCO-Tasks BCE preference labels.

The end-of-Phase-1 artifact is a single `tadop_reference.py` script that
takes (image, task_string) and outputs (bbox, class) — your golden model
for everything FPGA-side to validate against.
