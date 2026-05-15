# Phase 1.3 — YOLOv8n FP32 Baseline + C2f Stride-8 Hook · Completion Summary

**Status:** ✅ Complete
**Date completed:** Phase 1.3 of TADOP Stage 2 build
**Owner:** Arnav

---

## What this phase delivered

A working YOLOv8n FP32 detection pipeline with a forward hook on the
backbone's stride-8 C2f layer (`model.4`). For each input image the
pipeline produces:

- Post-NMS detections (bbox, class, confidence)
- The intermediate stride-8 feature map (the tensor RoI pooling reads from
  in the two-stage scorer)
- Image metadata (original size, scale factors, latency)

This is the **golden FP32 reference** that all downstream FPGA work will
be validated against. When Phase 3 RTL produces a detection or feature
map, we compare against these saved tensors to confirm bit-level (or
near-bit-level) match.

---

## Artifacts produced

Repo files (`Stage2/` root):

| File | Purpose |
|---|---|
| `yolov8n_baseline.py` | Main runner: loads model, hooks C2f, processes images |
| `test_yolov8n_baseline.py` | 9-test pytest suite locking in output contract |
| `conftest.py` | Registers `--baseline-dir` option for pytest |
| `get_coco_images.sh` | One-shot download of COCO val2017 (~1 GB) |

Per-image outputs (in `data/baseline/<image_id>/`):

| File | Shape / Format |
|---|---|
| `c2f_stride8.npy` | float32 `(64, 80, 80)` — the hooked feature map |
| `detections.json` | list of `{bbox_xyxy, class_id, confidence, class_name}` |
| `image_meta.json` | original size, scale factors, latency |

Aggregate output (`data/baseline/summary.json`):
- Model info, layer name, latency percentiles, per-image counts

---

## Validation results

### Test suite

All **9 of 9 tests pass** on the 20-image validation run:

- `test_summary_has_required_fields` — all expected keys present
- `test_stride8_shape_is_canonical` — confirms `(B, 64, 80, 80)`
- `test_each_image_has_required_artifacts` — 3 files per image
- `test_c2f_feature_shape_and_dtype` — `(64, 80, 80)` float32
- `test_c2f_features_are_finite` — no NaN/Inf
- `test_c2f_features_have_signal` — non-trivial activation variance
- `test_detections_format` — bbox/class/confidence ranges
- `test_image_meta_format` — metadata schema valid
- `test_latency_is_recorded` — latency captured, sane range

### Performance on Arnav's Mac (M-series CPU)

| Metric | Value |
|---|---|
| Mean latency | 50.4 ms |
| Median latency | 48.3 ms |
| Images processed | 20 |
| Total detections | 89 (mean 4.45 per image) |

This is **CPU baseline**, not a target. The FPGA accelerator's job in
Stage 2 Phase 5 will be to beat this latency. The reference value just
confirms the pipeline functions correctly.

---

## Key design decisions

### Force square 640×640 input (`rect=False`)

Initial test failures revealed that ultralytics' default rectangular
inference mode preserves aspect ratio, producing variable feature shapes
(e.g. `(64, 56, 80)` for a 426×640 image). For the FPGA accelerator we
need a deterministic feature shape. **`rect=False` forces square
letterboxing**, guaranteeing `(64, 80, 80)` for every image regardless
of input aspect ratio. Trade-off: ~10 ms extra latency from larger
input on landscape images. Accepted — predictable RTL > marginal CPU speedup.

### Auto-discovery of the stride-8 layer

The script walks the model graph and identifies the C2f layer producing
the stride-8 feature by **shape signature**, not by hardcoded name. This
makes the code robust to ultralytics module-name changes across versions.
The discovered layer is logged (`model.4` in current runs) and verified
in tests.

### Save pre-quantized FP32 features

Features are saved as float32, not pre-quantized to INT8. Rationale: the
quantization parameters for the activation path will be calibrated in
Phase 2.1 using these same images. Saving FP32 keeps the reference
"frozen" so we can re-quantize later without re-running YOLOv8n.

---

## Key insight surfaced during this phase: feature dimensions

**Stage 1 report claim:** Stride-8 feature is 80×80×**128**, RoI feature
vector is 256-D, W projection is **256→384**.

**Stage 2 reality:** Stride-8 feature is 80×80×**64**. YOLOv8n is the
*nano* variant with width multiplier 0.25 — its stride-8 backbone output
has 64 channels, not the base architecture's 128.

**Impact on the design:**
- RoI features (GAP over box extent in feature space) are **64-D**, not 256-D
- The W projection becomes **64→384** (or we add an intermediate
  expansion stage)
- Net effect: the projection is **4× cheaper** than budgeted — 64×384 =
  24,576 INT8 MACs per RoI vs the budgeted 256×384 = 98,304
- DSP utilization estimate in Stage 1 was conservative; we have margin

This is good news for the FPGA. It also frees up budget for things like
deeper masking logic or more aggressive RoI batching.

---

## Open items / future work

- **Class logits export.** Current detections.json captures post-NMS
  outputs only. For task-conditioned masking in Phase 1.4 we'll need
  pre-NMS, pre-sigmoid class logits. Will add to the hook in Phase 1.4
  when integrating with the scorer. Deferred because (a) it requires
  a second hook deeper in the head, and (b) we want to first decide
  whether to do masking pre- or post-NMS in the FPGA pipeline.
- **Real COCO val2017 mAP measurement.** Out of scope for this phase but
  worth a one-shot run with the standard YOLOv8 val command before
  Phase 2.1, to lock in the FP32 mAP baseline that INT8 PTQ will be
  compared against.

---

## Phase 1 progress so far

- ✅ Phase 1.1 — Affordance LUT (real COCO-Tasks data, 2 noisy entries clamped)
- ✅ Phase 1.2 — MiniLM task embedding cache (all checks pass)
- ✅ Phase 1.3 — YOLOv8n FP32 baseline + C2f stride-8 hook (9/9 tests pass)
- ⬜ Phase 1.4 — Two-stage scorer ← next
- ⬜ Phase 1.5 — Train the W projection

---

## What's next

**Phase 1.4 — Two-stage scorer.**

Goals: tie together everything we've built so far. For each detection in
each image:

1. RoI-pool the (64, 80, 80) feature inside the detection's box → 64-D vector
2. Project to 384-D via `W` (initialized as zero-pad identity for now;
   will be trained in Phase 1.5)
3. Compute the three score factors:
   - `conf` (from YOLOv8n)
   - `LUT[class][task]` (Phase 1.1 artifact)
   - `cos_sim(W·roi, task_embedding[task])` (Phase 1.2 artifact)
4. Combine: `score = conf × LUT × cos_sim`
5. Rank and pick top-K (with task-conditioned class masking applied first)

Deliverable: a single `tadop_reference.py` that takes `(image_path, task_id)`
and returns ranked, task-aware detections. This is the **end-of-Phase-1
golden model** — the integrated software reference everything FPGA-side
will validate against.
