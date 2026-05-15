# TADOP MiniLM Task Embedding Cache

Phase 1.2 of the Stage 2 build. Produces the 14×384 task embedding cache
that lives in BRAM on the FPGA and is matched against live user-query
embeddings to identify which task the user meant.

## What this is

A precomputed matrix where row `t` is the L2-normalized MiniLM embedding
of task `t`'s canonical description. At inference time:

    user_query_embedding = MiniLM("how do I serve a glass of wine?")
    task_id = argmax_t cosine_sim(user_query_embedding, cache[t])
    # task_id will be 9 (= serve_wine, index from 0)

Stage 1 design uses these as the *task vector* in the two-stage scorer's
RoI×task cosine similarity:

    score(instance) = conf × LUT[class] × cos_sim(W·roi_feat, task_emb)

Note: the task_emb here is the *cached* row at the recognized task index,
not the user's free-form query. The user query is only used to *pick*
which row to use — once we know the task index, we use the canonical
embedding for stability.

## Files

    build_task_embeddings.py    Builder. Encodes the 14 task prompts.
    test_task_embeddings.py     pytest suite. Synthetic tests run anywhere;
                                end-to-end tests require network for model
                                download (skipped automatically if offline).
    README.md                   This file.

## How to run

First-time setup needs the sentence-transformers package and downloads
the ~80 MB MiniLM model from HuggingFace:

    pip install sentence-transformers
    python build_task_embeddings.py --out-dir ./data/embeddings

The model download happens once; subsequent runs use the local cache.

## Outputs (in --out-dir)

    task_embeddings_fp32.npy    14×384 float32 L2-normalized embeddings.
                                Reference for the PyTorch software pipeline.

    task_embeddings_int8.npy    14×384 int8. Bit-exact match for what VEGA
                                will read from BRAM.

    task_embedding_scales.npy   14 float32. Per-row scale factors:
                                real_value = int8_value × scale.

    task_embeddings.coe         Vivado BRAM init file (5,376 bytes, row-major).
    task_embeddings.mem         Verilog $readmemh init (one byte per line).
    task_scales.mem             14 fp32 scales as 4-byte hex (little-endian).

    task_embeddings.json        Task name + prompt + first 8 dims, for humans.
    embedding_stats.json        Norms, scales, quantization errors, cosine
                                matrix info.

## Memory footprint

    Embeddings:  14 × 384 = 5,376 bytes
    Scales:      14 × 4   = 56 bytes
    Total:                  5,432 bytes  (fits comfortably in one BRAM36)

## Quantization scheme

Per-row symmetric INT8. Each task gets its own scale factor, computed
from its row's max absolute value so the largest entry maps to ±127.

    scale[t] = max(|emb[t]|) / 127
    q[t]     = round(emb[t] / scale[t])   clipped to [-128, 127]

Cosine similarity at FPGA inference:

    dot_int32 = sum(q_user[i] × q_t[i])          # 384 INT8×INT8 → INT32
    cos       = dot_int32 × scale_user × scale_t  # fp32 post-multiply

Both query and cached rows are L2-normalized in fp32 before quantization,
so the post-multiplied cosine value is directly in [-1, 1].

## The task prompts

These are deliberately written as natural first-person sentences. The
COCO-Tasks labels themselves (e.g. `get_potatoes_out_of_fire`) are
internal identifiers, not what users would type. Encoding the labels
verbatim would produce embeddings that don't match real query phrasings.

Current prompts (see build_task_embeddings.py for the full list):

  1.  I want to step on something to reach a higher place
  2.  I want to sit down comfortably
  3.  I want to place flowers in a container
  ...
  10. I want to serve wine in a glass
  ...
  14. I want to pound a carpet to clean it

The builder prints a confusability check at the end: any two task prompts
with cosine > 0.55 are flagged. Rewrite prompts if a critical pair (e.g.
serve_wine vs pour_sugar) shows up — those are tasks a user could plausibly
ask about, and we don't want them aliased.

## Tests

    python -m pytest test_task_embeddings.py -v

10 tests run without network. 4 more run if MiniLM downloads successfully.
Test coverage:

  - Quantization preserves dynamic range and reconstructs within scale/2
  - Cosine similarity survives quantization (< 0.02 error)
  - .coe / .mem / .scales.mem format correctness, including two's complement
  - Prompts are natural sentences (no underscores, no labels)
  - End-to-end: rows are unit-norm, no confusable task pairs

## Known caveats

  - **The model download is large-ish** (~80 MB). First run takes a minute
    on decent internet; subsequent runs are instant.
  - **MiniLM at FPGA runtime** is out of scope for Stage 2 (it's 22M params,
    not feasible on VEGA in real time). The plan: run MiniLM on host for
    the live demo, or fall back to the 14 cached embeddings only and
    require the user to pick from a dropdown of task descriptions. See the
    Stage 1 report's risk discussion.
  - **Prompt phrasing affects scoring quality**. If empirical results
    show poor task discrimination, rewriting the prompts is the first
    knob to turn — much cheaper than retraining anything.

## Next steps in the pipeline

  - Phase 1.3: YOLOv8n FP32 baseline with C2f stride-8 feature hook
  - Phase 1.4: Two-stage scorer combining the LUT and these embeddings
  - Phase 1.5: Train the W projection (256→384) on COCO-Tasks
