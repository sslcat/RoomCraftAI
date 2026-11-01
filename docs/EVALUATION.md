# Reproducible evaluation

There are no quality scores or benchmark claims in V0.1. The fixture is excluded from any model-quality benchmark. Establish a consented held-out test set first, then freeze its manifest SHA, prompts, masks, model revisions, software lock, GPU, resolution, strength, guidance, conditioning and seed list (for example 17, 42, 123).

For every room/style/seed, run the unchanged foundation and the candidate adapter with identical settings. Keep mask-only, Canny and depth comparisons in separate ablation tracks. Save all outputs, including failed generations; do not select only favorable seeds.

```bash
uv run roomcraft compare --baseline outputs/base --candidate outputs/adapted --regions data/eval/room01/regions --output outputs/comparison
uv run roomcraft human-summary outputs/comparison/human_scores.csv

# Fixed-seed test split, once a real adapter and licensed test set exist:
uv run python scripts/benchmark.py --manifest data/rooms/manifest.jsonl --adapter runs/first/step-001000 --output outputs/benchmark
```

`compare` validates model, input/mask/depth hashes, prompts and settings, and rejects a baseline carrying an adapter. It writes `comparison.png`, structured `metrics.json`, blinded `A.png`/`B.png`, `review.html`, `human_scores.csv`, and a separate `blinding_key.json`. Keep the key and labeled contact sheet from reviewers until scoring is finished. An identity comparison is a pipeline control, never evidence of improvement.

| Dimension | Implemented measurement | Interpretation / limits |
| --- | --- | --- |
| Overall quality | Blind human 1–5 score | Artifacts, photorealism, lighting. No automated aesthetic score is presented as truth. |
| Walls, windows, doors | Region-specific raw-output normalized pixel MAE and Canny edge F1 within 2 pixels; independent human 1–5 scores | Requires manually reviewed working-resolution binary `walls.png`, `windows.png`, `doors.png`. Missing masks yield no semantic metric. Edge agreement is a proxy, not a semantic detector. |
| Style adherence | Blind human 1–5 against the exact prompt | Target image may show only one of many valid styles/layouts. |
| Furniture plausibility | Blind human 1–5 | Score scale, support/contact, object shape, access to openings and useful circulation. |
| Latency | CUDA-synchronized generation seconds and total request seconds from metadata | First load/download is separate; use identical offload configuration and warmup policy. |
| Peak GPU memory | Process peak allocated and reserved bytes | Does not measure other processes or driver overhead; optional external `nvidia-smi` readings should be labeled separately. |

Raw and composited geometry scores are separate. Composited outside-mask MAE is usually zero by construction and must not be reported as learned preservation. Region edges are eroded away from boundaries. No reference edges produces `null`, not a perfect score. Furniture legitimately occludes architecture; annotate visible reference regions and record occlusions in human notes.

Use at least three independent reviewers, with 1=unusable/severe violations, 3=usable with visible flaws, 5=convincing/no visible issue. Leave absent windows/doors blank, never award automatic full marks. Ask for comments on failures and retain per-reviewer data. The CSV summary reports count and mean per dimension, not statistical significance. Aggregate rooms first, then bootstrap confidence intervals by room in a later analysis, rather than treating seeds as independent rooms.

Planned after sufficient licensed test data: CLIP-based style similarity, LPIPS/SSIM on aligned protected regions, segmentation/line-junction drift, and distributional quality (FID/KID only with adequate samples). These are not implemented or measured. Human evaluation is the current mechanism for quality, style and plausible furniture. No automatic metric establishes physical buildability.
