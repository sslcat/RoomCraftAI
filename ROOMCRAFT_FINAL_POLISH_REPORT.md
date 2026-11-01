# RoomCraft Studio final polish report

2026-10-31. **RC2 prepared for owner review; the exact fresh RGA-versus-baseline image comparison remains pending separate paid-GPU approval.** No new paid compute, retraining, checkpoint changes, Brasik/Rulbi changes or publication occurred during this polish mission. Original V1.1 RC1 ZIP is preserved; its historical evidence is not overwritten.

## Delivered

- Broader RoomCraft Studio: independent Gaming / Work / Relaxation / Mixed-purpose controls and six themes plus custom. Gaming retains detailed equipment. Work has workstation/display choices; relaxation has seating/reading choices; mixed-purpose selects functions and requests shared furniture and clear circulation. Exact-prompt preview, model selection, bounded worker transport, comparison, history, reset and downloads are retained.
- Optional local selective preservation: manual architecture mask, optional foreground exclusion, inward feathering and broad lighting matching. Generates a separately labeled derivative with raw-output and mask hashes; preserves raw history. No GPU call. Default generation remains raw.
- Three free mask experiments against the genuine Underwater output, full PNGs/masks/alphas and review gallery at `outputs/final-polish/index.html`. Broad copying and the initial window rectangle were rejected for obvious seams and lost decorations. Refined feathering/lighting matching reduced the local curtain boundary without erasing furniture. It does not repair ceiling, wall finish or global viewpoint. Feature remains experimental; masks must be inspected for occlusion and alignment.
- Exact-prompt RGA pair runner `scripts/verify_rga_pair.py` and matched-settings plan at `outputs/final-polish/rga-comparison-plan.json`. Local comparison confirms the new composer reproduces the original saved Underwater prompt exactly. Runner fails closed on prompt/source/model/sampling mismatches and does not provision. **No fresh RGA-on image exists from this mission; no architectural benefit is claimed.** Separate GPU approval was requested while local work continued.

## Verification

- Qwen CPU suite: 156 passed; staged RC2 candidate suite: 156 passed. Base/SDXL environment: 142 passed, 14 expected skips.
- Lightweight final UI/purpose/preservation suite: 44 passed.
- Mypy: 30 source modules passed. Ruff lint and repository formatting passed; all three locks consistent offline. Runtime dependencies remain unchanged.
- Studio browser: all four purposes, independent theme, workstation/display/seating/reading choices, reset, saved-result preservation and metadata download passed. 1440 / 390 / 320 layouts passed without overflow or JavaScript errors.
- Existing gaming browser: six themes/custom, setups/equipment/light/model/RGA controls and reset passed. Existing base/LoRA saved journey: comparison drag, verified PNG/settings downloads, history transition and clear passed. These are historical saved pixels, not fresh GPU inference.
- Native V1.1 three-theme GPU outputs remain genuine prior evidence; all monitor counts were correct, but architectural preservation was partial. New Work/Relaxation/Mixed-purpose GPU image quality is untested.

## Research, provenance and security

Original 100-update V0.6 LoRA, calibrated V0.9 RGA, Qwen revision, SDXL and training mathematics remain intact. Architecture/math, provenance, licensing and reproducibility docs remain included and are linked from README and `docs/FINAL_POLISH.md`. No claim of inventing LoRA or foundation-model sampling. The RGA has 728,512 trained parameters; historical causality studies found no demonstrated useful architectural control.

The 2026-10-31 installed dependency audit remains disclosed: UI zero known advisories, Qwen 88 entries / 66 distinct IDs, SDXL 92 / 70; this mission did not rerun the advisory feed or resolve model-worker vulnerabilities. UI Gradio/Pillow bounds override remains disclosed. No dependencies were upgraded. Models/adapters retain separate license and hosting decisions. Source release excludes neural weights, datasets, caches, credentials and private operational receipts and is scanned by the existing release builder.

Photo and derived screenshots/images: upyernoz, Our Hotel Suite-1, CC BY 2.0, resized and AI edited/composited; attribution retained. Conceptual visualization only: no guaranteed physical fit or unchanged architecture.

## Owner review

Candidate: `dist/RoomCraftAI-v1.1.0-rc2.zip`; per-file manifest inside the staged candidate and ZIP. Archive size/hash receipt: `dist/release_receipt-v1.1.0-rc2.json`. Original RC1 ZIP and receipt remain preserved. Nothing pushed or published. Review `docs/FINAL_POLISH.md`, screenshots, mask gallery, disclosed dependency risks and adapter-distribution decision before publication.

The source candidate is ready to review with these explicit limitations. Phase 1 remains unfinished pending separate approval; do not label the full mission or architecture-preservation acceptance complete. No further research cycle or paid retry is started automatically.
