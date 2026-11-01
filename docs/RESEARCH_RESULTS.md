# Verified research record

These are executed findings. The final Gradio application also passed fresh GPU acceptance for unchanged Qwen and trained LoRA on one licensed real photograph. `scripts/verify_app.py` records acceptance separately: default mode uses previous real outputs; `--live` invokes new inference.

| Milestone | Executed evidence | Limit |
|---|---|---|
| V0.6 | Full loading, two real optimizer updates, checkpoint save/reload/resume, 100-update LoRA pilot and six synthetic images | Small synthetic data; no broad quality proof |
| V0.7 | 36 new real/synthetic generations plus six reused controls; native 512/768/1024, three licensed photos, matched base/LoRA | Small qualitative AI-assisted review |
| V0.9 | Full-model RGA training/checkpoint replay, calibration and 24 new native-1024 comparisons | Active residuals/gradients do not establish useful geometry |
| V1.0 | 36 geometry-intervention images plus three subtraction ablations | Useful architectural control not demonstrated |
| Final app | Both models generated native 768x1024 images through the running UI; comparison, downloads and history verified | One real photograph and preset; functional acceptance, not a quality benchmark |
| Gaming data | 100 reviewed original pairs, 20 families, 60/20/20 disjoint split | Synthetic visualization, not trained gaming weights |

The private workspace preserves full reports, logs, original locks, hashes, data and checkpoints. Account-bound receipts and full image collections are excluded from the public candidate.

## Geometry causality

Moving edge cues 64 pixels right yielded upper-window translations of **0, 0 and -1 pixels** in three synthetic cases. These are translation-only edge-registration proxies with visual review, not semantic landmark annotations. Apartment doorway/closet framing still changed with correct cues. Other photos retained windows even with geometry disabled, zeroed or unrelated. Correct cues did not consistently improve preservation.

At 512, zeroing edges changed calibrated encoder features only about 0.59-1.78% in relative L2. Ordinary full-model residual RMS was around 23,000. Subtracting the zero-edge field reduced it to 78-172; 97.90-99.55% of last-block additions were unchanged in BF16 in the first conditional forward. Three synthetic outputs approached LoRA-only appearance. Candidate real-photo and shifted-cue runs were not executed. This diagnoses a weak cue-dependent component; it is not a validated fix.

## Real example

The bundled hotel [source/base/LoRA files](assets/example/manifest.json) are actual V0.7 outputs at native 768x1024, 40 steps, CFG 4 and seed 42. Per-image metadata records exact runtime and hashes. They are not gaming renders or new application generations. Source: upyernoz, [CC BY 2.0 photograph](https://commons.wikimedia.org/wiki/File:Our_Hotel_Suite-1_(4983234010).jpg); resized and AI edited.

Assess furniture, framing, openings, perspective, lighting and unwanted changes visually. Hidden/occluded landmarks cannot count as perfectly preserved. More seeds, authorized real pairs and independent review are needed before commercial reliability claims.


## Early V1.1 product verification (before the later Underwater comparison)

All six theme definitions and setup/equipment/light controls were exercised in the real browser, including 1440/390/320px layouts. Historical base/LoRA comparison, verified downloads and history/clear still pass. No fresh themed outputs were generated and no training occurred. This is UI/integration evidence, not a themed-quality benchmark. [Receipt](security/gaming_acceptance.json).


## Final RoomCraft Studio polish

Independent room-purpose/theme controls and optional local selective-preservation derivatives are documented in [FINAL_POLISH.md](FINAL_POLISH.md). Original model checkpoints and research methods remain unchanged. The matched Underwater comparison now demonstrates one positive example with active calibrated V0.9 RGA: improved wallpaper and framing, with remaining shifts. This does not establish consistent improvement. See [verified RGA comparison](RGA_VALIDATION.md).


## V1.1 RC3 geometry integration

**Preserve Room Geometry — RoomCraft RGA** is visible beside model selection and is automatically enabled when RoomCraft LoRA is selected. Switch it off for LoRA-only inference. Unchanged Qwen remains the initial/reset model choice, with RGA disabled. Selecting LoRA again restores its default RGA-on state. The pinned base, LoRA and calibrated RGA checkpoints and inference mathematics are unchanged. Downloaded metadata records actual geometry conditioning; saved examples retain their historical settings rather than inheriting the current checkbox.

The matched Underwater comparison shows better wallpaper, camera/room framing and visible preservation with active RGA. This is one positive example, not proof of reliable preservation across rooms. Monitor count, small geometric shifts, materials, occlusion and furniture clearance remain limitations. [Full evidence and gallery](RGA_VALIDATION.md). No new GPU generation occurred for this release audit.
