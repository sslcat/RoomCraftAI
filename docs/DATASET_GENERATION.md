# RoomCraft V0.5 paired synthetic interiors

This dataset is authored in Blender, not photographed and not produced by Qwen. An empty input and furnished target share the same camera, architecture, materials, exposure, lighting configuration and Cycles seed. Furniture naturally changes shadows, reflections and indirect illumination. The geometry generator, pairing checks and review tools are RoomCraft code. Blender/Cycles, OpenImageDenoise and the credited CC0 furniture are external components.

The actual accepted count and verification results are in ROOMCRAFT_V05_REPORT.md (`../ROOMCRAFT_V05_REPORT.md`, local research artifact excluded from the source candidate). Generated data lives under ignored `data/v05/`; model code and model locks are unchanged. Do not train directly from unreviewed render folders.

## Recipe and requirements

- Blender **5.2.2 LTS**, build **d13f752e3b9c**. Local binary: `D:/Blender Foundation/Blender 5.2/blender.exe`. Blender Python runs the scene generator; the project's Python environment runs review/sealing utilities.
- RTX 4070 SUPER, 12 GB, OptiX. The generator fails if the requested GPU is unavailable; there is no silent CPU fallback. Local rendering needs no Runpod resource.
- [Frozen configuration](../configs/dataset_v05.json): 20 scene families × 5 variants, seed 53021, RGB 1024×768, 128 Cycles samples, adaptive threshold 0.015, OpenImageDenoise, AgX Medium High Contrast.
- Living rooms, bedrooms and dining rooms; two furniture arrangements per room type where geometry permits, variable room proportions, window dimensions/offsets, one/two windows, left/right doors, five wall palettes, timber/limestone floors, ceiling palettes, camera positions and daylight/artificial-light variations.
- Furniture dimensions are in meters. Placement checks reserve door clearance, reject primary assembly footprint intersections, check all furniture bounds against the room envelope, and fit furniture within a plausible camera field of view. Rugs can naturally extend outside the camera frame. These checks are conservative proxies, not a full mesh collision solver; visual review remains mandatory.
- The 20 room families are separate sampled room shells. They share procedural templates and furniture assets. The split is **room-family disjoint**, not template-, style- or asset-disjoint. Five variants of a family always remain together. Do not interpret 100 renders as 100 independently designed homes.

## Asset provenance and commercial use

Original room geometry, procedural materials, bed/table/rug/lamp geometry and rendering recipes were authored in this project. They use the repository's [MIT license](../LICENSE); the original contributions to the generated dataset are offered on the same terms, to the extent rights subsist. Keep the MIT notice when redistributing that contribution. No exclusive ownership of third-party furniture is claimed.

Each external asset has an individually checked CC0 notice, author, original page, archived evidence and locked file hashes in [asset_lock.json](sources/v05/asset_lock.json). The actual CC0 legal text (`sources/v05/CC0.txt`, local research artifact excluded from the source candidate) permits commercial reuse, modification and redistribution without an attribution condition. The downloaded assets are not site preview photographs. No human photographs, trademarks or paid scene libraries were introduced.

| Asset | Creator | Verified asset page |
|---|---|---|
| modern_arm_chair_01 | Vibrant Nordic | [individual asset](https://polyhaven.com/a/modern_arm_chair_01) |
| dining_chair_02 | James Ray Cock | [individual asset](https://polyhaven.com/a/dining_chair_02) |
| sofa_02 | Kirill Sannikov | [individual asset](https://polyhaven.com/a/sofa_02) |
| potted_plant_01 | Rico Cilliers | [individual asset](https://polyhaven.com/a/potted_plant_01) |
| potted_plant_02 | Rico Cilliers | [individual asset](https://polyhaven.com/a/potted_plant_02) |

**Powered by Poly Haven.** The API's separate terms (`sources/v05/API_TERMS.md`, local research artifact excluded from the source candidate) require application identification and credit; the downloader sends `RoomCraftAI/0.5 dataset-preparation` as its User-Agent. Asset attribution is voluntary under CC0; API credit is retained. Archived website pages are licensing evidence, not assets offered under CC0. Review website-evidence redistribution separately before a public release; no publication is performed here. [Blender's license](https://www.blender.org/about/license/) covers the application, not a requirement to license rendered artwork under GPL; see its [FAQ](https://www.blender.org/support/faq/).

The downloader verifies SHA256 against the reviewed lock, plus the initial acquisition checked provider MD5. Changed remote bytes fail verification. It does not silently select updated assets. The renderer verifies every locked asset before building scenes. `data/v05/provenance/rights.json` binds the asset ledger, generator, config and license evidence used by the training manifest.

## Reproduce locally (PowerShell)

Run from `D:\RoomCraftAI` with the existing base development environment installed. These commands download only the allowlisted free CC0 assets and use the local GPU; they do not train or start cloud resources.

```powershell
.venv/Scripts/python.exe scripts/v05_download_assets.py
$blenderPath = 'D:/Blender Foundation/Blender 5.2/blender.exe'
& $blenderPath --background --factory-startup --python-exit-code 1 --python scripts/generate_room_dataset.py -- --output data/v05_reproduced --preflight
& $blenderPath --background --factory-startup --python-exit-code 1 --python scripts/generate_room_dataset.py -- --output data/v05_reproduced --save-scenes
.venv/Scripts/python.exe scripts/review_room_dataset.py gallery data/v05_reproduced
.venv/Scripts/python.exe scripts/prepare_render_provenance.py data/v05_reproduced
```

For a benchmark subset, append `--families 0,1,2 --variants 0`. `--families`/`--variants` accept comma-separated numeric IDs. Keep the exact Blender build/config/asset lock. Matching seeds and fixed bytes support reproducibility; bitwise image equivalence is not promised even on the same machine. The executed repeat test had identical scene state and at most one 8-bit channel level of pixel variation; see the report. The source generator SHA256 and all settings are recorded per pair. Existing completed outputs are only skipped if their recipe, settings and image hashes agree; partial or stale outputs fail. Choose a new output directory for changed recipes, never overwrite reviewed images.

Each completed pair contains `source.png`, `target.png`, `render.json`, `invariant_state.json`, and (with `--save-scenes`) a compressed, texture-packed `scene.blend`. The saved scene is furnished; `RoomCraftFurniture.hide_render=True` recovers the empty state. The invariant includes architecture vertices/faces/transforms, material node values/links, relevant modifiers, light settings/visibility, world nodes, camera transform/lens and render seed. It is checked before both renders. Save files and provenance are hashed; they are not training checkpoints.

## Visual review, sealing and splits

`gallery/index.html` links every full-resolution input/target and render record; each family has a paired contact sheet. The gallery/audit computes hashes, dimensions, mean brightness and pair change. Those are integrity filters, **not realism scores**. Review actual images for floating objects, wall/furniture intersections, door access, framing, lighting, artifacts, instruction accuracy and unchanged architecture. Whole rugs need not be visible. Reject unusable outputs; do not infer acceptance from exit code zero.

Create `quality_reviews.json` with `reviewer`, ISO `date`, `method` (human or AI-assisted, stated honestly), and explicit per-pair `{id, decision: "accept"|"reject", sha256: {source, target}, notes}`. Hashes must refer to the images actually reviewed. Re-rendered images require a fresh review. `prepare_render_provenance.py` snapshots matching recipe/config/license evidence and refuses mixed recipes. Never reuse a review of different image bytes.

```powershell
.venv/Scripts/python.exe scripts/review_room_dataset.py ledger data/v05
.venv/Scripts/python.exe scripts/prepare_qwen_pairs.py data/v05/reviewed.json --output data/v05/sealed.json
.venv/Scripts/roomcraft.exe split-pairs data/v05/sealed.json --output data/v05/manifest.json --seed 42
.venv/Scripts/roomcraft.exe validate-pairs data/v05/manifest.json
```

The existing V0.4 sealer and validator remain authoritative. The manifest includes relative input/target paths, furnishing instructions, scene/family IDs, dimensions, hashes, rights evidence, alignment evidence, review identity and splits. With all 20 families accepted, the split is 16/2/2 families, or 80/10/10 pairs. Splits are computed at family level; do not randomly split images. Rejected renders remain outside the manifest. Test images must not be used for adapter fitting or validation-driven tuning.

## Future Qwen experiment (not executed by this mission)

The RGB paired manifest is compatible with the unchanged V0.4 `qwen-cache` and `train-qwen` commands. On an owner-approved GPU meeting [V0.4 requirements](QWEN_TRAINING_V04.md), first encode the frozen VAE/VLM:

```bash
uv run --project environments/qwen roomcraft validate-pairs data/v05/manifest.json
uv run --project environments/qwen roomcraft qwen-cache --manifest data/v05/manifest.json --output data/v05/cache-512 --size 512
```

Do not run training merely because data validation passes. Review the gallery, assess synthetic-to-real domain shift and use the existing approved training protocol. This dataset uses a small catalog and simple rectangular rooms; it cannot establish general real-room staging capability. No model improvement, pretrained adapter or training result follows from generating these pairs.
