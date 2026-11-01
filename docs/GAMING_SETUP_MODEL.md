# Gaming setup specialization: original paired dataset

RoomCraft's intended product is a realistic gaming-setup visualization inside a user's photographed room. V1.0 contributes original paired synthetic data and geometry diagnostics. **There is no gaming-trained RoomCraft checkpoint yet.** The preserved Qwen/V0.6 LoRA/calibrated V0.9 RGA remain research baselines; synthetic render quality is not evidence of real-photo generation quality.

## Generator and actual scene content

`scripts/generate_gaming_dataset.py` extends the preserved Blender renderer through a separate `GamingBuilder`. It reuses camera fitting, Cycles rendering, state hashing and placement validation instead of replacing the original V0.5 implementation. `configs/dataset_v10_gaming.json` fixes seeds, Blender build and rendering settings.

The target is 20 distinct scene families with five variants each, 1024x768 native output, 128 Cycles samples, adaptive threshold 0.015, OptiX and OpenImageDenoise. Local renderer: Blender **5.2.2 LTS**, build `d13f752e3b9c`; RTX 4070 SUPER, 12 GB. Room dimensions, window layout, wall/floor/ceiling palette, desk wall, view and lighting vary. Five compact families add a shallow architectural service pier. These remain mostly rectangular rooms, not a representative distribution of all real homes.

Procedural furniture includes 1.6/1.95/2.3 m wide, 0.8 m deep, 0.75 m high desks; straight and L-shaped tops; supported single/dual/triple monitor assemblies; PC cases with stylized fans; swivel chairs with gas lift and casters; individual keyboard keys, mice, desk mats and pen cups; monitor clamps/arms, cable trays and a supported power cable; shelves/books and a streamer microphone. Styles are minimalist, professional, streamer and RGB-heavy. Original abstract bars appear on screens; no game screenshots, brand logos or copyrighted UI are used.

Room-use labels are bedroom, spare room, home office and compact apartment. These describe recipe proportions and intended use; bedrooms do not contain pre-existing beds, and apartment families are one-room views rather than whole apartments. Desk shape is currently correlated with room-use category. Chair, monitor and tower shapes share a small procedural vocabulary. Those are dataset limitations, not hidden diversity claims.

All desk/PC/chair primary footprints are checked against the room envelope, door clearance and one another. Mesh bounds and projected camera framing are checked before rendering. Floor contacts and supported tabletop objects are authored to metric dimensions. This is not a general collision solver or building-code certification. Cable curves and intrinsic furniture lights are not included in mesh-bounds checks; their paths are authored inside the setup envelope and reviewed visually.

## Pair alignment and light behavior

Each scene is built furnished once. Furniture is hidden for the source and shown for the target. Camera, room geometry, room materials, world and room lights have identical serialized hashes across both renders. The actual `.blend`, camera transforms, placements, instruction and seed are saved for every pair. Original screen emission and RGB lights belong to the introduced setup and appear only in the target, explicitly recorded in `lighting_note`; they can change illumination and reflections. That is an intrinsic part of adding a powered gaming setup, not unchanged pixel lighting.

Each pair includes `source.png`, `target.png`, `render.json`, `invariant_state.json`, `scene.blend` and `landmarks.json`. Projected window/door/wall coordinates derive from the actual camera, not estimated annotations. They are source opening/corner projections, not full semantic masks or occlusion labels. The extra service pier and window moldings are not covered by the analytic landmark set.

## Rights and provenance

No external furniture, image, texture or HDRI asset was downloaded for this dataset. Scene geometry, shaders and screen patterns are original procedural contributions, using the existing project **MIT** license to the extent rights subsist. Retain the MIT notice when distributing original contributions. This is a documented permission basis, not a guarantee of copyrightability or a legal warranty.

`data/v10_gaming/provenance/` archives the actual base/extension/projector source, configuration, split assignment and repository LICENSE with SHA256 hashes. `assets.json` lists original asset IDs, creator, source URNs, generator hashes, permissions and attribution obligation; external URLs are explicitly null because there are no external assets. `rights.json` records commercial training, derivatives, adapter distribution and image redistribution separately. V0.5 CC0 provenance is not borrowed to license these original-only assets. Real evaluation photographs are not included in this training dataset.

Blender is the rendering application. The project's existing licensing research references its [license page](https://www.blender.org/about/license/) and [FAQ](https://www.blender.org/support/faq/). A fresh V1.0 retrieval failed in both the web tool and local URL client; `docs/sources/v10/fetch.json` records that limitation. No claim of a fresh primary-page review is made. The archived local project license and original-only asset provenance are the new dataset's permission evidence.

## Acceptance and splits

`render_dataset.audit_render` checks source/target dimensions, RGB mode, hashes, architectural-state hashes, exposure and a minimum source-target difference. These checks **do not approve visual quality**. An explicit, dated, image-hash-bound `quality_reviews.json` is required before the existing V0.4 ledger/sealer accepts a pair. Review covers paired contact sheets for every family plus native-resolution representative targets and any questionable image. Rejected and benchmark renders stay outside the sealed manifest.

The separate predeclared `configs/dataset_v10_split.json` assigns whole families: 00–03 test, 04–07 validation, 08–19 train. With all 100 accepted this gives **60/20/20**, intentionally reserving one family of each room-use category in both held-out splits. No camera/style variant crosses splits. This reduces training count in exchange for broader held-out coverage; it does not alter the prior V0.6 70/15/15 dataset. Exact completed/accepted counts and actual verification results are in `ROOMCRAFT_V10_REPORT.md`.

The renders are synthetic architectural visualizations with simplified furniture and materials. They are not labeled photorealistic. They can support controlled geometry and prompt experiments but do not establish commercial-quality real-photo staging.

## Regeneration

Keep the delivered version immutable. For a fresh target directory, from `D:\RoomCraftAI`:

```powershell
& 'D:/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --python-exit-code 1 --python scripts/generate_gaming_dataset.py -- --output data/v10_gaming_reproduction --preflight
& 'D:/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --python-exit-code 1 --python scripts/generate_gaming_dataset.py -- --output data/v10_gaming_reproduction
.venv/Scripts/python.exe scripts/prepare_gaming_provenance.py provenance data/v10_gaming_reproduction
.venv/Scripts/python.exe scripts/review_room_dataset.py gallery data/v10_gaming_reproduction
```

Inspect the generated images and create an explicit `quality_reviews.json` following the delivered review schema, bound to the new image hashes. Do not automatically copy acceptance from different bytes. Then:

```powershell
.venv/Scripts/python.exe scripts/review_room_dataset.py ledger data/v10_gaming_reproduction
.venv/Scripts/python.exe scripts/prepare_qwen_pairs.py data/v10_gaming_reproduction/reviewed.json --output data/v10_gaming_reproduction/sealed.json
.venv/Scripts/python.exe scripts/prepare_gaming_provenance.py split data/v10_gaming_reproduction
.venv/Scripts/python.exe -c "from pathlib import Path; from roomcraft.qwen_data import validate_pairs; print(len(validate_pairs(Path('data/v10_gaming_reproduction/manifest.json'))))"
```

`--families 0,3 --variants 0` renders a short benchmark; `--preflight` only constructs/checks scenes. Existing pairs are skipped only when their settings, generator and image hashes match. Partial/changed folders fail closed; use a new version directory rather than silently replacing scenes. Reproducibility means pinned recipe/build/seed and verified alignment; bit-identical Cycles output across other GPUs, drivers or Blender builds is not promised.

## Next model work, in evidence order

1. Complete the bounded full-model geometry interventions before further RGA training. The current encoder has weak edge dependence; a gaming dataset alone does not prove a fix.
2. Evaluate unchanged Qwen, V0.6 LoRA and calibrated RGA on the same authorized real gaming-room prompts. Keep monitor count, desk position, architecture and texture realism as separate judgments.
3. If the causal test confirms an unhelpful cue-independent residual, run one controlled centered-edge candidate experiment, with measured residual units and the original RGA preserved. The candidate is implemented/tested locally but not activated.
4. Improve dataset coverage based on visible weaknesses: independently vary chair/case designs, free desk shape from room-use labels, show more door boundaries, add less rectangular rooms and more realistic materials/backgrounds. Higher render sampling alone will not solve the limited asset vocabulary.
5. Obtain authorized aligned real-photo before/after gaming setups with explicit commercial training and distribution permissions. Use them for a staged synthetic-to-real pilot, maintaining property-disjoint splits. Do not use unrelated photos as aligned pairs or presume the three evaluation-photo licenses authorize every future training/distribution plan.

No training, paid rendering, publication or resource deployment was performed for this dataset mission.
