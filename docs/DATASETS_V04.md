# Paired training data and provenance — V0.4

**Approved training pairs currently available: zero.** No examined public source is both established as commercially cleared and immediately usable as aligned empty/furnished photographs. This is a research finding about the sources reviewed, not proof that no suitable dataset exists anywhere. The three licensed V0.2 photos remain external evaluation inputs without target pairs.

## Primary-source findings, checked 2026-10-28 (Chicago)

| Source | Pairing and provenance | Commercial use / redistribution | Decision |
|---|---|---|---|
| [ApplyDesign bedroom dataset](https://huggingface.co/datasets/ApplyDesign/bedroom-dataset) | Card describes 15,471 examples with empty, furniture, shadow and staged images. Whether targets are photographs or generated/rendered edits and whether camera geometry matches require inspection. | HF API revision `c6d4fc31032c3254533b46f0b0ac52fa44321c7f`: manual gate, no license/card metadata and no LICENSE among repository files. Public availability is not a grant. | Request explicit commercial training, adapter distribution and image redistribution terms plus original-image provenance. No gate accepted, no data downloaded. |
| [Structured3D](https://structured3d-dataset.org/) | Rendered house scenes with empty/full perspective images indexed by camera position and multiple furniture states. Potentially aligned **synthetic**, not photographic pairs; verify pose files and lighting. | Actual linked Terms of Use §§1–2 restrict use to noncommercial research/education and prohibit data redistribution. Repository code MIT is separate. | Excluded absent a separate commercial agreement. Actual PDF and extraction archived. |
| [Apple Hypersim](https://github.com/apple/ml-hypersim) | Furnished photorealistic renders, cameras and geometry labels. Released renders do not provide paired empty-room targets. | Dataset README specifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode): commercial reuse is permitted with attribution and applicable share-alike terms for redistributed adaptations. Original editable scene assets must be separately purchased/licensed. Code license is not the dataset license. | Not a ready paired set. Do not assume rendered-image rights grant editable asset rights or silently relicense an adapter; determine applicable derivative obligations before training/release. |
| [3D-FRONT official project](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) | Furnished scene layouts could support controlled rerendering if underlying furniture/material rights permit. Not an aligned photograph collection. | Official landing page returned an unavailable/404 page. Actual current dataset/asset terms could not be verified. A third-party loader's MIT license does not clear data. | Not approved; obtain current original terms before acquisition. |
| [Poly Haven assets](https://polyhaven.com/license) | Staff/artist-provided models, textures and HDRIs. Assemble authorized room scenes and render both states with the same camera. These are synthetic pairs. | Asset license is [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode), permitting commercial reuse and redistribution. Site example renders/text/logos are explicitly outside the asset grant; website scraping/API use has separate terms. | Preferred no-purchase asset route for authored scenes. Download chosen assets through permitted channels, not site preview photographs or bulk scraping. No assets acquired here. |
| Owner/partner capture | Photograph the same room before and after furnishing from a locked tripod, with fixed focal length, exposure and lighting. | Written authorization must cover the photographer's rights, commercial ML training, derivatives and adapter distribution. Image publication is a separate permission. | Preferred real-photo pilot; no such authorized pairs supplied yet. |

Evidence locations: `docs/sources/v04/index.json` contains URLs, dates and SHA256 hashes; `structured3d-terms.pdf`/`.txt` contain the actual agreement; `cc0-1.0.txt` and `cc-by-sa-3.0.txt` contain actual license texts. Unknown permissions stay unknown. No subscriptions, purchases, scraping, dataset contact messages or access-form submissions were made.

## Practical collection plan

Start with an independently reviewed pilot of approximately 30–100 pairs across at least ten unrelated homes or scene families; this is a collection target, not an existing dataset or a guaranteed sample-size requirement. Prefer real authorized before/after captures. Hold camera position, crop, resolution, lens, wall finish and openings fixed. Allow furniture-dependent shadows/reflections to change naturally. Record capture dates and camera details, photograph ownership/consent, and an accurate furnishing instruction. Reject viewpoint changes, altered walls/windows, object-removal residue, blurry targets and hidden watermarks. Do not pair unrelated listing photographs or use generative furnishing outputs as unlabelled ground truth.

For the synthetic route, author/obtain a rights-cleared `.blend` scene. Put furniture only in `RoomCraftFurniture`; keep architecture, camera and environment lighting outside. Keep a ledger for **every** model, texture and HDRI, including creator, source URL, license URL/text, asset SHA256 and commercial-training permission. Render with the fixed-camera script:

```bash
blender --background data/scenes/authorized-room.blend --python scripts/render_qwen_pair.py -- --assets data/scenes/assets.json --output data/qwen/render-room01 --seed 42
```

The script toggles furniture visibility, keeps camera/scene/lighting settings, records the Blender version, scene hash, camera matrix/lens, render settings and asset ledger, and writes `source.png`, `target.png`, `render_provenance.json`. It never saves the source scene. A reviewer must still confirm that the collection contains no structural objects, all asset rights are valid, furniture appears plausible and pairs are aligned. The renderer is executable but **not Blender-tested locally** (Blender is unavailable); no renders were produced or presented as photographs. Render-to-photo domain shift must be evaluated separately.

## Manifest and review

Copy `configs/qwen_pairs_template.json` into ignored `data/qwen/reviewed.json`. Populate `pairs` only after reviewing real files. Each row has:

| Field | Required meaning |
|---|---|
| `id`, `scene_id`, `group_id` | Safe unique ID, room identity, property/scene-family identity. Group all camera views and furniture variants from one property together. |
| `source`, `target`, `instruction` | Explicit relative file paths and truthful edit instruction; no filename-based auto-pairing. |
| `kind` | `photograph_pair` or `render_pair`; never label renders as photos. |
| `creator`, `source_url`, `license`, `license_url` | Image/scene origin, original rights holder and actual governing terms. For private permission, a stable local permission-record reference is acceptable. |
| `evidence`, `alignment_evidence` | Relative local files: permission/license record and camera/reviewer evidence. Keep private signed records out of Git. |
| `rights` | Boolean `commercial_training`, `derivatives`, `adapter_distribution` must all be true based on evidence. `image_redistribution` must explicitly be true or false. |
| `review` | Named `reviewer`, `date`, true `same_camera`, `architecture_unchanged`, `source_empty`, `target_furnished`. These are accountable review assertions, not automatic vision judgments. |
| `sha256` | Hashes of `source`, `target`, `evidence`, `alignment_evidence`; filled by the sealing script. |
| `split` | Assigned once by the splitter: `train`, `validation`, `test`. |

`scripts/prepare_qwen_pairs.py` computes hashes, validates reviewed records and preserves explicit pair correspondence. `roomcraft split-pairs` shuffles sorted property groups with a fixed seed, reserves roughly 10% each for validation/test (at least one group each), and keeps the remainder for training. At least three independent groups are required; three is a software minimum, not adequate evidence for generalization. Output stays next to input so relative paths remain valid.

Validation checks file containment, hashes, distinct pair pixels, same dimensions, approval fields, nonempty splits, scene/group consistency and decoded-image duplicates across groups. These checks catch exact duplicates saved with different metadata; they do not detect all near-duplicates or prove alignment. Review shared layouts/assets and property identity manually before sealing. Freeze the final test split before model selection. Do not reshuffle until a favorable result appears.

There is intentionally no production paired-data fixture. Unit tests create original synthetic pixels in temporary test directories to exercise the validator; those are software tests and cannot establish room quality or populate a real training manifest.
