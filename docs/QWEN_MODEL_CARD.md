# RoomCraft Qwen staging adapter — model card preparation

**Status: no trained RoomCraft Qwen adapter exists.** This is a release-ready record structure, not a weight release. Tiny random-model optimizer tests are software verification only.

| Field | Current value / required release evidence |
|---|---|
| Foundation | `Qwen/Qwen-Image-Edit-2511` |
| Immutable revision | `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` |
| Foundation license | Apache-2.0 as declared by the Qwen model card; official Qwen repository license archived in `docs/sources/qwen-LICENSE`. Preserve notices. Model repository has no separate LICENSE file at the audited revision. |
| Adapter artifact / SHA256 | Not available; fill after genuine training. |
| Adapter license | Undecided until the selected dataset's training/distribution obligations are reviewed. MIT code does not license input images or weights. |
| Method | Paired supervised flow matching; frozen foundation/VAE/VLM; attention LoRA, uniform sigma, unit weighting, target-token-only loss. |
| Default trainable count | Rank 8 / alpha 8: 23,543,808; 479 projections. Final text-only output projection excluded. |
| Intended use | Virtual staging and interior-design ideation on authorized photographs. |
| Unsupported guarantees | Architectural fidelity, dimensions, structural safety, furniture fit, ownership/valuation conclusions. |
| Dataset | Zero approved pairs at preparation time. Fill counts, photo/render mix, provenance manifest hash, licenses, permissions and collection methods. |
| Splits | Property/scene-family disjoint train/validation/test. Supply counts and manual near-duplicate review. |
| Compute | Planned single 80 GB A100/H100, BF16; no pretrained training run or measured full-model memory/throughput. |
| Reproduction | Record code revision/archive hash, both lock hashes, config, cache/provenance hashes, GPU/driver, seed, logs and checkpoint steps. |
| Training outcome | Pending: finite losses, gradients, resume verification and learning curves on real reviewed data. |
| Evaluation | Pending: matched unchanged Qwen/adapter outputs on sealed test cases, failures, multiple seeds and human reviews. No improvement claim. |
| Owner observations | Owner reports convincing base-Qwen Replicate staging. Not independently audited or a RoomCraft adapter result. |
| Limitations | Openings/walls may change; distorted furniture, floating objects, shadows, textures and layout bias remain possible. Synthetic training can create a real-photo domain gap. |

Artifacts use the RoomCraft `.a/.b` adapter format and require `roomcraft edit --adapter CHECKPOINT_DIRECTORY`. They are not advertised as native Diffusers/PEFT LoRA files. Keep private photographs and consent records private even if the adapter is later released. Publish only after dataset obligations, full-model validation and explicit owner publication approval are satisfied.
