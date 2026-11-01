# Attribution and license boundaries

Original RoomCraft source code is MIT: application/presets, paired-data validation, project-owned LoRA/flow training, geometry encoder/residual integration, Blender recipes and evaluation utilities. MIT does not cover external weights, photos, assets or dependencies.

**Qwen-Image-Edit-2511**, Qwen team: the exact revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` declares `apache-2.0` in its official model card. That snapshot has no standalone LICENSE file. [Declaration evidence](docs/licenses/qwen_license_record.json), [canonical Apache-2.0 terms](docs/licenses/APACHE_2.txt). No foundation weights are redistributed.

**SDXL inpainting and ControlNets**, Hugging Face Diffusers team/Stability AI: separate Open RAIL++-M weight terms apply. **PyTorch** uses BSD-style terms. **Diffusers, Transformers, Accelerate, PEFT, Hugging Face Hub, safetensors and Gradio** use Apache-2.0. Pillow has HPND-derived terms; NumPy uses BSD-3-Clause. Wheels/native libraries may have additional notices; no binary environment is bundled.

LoRA follows [Hu et al.](https://arxiv.org/abs/2106.09685). Conditional residuals and zero initialization have established precedents including [ControlNet](https://arxiv.org/abs/2302.05543). RoomCraft claims its implementation and tests, not invention of these methods or Qwen's architecture.

The bundled photo and AI derivatives credit **upyernoz**, [Our Hotel Suite-1](https://commons.wikimedia.org/wiki/File:Our_Hotel_Suite-1_(4983234010).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Input was resized/converted; outputs were AI edited. Screenshots containing them retain attribution obligations. [Image notice](docs/assets/example/LICENSE.md). No endorsement is implied.

Original procedural test fixtures are CC0 and are not quality evidence. Original scene/dataset contributions use MIT to the extent rights subsist. Blender/Cycles/OpenImageDenoise are external tools, not bundled binaries. V0.5/V0.6 used reviewed CC0 assets: modern_arm_chair_01 (Vibrant Nordic), dining_chair_02 (James Ray Cock), sofa_02 (Kirill Sannikov), potted_plant_01 and potted_plant_02 (Rico Cilliers). **Powered by Poly Haven.** IDs, authors, URLs and hashes are retained in the asset lock. Website previews and archived HTML are excluded from the public candidate; they are not automatically CC0.

V1.0 gaming scenes use original procedural geometry. Complete datasets, packed scenes and third-party assets are not distributed here. Dataset terms remain independent of code licenses. RoomCraft adapters remain unpublished and require a separately reviewed model license and notices. [Adapter distribution](docs/ADAPTER_DISTRIBUTION.md).

The verified Underwater RGA demo is included under the same upyernoz CC BY 2.0 source/modification notice; see [demo license](docs/assets/rga-underwater/LICENSE.md). No additional room photo or unauthorized external theme artwork is included.

The preserved model catalog also references **FLUX.1 Fill [dev]** for historical comparison. It is not the production backend and no weights are bundled. The reviewed FLUX.1 [dev] Non-Commercial v1.1.1 terms are separate from this MIT code license; no commercial/production permission was acquired. [Model-selection and license boundaries](docs/MODEL_SELECTION.md).
