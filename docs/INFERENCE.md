# Installation and inference

Use Python 3.12 and uv 0.12.23. On Windows set `UV_CACHE_DIR` and `HF_HOME` to directories under this project. On Linux:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
uv sync --frozen
uv run roomcraft --help
```

The lock targets CUDA 12.8 PyTorch on Windows/Linux. An NVIDIA driver supporting CUDA 12.8 is required for GPU execution; a separate local CUDA toolkit is not required by the wheel. CPU tests use this wheel without a GPU. No credentials are necessary for the selected public model. If needed, provide Hugging Face credentials through the environment, never a checked-in file.

```bash
# No model download, no generation:
uv run roomcraft preprocess --image data/room.png --mask data/placement.png --output outputs/prepared --size 1024

# Standard baseline:
uv run roomcraft infer --image data/room.png --mask data/placement.png --style "warm minimalist living room" --seed 42 --output outputs/base

# Geometry ablation, adds a model download:
uv run roomcraft infer --image data/room.png --mask data/placement.png --style "warm minimalist living room" --seed 42 --conditioning canny --control-scale 0.5 --output outputs/canny

# Aligned grayscale relative inverse depth, brighter near the camera:
uv run roomcraft infer --image data/room.png --mask data/placement.png --style "warm minimalist living room" --seed 42 --conditioning depth --depth data/depth.png --output outputs/depth

# Trained project adapter (only after actual training):
uv run roomcraft infer --image data/room.png --mask data/placement.png --style "warm minimalist living room" --seed 42 --adapter runs/experiment/step-001000 --output outputs/adapted
```

For limited VRAM use `--size 512 --offload`; this trades time for memory. Start with the supplied-mask baseline, then compare `--control-scale 0.3`, `0.5`, `0.7` using the same seed list. Strong depth can reinforce an empty-room appearance and inhibit furniture; edges can overconstrain furniture intersections. Structure inside the editable region is not protected by compositing.

Masks must have the input's EXIF-oriented dimensions. Threshold is 128. Omitted masks produce a lower-center rectangle, not a learned furniture mask. Blank masks fail fast. Output folders must be new/empty for inference. Images are saved as PNG without copying EXIF metadata. Raw and final images use the working resolution, not the original resolution.

Seeded runs are reproducible within a fixed software/hardware/kernel setup; bitwise equality across GPU generations or platforms is not promised. Metadata reports CUDA-allocated and reserved peak memory for this process, not global device usage. Generation time excludes model loading; total time includes model loading and writing image outputs but not metadata serialization. First model-download latency should be reported separately from steady-state timing.
## V0.2 controlled prompt comparisons

`roomcraft infer --prompt "A blue upholstered sofa ..."` supplies the exact positive prompt. Omit it to retain the V0.1 style template. The exact prompt is saved in metadata. This option does not change the checkpoint or default strength.

See the V0.2 report (`../ROOMCRAFT_V02_REPORT.md`, local research artifact excluded from the source candidate) for the local real-photo experiment and its failures. `scripts/v02_run.py` runs a recorded serial comparison; `scripts/v02_gallery.py` assembles actual input, mask, raw and composited images. Those local image artifacts are ignored by Git and have separate photo licenses.

