"""CPU check of the real pinned processor; downloads only tokenizer/processor files."""

from pathlib import Path

from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions
from transformers import Qwen2Tokenizer, Qwen2VLProcessor

from roomcraft.common import model_spec, write_json
from roomcraft.preprocess import read_rgb

ROOT = Path(__file__).resolve().parents[1]
spec = model_spec("qwen")
processor = Qwen2VLProcessor.from_pretrained(
    spec["repo"], subfolder="processor", revision=spec["revision"]
)
tokenizer = Qwen2Tokenizer.from_pretrained(
    spec["repo"], subfolder="tokenizer", revision=spec["revision"]
)
pipe = QwenImageEditPlusPipeline(
    scheduler=None,
    vae=None,
    text_encoder=None,
    tokenizer=tokenizer,
    processor=processor,
    transformer=None,
)
image = read_rgb(ROOT / "data/v02/windows.jpg")
width, height = calculate_dimensions(384 * 384, image.width / image.height)
image = pipe.image_processor.resize(image, height, width)
text = pipe.prompt_template_encode.format(
    "Picture 1: <|vision_start|><|image_pad|><|vision_end|>Add a blue sofa below the small window."
)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
record = {
    "status": "CPU processor passed; no text encoder, denoiser or VAE weights loaded; no generation",
    "base": spec,
    "condition_size": list(image.size),
    "tensor_shapes": {key: list(value.shape) for key, value in inputs.items()},
    "image_grid_thw": inputs.image_grid_thw.tolist(),
    "tokenizer_type": type(tokenizer).__name__,
    "processor_type": type(processor).__name__,
}
write_json(ROOT / "experiments/v03_processor_check.json", record)
print(record)
