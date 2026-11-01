"""Cache provenance and fair-evaluation contracts; generated test pixels are not quality evidence."""

import json
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from roomcraft.common import model_spec, sha256, write_json
from roomcraft.qwen_cache import encode_pair, validate_cache
from roomcraft.qwen_evaluation import compare_qwen
from roomcraft.qwen_training import load_training_adapter


def test_cache_hashes_provenance_and_shapes(tmp_path):
    rows = []
    tensors = {
        "target": torch.zeros(1, 4, 64),
        "source": torch.ones(1, 4, 64),
        "text": torch.ones(1, 3, 3584),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
        "shapes": torch.tensor([[1, 2, 2], [1, 2, 2]]),
    }
    for split in ("train", "validation", "test"):
        file = tmp_path / f"{split}.safetensors"
        save_file(tensors, str(file))
        rows.append(
            {
                "id": split,
                "group_id": split,
                "split": split,
                "file": file.name,
                "sha256": sha256(file),
            }
        )
    write_json(tmp_path / "provenance.json", {"pairs": rows})
    record = {
        "schema": "roomcraft-qwen-cache-v1",
        "base": model_spec("qwen"),
        "provenance_sha256": sha256(tmp_path / "provenance.json"),
        "rows": rows,
    }
    cache = tmp_path / "cache.json"
    write_json(cache, record)
    assert len(validate_cache(cache)["rows"]) == 3
    tensors["target"] += 1
    save_file(tensors, str(tmp_path / "train.safetensors"))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_cache(cache)


def test_incompatible_adapter_rejected_before_mutation(tmp_path):
    write_json(tmp_path / "config.json", {"format": "sdxl"})
    with pytest.raises(ValueError, match="adapter does not match"):
        load_training_adapter(None, tmp_path)


def fake_generation(path: Path, adapter: str | None):
    path.mkdir()
    Image.new("RGB", (64, 64), "white").save(path / "input.png")
    Image.new("RGB", (64, 64), "blue" if adapter else "red").save(path / "output.png")
    meta = {
        "backend": "qwen-image-edit-2511",
        "base": model_spec("qwen"),
        "input_sha256": "ab" * 32,
        "prompt": "Add a sofa",
        "settings": {"seed": 42},
        "working_size": [64, 64],
        "dtype": "bfloat16",
        "quantization": "none",
        "environment": {},
        "scheduler_config": {},
        "adapter_sha256": adapter,
        "output_sha256": sha256(path / "output.png"),
        "generation_seconds": 1,
        "peak_gpu_allocated_bytes": 1,
        "peak_gpu_reserved_bytes": 1,
    }
    write_json(path / "metadata.json", meta)
    return meta


def test_matched_comparison_has_no_invented_quality_scores(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    fake_generation(base, None)
    meta = fake_generation(candidate, "123")
    result = compare_qwen(base, candidate, tmp_path / "review")
    assert result["quality_improvement"] is None and result["human_review"] == "pending"
    assert result["automated"]["baseline"]["regions"] == {}
    assert (tmp_path / "review/comparison.png").is_file()
    meta["settings"]["seed"] = 43
    write_json(candidate / "metadata.json", meta)
    with pytest.raises(ValueError, match="settings"):
        compare_qwen(base, candidate, tmp_path / "unfair")
    assert not (tmp_path / "unfair").exists()


def test_qwen_manifest_is_intentionally_empty():
    path = Path(__file__).resolve().parents[1] / "configs/qwen_pairs_template.json"
    assert json.loads(path.read_text())["pairs"] == []


@pytest.mark.skipif(
    metadata.version("diffusers") != "0.37.1", reason="requires isolated Qwen environment"
)
def test_encode_pair_uses_real_upstream_resize_pack_and_template_entrypoint():
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

    seen = {}

    class EncoderContract:
        image_processor = VaeImageProcessor(vae_scale_factor=16)
        vae = SimpleNamespace(dtype=torch.float32)
        _pack_latents = staticmethod(QwenImageEditPlusPipeline._pack_latents)

        def encode_prompt(self, **kwargs):
            seen.update(kwargs)
            return torch.ones(1, 3, 3584), None

        def _encode_vae_image(self, pixels, generator):
            # Stub neural encoding only; real upstream resize/packing exercised above.
            assert pixels.shape[:3] == (1, 3, 1)
            return torch.zeros(1, 16, 1, pixels.shape[-2] // 8, pixels.shape[-1] // 8)

    data = encode_pair(
        EncoderContract(),
        Image.new("RGB", (1280, 960)),
        Image.new("RGB", (1280, 960)),
        "Add a sofa",
        512,
        "cpu",
    )
    assert data["target"].shape == (1, 768, 64)
    assert data["source"].shape[1] > data["target"].shape[1]
    assert data["text_mask"].all()
    assert seen["prompt"] == "Add a sofa" and len(seen["image"]) == 1
    assert seen["image"][0].size == (448, 320)
