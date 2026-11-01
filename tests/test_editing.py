"""Contract and resource tests; fake pixels are never quality evidence."""

import inspect
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from roomcraft import editing


def test_resource_gate_accounts_for_transformer_and_host() -> None:
    assert editing.memory_reasons(12, 31, False)
    assert editing.memory_reasons(12, 31, True)
    assert editing.memory_reasons(80, 31, False)
    assert editing.memory_reasons(48, 100, False)
    assert editing.memory_reasons(48, 100, True) == []
    assert editing.memory_reasons(75, 100, False) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"true_cfg_scale": float("nan")},
        {"true_cfg_scale": 0},
        {"size": 1000},
        {"seed": -1},
    ],
)
def test_edit_configuration_rejects_invalid_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        editing.EditConfig(**kwargs).validate()


def test_edit_rejects_before_loading_weights(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "room.png"
    Image.new("RGB", (1280, 960)).save(image)
    called = []
    monkeypatch.setattr(editing, "load_edit_pipeline", lambda config: called.append(True))
    monkeypatch.setattr(
        editing, "edit_preflight", lambda offload: {"can_attempt": False, "reasons": ["too small"]}
    )
    with pytest.raises(ValueError, match="instruction"):
        editing.edit(image, " ", tmp_path / "blank", editing.EditConfig())
    with pytest.raises(RuntimeError, match="too small"):
        editing.edit(image, "Add a sofa", tmp_path / "blocked", editing.EditConfig())
    assert not called
    assert not (tmp_path / "blocked").exists()


def test_instruction_forwarding_and_uncomposited_artifacts(tmp_path: Path, monkeypatch) -> None:
    original = tmp_path / "room.png"
    Image.new("RGB", (1280, 960), "white").save(original)
    seen = {}

    class ContractPipeline:
        transformer = SimpleNamespace(config=SimpleNamespace(zero_cond_t=True))
        scheduler = SimpleNamespace(config={"shift": 1.0})

        def __call__(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")]
            )

    monkeypatch.setattr(editing, "edit_preflight", lambda offload: {"can_attempt": True})
    monkeypatch.setattr(editing, "load_edit_pipeline", lambda config: ContractPipeline())
    monkeypatch.setattr(editing, "seed_all", lambda seed: None)
    generator = torch.Generator
    monkeypatch.setattr(torch, "Generator", lambda device: generator(device="cpu"))
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    out = tmp_path / "result"
    result = editing.edit(original, "Add a blue sofa", out, editing.EditConfig())
    assert seen["image"][0].size == (1280, 960)
    assert (seen["width"], seen["height"]) == (1024, 768)
    assert seen["prompt"] == "Add a blue sofa"
    assert seen["true_cfg_scale"] == 4 and seen["negative_prompt"] == " "
    assert "guidance_scale" not in seen and "mask_image" not in seen and "strength" not in seen
    assert result["compositing"] == "none" and result["base"] == editing.model_spec("qwen")
    assert (out / "raw.png").read_bytes() == (out / "output.png").read_bytes()
    with Image.open(out / "output.png") as generated:
        assert generated.getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.skipif(
    metadata.version("diffusers") != "0.37.1", reason="Requires isolated Qwen environment"
)
def test_real_qwen_api_and_zero_condition_transformer() -> None:
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    parameters = inspect.signature(QwenImageEditPlusPipeline.__call__).parameters
    assert {"image", "prompt", "true_cfg_scale", "negative_prompt", "width", "height"} <= set(
        parameters
    )
    # Actual tiny random architecture, forward only; no weights, download or training.
    model = QwenImageTransformer2DModel(
        in_channels=64,
        out_channels=16,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    ).eval()
    with torch.inference_mode():
        result = model(
            hidden_states=torch.randn(1, 8, 64),
            encoder_hidden_states=torch.randn(1, 3, 16),
            encoder_hidden_states_mask=torch.ones(1, 3, dtype=torch.bool),
            timestep=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2), (1, 2, 2)]],
            return_dict=False,
        )[0]
    assert result.shape == (1, 8, 64)
    assert torch.isfinite(result).all()
