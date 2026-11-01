import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from roomcraft.checkpoint import resume_checkpoint, save_checkpoint
from roomcraft.data import RoomDataset, local_path, validate_manifest
from roomcraft.evaluation import compare_runs, structure_metrics, summarize_human
from roomcraft.inference import InferenceConfig
from roomcraft.lora import LoRALinear, inject_lora, parameter_report
from roomcraft.objective import inpaint_input, region_loss
from roomcraft.preprocess import composite, prepare
from roomcraft.training import TrainConfig

FIXTURES = Path(__file__).parent / "fixtures"


def test_preprocessing_and_composite() -> None:
    source, mask = prepare(FIXTURES / "room.png", FIXTURES / "mask.png", 256)
    assert source.size == mask.size == (256, 256)
    assert set(np.unique(mask)) == {0, 255}
    output = composite(source, Image.new("RGB", source.size, "red"), mask)
    outside = np.asarray(mask) == 0
    assert np.array_equal(np.asarray(output)[outside], np.asarray(source)[outside])


def test_preprocess_aspect_ratio_and_bad_mask(tmp_path: Path) -> None:
    Image.new("RGB", (640, 320)).save(tmp_path / "source.png")
    source, mask = prepare(tmp_path / "source.png", None, 512)
    assert source.size == mask.size == (512, 256)
    Image.new("L", (640, 320), 0).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="no editable"):
        prepare(tmp_path / "source.png", tmp_path / "mask.png", 512)
    with pytest.raises(ValueError):
        prepare(tmp_path / "source.png", None, 511)


def test_fixture_gate_and_tensor_contracts() -> None:
    with pytest.raises(ValueError, match="fixtures"):
        validate_manifest(FIXTURES / "manifest.jsonl")
    data = RoomDataset(FIXTURES / "manifest.jsonl", "train", 256, True)
    row = data[0]
    assert row["source"].shape == row["target"].shape == (3, 256, 256)
    assert row["mask"].shape == (1, 256, 256)
    assert row["source"].min() >= -1 and row["source"].max() <= 1


def test_path_traversal() -> None:
    with pytest.raises(ValueError, match="inside"):
        local_path(FIXTURES, "../private.png")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("scene", "scene leakage"),
        ("hash", "checksum"),
        ("rights", "commercial rights"),
        ("content", "content leakage"),
    ],
)
def test_manifest_rejections(tmp_path: Path, mutation: str, match: str) -> None:
    import shutil

    for source in FIXTURES.glob("*.png"):
        shutil.copy(source, tmp_path / source.name)
    rows = [json.loads(line) for line in (FIXTURES / "manifest.jsonl").read_text().splitlines()]
    if mutation == "scene":
        rows[1]["scene_id"] = rows[0]["scene_id"]
    elif mutation == "hash":
        rows[0]["sha256"]["input"] = "wrong"
    elif mutation == "rights":
        rows[0]["commercial_use"] = False
    else:
        rows[1]["input"] = rows[0]["input"]
        rows[1]["sha256"]["input"] = rows[0]["sha256"]["input"]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match=match):
        validate_manifest(path, True)


def test_lora_initial_equivalence_and_gradients() -> None:
    torch.manual_seed(7)
    base = nn.Linear(5, 3)
    layer = LoRALinear(base, rank=2, alpha=2)
    x = torch.randn(4, 5)
    assert torch.equal(layer(x), base(x))
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert layer.b.grad is not None and layer.b.grad.abs().sum() > 0
    assert layer.a.grad is not None and layer.a.grad.abs().sum() == 0  # B starts at zero.
    optimizer = torch.optim.SGD([layer.a, layer.b], lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    layer(x).square().mean().backward()
    assert layer.a.grad is not None and layer.a.grad.abs().sum() > 0


def test_region_loss_gradient_and_shapes() -> None:
    predicted = torch.tensor([[[[2.0, 3.0]]]], requires_grad=True)
    target = torch.zeros_like(predicted)
    mask = torch.tensor([[[[1.0, 0.0]]]])
    loss = region_loss(predicted, target, mask, 0.5)
    loss.backward()
    assert loss.item() == pytest.approx((4 + 4.5) / 1.5)
    assert torch.allclose(predicted.grad, torch.tensor([[[[4 / 1.5, 3 / 1.5]]]]))
    latent = torch.randn(2, 4, 8, 8)
    assert inpaint_input(latent, torch.ones(2, 1, 8, 8), latent).shape == (2, 9, 8, 8)
    with pytest.raises(ValueError):
        inpaint_input(latent, torch.ones(2, 2, 8, 8), latent)


def test_checkpoint_resume_equivalence(tmp_path: Path) -> None:
    def setup():
        torch.manual_seed(3)
        model = nn.ModuleDict({"to_q": nn.Linear(5, 3)})
        inject_lora(model, 2, 2)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        return model, optimizer, scaler

    def step(model, optimizer):
        optimizer.zero_grad()
        model["to_q"](torch.ones(2, 5)).square().mean().backward()
        optimizer.step()

    model, optimizer, scaler = setup()
    step(model, optimizer)
    config = {"rank": 2, "steps": 2, "base": "test"}
    save_checkpoint(tmp_path / "step-1", model, optimizer, scaler, 1, config)
    step(model, optimizer)
    other, other_optimizer, other_scaler = setup()
    assert resume_checkpoint(tmp_path / "step-1", other, other_optimizer, other_scaler, config) == 1
    step(other, other_optimizer)
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), other.parameters(), strict=True)
    )
    assert parameter_report(model)["trainable"] == 16
    with pytest.raises(ValueError, match="mismatch"):
        resume_checkpoint(
            tmp_path / "step-1", other, other_optimizer, other_scaler, {**config, "rank": 4}
        )


def test_metrics_identity_and_empty_edges() -> None:
    source = Image.open(FIXTURES / "room.png").convert("RGB")
    mask = Image.new("L", source.size, 255)
    scores = structure_metrics(source, source, mask)
    assert scores["mae"] == 0 and scores["edge_f1"] == 1
    blank = Image.new("RGB", source.size, "white")
    assert structure_metrics(blank, blank, mask)["edge_f1"] is None
    assert structure_metrics(source, blank, mask)["edge_f1"] == 0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        InferenceConfig(steps=2, strength=0.1).validate()
    with pytest.raises(ValueError):
        TrainConfig(accumulation=0).validate()


def test_human_scores(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(
        "reviewer,image,quality,walls,windows,doors,style,furniture,notes\na,A,4,5,,,3,2,ok\n"
    )
    scores = summarize_human(path)
    assert scores["A"]["quality"] == {"mean": 4, "n": 1}


def test_comparison_artifacts_and_fairness(tmp_path: Path) -> None:
    """Synthetic metadata is only a test input, never a benchmark result."""
    import shutil

    meta = {
        "input_sha256": "ab" * 32,
        "mask_sha256": "cd" * 32,
        "depth_sha256": None,
        "prompt": "test <style>",
        "negative_prompt": "",
        "settings": {"seed": 42},
        "base": {"repo": "unit-test"},
        "control": None,
        "environment": {"gpu": "unit-test"},
        "adapter_sha256": None,
        "generation_seconds": 1,
        "total_seconds": 2,
        "peak_gpu_allocated_bytes": 0,
        "peak_gpu_reserved_bytes": 0,
    }
    for name in ("baseline", "candidate"):
        folder = tmp_path / name
        folder.mkdir()
        for asset in ("input.png", "raw.png", "output.png"):
            shutil.copy(FIXTURES / "room.png", folder / asset)
        shutil.copy(FIXTURES / "mask.png", folder / "mask.png")
        (folder / "metadata.json").write_text(json.dumps(meta))
    result = compare_runs(tmp_path / "baseline", tmp_path / "candidate", tmp_path / "compare")
    assert result["automated"]["foundation"]["raw"]["outside_edit_mask"]["mae"] == 0
    assert (tmp_path / "compare" / "comparison.png").exists()
    assert "&lt;style&gt;" in (tmp_path / "compare" / "review.html").read_text(encoding="utf-8")
    meta["settings"]["seed"] = 43
    (tmp_path / "candidate" / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unfair comparison"):
        compare_runs(tmp_path / "baseline", tmp_path / "candidate", tmp_path / "other")


def test_tiny_diffusers_unet_backward() -> None:
    """Real Diffusers attention/added-conditioning integration, random tiny CPU model."""
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=9,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=8,
        addition_embed_type="text_time",
        addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=32,
    )
    inject_lora(unet, rank=2, alpha=2)
    unet.enable_gradient_checkpointing()
    unet.train()
    output = unet(
        torch.randn(1, 9, 8, 8),
        torch.tensor([10]),
        encoder_hidden_states=torch.randn(1, 5, 16),
        added_cond_kwargs={"text_embeds": torch.randn(1, 8), "time_ids": torch.ones(1, 6)},
    ).sample
    region_loss(output, torch.randn_like(output), torch.ones(1, 1, 8, 8)).backward()
    assert output.shape == (1, 4, 8, 8)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in unet.named_parameters()
        if n.endswith(".b")
    )
    assert all(p.grad is None for p in unet.parameters() if not p.requires_grad)
