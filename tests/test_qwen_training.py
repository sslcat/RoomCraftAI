"""Tiny random-model tests only: these never produce a trained RoomCraft adapter."""

import copy
import json
from importlib import metadata
from pathlib import Path

import pytest
import torch
from PIL import Image

from roomcraft.checkpoint import resume_checkpoint, save_checkpoint
from roomcraft.common import sha256, write_json
from roomcraft.lora import adapter_state
from roomcraft.qwen_data import split_pairs, validate_pairs
from roomcraft.qwen_objective import (
    flow_path,
    gradient_report,
    inject_qwen_lora,
    paired_loss,
    target_flow_loss,
    validate_tensors,
)
from roomcraft.qwen_training import QwenTrainConfig, sample_seed

QWEN = pytest.mark.skipif(
    metadata.version("diffusers") != "0.37.1", reason="requires isolated Qwen environment"
)


def tiny_model():
    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(22)
    return QwenImageTransformer2DModel(
        in_channels=64,
        out_channels=16,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    )


def tiny_tensors():
    generator = torch.Generator().manual_seed(9)
    return {
        "target": torch.randn(1, 4, 64, generator=generator),
        "source": torch.randn(1, 4, 64, generator=generator),
        "text": torch.randn(1, 3, 16, generator=generator),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
        "shapes": torch.tensor([[1, 2, 2], [1, 2, 2]]),
    }


def test_flow_endpoints_analytic_gradient_and_reference_exclusion():
    clean, noise = torch.zeros(1, 4, 64), torch.ones(1, 4, 64)
    assert torch.equal(flow_path(clean, noise, torch.zeros(1))[0], clean)
    assert torch.equal(flow_path(clean, noise, torch.ones(1))[0], noise)
    prediction = torch.randn(1, 8, 64, requires_grad=True)
    velocity = flow_path(clean, noise, torch.tensor([0.3]))[1]
    target_flow_loss(prediction, velocity).backward()
    assert torch.allclose(
        prediction.grad[:, :4], 2 * (prediction.detach()[:, :4] - velocity) / velocity.numel()
    )
    assert torch.count_nonzero(prediction.grad[:, 4:]) == 0
    with pytest.raises(ValueError, match="sigma"):
        flow_path(clean, noise, torch.tensor([1.2]))


def test_conditioning_shape_contract():
    tensors = tiny_tensors()
    validate_tensors(tensors, text_width=16)
    tensors["shapes"][1, 1] = 3
    with pytest.raises(ValueError, match="grid"):
        validate_tensors(tensors, text_width=16)


@QWEN
def test_real_qwen_lora_backward_checkpointing_and_clean_condition():
    model = tiny_model()
    tensors = tiny_tensors()
    before = paired_loss(model.eval(), tensors, 1).detach()
    assert len(inject_qwen_lora(model, 2, 2)) == 15
    assert torch.allclose(paired_loss(model, tensors, 1).detach(), before)
    model.train().enable_gradient_checkpointing()
    seen = []
    times = []
    time_handle = model.time_text_embed.register_forward_pre_hook(
        lambda module, args: times.append(args[0].detach().clone())
    )
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True
    )
    loss = paired_loss(model, tensors, 1)
    loss.backward()
    handle.remove()
    time_handle.remove()
    assert gradient_report(model)["gradient_tensors"] == 30
    assert torch.equal(seen[0]["hidden_states"][:, 4:], tensors["source"])
    assert seen[0]["timestep"].shape == (1,)
    assert times[0].shape == (2,) and times[0][1] == 0
    assert torch.equal(times[0][:1], seen[0]["timestep"])
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(
        torch.count_nonzero(p.grad) == 0 for n, p in model.named_parameters() if n.endswith(".a")
    )
    assert any(
        torch.count_nonzero(p.grad) > 0 for n, p in model.named_parameters() if n.endswith(".b")
    )


@QWEN
def test_accumulation_equals_mean_gradient_and_resume(tmp_path):
    model = tiny_model()
    inject_qwen_lora(model, 2, 2)
    other = copy.deepcopy(model)
    data = tiny_tensors()
    for seed in (10, 11):
        (paired_loss(model, data, seed) / 2).backward()
    ((paired_loss(other, data, 10) + paired_loss(other, data, 11)) / 2).backward()
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        if p.requires_grad:
            assert torch.allclose(p.grad, q.grad, atol=1e-7)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    optimizer.step()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    config = {"steps": 1, "seed": 42}
    save_checkpoint(tmp_path / "step", model, optimizer, scaler, 1, config)
    resumed = tiny_model()
    inject_qwen_lora(resumed, 2, 2)
    opt2 = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=0.01)
    assert resume_checkpoint(tmp_path / "step", resumed, opt2, scaler, {**config, "steps": 2}) == 1
    for network, opt in ((model, optimizer), (resumed, opt2)):
        opt.zero_grad(set_to_none=True)
        paired_loss(network, data, sample_seed(42, 1, 0, 1)).backward()
        gradient_report(network)
        opt.step()
    for name, value in adapter_state(model).items():
        assert torch.equal(value, adapter_state(resumed)[name])
    with pytest.raises(ValueError, match="seed"):
        resume_checkpoint(tmp_path / "step", resumed, opt2, scaler, {"seed": 43})


def make_manifest(root: Path) -> Path:
    """Original synthetic pixels solely for validator tests, never a training fixture."""
    (root / "rights.txt").write_text("Synthetic test rights assertion; not real data.")
    rows = []
    for i in range(6):
        source, target = f"source{i}.png", f"target{i}.png"
        Image.new("RGB", (64, 64), (i * 10, 0, 0)).save(root / source)
        Image.new("RGB", (64, 64), (i * 10, 50, 0)).save(root / target)
        rows.append(
            {
                "id": f"pair{i}",
                "scene_id": f"scene{i}",
                "group_id": f"property{i}",
                "source": source,
                "target": target,
                "instruction": "Add furniture",
                "kind": "render_pair",
                "creator": "test",
                "source_url": "test-only",
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "evidence": "rights.txt",
                "alignment_evidence": "rights.txt",
                "rights": {
                    "commercial_training": True,
                    "derivatives": True,
                    "adapter_distribution": True,
                    "image_redistribution": False,
                },
                "review": {
                    "reviewer": "unit-test",
                    "date": "2026-10-28",
                    "same_camera": True,
                    "architecture_unchanged": True,
                    "source_empty": True,
                    "target_furnished": True,
                },
                "sha256": {
                    "source": sha256(root / source),
                    "target": sha256(root / target),
                    "evidence": sha256(root / "rights.txt"),
                    "alignment_evidence": sha256(root / "rights.txt"),
                },
            }
        )
    path = root / "pairs.json"
    write_json(path, {"schema": "roomcraft-pairs-v1", "pairs": rows})
    return path


def test_rights_splits_evidence_and_leakage(tmp_path):
    source = make_manifest(tmp_path)
    out = tmp_path / "split.json"
    first = split_pairs(source, out, seed=42)
    assert len(validate_pairs(out)) == 6
    assert split_pairs(source, tmp_path / "split2.json", seed=42) == first
    row = first["pairs"][0]
    row["rights"]["commercial_training"] = False
    write_json(out, first)
    with pytest.raises(ValueError, match="rights"):
        validate_pairs(out)
    row["rights"]["commercial_training"] = True
    row["review"]["same_camera"] = False
    write_json(out, first)
    with pytest.raises(ValueError, match="same_camera"):
        validate_pairs(out)
    row["review"]["same_camera"] = True
    other = next(r for r in first["pairs"] if r["split"] != row["split"])
    other["group_id"] = row["group_id"]
    write_json(out, first)
    with pytest.raises(ValueError, match="leakage"):
        validate_pairs(out)


def test_unreviewed_or_empty_data_cannot_train(tmp_path):
    path = tmp_path / "empty.json"
    write_json(path, {"schema": "roomcraft-pairs-v1", "pairs": []})
    with pytest.raises(ValueError, match="no approved pairs"):
        validate_pairs(path)
    for config in (
        QwenTrainConfig(accumulation=0),
        QwenTrainConfig(learning_rate=float("nan")),
        QwenTrainConfig(rank=-1),
    ):
        with pytest.raises(ValueError):
            config.validate()
    assert json.loads(path.read_text())["pairs"] == []
