"""Random tiny Qwen only: learning/gradient evidence is not image-quality evidence."""

import copy
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from roomcraft.checkpoint import resume_checkpoint, save_checkpoint
from roomcraft.common import model_spec
from roomcraft.editing import EditConfig, edit
from roomcraft.qwen_data import split_pairs
from roomcraft.qwen_objective import gradient_report, inject_qwen_lora, paired_loss
from roomcraft.rga import (
    GeometryConditionedQwen,
    GeometryEncoder,
    RGAConfig,
    architecture_weights,
    geometry_conditioning,
    geometry_input,
    masked_velocity_loss,
    rga_loss,
    rga_record,
    token_residual,
)
from roomcraft.rga_data import prepare_geometry, validate_geometry
from roomcraft.rga_inference import load_geometry_adapter
from roomcraft.rga_training import RGATrainConfig

QWEN = pytest.mark.skipif(
    metadata.version("diffusers") != "0.37.1", reason="isolated Qwen environment required"
)


def geometry():
    image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(image).rectangle((8, 5, 27, 23), outline="black", width=2)
    return geometry_input(image, (32, 32))


def tiny():
    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(22)
    model = QwenImageTransformer2DModel(
        in_channels=64,
        out_channels=16,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    )
    inject_qwen_lora(model, 2, 2)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(".b"):
                p.normal_(std=0.01)  # Simulated nonzero LoRA, not pretrained RoomCraft weights.
    return GeometryConditionedQwen(model, RGAConfig(width=8, hidden_dim=16))


def tensors():
    gen = torch.Generator().manual_seed(9)
    return {
        "target": torch.randn(1, 4, 64, generator=gen),
        "source": torch.randn(1, 6, 64, generator=gen),
        "text": torch.randn(1, 3, 16, generator=gen),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
        "shapes": torch.tensor([[1, 2, 2], [1, 2, 3]]),
    }


def forward_args():
    data = tensors()
    return {
        "hidden_states": torch.cat([data["target"], data["source"]], 1),
        "timestep": torch.tensor([0.4]),
        "encoder_hidden_states": data["text"],
        "encoder_hidden_states_mask": data["text_mask"],
        "img_shapes": [[(1, 2, 2), (1, 2, 3)]],
        "return_dict": False,
    }


def test_source_cues_flags_alignment_and_depth():
    g = geometry()
    assert g.shape == (1, 7, 32, 32) and g[:, 0].sum() > 0
    assert g[:, 1:5].count_nonzero() == 0
    assert torch.all(g[0, 5, :, 0] < g[0, 5, :, -1])
    image = Image.new("RGB", (32, 32))
    mask = Image.new("L", (32, 32))
    ImageDraw.Draw(mask).rectangle((0, 0, 15, 15), fill=255)
    g = geometry_input(image, (32, 32), mask, np.arange(1024).reshape(32, 32))
    assert torch.all(g[:, (2, 4)] == 1)
    assert architecture_weights(g).flatten().tolist() == [1, 0, 0, 0]
    with pytest.raises(ValueError, match="align"):
        geometry_input(image, (32, 32), depth=np.zeros((16, 16)))
    with pytest.raises(ValueError, match="binary"):
        geometry_input(image, (32, 32), mask=Image.new("L", (32, 32), 120))


def test_encoder_grid_and_zero_residual_reference_exclusion():
    encoder = GeometryEncoder(RGAConfig(width=8, hidden_dim=16))
    features = encoder(geometry())
    assert features.shape == (1, 4, 16) and torch.count_nonzero(features) == 0
    residual = token_residual(
        features + 1, torch.zeros(1, 10, 64), [[(1, 2, 2), (1, 2, 3)]], 2, 1, (2, 2)
    )
    assert torch.all(residual[:, :4] == 0.5) and torch.count_nonzero(residual[:, 4:]) == 0
    with pytest.raises(ValueError, match="lattice"):
        token_residual(
            torch.ones(1, 6, 16), torch.zeros(1, 10, 64), [[(1, 2, 3), (1, 2, 2)]], 2, 1, (3, 2)
        )
    with pytest.raises(ValueError, match="geometry"):
        encoder(torch.zeros(1, 7, 33, 32))


def test_masked_velocity_analytic_gradient_and_empty_mask_rejection():
    prediction = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
    target = torch.randn(1, 4, 3, dtype=torch.float64)
    weights = torch.tensor([[[1.0], [0.0], [0.5], [0.0]]])
    loss = masked_velocity_loss(prediction, target, weights)
    loss.backward()
    expected = (
        2 * weights * (prediction.detach()[:, :4].float() - target.float()) / (3 * weights.sum())
    )
    assert torch.allclose(prediction.grad[:, :4].float(), expected, atol=1e-7)
    assert prediction.grad[:, 4:].count_nonzero() == 0
    with pytest.raises(ValueError, match="nonempty"):
        masked_velocity_loss(prediction, target, torch.zeros_like(weights))


def test_geometry_sidecars_preserve_splits_and_reject_tampering(tmp_path):
    from test_qwen_training import make_manifest

    manifest = tmp_path / "split.json"
    split_pairs(make_manifest(tmp_path), manifest, seed=42)
    output = tmp_path / "geometry"
    record = prepare_geometry(manifest, output, size=256)
    cache = {"manifest_sha256": record["manifest_sha256"], "size": 256, "rows": record["rows"]}
    assert len(validate_geometry(output / "geometry.json", cache)["rows"]) == 6
    with pytest.raises(ValueError, match="resolution"):
        validate_geometry(output / "geometry.json", {**cache, "size": 512})
    with pytest.raises(ValueError, match="new"):
        prepare_geometry(manifest, output, size=256)
    file = output / record["rows"][0]["file"]
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        validate_geometry(output / "geometry.json")
    for cfg in (RGATrainConfig(accumulation=0), RGATrainConfig(architecture_lambda=-1)):
        with pytest.raises(ValueError):
            cfg.validate()


def test_invalid_rga_inference_arguments_fail_before_image_or_gpu_access(tmp_path):
    for kwargs in (
        {"rga_mask": Path("missing-mask.png")},
        {"rga_depth": Path("missing-depth.npy")},
        {"rga_strength": float("nan")},
        {"rga_checkpoint": Path("missing-checkpoint")},
    ):
        with pytest.raises(ValueError):
            edit(
                Path("missing-room.png"), "Add furniture", tmp_path / "out", EditConfig(), **kwargs
            )


@QWEN
def test_zero_initialization_preserves_real_qwen_and_existing_lora():
    model = tiny().eval()
    before = copy.deepcopy(model.transformer.state_dict())
    baseline = model.transformer(**forward_args())[0]
    conditioned = model(geometry=geometry(), **forward_args())[0]
    assert torch.equal(baseline, conditioned)
    assert torch.equal(
        paired_loss(model.transformer, tensors(), 123),
        rga_loss(model, tensors(), geometry(), 123)[0],
    )
    assert all(not p.requires_grad for p in model.transformer.parameters())
    assert all(torch.equal(v, model.transformer.state_dict()[k]) for k, v in before.items())


@QWEN
def test_real_qwen_learning_gradients_condition_effect_and_checkpointing():
    model = tiny()
    frozen = copy.deepcopy(model.transformer.state_dict())
    model.transformer.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(model.geometry.parameters(), lr=0.005, weight_decay=0)
    history = []
    encoder_norms = []
    for _ in range(8):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = rga_loss(model, tensors(), geometry(), 123)
        loss.backward()
        gradient_report(model)
        encoder_norms.append(
            sum(float(p.grad.square().sum()) for p in model.geometry.encoder.parameters())
        )
        assert all(torch.isfinite(p.grad).all() for p in model.geometry.parameters())
        history.append(float(loss.detach()))
        optimizer.step()
    assert encoder_norms[0] == 0 and encoder_norms[1] > 0
    assert history[-1] < history[0]
    changed = geometry().clone()
    changed[:, 0] = 0
    a = model(geometry=geometry(), **forward_args())[0]
    b = model(geometry=changed, **forward_args())[0]
    assert (a - b).abs().max() > 1e-6
    assert all(p.grad is None for p in model.transformer.parameters())
    assert all(torch.equal(v, model.transformer.state_dict()[k]) for k, v in frozen.items())
    print(
        json.dumps(
            {
                "random_tiny_qwen_only": True,
                "losses": history,
                "encoder_squared_gradient": encoder_norms,
                "edge_ablation_max_delta": float((a - b).abs().max().detach()),
            }
        )
    )


@QWEN
def test_rga_accumulation_and_exact_optimizer_resume(tmp_path):
    model = tiny()
    other = copy.deepcopy(model)
    for seed in (3, 4):
        (rga_loss(model, tensors(), geometry(), seed)[0] / 2).backward()
    (
        (
            rga_loss(other, tensors(), geometry(), 3)[0]
            + rga_loss(other, tensors(), geometry(), 4)[0]
        )
        / 2
    ).backward()
    for p, q in zip(model.geometry.parameters(), other.geometry.parameters(), strict=True):
        assert torch.allclose(p.grad, q.grad, atol=1e-7)
    opt = torch.optim.AdamW(model.geometry.parameters(), lr=0.005)
    opt.step()
    record = {
        **rga_record(model.geometry.config),
        "base": model_spec("qwen"),
        "frozen_lora_sha256": "random-tiny-test-only",
        "steps": 1,
        "seed": 42,
    }
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(tmp_path / "step", model, opt, scaler, 1, record)
    saved_geometry = copy.deepcopy(model.geometry.state_dict())
    resumed = tiny()
    opt2 = torch.optim.AdamW(resumed.geometry.parameters(), lr=0.005)
    assert resume_checkpoint(tmp_path / "step", resumed, opt2, scaler, {**record, "steps": 2}) == 1
    assert torch.equal(
        model(geometry=geometry(), **forward_args())[0],
        resumed(geometry=geometry(), **forward_args())[0],
    )
    for m, o in ((model, opt), (resumed, opt2)):
        o.zero_grad(set_to_none=True)
        rga_loss(m, tensors(), geometry(), 456)[0].backward()
        o.step()
    for p, q in zip(model.geometry.parameters(), resumed.geometry.parameters(), strict=True):
        assert torch.equal(p, q)
    loaded = load_geometry_adapter(tmp_path / "step", resumed.transformer, "random-tiny-test-only")
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in saved_geometry.items())
    with pytest.raises(ValueError, match="identity"):
        load_geometry_adapter(tmp_path / "step", resumed.transformer, "different-lora")
    with pytest.raises(ValueError, match="seed"):
        resume_checkpoint(tmp_path / "step", resumed, opt2, scaler, {**record, "seed": 43})


@QWEN
def test_inference_context_matches_training_path_cfg_calls_disable_and_cleanup():
    model = tiny()
    torch.nn.init.normal_(model.geometry.projection.weight, std=0.01)
    expected = model(geometry=geometry(), **forward_args())[0]
    with torch.inference_mode():
        baseline = model.transformer(**forward_args())[0]
        with geometry_conditioning(model.transformer, model.geometry, geometry()):
            assert torch.equal(model.transformer(**forward_args())[0], expected)
            assert torch.equal(model.transformer(**forward_args())[0], expected)
            with pytest.raises(RuntimeError, match="nested"):
                with geometry_conditioning(model.transformer, model.geometry, geometry()):
                    pass
        with geometry_conditioning(model.transformer, model.geometry, geometry(), 0):
            assert torch.equal(model.transformer(**forward_args())[0], baseline)
        with pytest.raises(ValueError, match="test cleanup"):
            with geometry_conditioning(model.transformer, model.geometry, geometry()):
                raise ValueError("test cleanup")
        assert torch.equal(model.transformer(**forward_args())[0], baseline)
    assert not getattr(model.transformer, "_roomcraft_rga_active", False)


@QWEN
def test_cpu_bf16_zero_path_and_finite_master_gradients():
    model = tiny()
    data = tensors()
    for key in ("target", "source", "text"):
        data[key] = data[key].bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        baseline = paired_loss(model.transformer, data, 123)
        loss, _ = rga_loss(model, data, geometry(), 123)
    assert torch.equal(baseline, loss)
    loss.backward()
    gradient_report(model)
    assert all(p.dtype == torch.float32 for p in model.geometry.parameters())
    assert all(p.grad is None for p in model.transformer.parameters())


@QWEN
def test_optional_architecture_loss_is_connected_and_refuses_empty_masks():
    model = tiny()
    with pytest.raises(ValueError, match="nonempty"):
        rga_loss(model, tensors(), geometry(), 123, 0.2)
    g = geometry()
    g[:, 1:3] = 1  # Unit-test annotation only, not a dataset label.
    loss, parts = rga_loss(model, tensors(), g, 123, 0.2)
    assert parts["architecture"] > 0
    assert float(loss.detach()) == pytest.approx(parts["flow"] + 0.2 * parts["architecture"])
    loss.backward()
    gradient_report(model)


def test_frozen_tensor_digest_detects_bf16_weight_mutation():
    from roomcraft.rga_training import frozen_digest

    model = torch.nn.Linear(4, 3).to(dtype=torch.bfloat16).requires_grad_(False)
    before = frozen_digest(model)
    assert before == frozen_digest(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert before != frozen_digest(model)


@QWEN
def test_residual_diagnostics_observe_without_changing_predictions():
    from roomcraft.rga_inference import residual_diagnostics

    model = tiny()
    torch.nn.init.normal_(model.geometry.projection.weight, std=0.01)
    record = {}
    with torch.inference_mode():
        with geometry_conditioning(model.transformer, model.geometry, geometry()):
            expected = model.transformer(**forward_args())[0]
            with residual_diagnostics(model.transformer, record):
                actual = model.transformer(**forward_args())[0]
    assert torch.equal(actual, expected)
    assert set(record) == {"first_block", "last_block"}
    assert all(0 <= v["fraction_unchanged_after_add"] <= 1 for v in record.values())
    assert all(v["residual_rms"] > 0 for v in record.values())


@QWEN
def test_calibrated_gain_preserves_zero_baseline_and_scales_projection_gradient():
    import copy
    from dataclasses import replace

    original = tiny()
    calibrated = copy.deepcopy(original)
    calibrated.geometry.config = replace(original.geometry.config, residual_gain=100)
    a, _ = rga_loss(original, tensors(), geometry(), 42)
    b, _ = rga_loss(calibrated, tensors(), geometry(), 42)
    assert torch.equal(a, b)
    a.backward()
    b.backward()
    for p, q in zip(
        original.geometry.projection.parameters(),
        calibrated.geometry.projection.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(q.grad, 100 * p.grad, rtol=1e-4, atol=1e-6)
    assert RGAConfig(**{"width": 32, "hidden_dim": 3072, "strength": 1}).residual_gain == 1
    for gain in (0, -1, float("nan"), 1e10):
        with pytest.raises(ValueError, match="residual gain"):
            RGAConfig(residual_gain=gain).validate()


def test_validation_safety_configuration_rejects_invalid_limits():
    from roomcraft.rga_training import RGATrainConfig

    RGATrainConfig(validation_loss_ceiling=0.1, validation_patience=2).validate()
    for ceiling in (-1, 0, float("nan")):
        with pytest.raises(ValueError, match="validation loss ceiling"):
            RGATrainConfig(validation_loss_ceiling=ceiling).validate()
    with pytest.raises(ValueError):
        RGATrainConfig(validation_patience=0).validate()
