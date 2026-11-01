"""Cue interventions and actual tiny-Qwen integration, not visual-quality evidence."""

import json
import runpy
import time
from pathlib import Path

import pytest
import torch
from PIL import Image
from test_rga import QWEN, forward_args, geometry, tiny

from roomcraft.causality import (
    edge_contrast_features,
    intervene,
    load_intervention,
    save_intervention,
)
from roomcraft.rga import GeometryEncoder, RGAConfig, geometry_conditioning, token_residual


def test_interventions_do_not_move_coordinates_or_wrap_edges():
    g = geometry()
    g[:, 0] = 0
    g[0, 0, 3, 30] = 1
    g[0, 0, 8, 2] = 1
    shifted = intervene(g, "shifted", shift=(0, 4))
    assert shifted[0, 0, 8, 6] == 1 and shifted[:, 0].sum() == 1
    assert torch.equal(g[:, 1:], shifted[:, 1:]) and g[0, 0, 3, 30] == 1
    assert torch.equal(intervene(g, "disabled"), g)
    assert intervene(g, "zero_edges")[:, 0].count_nonzero() == 0
    with pytest.raises(ValueError, match="donor"):
        intervene(g, "unrelated")


def test_count_matched_removal_and_fail_closed():
    g = geometry()
    region = torch.zeros(32, 32, dtype=torch.bool)
    region[4:7, 10:20] = True
    removed = intervene(g, "removed", region=region)
    random = intervene(g, "random_removed", region=region)
    assert (g[:, 0] != removed[:, 0]).sum() == (g[:, 0] != random[:, 0]).sum() > 0
    assert torch.equal(random, intervene(g, "random_removed", region=region))
    assert torch.equal(random[0, 0][region], g[0, 0][region])
    with pytest.raises(ValueError, match="no source edges"):
        intervene(g, "removed", region=torch.zeros_like(region))


def test_intervention_packages_bind_source_size_and_tensor(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32)).save(source)
    path = tmp_path / "cue.json"
    g = geometry()
    save_intervention(path, g, source, {"arm": "correct"})
    actual, _ = load_intervention(path, source, (32, 32))
    assert torch.equal(actual, g)
    with pytest.raises(ValueError, match="immutable"):
        save_intervention(path, g, source, {})
    with pytest.raises(ValueError, match="dimensions"):
        load_intervention(path, source, (64, 32))
    Image.new("RGB", (32, 32), "red").save(source)
    with pytest.raises(ValueError, match="source"):
        load_intervention(path, source, (32, 32))


def test_target_grid_alignment_excludes_reference_tokens():
    features = torch.arange(4 * 16, dtype=torch.float32).reshape(1, 4, 16)
    residual = token_residual(
        features, torch.zeros(1, 10, 64), [[(1, 2, 2), (1, 2, 3)]], 2, 1, (2, 2)
    )
    assert torch.equal(residual[:, :4], features / 2)
    assert residual[:, 4:].count_nonzero() == 0
    with pytest.raises(ValueError, match="lattice"):
        token_residual(features, torch.zeros(1, 10, 64), [[(1, 1, 4), (1, 2, 3)]], 2, 1, (2, 2))


def test_contrast_candidate_zero_cue_and_parameter_gradients():
    torch.manual_seed(9)
    encoder = GeometryEncoder(RGAConfig(width=8, hidden_dim=16))
    with torch.no_grad():
        encoder.projection.weight.normal_(std=0.03)
        encoder.projection.bias.fill_(0.7)
    g = geometry()
    features = edge_contrast_features(encoder, g)
    assert (
        features.norm() > 0
        and edge_contrast_features(encoder, intervene(g, "zero_edges")).count_nonzero() == 0
    )
    features.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert sum(float(p.grad.abs().sum()) for p in encoder.encoder.parameters()) > 0
    assert encoder.projection.bias.grad.count_nonzero() == 0


@QWEN
def test_tiny_qwen_intervention_effect_and_disabled_baseline():
    model = tiny()
    args = forward_args()
    g = geometry()
    with torch.no_grad():
        model.geometry.projection.weight.normal_(std=0.1)
    frozen = {k: v.clone() for k, v in model.transformer.state_dict().items()}
    with torch.no_grad():
        baseline = model.transformer(**args)[0]
        with geometry_conditioning(model.transformer, model.geometry, g, 0):
            disabled = model.transformer(**args)[0]
        with geometry_conditioning(model.transformer, model.geometry, g, 1):
            correct = model.transformer(**args)[0]
        with geometry_conditioning(
            model.transformer, model.geometry, intervene(g, "shifted", shift=(0, 8)), 1
        ):
            shifted = model.transformer(**args)[0]
    assert torch.equal(baseline, disabled) and not torch.equal(correct, shifted)
    assert all(not p.requires_grad for p in model.transformer.parameters())
    assert all(torch.equal(v, model.transformer.state_dict()[k]) for k, v in frozen.items())


def test_gaming_recipes_reproducible_and_diverse():
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/generate_gaming_dataset.py"))
    family, variant = module["family_parameters"], module["variant_parameters"]
    families = [family(i, 100731) for i in range(20)]
    assert families == [family(i, 100731) for i in range(20)]
    assert len({(f["width"], f["depth"], f["window_offset"]) for f in families}) == 20
    rows = [variant(f, j, 100731) for f in families for j in range(5)]
    assert len({r["id"] for r in rows}) == 100 and {r["monitors"] for r in rows} == {1, 2, 3}
    assert {r["style"] for r in rows} == {"minimalist", "professional", "streamer", "RGB-heavy"}
    assert {f["desk_wall"] for f in families} == {"back", "left", "right"}


def test_gaming_split_plan_separates_families_and_covers_room_uses():
    root = Path(__file__).parents[1]
    plan = json.loads((root / "configs/dataset_v10_split.json").read_text())
    module = runpy.run_path(str(root / "scripts/generate_gaming_dataset.py"))
    seen = set()
    for split, count in (("train", 12), ("validation", 4), ("test", 4)):
        groups = set(plan[split])
        assert len(groups) == count and not seen & groups
        uses = {
            module["family_parameters"](int(g.split("_")[1]), 100731)["room_use"] for g in groups
        }
        assert len(uses) == 4
        seen |= groups
    assert len(seen) == 20


def test_cloud_runner_requires_bounded_deadline_before_model_loading(tmp_path):
    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/v10_causality.py"))
    for deadline in (None, time.time() - 1, time.time() + 7200):
        with pytest.raises(ValueError, match="deadline"):
            script["run"](tmp_path / "missing-plan.json", tmp_path / "output", deadline)
    assert not (tmp_path / "output").exists()


@QWEN
def test_inference_contrast_hook_matches_candidate_and_cleans_up(tmp_path, monkeypatch):
    from roomcraft import rga_inference

    model = tiny()
    adapter, transformer = model.geometry, model.transformer
    with torch.no_grad():
        adapter.projection.weight.normal_(std=0.1)
        adapter.projection.bias.fill_(0.7)
    saved = {k: v.clone() for k, v in adapter.state_dict().items()}
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "adapter.safetensors").write_bytes(b"test identity only")
    monkeypatch.setattr(rga_inference, "load_geometry_adapter", lambda *args: adapter)
    g = geometry()
    captured = []
    with torch.inference_mode():
        expected = edge_contrast_features(adapter, g)
        with rga_inference.rga_edit_context(
            transformer,
            tmp_path,
            "test",
            Image.new("RGB", (32, 32)),
            (32, 32),
            1,
            intervention=(g, {}),
            edge_contrast=True,
        ) as record:
            handle = transformer.register_forward_pre_hook(
                lambda _m, _a, kwargs: captured.append(
                    kwargs["controlnet_block_samples"][0].clone()
                ),
                with_kwargs=True,
            )
            try:
                transformer(**forward_args())
            finally:
                handle.remove()
            assert record["edge_contrast_ablation"] is True
        assert torch.allclose(captured[0][:, :4], expected / 2)
        assert captured[0][:, 4:].count_nonzero() == 0
    assert not adapter._forward_hooks
    assert all(torch.equal(v, adapter.state_dict()[k]) for k, v in saved.items())
