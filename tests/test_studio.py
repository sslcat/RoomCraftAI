"""Room-purpose transport and mask occlusion regression tests."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from roomcraft.app_service import AppConfig, AppService
from roomcraft.gaming import GamingConfig, compose_gaming
from roomcraft.preservation import selective_composite


def test_purposes_have_distinct_equipment_and_receipts():
    prompts = {}
    for purpose in ("Gaming", "Work", "Relaxation", "Mixed-purpose"):
        prompt, receipt = compose_gaming(GamingConfig(purpose=purpose))
        assert receipt["configuration"]["purpose"] == purpose
        assert "windows" in prompt and "camera viewpoint" in prompt
        prompts[purpose] = prompt
    assert len(set(prompts.values())) == 4
    assert "PC tower" not in prompts["Relaxation"]
    assert "task chair" in prompts["Work"]
    assert "share the desk" in prompts["Mixed-purpose"]


@pytest.mark.parametrize(
    "field,value,purpose",
    [
        ("workstation", "Sit-stand desk", "Work"),
        ("work_screens", "Dual monitors", "Work"),
        ("seating", "Compact sofa", "Relaxation"),
        ("reading", False, "Relaxation"),
        ("mixed_functions", "Gaming + relaxation", "Mixed-purpose"),
    ],
)
def test_visible_purpose_controls_change_prompt(field, value, purpose):
    c = GamingConfig(purpose=purpose)
    assert compose_gaming(c)[0] != compose_gaming(replace(c, **{field: value}))[0]


def test_feather_cannot_restore_over_foreground_or_outside_mask():
    src = Image.new("RGB", (80, 80), "white")
    raw = Image.new("RGB", (80, 80), "black")
    mask = Image.new("L", src.size, 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 60, 60), fill=255)
    fg = Image.new("L", src.size, 0)
    ImageDraw.Draw(fg).rectangle((20, 20, 40, 40), fill=255)
    result, alpha = selective_composite(src, raw, mask, fg, 8)
    assert result.getpixel((30, 30)) == (0, 0, 0)
    assert result.getpixel((70, 70)) == (0, 0, 0)
    assert alpha.getpixel((30, 30)) == 0
    assert result.getpixel((2, 2)) == (255, 255, 255)
    with pytest.raises(ValueError):
        selective_composite(src, raw, Image.new("L", (10, 10)), None)


def test_preservation_creates_labeled_derivative_and_keeps_raw(tmp_path):
    service = AppService(tmp_path / "app", AppConfig())
    token = service.session()
    raw = service.example(token, "roomcraft")
    original_hash = raw["metadata"]["output_sha256"]
    with Image.open(raw["source"]) as image:
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((0, 0, 100, 100), fill=255)
        mask.save(tmp_path / "mask.png")
    result = service.preserve(token, raw["metadata"], str(tmp_path / "mask.png"), None, 8)
    assert len(service.records(token)) == 2
    assert raw["metadata"]["output_sha256"] == original_hash
    assert result["metadata"]["selective_preservation"]["raw_output_sha256"] == original_hash
    assert (
        json.loads(Path(result["metadata_path"]).read_text())["application"]["origin"]
        == "selective_composite"
    )
