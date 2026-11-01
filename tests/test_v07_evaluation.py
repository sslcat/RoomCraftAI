"""CPU checks for honest native-resolution evaluation and safe result resumption."""

import importlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from roomcraft.common import model_spec, sha256
from roomcraft.editing import EditConfig, edit_size


@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("v07_evaluate")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def existing(evaluator, tmp_path):
    source = tmp_path / "room.png"
    Image.new("RGB", (1024, 768), "gray").save(source)
    case = {"source": "room.png", "source_sha256": sha256(source), "instruction": "Add a sofa"}
    folder = tmp_path / "output"
    folder.mkdir()
    Image.new("RGB", (512, 384), "white").save(folder / "output.png")
    config = EditConfig(size=512)
    meta = {
        "input_sha256": sha256(source),
        "prompt": case["instruction"],
        "settings": asdict(config),
        "adapter_sha256": None,
        "base": model_spec("qwen"),
        "working_size": [512, 384],
        "output_sha256": sha256(folder / "output.png"),
    }
    (folder / "metadata.json").write_text(json.dumps(meta))
    return case, folder, config, meta


def test_reuses_only_matching_complete_native_output(evaluator, tmp_path):
    case, folder, config, _ = existing(evaluator, tmp_path)
    evaluator.verified_existing(folder, case, config, None)


@pytest.mark.parametrize(
    "size,expected", [(512, (512, 384)), (768, (768, 576)), (1024, (1024, 768))]
)
def test_requested_native_target_sizes(size, expected):
    assert edit_size(Image.new("RGB", (1024, 768)), size) == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter_sha256", "wrong"),
        ("input_sha256", "wrong"),
        ("prompt", "Change the room"),
        ("working_size", [1024, 768]),
        ("base", {"repo": "wrong", "revision": "wrong"}),
    ],
)
def test_rejects_mismatched_reused_results(evaluator, tmp_path, field, value):
    case, folder, config, meta = existing(evaluator, tmp_path)
    meta[field] = value
    (folder / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        evaluator.verified_existing(folder, case, config, None)


def test_rejects_upscaled_output_with_old_metadata(evaluator, tmp_path):
    case, folder, config, meta = existing(evaluator, tmp_path)
    Image.new("RGB", (1024, 768), "white").save(folder / "output.png")
    meta["output_sha256"] = sha256(folder / "output.png")
    (folder / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="native dimensions"):
        evaluator.verified_existing(folder, case, config, None)


def test_gallery_labels_pending_images_without_fake_results(evaluator, tmp_path):
    Image.new("RGB", (1024, 768)).save(tmp_path / "photo.jpg")
    case = {
        "id": "real",
        "domain": "real",
        "source": "photo.jpg",
        "instruction": "Add chair",
        "provenance": {
            "author": "Owner",
            "source_url": "urn:owner",
            "license": "permission",
            "license_url": "urn:permission",
        },
    }
    protocol = {"cases": [case], "runs": [{"case": "real", "size": 1024, "seed": 42}]}
    evaluator.gallery(protocol, tmp_path / "gallery")
    page = (tmp_path / "gallery/real-photo.html").read_text(encoding="utf-8")
    assert page.count("NOT EXECUTED") == 2
    assert "output.png" not in page
