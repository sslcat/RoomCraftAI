"""Tiny artificial images test bookkeeping only, never render-quality evidence."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

from roomcraft.common import sha256, write_json
from roomcraft.qwen_data import split_pairs, validate_pairs
from roomcraft.render_dataset import audit_render, make_gallery, reviewed_ledger


def fixture_pair(root: Path, index: int) -> Path:
    folder = root / "pairs" / f"family_{index:02d}_v00"
    folder.mkdir(parents=True)
    for key, level in (("source", 80 + index), ("target", 180 + index)):
        Image.new("RGB", (32, 24), (level, level, level)).save(folder / f"{key}.png")
    state = {"camera": [0, 0, 0], "test_only": True}
    write_json(folder / "invariant_state.json", state)
    write_json(
        folder / "render.json",
        {
            "variant": {"id": folder.name},
            "family": {"family": f"family_{index:02d}"},
            "paired_invariant_verified": True,
            "architecture_camera_lighting_sha256": hashlib.sha256(
                json.dumps(state, sort_keys=True).encode()
            ).hexdigest(),
            "sha256": {key: sha256(folder / f"{key}.png") for key in ("source", "target")},
            "settings": {"resolution": [32, 24]},
            "generation_seconds": {"source": 0, "target": 0},
            "instruction": "Test fixture only",
            "asset_provenance_sha256": "fixture",
        },
    )
    return folder


def test_generation_parameters_are_deterministic_and_diverse():
    generator = runpy.run_path(str(Path(__file__).parents[1] / "scripts/generate_room_dataset.py"))
    family = generator["family_parameters"]
    variant = generator["variant_parameters"]
    families = [family(i, 53021) for i in range(20)]
    assert families == [family(i, 53021) for i in range(20)]
    assert len({(f["width"], f["depth"], f["window_offset"]) for f in families}) == 20
    assert {f["kind"] for f in families} == {"living", "bedroom", "dining"}
    assert {f["door_side"] for f in families} == {"left", "right"}
    assert len({variant(f, v, 53021)["seed"] for f in families for v in range(5)}) == 100
    assert len({variant(f, v, 53021)["id"] for f in families for v in range(5)}) == 100


def test_render_audit_rejects_modified_image_and_state(tmp_path):
    folder = fixture_pair(tmp_path, 0)
    assert audit_render(folder)["visual_quality"] == "requires explicit review"
    state_path = folder / "invariant_state.json"
    state = state_path.read_bytes()
    write_json(state_path, {"camera": [0, 1, 0]})
    with pytest.raises(ValueError, match="architectural state"):
        audit_render(folder)
    state_path.write_bytes(state)
    Image.new("RGB", (32, 24), "red").save(folder / "target.png")
    with pytest.raises(ValueError, match="changed target"):
        audit_render(folder)


def test_render_review_to_existing_split_validator(tmp_path):
    folders = [fixture_pair(tmp_path, i) for i in range(3)]
    (tmp_path / "provenance").mkdir()
    write_json(
        tmp_path / "provenance/rights.json",
        {
            "asset_provenance_sha256": "fixture",
            "rights": {
                k: True
                for k in (
                    "commercial_training",
                    "derivatives",
                    "adapter_distribution",
                    "image_redistribution",
                )
            },
        },
    )
    reviews = {
        "reviewer": "unit test",
        "date": "2026-10-28",
        "method": "fixture only",
        "pairs": [
            {
                "id": p.name,
                "decision": "accept",
                "sha256": audit_render(p)["sha256"],
                "notes": "Bookkeeping fixture",
            }
            for p in folders
        ],
    }
    review_path = tmp_path / "quality_reviews.json"
    write_json(review_path, reviews)
    data = reviewed_ledger(tmp_path, review_path)
    for row in data["pairs"]:
        row["sha256"] = {
            key: sha256(tmp_path / row[key])
            for key in ("source", "target", "evidence", "alignment_evidence")
        }
    manifest = tmp_path / "sealed.json"
    write_json(manifest, data)
    split_pairs(manifest, tmp_path / "manifest.json")
    assert len(validate_pairs(tmp_path / "manifest.json")) == 3
    assert len(make_gallery(tmp_path)) == 3
    assert (tmp_path / "gallery/index.html").exists()
    reviews["pairs"][0]["sha256"]["source"] = "changed"
    write_json(review_path, reviews)
    with pytest.raises(ValueError, match="bind"):
        reviewed_ledger(tmp_path, review_path)
    with pytest.raises(ValueError, match="stale"):
        make_gallery(tmp_path)


def test_review_cannot_autoapprove_pending_or_duplicate_decisions(tmp_path):
    folder = fixture_pair(tmp_path, 0)
    row = {
        "id": folder.name,
        "decision": "pending",
        "sha256": audit_render(folder)["sha256"],
        "notes": "unreviewed",
    }
    reviews = {"reviewer": "test", "date": "2026-10-28", "method": "fixture", "pairs": [row]}
    path = tmp_path / "quality_reviews.json"
    write_json(path, reviews)
    with pytest.raises(ValueError, match="explicit"):
        reviewed_ledger(tmp_path, path)
    row["decision"] = "reject"
    reviews["pairs"] = [row, row]
    write_json(path, reviews)
    with pytest.raises(ValueError, match="duplicate"):
        reviewed_ledger(tmp_path, path)
