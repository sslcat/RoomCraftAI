"""Stage corrected V0.6 pairs, then seal only after explicit hash-bound visual review."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from roomcraft.common import sha256, write_json
from roomcraft.qwen_data import validate_pairs
from roomcraft.render_dataset import audit_render, make_gallery


def stage(root):
    if root.exists():
        raise ValueError("staging root must be new")
    parent = Path("data/v06")
    document = json.loads((parent / "manifest.json").read_text())
    validate_pairs(parent / "manifest.json")
    root.mkdir(parents=True)
    shutil.copytree(parent / "provenance", root / "provenance")
    repair_script = root / "provenance/repair_v06_plants.py"
    shutil.copy2("scripts/repair_v06_plants.py", repair_script)
    rights_path = root / "provenance/rights.json"
    rights = json.loads(rights_path.read_text())
    rights["evidence_files"]["provenance/repair_v06_plants.py"] = sha256(repair_script)
    rights["v06_change"] = (
        "Replaced floating plant 02 with already licensed CC0 plant 01; original contributions retain MIT terms. V0.5 rights and per-asset evidence retained."
    )
    write_json(rights_path, rights)
    changed = []
    for row in document["pairs"]:
        original = parent / "pairs" / row["id"]
        repaired = Path("data/v06_repaired_renders/pairs") / row["id"]
        old = json.loads((original / "render.json").read_text())
        needs_repair = "potted_plant_02" in old["cc0_asset_ids"]
        source = repaired if needs_repair else original
        audit_render(source)
        record = json.loads((source / "render.json").read_text())
        if needs_repair:
            if record["repair"]["script_sha256"] != sha256(repair_script):
                raise ValueError("repair recipe changed")
            if sha256(source / "scene.blend") != record["scene_sha256"]:
                raise ValueError("repaired scene changed")
            if (
                record["architecture_camera_lighting_sha256"]
                != old["architecture_camera_lighting_sha256"]
            ):
                raise ValueError("architecture changed")
            if sha256(source / "source.png") != row["sha256"]["source"]:
                raise ValueError("empty input changed")
            row["review"] = {}  # No automatic acceptance of newly rendered pixels.
            row["source_url"] = f"urn:roomcraft:procedural:v06-repair:{row['id']}"
            changed.append(row["id"])
        destination = root / "pairs" / row["id"]
        destination.mkdir(parents=True)
        for name in ("source.png", "target.png", "render.json", "invariant_state.json"):
            shutil.copy2(source / name, destination / name)
        row["sha256"] = {
            key: sha256(root / row[key])
            for key in ("source", "target", "evidence", "alignment_evidence")
        }
    document.update(parent_sha256=sha256(parent / "manifest.json"), repaired_ids=changed)
    write_json(root / "pending_review.json", document)
    make_gallery(root)
    print(
        f"Staged {len(document['pairs'])} pairs; {len(changed)} repaired targets require visual acceptance"
    )


def seal(root, reviews):
    document = json.loads((root / "pending_review.json").read_text())
    decisions = json.loads(reviews.read_text())
    rows = decisions["pairs"]
    if len(rows) != len({r["id"] for r in rows}) or {r["id"] for r in rows} != set(
        document["repaired_ids"]
    ):
        raise ValueError("need exactly one explicit review per repaired pair")
    by_id = {r["id"]: r for r in rows}
    for row in document["pairs"]:
        if row["id"] not in by_id:
            continue
        decision = by_id[row["id"]]
        if decision["decision"] != "accept" or decision["sha256"] != {
            k: row["sha256"][k] for k in ("source", "target")
        }:
            raise ValueError("rejected pair or stale visual review")
        row["review"] = {**decisions["review"], "notes": decision["notes"]}
    if (root / "reviewed.json").exists() or (root / "manifest.json").exists():
        raise ValueError("already sealed")
    write_json(root / "reviewed.json", document)
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_qwen_pairs.py",
            str(root / "reviewed.json"),
            "--output",
            str(root / "manifest.json"),
        ],
        check=True,
    )
    validate_pairs(root / "manifest.json")
    write_json(root / "v06_visual_reviews.json", decisions)
    write_json(
        root / "quality_reviews.json",
        {
            "reviewer": "OpenAI Codex (AI review; not independent human evaluation)",
            "date": "2026-10-29",
            "method": "50 repaired targets explicitly inspected in V0.6; 50 unchanged targets inherit V0.5 reviews. See per-row manifest review and v06_visual_reviews.json.",
            "pairs": [
                {
                    "id": r["id"],
                    "decision": "accept",
                    "sha256": {k: r["sha256"][k] for k in ("source", "target")},
                    "notes": r["review"]["notes"],
                }
                for r in document["pairs"]
            ],
        },
    )
    make_gallery(root)
    write_json(
        Path("experiments/v06_final_dataset.json"),
        {
            "status": "100_validated_pairs_after_50_target_repairs",
            "manifest": str(root / "manifest.json"),
            "manifest_sha256": sha256(root / "manifest.json"),
            "repaired_targets": len(rows),
            "unchanged_empty_inputs": 100,
            "splits": {
                s: sum(r["split"] == s for r in document["pairs"])
                for s in ("train", "validation", "test")
            },
            "review_sha256": sha256(reviews),
            "source_datasets_preserved": ["data/v05", "data/v06"],
            "repaired_scene_root": "data/v06_repaired_renders",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "seal"))
    parser.add_argument("--root", type=Path, default=Path("data/v06_ready"))
    parser.add_argument("--reviews", type=Path, default=Path("experiments/v06_repair_reviews.json"))
    args = parser.parse_args()
    if args.action == "stage":
        stage(args.root)
    else:
        seal(args.root, args.reviews)


if __name__ == "__main__":
    main()
