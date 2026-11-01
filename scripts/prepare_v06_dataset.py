"""Audit immutable V0.5 and make a separate, explicitly stratified V0.6 training bundle."""

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

from roomcraft.common import sha256, write_json
from roomcraft.data import local_path
from roomcraft.qwen_data import validate_pairs
from roomcraft.render_dataset import audit_render

PARENT_SHA256 = "0b331516787c50891ff0761329f61db03fc6ea0ec96e6c685573cc0866d97739"
MOVES = {"family_06": "validation", "family_18": "test"}


def corrected_rows(rows):
    result = copy.deepcopy(rows)
    for row in result:
        if row["group_id"] in MOVES:
            if row["split"] != "train":
                raise ValueError("only previously training families may move")
            row["split"] = MOVES[row["group_id"]]
    return result


def prepare(source: Path, output: Path, assets: Path):
    manifest = source / "manifest.json"
    if sha256(manifest) != PARENT_SHA256:
        raise ValueError("V0.5 parent differs from the frozen, reviewed manifest")
    if output.exists():
        raise ValueError("output must be new; preserve all dataset versions")
    rows = validate_pairs(manifest)
    if len(rows) != 100 or len({r["group_id"] for r in rows}) != 20:
        raise ValueError("expected 100 pairs in 20 families")
    rights = json.loads((source / "provenance/rights.json").read_text(encoding="utf-8"))
    for name, digest in rights["evidence_files"].items():
        if sha256(local_path(source, name)) != digest:
            raise ValueError(f"rights evidence changed: {name}")
    lock = json.loads((source / "provenance/asset_lock.json").read_text(encoding="utf-8"))
    asset_files = 0
    for asset in lock["assets"]:
        if asset["license"] != "CC0-1.0" or not asset["commercial_use"]:
            raise ValueError("unexpected asset rights")
        for file in asset["files"]:
            if sha256(local_path(assets, file["path"])) != file["sha256"]:
                raise ValueError("asset bytes changed")
            asset_files += 1
    audits, kinds = [], {}
    for row in rows:
        folder = source / "pairs" / row["id"]
        audit = audit_render(folder)
        record = json.loads((folder / "render.json").read_text(encoding="utf-8"))
        if sha256(folder / "scene.blend") != record["scene_sha256"]:
            raise ValueError("saved scene changed")
        kinds[row["group_id"]] = record["family"]["kind"]
        audits.append(audit)
    revised = corrected_rows(rows)
    coverage = Counter((r["split"], kinds[r["group_id"]]) for r in revised)
    if len(coverage) != 9 or any(n < 5 for n in coverage.values()):
        raise ValueError("each split needs all three room types")
    output.mkdir(parents=True)
    shutil.copytree(source / "provenance", output / "provenance")
    shutil.copytree(source / "gallery", output / "gallery")
    shutil.copy2(source / "quality_reviews.json", output / "quality_reviews.json")
    for row in rows:
        dest = output / "pairs" / row["id"]
        dest.mkdir(parents=True)
        for name in ("source.png", "target.png", "render.json", "invariant_state.json"):
            shutil.copy2(source / "pairs" / row["id"] / name, dest / name)
    document = {
        "schema": "roomcraft-pairs-v1",
        "parent_sha256": PARENT_SHA256,
        "split_policy": "v06-predeclared-family-promotion; original validation/test retained",
        "moves": MOVES,
        "pairs": revised,
    }
    # Preserve already-sealed byte hashes. The unchanged validator rechecks every copied file.
    write_json(output / "manifest.json", document)
    validate_pairs(output / "manifest.json")
    if sha256(manifest) != PARENT_SHA256:
        raise ValueError("parent changed during audit")
    result = {
        "status": "validated_no_training",
        "parent_manifest_sha256": PARENT_SHA256,
        "manifest_sha256": sha256(output / "manifest.json"),
        "pairs": len(rows),
        "families": len(kinds),
        "split_counts": dict(Counter(r["split"] for r in revised)),
        "split_room_type_counts": {f"{s}/{k}": n for (s, k), n in sorted(coverage.items())},
        "moves": MOVES,
        "verified_scene_files": len(audits),
        "verified_asset_files": asset_files,
        "verified_rights_evidence_files": len(rights["evidence_files"]),
        "images_and_instructions_changed": False,
        "scene_files": "Retained unchanged under data/v05; not duplicated in training bundle",
        "visual_review": "See experiments/v06_visual_audit.json; synthetic, not photorealism-certified",
        "render_audits": audits,
    }
    write_json(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/v05"))
    parser.add_argument("--output", type=Path, default=Path("data/v06"))
    parser.add_argument("--assets", type=Path, default=Path("data/v05_assets"))
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.assets)
    write_json(Path("experiments/v06_dataset_audit.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "render_audits"}, indent=2))


if __name__ == "__main__":
    main()
