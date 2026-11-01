"""Archive original-only gaming asset evidence; seal only separately reviewed pairs."""

import argparse
import datetime
import json
import shutil
from pathlib import Path

from roomcraft.common import sha256, write_json
from roomcraft.qwen_data import validate_pairs

ROOT = Path(__file__).resolve().parents[1]


def provenance(root):
    destination = root / "provenance"
    destination.mkdir(exist_ok=False)
    files = (
        "scripts/generate_room_dataset.py",
        "scripts/generate_gaming_dataset.py",
        "scripts/project_room_landmarks.py",
        "configs/dataset_v10_gaming.json",
        "configs/dataset_v10_split.json",
        "LICENSE",
    )
    evidence = {}
    for name in files:
        target = destination / Path(name).name
        shutil.copyfile(ROOT / name, target)
        evidence[target.relative_to(root).as_posix()] = sha256(target)
    generator = sha256(ROOT / files[1])
    catalog = {
        "schema": "roomcraft-original-assets-v1",
        "external_assets": [],
        "author": "RoomCraft AI contributors",
        "license": "MIT, to the extent copyright or related rights subsist",
        "license_evidence": "LICENSE",
        "attribution_obligation": "Retain the RoomCraft MIT copyright and permission notice.",
        "assets": [
            {
                "id": "roomcraft-v10-" + name,
                "source": "generate_gaming_dataset.py",
                "source_uri": "urn:roomcraft:original:v10:" + name,
                "sha256": generator,
                "external_url": None,
                "commercial_use": True,
                "redistribution": True,
            }
            for name in (
                "desk-straight",
                "desk-l",
                "monitor-arm",
                "abstract-monitor-display",
                "pc-tower",
                "chair",
                "keyboard",
                "mouse",
                "cables",
                "microphone",
                "shelf",
                "books",
                "desk-accessories",
                "rgb-lights",
            )
        ],
        "architecture_and_materials": {
            "source": "generate_room_dataset.py",
            "sha256": sha256(ROOT / files[0]),
            "mode": "original_procedural; no CC0 furniture or texture downloads used",
        },
    }
    write_json(destination / "assets.json", catalog)
    evidence["provenance/assets.json"] = sha256(destination / "assets.json")
    write_json(
        destination / "rights.json",
        {
            "schema": "roomcraft-render-rights-v1",
            "recorded_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "creator": "RoomCraft AI contributors, original procedural geometry and materials",
            "source_namespace": "urn:roomcraft:procedural:v10-gaming",
            "license": "MIT (original contributions, to the extent rights subsist)",
            "license_url": "https://opensource.org/license/mit",
            "asset_provenance_sha256": None,
            "statement": "Original algorithmic room geometry, furniture, abstract screen graphics, shaders and renders. No third-party photographs, logos, game screenshots, textures or furniture models. Project MIT notice archived. This records provenance and permission, not a warranty of copyright ownership or photorealism.",
            "rights": {
                k: True
                for k in (
                    "commercial_training",
                    "derivatives",
                    "adapter_distribution",
                    "image_redistribution",
                )
            },
            "attribution": "Retain the RoomCraft MIT notice. Blender is the rendering application, not a source of bundled furniture assets.",
            "evidence_files": evidence,
        },
    )


def assign(root):
    source = root / "sealed.json"
    output = root / "manifest.json"
    if output.exists():
        raise ValueError("manifest is immutable")
    rows = validate_pairs(source, require_splits=False)
    split_path = root / "provenance/dataset_v10_split.json"
    plan = json.loads(split_path.read_text())
    mapping = {}
    for split in ("train", "validation", "test"):
        for family in plan[split]:
            if family in mapping:
                raise ValueError("split plan repeats a family")
            mapping[family] = split
    for row in rows:
        row["split"] = mapping[row["group_id"]]
        record = json.loads((root / row["alignment_evidence"]).read_text())
        for key, filename in (
            ("generator_sha256", "generate_room_dataset.py"),
            ("gaming_generator_sha256", "generate_gaming_dataset.py"),
            ("landmark_generator_sha256", "project_room_landmarks.py"),
        ):
            if record[key] != sha256(root / "provenance" / filename):
                raise ValueError("render used a different generator")
        folder = root / "pairs" / row["id"]
        if sha256(folder / "landmarks.json") != record["landmarks_sha256"]:
            raise ValueError("landmark record changed")
        if sha256(folder / "scene.blend") != record["scene_sha256"]:
            raise ValueError("saved Blender scene changed")
    write_json(
        output,
        {
            "schema": "roomcraft-pairs-v1",
            "parent_sha256": sha256(source),
            "split_plan_sha256": sha256(split_path),
            "pairs": rows,
        },
    )
    validate_pairs(output)
    print(f"Validated {len(rows)} gaming pairs with disjoint scene families.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("provenance", "split"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    provenance(args.root) if args.action == "provenance" else assign(args.root)
