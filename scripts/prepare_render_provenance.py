"""Snapshot the exact local recipe and reviewed asset rights beside rendered pairs."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from roomcraft.common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    destination = args.root / "provenance"
    if destination.exists():
        raise ValueError("refusing to replace provenance; use a new dataset for changed recipes")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.root / "pairs").glob("*/render.json")
    ]
    if not records:
        raise ValueError("no completed renders")
    recipe = ROOT / "scripts/generate_room_dataset.py"
    lock = ROOT / "docs/sources/v05/asset_lock.json"
    config = ROOT / "configs/dataset_v05.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    for record in records:
        if record["generator_sha256"] != sha256(recipe) or record["settings"] != settings:
            raise ValueError("render recipe differs from current source/config")
        if record["asset_provenance_sha256"] != sha256(lock):
            raise ValueError("render did not use the reviewed asset lock")
    destination.mkdir()
    files = {
        "generator.py": recipe,
        "config.json": config,
        "LICENSE": ROOT / "LICENSE",
        "asset_lock.json": lock,
        "CC0.txt": ROOT / "docs/sources/v05/CC0.txt",
        "API_TERMS.md": ROOT / "docs/sources/v05/API_TERMS.md",
    }
    for asset in json.loads(lock.read_text(encoding="utf-8"))["assets"]:
        relative = asset["individual_license_evidence"]
        files[relative] = ROOT / "docs/sources/v05" / relative
    hashes = {}
    for relative, source in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[target.relative_to(args.root).as_posix()] = sha256(target)
    write_json(
        destination / "rights.json",
        {
            "schema": "roomcraft-render-rights-v1",
            "recorded_utc": datetime.now(UTC).isoformat(),
            "original_creator": "RoomCraft AI contributors",
            "original_contribution_license": "MIT",
            "statement": "RoomCraft-authored scene recipes, geometry, materials and original rendered-dataset contributions use the repository MIT terms to the extent rights subsist. External furniture/textures remain CC0-1.0 under the individually archived asset notices. No photographs or site preview images are used as training images.",
            "asset_provenance_sha256": sha256(lock),
            "generator_sha256": sha256(recipe),
            "evidence_files": hashes,
            "rights": {
                k: True
                for k in (
                    "commercial_training",
                    "derivatives",
                    "adapter_distribution",
                    "image_redistribution",
                )
            },
            "attribution": "Retain the RoomCraft MIT notice. Powered by Poly Haven. Asset authors and URLs are in asset_lock.json.",
            "evidence_restriction": "Website HTML is archived rights evidence, not an asset licensed CC0. Do not redistribute that HTML under the dataset license without a separate review.",
            "blender_output_sources": [
                "https://www.blender.org/about/license/",
                "https://www.blender.org/support/faq/",
            ],
        },
    )
    print(f"Snapshotted rights and recipe for {len(records)} completed renders.")


if __name__ == "__main__":
    main()
