"""Download an explicitly reviewed CC0 asset allowlist. Powered by Poly Haven.

Each item's own page must contain a CC0 notice; do not infer rights from the catalog alone.
"""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("modern_arm_chair_01", "dining_chair_02", "sofa_02", "potted_plant_01", "potted_plant_02")
HEADERS = {"User-Agent": "RoomCraftAI/0.5 dataset-preparation"}


def fetch(url):
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=HEADERS), timeout=90
    ) as response:
        return response.read()


def main():
    destination = ROOT / "data/v05_assets"
    destination.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "docs/sources/v05"
    lock = evidence / "asset_lock.json"
    if lock.exists():
        # Reproduction uses the reviewed bytes, never silently upgrades the catalog selection.
        records = json.loads(lock.read_text(encoding="utf-8"))["assets"]
        for record in records:
            for item in record["files"]:
                target = (destination / item["path"]).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("unsafe locked path")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(fetch(item["url"]))
                if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError(f"locked asset has changed: {target}")
            relative = record["individual_license_evidence"]
            page = (evidence / relative).read_bytes()
            if hashlib.sha256(page).hexdigest() != record["evidence_sha256"]:
                raise ValueError("archived individual license evidence changed")
            (destination / relative).write_bytes(page)
            print(f"Verified locked CC0 asset: {record['id']}", flush=True)
        for name in ("CC0.txt", "API_TERMS.md"):
            shutil.copyfile(evidence / name, destination / name)
        shutil.copyfile(lock, destination / "provenance.json")
        return
    records = []
    for asset in ASSETS:
        folder = destination / asset
        folder.mkdir(exist_ok=True)
        page_url = f"https://polyhaven.com/a/{asset}"
        page = fetch(page_url)
        if b"CC0" not in page:
            raise ValueError(f"individual CC0 notice not found for {asset}")
        (folder / "asset_page.html").write_bytes(page)
        info_data, files_data = [
            fetch(f"https://api.polyhaven.com/{endpoint}/{asset}") for endpoint in ("info", "files")
        ]
        (folder / "info.json").write_bytes(info_data)
        (folder / "files.json").write_bytes(files_data)
        info, files = json.loads(info_data), json.loads(files_data)
        entry = files["blend"]["2k"]["blend"]
        downloads = {f"{asset}_2k.blend": entry, **entry.get("include", {})}
        saved = []
        for relative, item in downloads.items():
            target = (folder / relative).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("unexpected asset path")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(fetch(item["url"]))
            data = target.read_bytes()
            if hashlib.md5(data).hexdigest() != item["md5"]:
                raise ValueError(f"provider hash mismatch: {target}")
            saved.append(
                {
                    "path": target.relative_to(destination).as_posix(),
                    "url": item["url"],
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "md5": item["md5"],
                    "bytes": len(data),
                }
            )
        records.append(
            {
                "id": asset,
                "source_url": page_url,
                "authors": info["authors"],
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
                "commercial_use": True,
                "redistribution": True,
                "asset_attribution_required": False,
                "api_credit": "Powered by Poly Haven",
                "individual_license_evidence": f"{asset}/asset_page.html",
                "evidence_sha256": hashlib.sha256(page).hexdigest(),
                "files": saved,
            }
        )
        print(f"Verified/downloaded {asset}: {sum(f['bytes'] for f in saved)} bytes", flush=True)
    for name, url in {
        "CC0.txt": "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt",
        "API_TERMS.md": "https://raw.githubusercontent.com/Poly-Haven/Public-API/master/ToS.md",
    }.items():
        (destination / name).write_bytes(fetch(url))
    (destination / "provenance.json").write_text(
        json.dumps({"assets": records, "credit": "Powered by Poly Haven"}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
