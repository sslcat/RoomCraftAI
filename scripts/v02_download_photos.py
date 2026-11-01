"""Fetch a small reviewed Commons photo set; one candidate was rejected visually."""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "v02"
SOURCES = [
    ("apartment", "Empty room in apartment.jpg"),
    ("hotel", "Our Hotel Suite-1 (4983234010).jpg"),
    ("school", "Hollow room.jpg"),
    ("windows", "Empty apartment room with corner windows.jpg"),
]
EXPECTED_HASHES = {
    "apartment": "72a942820e607388eb769a9f095156455ba5c1b49dde285a5e91704d4f47762c",
    "hotel": "6832cabc8d8b44b435e1d3ad6d4ffe0130780e1eea86ec346739377fdf927a18",
    "school": "e65b0ebb3e19f68542d6137441aaade1e676f2f57792af36586ef365aa513f2e",
    "windows": "7a700d5b1fd01f40be178acb74d36021188908dbeb82a354e1ca7ff1fababfdd",
}
DEST.mkdir(parents=True, exist_ok=True)
records = []
for name, filename in SOURCES:
    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "titles": "File:" + filename,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|sha1|timestamp",
            "iiurlwidth": 1280,
        }
    )
    headers = {"User-Agent": "RoomCraftAI/0.2 (local licensed-image evaluation; no crawling)"}
    rights_path = DEST / f"{name}-rights.json"
    if rights_path.exists():
        info = json.loads(rights_path.read_text(encoding="utf-8"))
    else:
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php?" + query, headers=headers, timeout=60
        )
        response.raise_for_status()
        metadata = response.json()
        info = next(iter(metadata["query"]["pages"].values()))["imageinfo"][0]
    (DEST / f"{name}-rights.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    download_url = info.get("thumburl", info["url"])
    if (DEST / f"{name}.jpg").exists():
        content = (DEST / f"{name}.jpg").read_bytes()
    else:
        response = requests.get(download_url, headers=headers, timeout=60)
        response.raise_for_status()
        content = response.content
    (DEST / f"{name}.jpg").write_bytes(content)
    if hashlib.sha256(content).hexdigest() != EXPECTED_HASHES[name]:
        raise ValueError(f"{name}: source bytes changed; inspect and reapprove the provenance")
    ext = info["extmetadata"]
    records.append(
        {
            "id": name,
            "file_title": filename,
            "source_page": info["descriptionurl"],
            "original_url": info["url"],
            "download_date": "2026-10-27",
            "download_url": download_url,
            "derivation": "Commons 1280px thumbnail, resized for local evaluation",
            "author_html": ext.get("Artist", {}).get("value"),
            "license": ext.get("LicenseShortName", {}).get("value"),
            "license_url": ext.get("LicenseUrl", {}).get("value"),
            "sha256": hashlib.sha256(content).hexdigest(),
            "status": "excluded: visible desk/chair and ball; not empty"
            if name == "school"
            else "included",
            "restrictions": "Credit author, link source and license, identify resizing/masks/AI edits; share-alike for distributed adaptations where BY-SA applies. No endorsement. Photo rights do not license model weights.",
            "purpose": "local inference evaluation; not training; no paired target exists",
        }
    )
    print(name, records[-1]["license"], len(content), flush=True)
(ROOT / "experiments" / "v02_sources.json").write_text(
    json.dumps(records, indent=2) + "\n", encoding="utf-8"
)
