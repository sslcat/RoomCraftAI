"""Archive small official metadata/license sources. No dataset images or weights downloaded."""

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

SOURCES = {
    "applydesign-api.json": "https://huggingface.co/api/datasets/ApplyDesign/bedroom-dataset",
    "hypersim-readme.md": "https://raw.githubusercontent.com/apple/ml-hypersim/main/README.md",
    "cc-by-sa-3.0.txt": "https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt",
    "cc0-1.0.txt": "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt",
    "structured3d-terms.pdf": "https://drive.usercontent.google.com/download?id=13ZwWpU_557ZQccwOUJ8H5lvXD7MeZFMa&export=download",
    "structured3d-layout.md": "https://raw.githubusercontent.com/bertjiazheng/Structured3D/master/data_organization.md",
    "global-volume-overview.html": "https://docs.runpod.io/storage/globalvolume/overview",
    "global-volume-pods.html": "https://docs.runpod.io/storage/globalvolume/globalvolume-pods",
    "diffsynth-2511.sh": "https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/main/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh",
}


def main():
    root = Path(__file__).resolve().parents[1] / "docs/sources/v04"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "RoomCraft-research/0.4"})
            with urllib.request.urlopen(request, timeout=40) as response:
                data = response.read(2_000_000)
            if name.endswith(".pdf") and not data.startswith(b"%PDF"):
                raise ValueError("not a PDF response")
            (root / name).write_bytes(data)
            records.append(
                {
                    "file": name,
                    "url": url,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "status": "downloaded",
                }
            )
        except Exception as error:
            records.append({"file": name, "url": url, "status": "unavailable", "error": str(error)})
    record = {"retrieved_utc": datetime.now(UTC).isoformat(), "sources": records}
    (root / "index.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
