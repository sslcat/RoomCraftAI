"""Refresh public license evidence only. No data or model weights are fetched."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
sources = {
    "qwen-LICENSE": "https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/LICENSE",
    "hypersim-software-LICENSE.txt": "https://raw.githubusercontent.com/apple/ml-hypersim/main/LICENSE.txt",
}
for name, url in sources.items():
    with urlopen(url, timeout=60) as response:
        content = response.read()
    (root / "docs" / "sources" / name).write_bytes(content)
    print(name, hashlib.sha256(content).hexdigest())
index = []
for path in sorted((root / "docs" / "sources").iterdir()):
    if path.is_file() and path.name != "checksums.json":
        index.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(root / "docs" / "sources" / "checksums.json").write_text(json.dumps(index, indent=2) + "\n")
