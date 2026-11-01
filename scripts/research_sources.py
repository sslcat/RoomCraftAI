"""Download small primary-source texts and model metadata, never weights."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".cache" / "research"
DEST.mkdir(parents=True, exist_ok=True)
(ROOT / "docs" / "sources").mkdir(parents=True, exist_ok=True)
MODELS = {
    "inpaint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "flux": "black-forest-labs/FLUX.1-Fill-dev",
    "qwen": "Qwen/Qwen-Image-Edit-2511",
}
lock = {}
for name, repo in MODELS.items():
    with urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as response:
        meta = json.load(response)
    lock[name] = {"repo": repo, "revision": meta["sha"]}
    files = [f["rfilename"] for f in meta["siblings"]]
    for filename in files:
        if filename.lower() in {"readme.md", "license", "license.md", "license.txt"}:
            url = f"https://huggingface.co/{repo}/resolve/{meta['sha']}/{filename}"
            with urlopen(url, timeout=60) as response:
                content = response.read()
            path = (
                DEST if filename.lower() == "readme.md" else ROOT / "docs" / "sources"
            ) / f"{name}-{filename}"
            path.write_bytes(content)
            print(name, filename, hashlib.sha256(content).hexdigest())
lock_path = ROOT / "src" / "roomcraft" / "models.json"
lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
sources = {
    "sdxl-LICENSE.md": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/LICENSE.md",
    "hypersim-README.md": "https://raw.githubusercontent.com/apple/ml-hypersim/main/README.md",
    "hypersim-software-LICENSE.txt": "https://raw.githubusercontent.com/apple/ml-hypersim/main/LICENSE.txt",
    "structured3d-README.md": "https://raw.githubusercontent.com/bertjiazheng/Structured3D/master/README.md",
    "structured3d-LICENSE": "https://raw.githubusercontent.com/bertjiazheng/Structured3D/master/LICENSE",
}
for filename, url in sources.items():
    try:
        with urlopen(url, timeout=60) as response:
            destination = DEST if "README" in filename else ROOT / "docs" / "sources"
            (destination / filename).write_bytes(response.read())
        print(filename, "OK")
    except Exception as exc:
        print(filename, str(exc))
