"""Single real inference job in the unchanged pinned Qwen environment."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--rga", type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve()
    request = json.loads((job / "request.json").read_text())
    try:
        from roomcraft.app_service import AppConfig, require_checkpoints
        from roomcraft.editing import EditConfig, edit

        # Reuse the same identity gate on the actual machine holding the checkpoint.
        require_checkpoints(
            AppConfig(lora=args.lora, rga=args.rga), request["model"], request["rga"]
        )
        meta = edit(
            job / "upload.png",
            request["instruction"],
            job / "result",
            EditConfig(
                size=request["size"],
                steps=request["steps"],
                true_cfg_scale=request["cfg"],
                seed=request["seed"],
            ),
            args.lora if request["model"] == "roomcraft" else None,
            rga_checkpoint=args.rga if request["rga"] else None,
        )
        response = {"metadata": meta}
        code = 0
    except Exception as exc:
        message = str(exc)
        if message.startswith("Qwen preflight:"):
            message = (
                "This GPU cannot run the current Qwen model. "
                + message
                + ". Use an approved 80 GB Runpod worker; no image was generated."
            )
        elif isinstance(exc, FileNotFoundError):
            message = (
                "Required model files or checkpoint are missing. Check the worker installation."
            )
        elif "out of memory" in message.lower():
            message = "The GPU ran out of memory. Reduce resolution or use a larger GPU. No image was returned."
        elif not message.startswith(("The verified", "Choose", "RoomCraft RGA")):
            message = (
                "The Qwen worker failed. Check its installation and available GPU/model files."
            )
        response = {"error": message}
        code = 1
    (job / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
