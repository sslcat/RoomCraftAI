"""Reproduce the saved Underwater prompt via Gradio; never provisions resources.

Run only against an already separately approved bounded worker. Saves genuine
RGA-off/on downloads. A visual architecture review is still required.
"""

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright
from verify_app import wait_status
from verify_gaming_app import instruction_contains, select


def wait_new_download(page, previous, timeout=900):
    """Require a fresh job image; Gradio's DownloadButton has no href attribute."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status = page.locator("#generation-status").inner_text()
        if "Generation did not complete" in status:
            raise RuntimeError(status)
        current = page.locator("#recent-images img").evaluate_all(
            "elements => elements.map(e => e.getAttribute('src'))"
        )
        if current and current != previous:
            wait_status(page, "Your room is ready")
            return current
        page.wait_for_timeout(500)
    raise TimeoutError("No fresh result download appeared; refusing the previous output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7863")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("both", "on", "off"), default="both")
    args = parser.parse_args()
    original = json.loads(args.baseline.read_text(encoding="utf8"))
    assert original["settings"] == {
        "size": 1024,
        "steps": 40,
        "true_cfg_scale": 4.0,
        "seed": 42,
        "negative_prompt": " ",
        "offload": False,
    }
    assert original["application"]["studio"]["configuration"]["theme"] == "underwater_realm"
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault(
        "PLAYWRIGHT_BROWSERS_PATH", str(Path(__file__).resolve().parents[1] / ".cache/playwright")
    )
    receipts = []
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page(accept_downloads=True, viewport={"width": 1440, "height": 1100})
        page.goto(args.url)
        page.locator("#room-upload input[type=file]").set_input_files(str(args.image))
        expect(page.locator("#generation-status")).to_contain_text(
            "Photograph checked", timeout=15000
        )
        select(page, "Gaming setup", "Triple-monitor setup")
        select(page, "Creative theme", "Underwater Realm")
        select(page, "Lighting", "Immersive themed lighting")
        instruction_contains(page, "Underwater Realm")
        assert page.locator("#composed-prompt textarea").input_value() == original["prompt"], (
            "Prompt changed; refuse unmatched comparison"
        )
        page.get_by_role("radio", name="Qwen + RoomCraft LoRA", exact=True).check()
        page.get_by_text("Advanced inference settings", exact=True).click()
        arms = (False, True) if args.arm == "both" else (args.arm == "on",)
        for enabled in arms:
            page.get_by_role(
                "checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True
            ).set_checked(enabled)
            previous = page.locator("#recent-images img").evaluate_all(
                "elements => elements.map(e => e.getAttribute('src'))"
            )
            page.get_by_role("button", name="Visualize my room", exact=True).click()
            wait_new_download(page, previous)
            name = "rga-on" if enabled else "rga-off"
            for label, suffix in [("Download PNG", ".png"), ("Download settings", ".json")]:
                with page.expect_download() as dl:
                    page.get_by_role("button", name=label, exact=True).click()
                dl.value.save_as(args.output / (name + suffix))
            meta = json.loads((args.output / (name + ".json")).read_text())
            for key in (
                "prompt",
                "settings",
                "base",
                "adapter_sha256",
                "input_sha256",
                "working_size",
            ):
                assert meta[key] == original[key], key
            assert bool(meta.get("geometry_adapter")) == enabled
            with Image.open(args.output / (name + ".png")) as image:
                image.verify()
            page.screenshot(path=str(args.output / (name + "-studio.png")), full_page=True)
            receipts.append(
                {"arm": name, "output_sha256": meta["output_sha256"], "matched_baseline": True}
            )
            (args.output / "evidence.json").write_text(json.dumps(receipts, indent=2))
        b.close()


if __name__ == "__main__":
    main()
