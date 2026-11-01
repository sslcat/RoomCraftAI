"""Browser acceptance: real saved artifacts, or fresh real inference with --live.

No responses are mocked. --live requires an already approved GPU worker; this
script cannot provision one. Saved mode is not evidence of new generation.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache/playwright"))

from roomcraft.app_service import LORA_SHA, digest  # noqa: E402 -- source checkout bootstrap


def wait_status(page, expected, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        text = page.locator("#generation-status").inner_text()
        if "Generation did not complete" in text:
            raise RuntimeError(text)
        if expected in text:
            return text
        page.wait_for_timeout(500)
    raise TimeoutError("The UI did not return the expected model result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--image", type=Path, default=ROOT / "docs/assets/example/source.png")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/app-acceptance")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--keep-session",
        action="store_true",
        help="Operator preserves session job/log files before application shutdown",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "mode": "fresh_real_inference" if args.live else "previous_real_outputs_only",
        "started_unix": time.time(),
        "models": [],
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.locator("#room-upload input[type=file]").set_input_files(str(args.image))
        wait_status(page, "Photograph checked", timeout=30)
        page.get_by_role("combobox", name="Design category").click()
        page.get_by_role("option", name="Living rooms", exact=True).click()
        page.get_by_role("combobox", name="Preset").click()
        page.get_by_role("option", name="Quiet reading corner", exact=True).click()
        for model, label in (("base", "Unchanged Qwen"), ("roomcraft", "Qwen + RoomCraft LoRA")):
            page.get_by_role("radio", name=label, exact=True).check()
            if args.live:
                page.get_by_role("button", name="Visualize my room", exact=True).click()
                expected = "Your room is ready"
                wait_status(page, expected)
                # A different model must also appear; the prior result cannot satisfy this wait.
                wait_status(page, "Unchanged Qwen" if model == "base" else "RoomCraft LoRA")
            else:
                page.get_by_role(
                    "button", name="Explore a saved real-room result", exact=True
                ).click()
                wait_status(page, "Saved example")
                wait_status(page, "Unchanged Qwen" if model == "base" else "RoomCraft LoRA")
            with page.expect_download() as png_event:
                page.get_by_role("button", name="Download PNG", exact=True).click()
            png = args.output / (model + ".png")
            png_event.value.save_as(png)
            with page.expect_download() as json_event:
                page.get_by_role("button", name="Download settings", exact=True).click()
            settings = args.output / (model + ".json")
            json_event.value.save_as(settings)
            meta = json.loads(settings.read_text())
            assert meta["adapter_sha256"] == (LORA_SHA if model == "roomcraft" else None)
            assert meta["application"]["origin"] == ("generated" if args.live else "saved_example")
            assert digest(png) == meta["output_sha256"]
            with Image.open(png) as image:
                image.verify()
            with Image.open(png) as image:
                assert list(image.size) == meta["working_size"] and max(image.size) == 1024
            fixed = page.locator('#room-comparison img[data-testid="imageslider-image"].fixed')
            handle = page.locator('#room-comparison [data-testid="slider"]')
            handle.scroll_into_view_if_needed()
            before = fixed.get_attribute("style")
            box = handle.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(
                box["x"] + box["width"] / 2 + 75, box["y"] + box["height"] / 2, steps=10
            )
            page.mouse.up()
            assert fixed.get_attribute("style") != before, "Comparison slider did not move"
            page.screenshot(path=str(args.output / (model + "-studio.png")), full_page=True)
            evidence["models"].append(
                {
                    "model": model,
                    "native_size": meta["working_size"],
                    "png_sha256": digest(png),
                    "adapter_sha256": meta["adapter_sha256"],
                    "origin": meta["application"]["origin"],
                    "total_seconds": meta["total_seconds"],
                    "comparison_drag": True,
                    "png_and_metadata_download": True,
                }
            )
            (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        page.get_by_role("button", name="Thumbnail 2 of 2", exact=True).click()
        wait_status(page, "Unchanged Qwen")
        evidence["session_history_verified"] = True
        if not args.keep_session:
            page.get_by_role("button", name="Clear session", exact=True).click()
            expect(page.get_by_role("button", name=re.compile("Thumbnail"))).to_have_count(
                0, timeout=15000
            )
            expect(page.get_by_role("button", name="Download PNG", exact=True)).to_be_disabled()
        evidence["session_clear_verified"] = not args.keep_session
        evidence["session_history_and_clear"] = not args.keep_session
        assert not errors, errors
        evidence["browser_errors"] = errors
        browser.close()
    if args.live:
        base = json.loads((args.output / "base.json").read_text())
        lora = json.loads((args.output / "roomcraft.json").read_text())
        assert all(
            base[k] == lora[k]
            for k in ("input_sha256", "prompt", "settings", "working_size", "base")
        )
        evidence["matched_base_lora_inputs_and_settings"] = True
    evidence["finished_unix"] = time.time()
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
