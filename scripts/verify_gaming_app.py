"""Real browser checks for Gaming Room Studio. Default mode cannot provision or generate.

--live clicks Generate for three themed configurations using an already approved
worker. Saved pixels remain explicitly historical, never themed quality evidence.
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
from verify_app import wait_status

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache/playwright"))

from roomcraft.app_service import LORA_SHA, digest  # noqa: E402
from roomcraft.gaming import THEMES  # noqa: E402


def select(page, label, option):
    page.get_by_role("combobox", name=label, exact=True).click()
    page.get_by_role("option", name=option, exact=True).click()


def instruction_contains(page, text):
    if page.locator("#composed-prompt textarea").count() == 0:
        page.get_by_text("Preview the exact instruction", exact=True).click()
    expect(page.locator("#composed-prompt textarea")).to_have_value(
        re.compile(re.escape(text)), timeout=15000
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/gaming-app-acceptance")
    parser.add_argument("--image", type=Path, default=ROOT / "docs/assets/example/source.png")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "mode": "fresh_themed_inference"
        if args.live
        else "free_browser_controls_and_saved_artifacts",
        "started_unix": time.time(),
        "themes": [],
        "live_models": [],
        "viewports": [],
    }
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url)
        page.locator("#room-upload input[type=file]").set_input_files(str(args.image))
        expect(page.locator("#generation-status")).to_contain_text(
            "Photograph checked", timeout=15000
        )
        for key, theme in THEMES.items():
            select(page, "Creative theme", theme["name"])
            instruction_contains(page, theme["materials"])
            evidence["themes"].append({"id": key, "instruction_verified": True})
        select(page, "Creative theme", "Custom theme")
        instruction_contains(page, "Describe your custom theme")
        page.get_by_role("button", name="Visualize my room", exact=True).click()
        expect(page.locator("#generation-status")).to_contain_text(
            "Describe your custom theme", timeout=15000
        )
        page.get_by_role("textbox", name="Describe your theme", exact=True).fill(
            "Retro observatory"
        )
        instruction_contains(page, "Retro observatory")
        page.get_by_role("button", name="Reset studio", exact=True).click()
        instruction_contains(page, "Minimalist Luxury")
        expect(
            page.get_by_role("textbox", name="Describe your theme", exact=True)
        ).not_to_be_visible()
        assert page.locator("#room-upload input[type=file]").input_value() == ""
        page.get_by_role("button", name="Visualize my room", exact=True).click()
        expect(page.locator("#generation-status")).to_contain_text(
            "Upload a room photograph first", timeout=15000
        )
        page.locator("#room-upload input[type=file]").set_input_files(str(args.image))
        page.get_by_text("Equipment & placement", exact=True).click()
        for name, marker in [
            ("Compact gaming corner", "compact desk"),
            ("Standard single-monitor setup", "one monitor"),
            ("Dual-monitor setup", "two monitors side by side"),
            ("Triple-monitor setup", "three monitors in a shallow arc"),
            ("Streaming / creator setup", "microphone boom"),
            ("Premium ultrawide setup", "one premium ultrawide monitor"),
        ]:
            select(page, "Gaming setup", name)
            instruction_contains(page, marker)
        select(page, "Desk type", "Compact desk")
        instruction_contains(page, "compact desk")
        select(page, "PC tower placement", "Under desk")
        instruction_contains(page, "under the desk")
        select(page, "Monitor count & arrangement", "Two monitors in a supported vertical stack")
        instruction_contains(page, "supported vertical stack")
        for name, marker in [
            ("Gaming chair", "Do not introduce a chair"),
            ("Keyboard, mouse & mousepad", "Do not add keyboard"),
            ("Headset & stand", "Omit headsets"),
            ("Desktop speakers", "two compact desktop speakers"),
            ("Cable management", "without decorative cable-management hardware"),
            ("Freestanding shelving", "small freestanding shelf"),
            ("Theme decorations", "Omit additional themed ornaments"),
        ]:
            page.get_by_role("checkbox", name=name, exact=True).click()
            instruction_contains(page, marker)
        for option, marker in [
            ("No RGB", "No RGB or colored lighting"),
            ("RGB accent lighting", "restrained RGB accents"),
            ("Immersive themed lighting", "immersive themed electric"),
            ("Subtle ambient lighting", "no saturated RGB glow"),
        ]:
            select(page, "Lighting", option)
            instruction_contains(page, marker)
        page.get_by_text("Advanced inference settings", exact=True).click()
        expect(
            page.get_by_role("checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True)
        ).to_be_disabled()
        page.get_by_role("radio", name="Qwen + RoomCraft LoRA", exact=True).check()
        expect(
            page.get_by_role("checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True)
        ).to_be_enabled()
        expect(
            page.get_by_role("checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True)
        ).to_be_checked()
        page.get_by_role(
            "checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True
        ).check()
        page.get_by_role("radio", name="Unchanged Qwen", exact=True).check()
        expect(
            page.get_by_role("checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True)
        ).not_to_be_checked()
        expect(
            page.get_by_role("checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True)
        ).to_be_disabled()
        page.get_by_role("button", name="Reset studio", exact=True).click()
        instruction_contains(page, "Minimalist Luxury")
        expect(page.get_by_role("button", name="Download PNG", exact=True)).to_be_disabled()
        expect(page.get_by_role("radio", name="Unchanged Qwen", exact=True)).to_be_checked()
        expect(
            page.get_by_role("checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True)
        ).to_be_visible()
        page.locator("#room-upload input[type=file]").set_input_files(str(args.image))
        expect(page.locator("#generation-status")).to_contain_text(
            "Photograph checked", timeout=15000
        )
        # A real saved staging result supplies pixels for comparison/download/layout tests.
        page.get_by_role("button", name="Explore a saved real-room result", exact=True).click()
        expect(page.locator("#generation-status")).to_contain_text("Saved example", timeout=15000)
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download settings", exact=True).click()
        dl.value.save_as(args.output / "saved-example.json")
        assert (
            json.loads((args.output / "saved-example.json").read_text())["application"]["origin"]
            == "saved_example"
        )
        select(page, "Creative theme", "Pirate's Cove")
        instruction_contains(page, "Pirate's Cove")
        page.locator("#recent-images").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.locator(".rc-hero h1").click()
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
        for width in (1440, 390, 320):
            page.set_viewport_size({"width": width, "height": 1100})
            page.wait_for_timeout(400)
            bounds = page.evaluate(
                "({width:innerWidth,document:document.documentElement.scrollWidth})"
            )
            assert bounds["document"] <= width + 1, bounds
            for label in ("Gaming setup", "Creative theme", "Lighting"):
                box = page.get_by_role("combobox", name=label, exact=True).bounding_box()
                assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 1, (label, box)
            evidence["viewports"].append(bounds)
            if width == 390:
                page.screenshot(path=str(args.output / "narrow.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1100})
        if args.live:
            page.get_by_role("button", name="Clear session", exact=True).click()
            page.locator("#room-upload input[type=file]").set_input_files(str(args.image))
            page.get_by_role("radio", name="Qwen + RoomCraft LoRA", exact=True).check()
            for key, theme, setup in [
                ("pirates_cove", "Pirate's Cove", "Dual-monitor setup"),
                ("cat_kingdom", "Cat Kingdom", "Compact gaming corner"),
                ("underwater_realm", "Underwater Realm", "Triple-monitor setup"),
            ]:
                select(page, "Gaming setup", setup)
                select(page, "Creative theme", theme)
                select(page, "Lighting", "Immersive themed lighting")
                instruction_contains(page, theme)
                expect(page.locator("#generation-status")).not_to_contain_text(
                    "Your room is ready", timeout=15000
                )
                page.get_by_role("button", name="Visualize my room", exact=True).click()
                wait_status(page, "Your room is ready")
                for button, suffix in [("Download PNG", ".png"), ("Download settings", ".json")]:
                    with page.expect_download() as dl:
                        page.get_by_role("button", name=button, exact=True).click()
                    dl.value.save_as(args.output / (key + suffix))
                meta = json.loads((args.output / (key + ".json")).read_text())
                assert meta["application"]["origin"] == "generated"
                assert meta["application"]["studio"]["configuration"]["theme"] == key
                assert meta["adapter_sha256"] == LORA_SHA
                assert meta["output_sha256"] == digest(args.output / (key + ".png"))
                with Image.open(args.output / (key + ".png")) as image:
                    image.verify()
                with Image.open(args.output / (key + ".png")) as image:
                    assert list(image.size) == meta["working_size"] and max(image.size) == 1024
                page.screenshot(path=str(args.output / (key + "-studio.png")), full_page=True)
                evidence["live_models"].append(
                    {
                        "theme": key,
                        "output_sha256": meta["output_sha256"],
                        "dimensions": meta["working_size"],
                    }
                )
                (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2))
        evidence["custom_theme_reset_equipment_lighting_rga_controls"] = "passed"
        evidence["browser_errors"] = errors
        assert (
            page.get_by_text(
                "This application is currently busy. Please try again.", exact=True
            ).count()
            == 0
        )
        assert not errors, errors
        browser.close()
    evidence["finished_unix"] = time.time()
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
