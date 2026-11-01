"""No-GPU browser acceptance for four purposes, six themes and RGA defaults."""

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from verify_gaming_app import instruction_contains, select

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/studio-acceptance")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache/playwright"))
    cases = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        rga = page.get_by_role(
            "checkbox", name="Preserve Room Geometry — RoomCraft RGA", exact=True
        )
        expect(rga).to_be_visible()
        expect(rga).to_be_disabled()
        expect(rga).not_to_be_checked()
        page.get_by_role("radio", name="Qwen + RoomCraft LoRA", exact=True).check()
        expect(rga).to_be_enabled()
        expect(rga).to_be_checked()
        rga.uncheck()
        expect(rga).not_to_be_checked()
        page.get_by_role("radio", name="Unchanged Qwen", exact=True).check()
        expect(rga).to_be_disabled()
        page.get_by_role("radio", name="Qwen + RoomCraft LoRA", exact=True).check()
        expect(rga).to_be_checked()
        themes = [
            "Pirate's Cove",
            "Cat Kingdom",
            "Underwater Realm",
            "Cyberpunk Neon",
            "Enchanted Forest",
            "Minimalist Luxury",
        ]
        for purpose, marker in [
            ("Gaming", "gaming setup"),
            ("Work", "task chair"),
            ("Relaxation", "reading armchair"),
            ("Mixed-purpose", "share the desk"),
        ]:
            select(page, "Room purpose", purpose)
            for theme in themes:
                select(page, "Creative theme", theme)
                instruction_contains(page, marker)
                instruction_contains(page, theme)
                expect(rga).to_be_checked()
                cases.append({"purpose": purpose, "theme": theme, "prompt_verified": True})
            if purpose == "Work":
                select(page, "Workstation", "Sit-stand desk")
                instruction_contains(page, "sit-stand desk")
                select(page, "Work displays", "Dual monitors")
                instruction_contains(page, "dual monitors")
            if purpose == "Relaxation":
                select(page, "Relaxation seating", "Compact sofa")
                instruction_contains(page, "compact sofa")
                page.get_by_role("checkbox", name="Reading lamp & books", exact=True).uncheck()
                instruction_contains(page, "Omit reading lamps")
            if purpose == "Mixed-purpose":
                for option in ("Gaming + work", "Gaming + relaxation", "Work + relaxation"):
                    select(page, "Combined functions", option)
                    instruction_contains(
                        page, "share the desk" if "work" in option.lower() else "gaming"
                    )
        page.get_by_role("button", name="Reset studio", exact=True).click()
        expect(page.get_by_role("radio", name="Unchanged Qwen", exact=True)).to_be_checked()
        expect(rga).to_be_disabled()
        expect(rga).not_to_be_checked()
        page.locator("#room-upload input[type=file]").set_input_files(
            str(ROOT / "docs/assets/example/source.png")
        )
        select(page, "Creative theme", "Underwater Realm")
        select(page, "Gaming setup", "Triple-monitor setup")
        page.get_by_role("radio", name="Qwen + RoomCraft LoRA", exact=True).check()
        instruction_contains(page, "Underwater Realm")
        page.get_by_text("Preview the exact instruction", exact=True).click()
        page.screenshot(path=str(args.output / "studio.png"), full_page=True)
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 1100})
            page.wait_for_timeout(200)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            expect(rga).to_be_visible()
        assert not errors, errors
        browser.close()
    result = {
        "mode": "free_browser_controls_only_no_inference",
        "purpose_theme_cases": cases,
        "rga_default_on_with_lora_off_switch_base_and_reset": "passed",
        "layouts": [1440, 390, 320],
        "browser_errors": errors,
    }
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
