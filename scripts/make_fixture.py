"""Original schematic images, dedicated to CC0, solely for software testing."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from roomcraft.common import sha256

root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
root.mkdir(parents=True, exist_ok=True)
rows = []
for index, split in enumerate(("train", "val")):
    prefix = "" if index == 0 else "val_"
    room = Image.new("RGB", (512, 512), (235 - index * 10, 231, 220))
    draw = ImageDraw.Draw(room)
    draw.polygon([(0, 512), (130, 320), (382, 320), (512, 512)], fill=(170, 145, 110))
    draw.line([(130, 0), (130, 320), (382, 320), (382, 0)], fill=(80, 80, 80), width=3)
    draw.line([(0, 512), (130, 320)], fill=(80, 80, 80), width=3)
    draw.line([(512, 512), (382, 320)], fill=(80, 80, 80), width=3)
    draw.rectangle(
        (170 + index * 15, 80, 290 + index * 15, 210),
        fill=(165, 210, 230),
        outline=(80, 80, 80),
        width=5,
    )
    room.save(root / f"{prefix}room.png")
    mask = Image.new("L", room.size, 0)
    ImageDraw.Draw(mask).rectangle((135, 325, 375, 470), fill=255)
    mask.save(root / f"{prefix}mask.png")
    target = room.copy()
    ImageDraw.Draw(target).rectangle((165, 355, 345, 430), fill=(65, 95 + index * 20, 130))
    target.save(root / f"{prefix}target.png")
    assets = {
        "input": f"{prefix}room.png",
        "target": f"{prefix}target.png",
        "mask": f"{prefix}mask.png",
    }
    rows.append(
        {
            "id": f"fixture-{index}",
            "scene_id": f"schematic-{index}",
            "split": split,
            **assets,
            "prompt": "A schematic room with a blue rectangular sofa",
            "license": "CC0-1.0",
            "commercial_use": True,
            "source": "Original programmatic geometric drawing by RoomCraft AI project",
            "rights_evidence": "tests/fixtures/README.md; scripts/make_fixture.py",
            "kind": "test_fixture",
            "sha256": {key: sha256(root / value) for key, value in assets.items()},
        }
    )
(root / "manifest.jsonl").write_text(
    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
)
