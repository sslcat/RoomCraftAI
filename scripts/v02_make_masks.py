"""Hand-drawn evaluation masks, in pixels of the licensed 1280px photos.

These are placement hypotheses, not segmentation or ground-truth annotations.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v02"
POLYGONS = {
    "windows": {
        "floor": [(180, 850), (650, 680), (1120, 810), (1110, 935), (200, 935)],
        "silhouette": [
            (735, 455),
            (1090, 510),
            (1105, 655),
            (1180, 690),
            (1170, 855),
            (750, 795),
            (680, 720),
            (680, 570),
            (735, 570),
        ],
        "broad": [(665, 415), (1230, 415), (1230, 935), (540, 935), (540, 670), (665, 600)],
    },
    "apartment": {
        "floor": [(710, 715), (1160, 850), (1160, 935), (700, 930)],
        "silhouette": [
            (790, 465),
            (1135, 555),
            (1145, 685),
            (1210, 740),
            (1190, 920),
            (735, 815),
            (730, 650),
            (785, 645),
        ],
    },
    "hotel": {
        "floor": [(170, 1380), (600, 1190), (1030, 1400), (1080, 1630), (200, 1650)],
        "silhouette": [
            (230, 850),
            (570, 890),
            (595, 1060),
            (660, 1120),
            (665, 1460),
            (160, 1540),
            (140, 1180),
            (220, 1090),
        ],
    },
}

for room, variants in POLYGONS.items():
    with Image.open(DATA / f"{room}.jpg") as source:
        original = source.convert("RGB")
    for name, points in variants.items():
        mask = Image.new("L", original.size)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        mask.save(DATA / f"{room}-{name}.png")
        tint = Image.new("RGB", original.size, (255, 70, 60))
        overlay = Image.composite(Image.blend(original, tint, 0.4), original, mask)
        overlay.save(DATA / f"{room}-{name}-overlay.png")

(ROOT / "experiments/v02_masks.json").write_text(
    json.dumps(
        {
            "coordinate_space": "downloaded image pixels; white=editable",
            "method": "manually drawn after image inspection; not segmentation",
            "polygons": POLYGONS,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
