"""CPU preprocessing. All transforms are shared by image and mask."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def binary_mask(image: Image.Image) -> Image.Image:
    return image.convert("L").point(lambda value: 255 if value >= 128 else 0)


def prepare(image_path: Path, mask_path: Path | None, size: int) -> tuple[Image.Image, Image.Image]:
    if size < 64 or size > 2048 or size % 64:
        raise ValueError("size must be a multiple of 64 between 64 and 2048")
    image = read_rgb(image_path)
    # Preserve field of view and aspect ratio; round the short side to a VAE-safe size.
    width, height = image.size
    scale = size / max(width, height)
    target = (max(64, round(width * scale / 64) * 64), max(64, round(height * scale / 64) * 64))
    if mask_path:
        with Image.open(mask_path) as source:
            mask = ImageOps.exif_transpose(source).convert("L")
        if mask.size != image.size:
            raise ValueError("mask must match the EXIF-oriented input dimensions")
        mask = binary_mask(mask.resize(target, Image.Resampling.NEAREST))
    else:
        mask = Image.new("L", target, 0)
        w, h = target
        ImageDraw.Draw(mask).rectangle((w * 0.12, h * 0.45, w * 0.88, h * 0.94), fill=255)
    if not np.asarray(mask).any():
        raise ValueError("mask has no editable pixels")
    return image.resize(target, Image.Resampling.LANCZOS), mask


def canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    edges = cv2.Canny(np.asarray(image.convert("L")), low, high)
    return Image.fromarray(edges).convert("RGB")


def composite(source: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    if source.size != generated.size or mask.size != source.size:
        raise ValueError("composite dimensions differ")
    return Image.composite(generated.convert("RGB"), source.convert("RGB"), binary_mask(mask))


def save_prepared(image: Image.Image, mask: Image.Image, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "input.png")
    mask.save(output / "mask.png")
    canny(image).save(output / "edges.png")
