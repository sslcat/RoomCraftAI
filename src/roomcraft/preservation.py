"""Optional CPU compositing; white mask restores source, white foreground excludes objects."""

from __future__ import annotations

from PIL import Image, ImageChops, ImageFilter


def selective_composite(
    source: Image.Image,
    generated: Image.Image,
    protection: Image.Image,
    foreground: Image.Image | None = None,
    feather: int = 8,
    match_lighting: bool = False,
) -> tuple[Image.Image, Image.Image]:
    """Feather inward only; never leak source pixels into excluded furniture."""
    if source.size != generated.size or protection.size != source.size:
        raise ValueError("Source, generated image and protection mask dimensions must match.")
    if foreground is not None and foreground.size != source.size:
        raise ValueError("Foreground mask dimensions must match the source.")
    if not 0 <= feather <= 24:
        raise ValueError("Feather must be between 0 and 24 pixels.")
    allowed = protection.convert("L").point(lambda v: 255 if v >= 128 else 0)
    if foreground is not None:
        exclusion = foreground.convert("L").point(lambda v: 255 if v >= 128 else 0)
        # Dilate around object edges before feathering to avoid erasing fine silhouettes.
        exclusion = exclusion.filter(ImageFilter.MaxFilter(5))
        allowed = ImageChops.subtract(allowed, exclusion)
    if allowed.getbbox() is None:
        raise ValueError("The protection mask has no usable white pixels.")
    alpha = (
        ImageChops.multiply(
            allowed,
            allowed.filter(ImageFilter.MinFilter(2 * feather + 1)).filter(
                ImageFilter.GaussianBlur(feather / 2)
            ),
        )
        if feather
        else allowed
    )
    original = source.convert("RGB")
    if match_lighting:
        # Transfer only broad illumination; the source supplies high-frequency texture.
        source_low = original.filter(ImageFilter.GaussianBlur(32))
        target_low = generated.convert("RGB").filter(ImageFilter.GaussianBlur(32))
        difference = ImageChops.subtract(target_low, source_low, scale=1, offset=128)
        original = ImageChops.add(original, difference, scale=1, offset=-128)
    return Image.composite(original, generated.convert("RGB"), alpha), alpha
