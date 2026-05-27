from enum import StrEnum
from pathlib import Path

import rawpy
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

RASTER_FORMAT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".heic": "HEIF",
    ".heif": "HEIF",
    ".png": "PNG",
    ".webp": "WEBP",
    ".avif": "AVIF",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}

RAW_FORMAT_MAP = {
    ".dng": "DNG",  # Adobe / generic RAW
    ".arw": "ARW",  # Sony
    ".cr2": "CR2",  # Canon
    ".cr3": "CR3",  # Canon
    ".nef": "NEF",  # Nikon
    ".rw2": "RW2",  # Panasonic
    ".orf": "ORF",  # Olympus / OM System
    ".raf": "RAF",  # Fujifilm
    ".srw": "SRW",  # Samsung
    ".pef": "PEF",  # Pentax
    ".x3f": "X3F",  # Sigma/Foveon
}
RAW_EXTENSIONS = frozenset(RAW_FORMAT_MAP)
IMAGE_EXTENSIONS = frozenset(RASTER_FORMAT_MAP) | RAW_EXTENSIONS
IMAGE_FORMATS = frozenset(frozenset(RASTER_FORMAT_MAP.values()) | frozenset(RAW_FORMAT_MAP.values()))


def load_image(path: Path) -> Image.Image:
    ext = path.suffix.lower()

    if ext in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess()
        image = Image.fromarray(rgb)
    else:
        with Image.open(path) as img:
            # We do this ourselves so as not to rely on HF or downstream models.
            image = ImageOps.exif_transpose(img)
            # Note: .convert() creates a new copy, so we'll never have the original img.
            image = image.convert("RGB")
    return image


class ExifOrientation(StrEnum):
    HORIZONTAL = "Horizontal (normal)"
    MIRRORED_HORIZONTAL = "Mirrored horizontal"
    ROTATED_180 = "Rotated 180"
    MIRRORED_VERTICAL = "Mirrored vertical"
    MIRRORED_HORIZONTAL_ROTATED_90_CCW = "Mirrored horizontal then rotated 90 CCW"
    ROTATED_90_CW = "Rotated 90 CW"
    MIRRORED_HORIZONTAL_ROTATED_90_CW = "Mirrored horizontal then rotated 90 CW"
    ROTATED_90_CCW = "Rotated 90 CCW"


def parse_orientation(value: object | None) -> ExifOrientation | None:
    if value is None:
        return None

    text = str(value)

    try:
        return ExifOrientation(text)
    except ValueError:
        raise ValueError(f"Unknown EXIF orientation: {text!r}") from None
