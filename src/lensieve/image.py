from pathlib import Path

import rawpy
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

RASTER_EXTENSIONS = frozenset(
    (
        ".jpg",
        ".jpeg",
        ".heic",
        ".heif",
        ".png",
        ".webp",
        ".avif",
        ".tif",
        ".tiff",
    )
)

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
IMAGE_EXTENSIONS = RASTER_EXTENSIONS | RAW_EXTENSIONS


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
