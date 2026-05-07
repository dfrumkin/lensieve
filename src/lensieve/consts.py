import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    LENSIEVE_DIR = ".lensieve"
    LANCEDB_DIR = "lancedb"
    LOGS_DIR = "logs"


def lensieve_root(root: Path) -> Path:
    return root / AppPaths.LENSIEVE_DIR


def logs_path(root: Path) -> Path:
    return lensieve_root(root) / AppPaths.LOGS_DIR


def lancedb_path(root: Path) -> Path:
    return root / AppPaths.LENSIEVE_DIR / AppPaths.LANCEDB_DIR


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass(frozen=True)
class TableName:
    IMAGES = "images"

    @staticmethod
    def clip_like_embeddings(model_name: str) -> str:
        return f"embeddings__clip_like__{normalize(model_name)}"

    @staticmethod
    def vision_embeddings(model_name: str) -> str:
        return f"embeddings__vision__{normalize(model_name)}"


@dataclass(frozen=True)
class BaseField:
    SHA256 = "sha256"


@dataclass(frozen=True)
class ImageField(BaseField):
    PATH = "path"

    FILE_SIZE_BYTES = "file_size_bytes"
    FILE_MTIME_NS = "file_mtime_ns"

    WIDTH = "width"
    HEIGHT = "height"
    IMAGE_FORMAT = "image_format"

    DATE_TAKEN = "date_taken"
    CAMERA_MAKE = "camera_make"
    CAMERA_MODEL = "camera_model"
    ORIENTATION = "orientation"

    STRUCTURE_ERROR = "structure_error"
    EXIF_ERROR = "exif_error"


@dataclass(frozen=True)
class EmbeddingField(BaseField):
    VECTOR = "vector"


@dataclass(frozen=True)
class AestheticField(BaseField):
    SCORE = "score"


@dataclass(frozen=True)
class FaceDetectionField(BaseField):
    X1 = "x1"
    Y1 = "y1"
    X2 = "x2"
    Y2 = "y2"
    SCORE = "score"


RASTER_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".heic",
    ".heif",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
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
}
RAW_EXTENSIONS = set(RAW_FORMAT_MAP)
IMAGE_EXTENSIONS = RASTER_EXTENSIONS | RAW_EXTENSIONS
