import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class TableName:
    IMAGES = "images"

    @staticmethod
    def embeddings(model_name: str) -> str:
        return f"embeddings__{normalize(model_name)}"


DISTANCE_COL = "_distance"


@dataclass(frozen=True, slots=True)
class BaseField:
    SHA256 = "sha256"


@dataclass(frozen=True, slots=True)
class DatedBaseField(BaseField):
    DATE_TAKEN = "date_taken"  # Denormalization for faster retrieval


@dataclass(frozen=True, slots=True)
class ImageField(DatedBaseField):
    PATH = "path"

    FILE_SIZE_BYTES = "file_size_bytes"
    FILE_MTIME_NS = "file_mtime_ns"

    WIDTH = "width"
    HEIGHT = "height"
    IMAGE_FORMAT = "image_format"

    CAMERA_MAKE = "camera_make"
    CAMERA_MODEL = "camera_model"
    ORIENTATION = "orientation"

    STRUCTURE_ERROR = "structure_error"
    EXIF_ERROR = "exif_error"


@dataclass(frozen=True, slots=True)
class EmbeddingField(DatedBaseField):
    VECTOR = "vector"
