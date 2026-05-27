import re
from pathlib import Path


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


class TableName:
    IMAGES = "images"
    IMAGES_VIEW = "images_view"

    @staticmethod
    def embeddings(model_name: str) -> str:
        return f"embeddings__{normalize(model_name)}"


DISTANCE_COL = "_distance"


class BaseField:
    SHA256 = "sha256"


class DatedBaseField(BaseField):
    DATE_TAKEN = "date_taken"  # Denormalization for faster retrieval


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


class ImageViewField(ImageField):
    DISPLAY_WIDTH = "display_width"
    DISPLAY_HEIGHT = "display_height"
    ORIENTATION_KIND = "orientation_kind"
    ASPECT_RATIO = "aspect_ratio"


class EmbeddingField(DatedBaseField):
    VECTOR = "vector"
