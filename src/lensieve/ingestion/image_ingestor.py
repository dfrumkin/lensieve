import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import exifread
import pyarrow as pa
import rawpy
from PIL import Image
from tqdm import tqdm

from lensieve.data_store import DataStore, duck_table_name, sql_ident
from lensieve.image import IMAGE_EXTENSIONS, RAW_EXTENSIONS, RAW_FORMAT_MAP
from lensieve.ingestion.utils import delete_rows, insert_rows, open_or_create_table
from lensieve.names import AppPaths
from lensieve.names import ImageField as IF
from lensieve.names import TableName as TN

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FileStat:
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _FileFormat:
    width: int
    height: int
    image_format: str | None


SCHEMA = pa.schema(
    [
        pa.field(IF.PATH, pa.string(), nullable=False),
        pa.field(IF.SHA256, pa.string()),
        pa.field(IF.FILE_SIZE_BYTES, pa.int64(), nullable=False),
        pa.field(IF.FILE_MTIME_NS, pa.int64(), nullable=False),
        pa.field(IF.WIDTH, pa.int64()),
        pa.field(IF.HEIGHT, pa.int64()),
        pa.field(IF.IMAGE_FORMAT, pa.string()),
        pa.field(IF.DATE_TAKEN, pa.timestamp("us")),
        pa.field(IF.CAMERA_MAKE, pa.string()),
        pa.field(IF.CAMERA_MODEL, pa.string()),
        pa.field(IF.ORIENTATION, pa.string()),
        pa.field(IF.STRUCTURE_ERROR, pa.bool_(), nullable=False),
        pa.field(IF.EXIF_ERROR, pa.bool_(), nullable=False),
    ]
)


def parse_exif_datetime(value):
    if value is None:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        sha = h.hexdigest()
    except OSError as e:
        logger.error("Error hashing file %s: %s", path, e)
        sha = None
    return sha


def normalize_exif_value(value: Any) -> str | None:
    return None if value is None else (str(value).strip() or None)


def extract_pillow_structure(path: Path) -> _FileFormat:
    with Image.open(path) as img:
        return _FileFormat(width=img.width, height=img.height, image_format=img.format)


def extract_raw_structure(path: Path) -> _FileFormat:
    with rawpy.imread(path) as raw:
        return _FileFormat(
            width=raw.sizes.width,
            height=raw.sizes.height,
            image_format=RAW_FORMAT_MAP.get(path.suffix.lower()),
        )


def extract_image_structure(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        IF.WIDTH: None,
        IF.HEIGHT: None,
        IF.IMAGE_FORMAT: None,
        IF.STRUCTURE_ERROR: False,
    }

    extract = extract_raw_structure if path.suffix.lower() in RAW_EXTENSIONS else extract_pillow_structure
    try:
        file_format = extract(path)
        result[IF.WIDTH] = file_format.width
        result[IF.HEIGHT] = file_format.height
        result[IF.IMAGE_FORMAT] = file_format.image_format
    except Exception as e:
        logger.error("Error extracting structure from %s: %s", path, e)
        result[IF.STRUCTURE_ERROR] = True

    return result


def extract_exif_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        IF.DATE_TAKEN: None,
        IF.CAMERA_MAKE: None,
        IF.CAMERA_MODEL: None,
        IF.ORIENTATION: None,
        IF.EXIF_ERROR: False,
    }

    try:
        with path.open("rb") as f:
            tags = exifread.process_file(
                f,
                details=False,
                strict=False,
            )

        result[IF.DATE_TAKEN] = parse_exif_datetime(
            tags.get("EXIF DateTimeOriginal") or tags.get("EXIF DateTimeDigitized") or tags.get("Image DateTime")
        )

        result[IF.CAMERA_MAKE] = normalize_exif_value(tags.get("Image Make"))
        result[IF.CAMERA_MODEL] = normalize_exif_value(tags.get("Image Model"))
        result[IF.ORIENTATION] = normalize_exif_value(tags.get("Image Orientation"))
    except Exception as e:
        logger.error("Error extracting exif data from %s: %s", path, e)
        result[IF.EXIF_ERROR] = True

    return result


def list_image_paths(data_store: DataStore) -> list[Path]:
    paths = []

    for dirpath, dirnames, filenames in data_store.root.walk():
        # prevent descending into the .lensieve directory
        dirnames[:] = [d for d in dirnames if d != AppPaths.LENSIEVE_DIR]

        for name in filenames:
            p = dirpath / name
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)

    return sorted(paths)


def make_image_row(root: Path, rel_path: str, stat: _FileStat) -> dict[str, Any]:
    path = root / rel_path
    return {
        IF.PATH: rel_path,
        IF.SHA256: sha256_file(path),
        IF.FILE_SIZE_BYTES: stat.size,
        IF.FILE_MTIME_NS: stat.mtime_ns,
        **extract_image_structure(path),
        **extract_exif_metadata(path),
    }


def summarize(data_store: DataStore) -> None:
    str_error_col = sql_ident(IF.STRUCTURE_ERROR)
    exif_error_col = sql_ident(IF.EXIF_ERROR)
    hash_col = sql_ident(IF.SHA256)
    with data_store.connect_duckdb_for_lance() as con:
        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE {str_error_col}) AS structure_errors,
                COUNT(*) FILTER (WHERE {exif_error_col}) AS exif_errors,
                COUNT(*) FILTER (WHERE {hash_col} IS NULL) AS hash_errors,
                COUNT(DISTINCT {hash_col}) FILTER (WHERE {hash_col} IS NOT NULL) AS unique_hashes
            FROM {duck_table_name(TN.IMAGES)}
            """
        ).fetchone()

    assert row is not None  # It will never be None for an aggregate query
    total, structure_errors, exif_errors, hash_errors, unique_hashes = row

    logger.info("Total images indexed: %s", total)
    logger.info("Total unique hashes: %s", unique_hashes)

    if hash_errors:
        logger.warning("Images with hash errors: %s", hash_errors)
    if structure_errors:
        logger.warning("Images with structure errors: %s", structure_errors)
    if exif_errors:
        logger.warning("Images with EXIF errors: %s", exif_errors)


def make_image_row_worker(args):
    root, rel, stat = args
    return make_image_row(root, rel, stat)


def ingest_images(
    *,
    data_store: DataStore,
    from_scratch: bool,
    delete_stale_data: bool,
    insert_batch_size: int,
    delete_batch_size: int,
    max_workers: int,
    worker_chunk_size: int,
) -> None:
    # Current filesystem snapshot
    paths = list_image_paths(data_store)
    current: dict[str, _FileStat] = {}
    for p in paths:
        stat = p.stat()
        rel = p.relative_to(data_store.root).as_posix()
        current[rel] = _FileStat(size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    # Existing table snapshot
    table = open_or_create_table(data_store.lancedb, TN.IMAGES, SCHEMA, from_scratch)

    # Find new and stale data
    existing_rows = {
        row[IF.PATH]: row for row in table.search().select([IF.PATH, IF.FILE_SIZE_BYTES, IF.FILE_MTIME_NS]).to_list()
    }
    to_delete: list[str] = []
    for rel, row in existing_rows.items():
        if rel in current:
            stat = current[rel]
            if row[IF.FILE_SIZE_BYTES] != stat.size or row[IF.FILE_MTIME_NS] != stat.mtime_ns:
                to_delete.append(rel)  # same path, but changed file (size or mtime mismatch) => always delete
            else:
                current.pop(rel)  # unchanged file, remove from current to skip re-insertion
        elif delete_stale_data:
            to_delete.append(rel)  # missing file

    delete_rows(table, IF.PATH, to_delete, delete_batch_size)

    # Insertions (new + changed)
    items = [(data_store.root, rel, stat) for rel, stat in current.items()]

    # There are exif parsing errors with multithreading => multiprocessing instead.
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        to_insert = list(
            tqdm(
                ex.map(make_image_row_worker, items, chunksize=worker_chunk_size),
                total=len(items),
                desc="Processing images",
                unit="img",
            )
        )

    insert_rows(table, to_insert, insert_batch_size)

    # Summarize final state
    summarize(data_store)
