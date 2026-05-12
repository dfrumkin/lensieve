import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import batched
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import lancedb
import pyarrow as pa
import rawpy
from PIL import Image, ImageOps
from tqdm import tqdm

from lensieve.consts import RAW_EXTENSIONS
from lensieve.consts import DatedBaseField as DBF
from lensieve.consts import ImageField as IF
from lensieve.consts import TableName as TN
from lensieve.data_store import DataStore, duck_table_name, sql_ident
from lensieve.ingestion.utils import delete_rows, insert_rows, open_or_create_table
from lensieve.models.inference_model import InferenceModel

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImageData:
    sha256: str
    path: Path
    date_taken: datetime


@dataclass(frozen=True, slots=True)
class LoadedImageData(ImageData):
    image: Image.Image


type ImageDataList = list[ImageData]
type LoadedBatch = list[LoadedImageData]
type QueueItem = LoadedBatch | Exception | None


class DerivedTableIngestor[T: InferenceModel](ABC):
    def __init__(
        self,
        *,
        resources: DataStore,
        table_name: str,
        model: T,
        workload_batch_size: int,
        from_scratch: bool,
        delete_stale_data: bool,
        insert_batch_size: int,
        delete_batch_size: int,
        max_workers: int,
        max_prefetch_batches: int,
        queue_timeout_s: int,
    ) -> None:
        self.resources = resources
        self.table_name = table_name
        self.model = model
        self.from_scratch = from_scratch
        self.delete_stale_data = delete_stale_data
        self.workload_batch_size = workload_batch_size
        self.insert_batch_size = insert_batch_size
        self.delete_batch_size = delete_batch_size
        self.max_workers = max_workers
        self.max_prefetch_batches = max_prefetch_batches
        self.queue_timeout_s = queue_timeout_s

    def run(self) -> None:
        table = open_or_create_table(
            self.resources.lancedb,
            self.table_name,
            self.schema(),
            self.from_scratch,
        )
        image_data = self._sync_with_images(table)
        q, producer_thread = self._start_loader(image_data)

        output_rows: list[dict] = []
        num_images = 0
        num_rows = 0

        with tqdm(total=len(image_data), desc=self.table_name) as pbar:
            while True:
                try:
                    loaded = q.get(timeout=self.queue_timeout_s)
                except Empty:
                    if not producer_thread.is_alive():
                        logger.exception("Producer thread died unexpectedly")
                        raise RuntimeError("Producer thread died unexpectedly") from None
                    continue

                if isinstance(loaded, Exception):
                    logger.error("Image loader failed", exc_info=loaded)
                    raise RuntimeError("Image loader failed") from loaded
                if loaded is None:
                    break

                num_images += len(loaded)

                rows = self.process_images(loaded)
                for item in loaded:
                    item.image.close()  # frees internal buffers

                output_rows.extend(rows)
                num_rows += len(rows)
                if len(output_rows) >= self.insert_batch_size:
                    insert_rows(table, output_rows)
                    output_rows.clear()
                pbar.update(len(loaded))

        if output_rows:
            insert_rows(table, output_rows)

        logger.info(
            "Finished populating %s: processed %s images, inserted %s rows",
            self.table_name,
            num_images,
            num_rows,
        )

        self._summarize()

    def _sync_with_images(self, table: lancedb.Table) -> ImageDataList:
        sha_col = sql_ident(DBF.SHA256)
        path_col = sql_ident(IF.PATH)
        date_col = sql_ident(DBF.DATE_TAKEN)
        t_name = duck_table_name(self.table_name)
        i_name = duck_table_name(TN.IMAGES)
        structure_error_col = sql_ident(IF.STRUCTURE_ERROR)

        with self.resources.connect_duckdb_for_lance() as con:
            if self.delete_stale_data:
                # Delete stale rows
                stale_rows = con.execute(
                    f"""
                    SELECT t.{sha_col}
                    FROM {t_name} AS t
                    ANTI JOIN {i_name} AS i
                    ON t.{sha_col} = i.{sha_col}
                    """
                ).fetchall()

                delete_rows(table, DBF.SHA256, [row[0] for row in stale_rows], self.delete_batch_size)

            # Find new distinct images
            new_rows = con.execute(
                f"""
                SELECT DISTINCT ON (i.{sha_col}) i.{sha_col}, i.{path_col}, i.{date_col}
                FROM {i_name} AS i
                ANTI JOIN {t_name} AS t
                ON i.{sha_col} = t.{sha_col}
                WHERE i.{sha_col} IS NOT NULL
                AND NOT i.{structure_error_col}
                ORDER BY i.{sha_col}, i.{path_col}
                """
            ).fetchall()

        return [
            ImageData(sha256=sha256, path=self.resources.root / path, date_taken=date_taken)
            for sha256, path, date_taken in new_rows
        ]

    def _start_loader(self, image_data: ImageDataList) -> tuple[Queue, Thread]:
        q = Queue(maxsize=self.max_prefetch_batches)

        def producer():
            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    for batch in batched(image_data, self.workload_batch_size, strict=False):
                        loaded = list(pool.map(self._load_one, batch))
                        q.put(loaded)
            except Exception as e:
                q.put(e)
            else:
                q.put(None)

        thread = Thread(target=producer, daemon=True)
        thread.start()
        return q, thread

    def _load_one(self, image_data: ImageData) -> LoadedImageData:
        return LoadedImageData(
            sha256=image_data.sha256,
            path=image_data.path,
            date_taken=image_data.date_taken,
            image=self._load_image(image_data.path),
        )

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
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

    def _summarize(self) -> None:
        hash_col = sql_ident(DBF.SHA256)
        with self.resources.connect_duckdb_for_lance() as con:
            row = con.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    COUNT(DISTINCT {hash_col}) FILTER (WHERE {hash_col} IS NOT NULL) AS unique_hashes
                FROM {duck_table_name(self.table_name)}
                """
            ).fetchone()

        assert row is not None  # It will never be None for an aggregate query
        total, unique_hashes = row

        logger.info("Total images indexed: %s", total)
        logger.info("Total unique hashes: %s", unique_hashes)
        if total != unique_hashes:
            logger.error("HASHES MUST BE UNIQUE!")

    @abstractmethod
    def process_images(self, batch: LoadedBatch) -> list[dict]: ...

    @abstractmethod
    def schema(self) -> pa.Schema: ...
