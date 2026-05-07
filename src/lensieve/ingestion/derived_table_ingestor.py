from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from logging import Logger
from pathlib import Path

import pyarrow as pa
import rawpy
from PIL import Image, ImageOps
from tqdm import tqdm

from lensieve.consts import RAW_EXTENSIONS
from lensieve.consts import BaseField as BF
from lensieve.consts import ImageField as IF
from lensieve.consts import TableName as TN
from lensieve.ingestion.utils import delete_rows, insert_rows, open_or_create_table, sql_ident
from lensieve.models.inference_model import InferenceModel
from lensieve.resources import Resources, duck_table_name

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

ImagePair = tuple[str, Path]
LoadedImagePair = tuple[str, Path, Image.Image]


class DerivedTableIngestor[T: InferenceModel](ABC):
    def __init__(
        self,
        *,
        resources: Resources,
        table_name: str,
        model: T,
        workload_batch_size: int,
        logger: Logger,
        insert_batch_size: int = 10_000,
        delete_batch_size: int = 1_000,
    ) -> None:
        self.resources = resources
        self.table_name = table_name
        self.logger = logger
        self.model = model
        self.workload_batch_size = workload_batch_size
        self.insert_batch_size = insert_batch_size
        self.delete_batch_size = delete_batch_size

        self.t_name = duck_table_name(self.table_name)
        self.i_name = duck_table_name(TN.IMAGES)

    def run(self) -> None:
        table = open_or_create_table(self.resources.lancedb, self.table_name, self.schema(), self.logger)
        self._delete_orphan_rows(table)

        pairs = self._find_new_image_pairs()

        output_rows: list[dict] = []
        num_images = 0
        num_rows = 0

        with tqdm(total=len(pairs), desc=self.table_name) as pbar:
            for pair_batch in batched(pairs, self.workload_batch_size, strict=False):
                pair_batch = list(pair_batch)
                num_images += len(pair_batch)

                rows = list(self._process_batch(pair_batch))
                output_rows.extend(rows)
                num_rows += len(rows)

                if len(output_rows) >= self.insert_batch_size:
                    insert_rows(table, output_rows, self.logger)
                    output_rows.clear()

                pbar.update(len(pair_batch))

        if output_rows:
            insert_rows(table, output_rows, self.logger)

        self.logger.info(
            "Finished populating %s: processed %s images, inserted %s rows",
            self.table_name,
            num_images,
            num_rows,
        )

    def _delete_orphan_rows(self, table) -> None:
        sha_col = sql_ident(BF.SHA256)

        rows = self.resources.duckdb.execute(
            f"""
            SELECT t.{sha_col}
            FROM {self.t_name} AS t
            ANTI JOIN {self.i_name} AS i
            ON t.{sha_col} = i.{sha_col}
            """
        ).fetchall()

        delete_rows(table, BF.SHA256, [row[0] for row in rows], self.logger, self.delete_batch_size)

    def _find_new_image_pairs(self) -> list[tuple[str, Path]]:
        sha_col = sql_ident(BF.SHA256)
        path_col = sql_ident(IF.PATH)

        # We want to process new distinct images.
        rows = self.resources.duckdb.execute(
            f"""
            SELECT DISTINCT ON (i.{sha_col}) i.{sha_col}, i.{path_col}
            FROM {self.i_name} AS i
            ANTI JOIN {self.t_name} AS t
            ON i.{sha_col} = t.{sha_col}
            ORDER BY i.{sha_col}, i.{path_col}
            """
        ).fetchall()

        return [(sha256, self.resources.root / path) for sha256, path in rows]

    def _process_batch(self, pairs: list[ImagePair]) -> list[dict]:
        loaded = self._load_images(pairs)
        return self.process_images(loaded)

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
                image = image.convert("RGB")

        return image

    def _load_images(self, pairs: list[ImagePair]) -> list[LoadedImagePair]:
        def load_one(pair: ImagePair) -> LoadedImagePair:
            sha256, path = pair
            image = self._load_image(path)
            return sha256, path, image

        with ThreadPoolExecutor(max_workers=4) as ex:
            loaded = list(ex.map(load_one, pairs))

        return loaded

    @abstractmethod
    def process_images(self, pairs: list[LoadedImagePair]) -> list[dict]: ...

    @abstractmethod
    def schema(self) -> pa.lib.Schema: ...
