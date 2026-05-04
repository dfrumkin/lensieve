from abc import ABC, abstractmethod
from itertools import batched
from logging import Logger
from pathlib import Path

import lancedb
import rawpy
from PIL import Image
from tqdm import tqdm

from lensieve.consts import RAW_EXTENSIONS
from lensieve.consts import BaseField as BF
from lensieve.consts import ImageField as IF
from lensieve.consts import TableName as TN
from lensieve.ingestion.utils import delete_rows, insert_rows, sql_ident
from lensieve.models.inference_model import InferenceModel
from lensieve.resources import Resources, duck_table

ImagePair = tuple[str, Path]
LoadedImagePair = tuple[str, Path, Image.Image]


class DerivedTableIngestor(ABC):
    def __init__(
        self,
        resources: Resources,
        table: lancedb.Table,
        logger: Logger,
        model: InferenceModel,
        workload_batch_size: int,
        insert_batch_size: int = 10_000,
    ) -> None:
        self.resources = resources
        self.table = table
        self.logger = logger
        self.model = model
        self.workload_batch_size = workload_batch_size
        self.insert_batch_size = insert_batch_size

        self.t_name = duck_table(self.table.name)
        self.i_name = duck_table(TN.IMAGES)
        self.sha_col = sql_ident(BF.SHA256)

    def run(self) -> None:
        self.delete_orphan_rows()

        pairs = self.find_new_image_pairs()

        output_rows: list[dict] = []
        num_images = 0
        num_rows = 0

        with tqdm(total=len(pairs), desc=self.table.name) as pbar:
            for pair_batch in batched(pairs, self.workload_batch_size, strict=False):
                pair_batch = list(pair_batch)
                num_images += len(pair_batch)

                rows = list(self.process_batch(pair_batch))
                output_rows.extend(rows)
                num_rows += len(rows)

                if len(output_rows) >= self.insert_batch_size:
                    insert_rows(self.table, output_rows, self.logger)
                    output_rows.clear()

                pbar.update(len(pair_batch))

        if output_rows:
            insert_rows(self.table, output_rows, self.logger)

        self.logger.info(
            "Finished %s: processed %s images, inserted %s rows",
            self.table.name,
            num_images,
            num_rows,
        )

    def delete_orphan_rows(self) -> None:

        rows = self.resources.duckdb.execute(
            f"""
            SELECT t.{self.sha_col}
            FROM {self.t_name} AS t
            ANTI JOIN {self.i_name} AS i
            ON t.{self.sha_col} = i.{self.sha_col}
            """
        ).fetchall()

        delete_rows(self.table, self.sha_col, [row[0] for row in rows], self.logger)

    def find_new_image_pairs(self) -> list[tuple[str, Path]]:
        path_col = sql_ident(IF.PATH)

        rows = self.resources.duckdb.execute(
            f"""
            SELECT i.{self.sha_col}, i.{path_col}
            FROM {self.i_name} AS i
            ANTI JOIN {self.t_name} AS t
            ON i.{self.sha_col} = t.{self.sha_col}
            """
        ).fetchall()

        return [(sha256, Path(path)) for sha256, path in rows]

    def process_batch(self, pairs: list[ImagePair]) -> list[dict]:
        loaded = self.load_images(pairs)
        return self.process_images(loaded)

    def load_image(self, path: Path) -> Image.Image:
        ext = path.suffix.lower()

        if ext in RAW_EXTENSIONS:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess()
            image = Image.fromarray(rgb)
        else:
            with Image.open(path) as img:
                image = img.convert("RGB")

        return image

    def load_images(self, pairs: list[ImagePair]) -> list[LoadedImagePair]:
        loaded = []

        for sha256, path in pairs:
            image = self.load_image(path)
            loaded.append((sha256, path, image))

        return loaded

    @abstractmethod
    def process_images(self, pairs: list[LoadedImagePair]) -> list[dict]: ...
