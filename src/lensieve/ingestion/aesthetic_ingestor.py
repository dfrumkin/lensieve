import logging

import pyarrow as pa

from lensieve.consts import AestheticField as AF
from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedImagePair

logger = logging.getLogger(__name__)


class AestheticIngestor(DerivedTableIngestor):
    def process_images(self, pairs: list[LoadedImagePair]) -> list[dict]:
        images = [image for _, _, image in pairs]
        scores = self.model.run(images=images)
        # TODO: We'll need an extra head on top of CLIP, perhaps multiple scores.
        return [
            {
                AF.SHA256: sha256,
                AF.MODEL_NAME: self.model.model_name,
                AF.SCORE: float(score),
            }
            for (sha256, _, _), score in zip(pairs, scores, strict=True)
        ]

    def schema(self) -> pa.lib.Schema:
        return pa.schema(
            [
                pa.field(AF.SHA256, pa.string()),
                pa.field(AF.MODEL_NAME, pa.string()),
                pa.field(AF.SCORE, pa.float32()),
            ]
        )
