import logging

import pyarrow as pa

from lensieve.consts import AestheticField as AF
from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedBatch

logger = logging.getLogger(__name__)


class AestheticIngestor(DerivedTableIngestor):
    # TODO Do we want a table per model?
    def process_images(self, batch: LoadedBatch) -> list[dict]:
        images = [image_data.image for image_data in batch]
        scores = self.model.run(images=images)
        # TODO: We'll need an extra head on top of CLIP, perhaps multiple scores.
        return [
            {
                AF.SHA256: image_data.sha256,
                AF.SCORE: float(score),
            }
            for image_data, score in zip(batch, scores, strict=True)
        ]

    def schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(AF.SHA256, pa.string()),
                pa.field(AF.SCORE, pa.float32()),
            ]
        )
