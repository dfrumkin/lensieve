from lensieve.consts import AestheticField as AF
from lensieve.consts import BaseField as BF
from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedImagePair


class AestheticIngestor(DerivedTableIngestor):
    def process_images(self, pairs: list[LoadedImagePair]) -> list[dict]:
        images = [image for _, _, image in pairs]
        scores = self.model.run(images=images)

        return [
            {
                BF.SHA256: sha256,
                AF.SCORE: float(score),
            }
            for (sha256, _, _), score in zip(pairs, scores, strict=True)
        ]
