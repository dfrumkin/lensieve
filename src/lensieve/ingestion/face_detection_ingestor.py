from lensieve.consts import BaseField as BF
from lensieve.consts import FaceDetectionField as FDF
from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedImagePair


class FaceDetectionIngestor(DerivedTableIngestor):
    def process_images(self, pairs: list[LoadedImagePair]) -> list[dict]:
        images = [image for _, _, image in pairs]
        detections_per_image = self.model.run(images=images)

        return [
            {
                BF.SHA256: sha256,
                FDF.X1: float(det.x1),
                FDF.Y1: float(det.y1),
                FDF.X2: float(det.x2),
                FDF.Y2: float(det.y2),
                FDF.SCORE: float(det.score),
            }
            for (sha256, _, _), detections in zip(pairs, detections_per_image, strict=True)
            for det in detections
        ]
