from lensieve.consts import BaseField as BF
from lensieve.consts import FaceDetectionField as FDF
from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedBatch


class FaceDetectionIngestor(DerivedTableIngestor):
    # TODO!
    def process_images(self, batch: LoadedBatch) -> list[dict]:
        images = [image_data.image for image_data in batch]
        detections_per_image = self.model.run(images=images)

        return [
            {
                BF.SHA256: image_data.sha256,
                FDF.X1: float(det.x1),
                FDF.Y1: float(det.y1),
                FDF.X2: float(det.x2),
                FDF.Y2: float(det.y2),
                FDF.SCORE: float(det.score),
            }
            for image_data, detections in zip(batch, detections_per_image, strict=True)
            for det in detections
        ]
