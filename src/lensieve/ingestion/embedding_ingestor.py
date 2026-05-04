from lensieve.consts import BaseField as BF
from lensieve.consts import EmbeddingField as EF
from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedImagePair
from lensieve.models.inference_model import InferenceModel


class EmbeddingIngestor(DerivedTableIngestor):
    def __init__(self, *, model: InferenceModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model

    def process_images(self, pairs: list[LoadedImagePair]) -> list[dict]:
        images = [image for _, _, image in pairs]
        vectors = self.model.run(images=images).numpy().astype("float32")
        return [
            {
                BF.SHA256: sha256,
                EF.VECTOR: vector.tolist(),
            }
            for (sha256, _, _), vector in zip(pairs, vectors, strict=True)
        ]
