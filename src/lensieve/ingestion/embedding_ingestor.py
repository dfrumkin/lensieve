import pyarrow as pa

from lensieve.consts import EmbeddingField as EF
from lensieve.consts import TableName as TN
from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedBatch
from lensieve.models.embedder import Embedder


class EmbeddingIngestor(DerivedTableIngestor[Embedder]):
    def __init__(self, model: Embedder, **kwargs) -> None:
        super().__init__(model=model, table_name=TN.embeddings(model.model_name), **kwargs)

    def process_images(self, batch: LoadedBatch) -> list[dict]:
        images = [image_data.image for image_data in batch]
        vectors = self.model.run(images=images).numpy().astype("float32")
        return [
            {
                EF.SHA256: image_data.sha256,
                EF.VECTOR: vector.tolist(),
                EF.DATE_TAKEN: image_data.date_taken,
            }
            for image_data, vector in zip(batch, vectors, strict=True)
        ]

    def schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(EF.SHA256, pa.string()),
                pa.field(EF.VECTOR, pa.list_(pa.float32(), self.model.embedding_dim)),
                pa.field(EF.DATE_TAKEN, pa.timestamp("us")),
            ]
        )
