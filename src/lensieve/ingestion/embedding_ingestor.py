import pyarrow as pa

from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedBatch
from lensieve.models.embedder import Embedder
from lensieve.names import EmbeddingField as EF
from lensieve.names import TableName as TN


class EmbeddingIngestor(DerivedTableIngestor[Embedder]):
    def __init__(self, model: Embedder, embedding_dim: int, **kwargs) -> None:
        super().__init__(model=model, table_name=TN.embeddings(model.model_name), **kwargs)
        self.embedding_dim = embedding_dim

    def process_images(self, batch: LoadedBatch) -> list[dict]:
        images = [image_data.image for image_data in batch]
        vectors = self.model.run(images=images)
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
                pa.field(EF.VECTOR, pa.list_(pa.float32(), self.embedding_dim)),
                pa.field(EF.DATE_TAKEN, pa.timestamp("us")),
            ]
        )
