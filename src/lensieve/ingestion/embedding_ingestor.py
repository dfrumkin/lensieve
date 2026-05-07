import pyarrow as pa

from lensieve.consts import EmbeddingField as EF
from lensieve.ingestion.derived_table_ingestor import DerivedTableIngestor, LoadedImagePair
from lensieve.models.embedder import Embedder


class EmbeddingIngestor(DerivedTableIngestor[Embedder]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process_images(self, pairs: list[LoadedImagePair]) -> list[dict]:
        images = [image for _, _, image in pairs]
        vectors = self.model.run(images=images).numpy().astype("float32")
        return [
            {
                EF.SHA256: sha256,
                EF.VECTOR: vector.tolist(),
            }
            for (sha256, _, _), vector in zip(pairs, vectors, strict=True)
        ]

    def schema(self) -> pa.lib.Schema:
        return pa.schema(
            [
                pa.field(EF.SHA256, pa.string()),
                pa.field(EF.VECTOR, pa.list_(pa.float32(), self.model.embedding_dim)),
            ]
        )
