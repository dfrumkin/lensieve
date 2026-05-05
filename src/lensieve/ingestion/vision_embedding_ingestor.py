import logging

from lensieve.consts import TableName as TN
from lensieve.ingestion.embedding_ingestor import EmbeddingIngestor
from lensieve.models.vision_embedder import VisionEmbedder
from lensieve.resources import Resources

logger = logging.getLogger(__name__)


class VisionEmbeddingIngestor(EmbeddingIngestor):
    def __init__(
        self,
        *,
        resources: Resources,
        model: VisionEmbedder,
        workload_batch_size: int,
        insert_batch_size: int = 10_000,
        delete_batch_size: int = 1_000,
    ) -> None:
        super().__init__(
            resources=resources,
            table_name=TN.VISION_EMBEDDINGS,
            model=model,
            workload_batch_size=workload_batch_size,
            logger=logger,
            insert_batch_size=insert_batch_size,
            delete_batch_size=delete_batch_size,
        )


if __name__ == "__main__":
    from pathlib import Path

    import hydra
    from omegaconf import DictConfig

    from lensieve.logging_config import setup_logging
    from lensieve.models.model_manager import ModelManager
    from lensieve.models.vision_embedder import VisionEmbedder
    from lensieve.resources import create_resources

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        data_root = Path(__file__).resolve().parents[3] / "data" / "small_samsung"

        setup_logging(data_root, verbose=False)
        resources = create_resources(data_root)

        model_name = cfg.models.vision.name
        batch_size = cfg.models.vision.batch_size
        embedding_dim = cfg.models.vision.embedding_dim

        manager = ModelManager()
        vision_model = VisionEmbedder(manager, model_name, embedding_dim)
        ingestor = VisionEmbeddingIngestor(resources=resources, model=vision_model, workload_batch_size=batch_size)
        ingestor.run()

    main()
