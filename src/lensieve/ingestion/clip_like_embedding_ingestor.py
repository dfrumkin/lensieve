import logging

from lensieve.consts import TableName as TN
from lensieve.ingestion.embedding_ingestor import EmbeddingIngestor
from lensieve.models.clip_like_embedder import ClipLikeEmbedder
from lensieve.resources import Resources

logger = logging.getLogger(__name__)


class ClipLikeEmbeddingIngestor(EmbeddingIngestor):
    def __init__(
        self,
        *,
        resources: Resources,
        model: ClipLikeEmbedder,
        workload_batch_size: int,
        insert_batch_size: int = 10_000,
        delete_batch_size: int = 1_000,
    ) -> None:
        super().__init__(
            resources=resources,
            table_name=TN.CLIP_LIKE_EMBEDDINGS,
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
    from lensieve.models.clip_like_embedder import ClipLikeEmbedder
    from lensieve.models.model_manager import ModelManager
    from lensieve.resources import create_resources

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        data_root = Path(__file__).resolve().parents[3] / "data" / "small_samsung"

        setup_logging(data_root, verbose=False)
        resources = create_resources(data_root)

        model_name = cfg.models.clip_like.name
        batch_size = cfg.models.clip_like.batch_size
        embedding_dim = cfg.models.clip_like.embedding_dim

        manager = ModelManager()
        clip_like_model = ClipLikeEmbedder(manager, model_name, embedding_dim)
        ingestor = ClipLikeEmbeddingIngestor(resources=resources, model=clip_like_model, workload_batch_size=batch_size)
        ingestor.run()

    main()
