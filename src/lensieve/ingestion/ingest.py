from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from lensieve.ingestion.embedding_ingestor import EmbeddingIngestor
from lensieve.ingestion.image_ingestor import ingest_images
from lensieve.logging_config import setup_logging
from lensieve.models.clip_like_embedder import ClipLikeEmbedder
from lensieve.models.model_manager import ModelManager
from lensieve.models.vision_embedder import VisionEmbedder
from lensieve.resources import Resources


@hydra.main(version_base=None, config_path="../../../configs", config_name="ing_config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.root, verbose=False)

    ing_conf = cfg.ingestion

    common_conf = dict(
        resources=Resources(cfg.root),
        from_scratch=cfg.from_scratch,
        delete_stale_data=cfg.delete_stale_data,
        insert_batch_size=ing_conf.insert_batch_size,
        delete_batch_size=ing_conf.delete_batch_size,
    )
    img_conf: dict[str, Any] = OmegaConf.to_container(ing_conf.images, resolve=True)  # type: ignore
    img_conf.update(common_conf)

    if "images" in cfg.run:
        ingest_images(**img_conf)

    manager = ModelManager()
    derived_conf: dict[str, Any] = OmegaConf.to_container(ing_conf.derived_table, resolve=True)  # type: ignore
    derived_conf.update(common_conf)

    if "clip_like" in cfg.run:
        clip_like_model = ClipLikeEmbedder(manager, cfg.models.clip_like.name, cfg.models.clip_like.embedding_dim)
        clip_like_ingestor = EmbeddingIngestor(
            model=clip_like_model, workload_batch_size=cfg.models.clip_like.batch_size, **derived_conf
        )
        clip_like_ingestor.run()

    if "vision" in cfg.run:
        vision_model = VisionEmbedder(manager, cfg.models.vision.name, cfg.models.vision.embedding_dim)
        vision_ingestor = EmbeddingIngestor(
            model=vision_model, workload_batch_size=cfg.models.vision.batch_size, **derived_conf
        )
        vision_ingestor.run()


if __name__ == "__main__":
    main()
