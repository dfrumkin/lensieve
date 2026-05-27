from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from lensieve.data.data_store import DataStore
from lensieve.ingestion.embedding_ingestor import EmbeddingIngestor
from lensieve.ingestion.image_ingestor import ingest_images
from lensieve.logging_config import setup_logging
from lensieve.models.clip_like_embedder import ClipLikeEmbedder
from lensieve.models.model_manager import EncoderInfo, ModelRole, get_model_manager
from lensieve.models.vision_embedder import VisionEmbedder


@hydra.main(version_base=None, config_path="../../../configs", config_name="ing_config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.root, app_name="ingest")

    ing_conf = cfg.ingestion

    common_conf = dict(
        data_store=DataStore(cfg.root),
        from_scratch=cfg.from_scratch,
        delete_stale_data=cfg.delete_stale_data,
        insert_batch_size=ing_conf.insert_batch_size,
        delete_batch_size=ing_conf.delete_batch_size,
    )
    img_conf: dict[str, Any] = OmegaConf.to_container(ing_conf.images, resolve=True)  # type: ignore
    img_conf.update(common_conf)

    if "images" in cfg.run:
        ingest_images(**img_conf)

    manager = get_model_manager(cfg)
    derived_conf: dict[str, Any] = OmegaConf.to_container(ing_conf.derived_table, resolve=True)  # type: ignore
    derived_conf.update(common_conf)

    model_role = ModelRole.CLIP_LIKE
    if model_role.value in cfg.run:
        model_info = manager.get_model_info(model_role)
        assert isinstance(model_info, EncoderInfo), f"Model for role {model_role.value} is not an encoder"
        clip_like_model = ClipLikeEmbedder(manager=manager)
        clip_like_ingestor = EmbeddingIngestor(
            model=clip_like_model,
            embedding_dim=model_info.embedding_dim,
            workload_batch_size=model_info.batch_size,
            **derived_conf,
        )
        clip_like_ingestor.run()

    model_role = ModelRole.VISION
    if model_role.value in cfg.run:
        model_info = manager.get_model_info(model_role)
        assert isinstance(model_info, EncoderInfo), f"Model for role {model_role.value} is not an encoder"
        vision_model = VisionEmbedder(manager=manager)
        vision_ingestor = EmbeddingIngestor(
            model=vision_model,
            embedding_dim=model_info.embedding_dim,
            workload_batch_size=model_info.batch_size,
            **derived_conf,
        )
        vision_ingestor.run()


if __name__ == "__main__":
    main()
