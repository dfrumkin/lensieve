import logging

import hydra
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from transformers import AutoModel, AutoProcessor

from lensieve.logging_config import setup_logging
from lensieve.models.model_manager import LLMInfo, get_model_manager

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging(root=cfg.root, app_name="download_models")
    model_manager = get_model_manager(cfg=cfg)

    for model_info in model_manager.unique_models:
        model_name = model_info.name

        logger.info("Downloading %s", model_name)
        if isinstance(model_info, LLMInfo):
            hf_hub_download(
                repo_id=model_info.repo_id,
                filename=model_name,
            )
        else:
            AutoProcessor.from_pretrained(model_name)
            AutoModel.from_pretrained(
                model_name,
                dtype=torch.float32,
            )
        logger.info("Finished downloading %s", model_name)


if __name__ == "__main__":
    main()
