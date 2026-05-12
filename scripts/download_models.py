import logging

import hydra

from lensieve.logging_config import setup_logging
from lensieve.models.model_manager import ModelKind, get_model_manager

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="ing_config", version_base=None)
def main(cfg) -> None:
    setup_logging(root=cfg.root)
    model_manager = get_model_manager(cfg=cfg, device="cpu")
    for kind in ModelKind:
        model_name = model_manager.get_model_name(kind)
        logger.info("Downloading %s", model_name)
        model_manager.load(kind)
        logger.info("Finished downloading %s", model_name)


if __name__ == "__main__":
    main()
