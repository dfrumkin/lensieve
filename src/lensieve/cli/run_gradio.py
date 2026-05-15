import hydra
from omegaconf import DictConfig

from lensieve.agent.photo_agent import PhotoAgent
from lensieve.app.gradio_app import build_app
from lensieve.data_store import DataStore
from lensieve.logging_config import setup_logging
from lensieve.models.model_manager import get_model_manager
from lensieve.names import lensieve_root


@hydra.main(version_base=None, config_path="../../../configs", config_name="agent_config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.root, verbose=False)
    model_manager = get_model_manager(cfg)
    data_store = DataStore(cfg.root)

    agent = PhotoAgent(
        model_manager=model_manager,
        data_store=data_store,
        northern_hemisphere=cfg.agent.northern_hemisphere,
        max_steps=cfg.agent.max_steps,
        max_results=cfg.agent.tools.search_photos.max_results,
    )

    cache_dir = lensieve_root(data_store.root) / ".cache" / "display"
    build_app(agent, cache_dir).launch(inbrowser=True)


if __name__ == "__main__":
    main()
