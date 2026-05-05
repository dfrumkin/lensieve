import hydra
from transformers import AutoModel, AutoProcessor


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg) -> None:
    for section_name, section_cfg in cfg.models.items():
        print(f"[{section_name}]")
        model_name = section_cfg.name
        print(f"Downloading {model_name}")

        AutoProcessor.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)

        print(f"Done: {model_name}")


if __name__ == "__main__":
    main()
