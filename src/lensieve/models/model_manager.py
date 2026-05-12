import gc
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import AutoModel, AutoProcessor


@dataclass(frozen=True, slots=True)
class ModelNames:
    clip_like: str
    vision: str


class ModelKind(StrEnum):
    CLIP_LIKE = "clip_like"
    VISION = "vision"


class ModelManager:
    def __init__(self, model_names: ModelNames, device: str | None = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model_name: str | None = None
        self.model = None
        self.processor = None
        self.models = {ModelKind.CLIP_LIKE: model_names.clip_like, ModelKind.VISION: model_names.vision}

    def unload(self):
        if self.model is not None:
            # Note: self.model.to("cpu") is not needed (and never really was).
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.model_name = None
        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def get_model_name(self, model_kind: ModelKind):
        return self.models[model_kind]

    def load(self, model_kind: ModelKind):
        model_name = self.get_model_name(model_kind)
        if self.model_name == model_name and self.model is not None:
            return self.model, self.processor

        self.unload()

        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        # Note: torch_dtype is deprecated in favor of dtype.
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device in {"cuda", "mps"} else torch.float32,
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(self.device)

        self.model_name = model_name
        return self.model, self.processor

    def move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def get_model_manager(cfg: DictConfig, device: str | None = None) -> ModelManager:
    model_names = ModelNames(clip_like=cfg.models.clip_like.name, vision=cfg.models.vision.name)
    return ModelManager(model_names=model_names, device=device)
