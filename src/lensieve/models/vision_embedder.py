from typing import Any

import torch

from lensieve.models.embedder import Embedder
from lensieve.models.model_manager import ModelRole


class VisionEmbedder(Embedder):
    def __init__(self, **kwargs):
        super().__init__(model_kind=ModelRole.VISION, **kwargs)

    def preprocess(self, processor: Any, *, images: Any) -> tuple[None, dict[str, Any]]:
        return None, processor(images=images, return_tensors="pt")

    def call_model(self, model: Any, _: None, inputs: dict[str, Any]) -> torch.Tensor:
        return model(**inputs).last_hidden_state[:, 0]
