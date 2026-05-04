from enum import Enum, auto
from typing import Any

import torch

from lensieve.models.inference_model import InferenceModel


class EmbedMode(Enum):
    IMAGE = auto()
    TEXT = auto()


class ClipLikeEmbedder(InferenceModel):
    def preprocess(self, processor: Any, *, images=None, texts=None):
        if (images is None) == (texts is None):
            raise ValueError("Pass exactly one of images or texts.")

        if images is not None:
            return EmbedMode.IMAGE, processor(images=images, return_tensors="pt", padding=True)
        return EmbedMode.TEXT, processor(text=texts, return_tensors="pt", padding=True)

    def call_model(self, model: Any, mode: EmbedMode, inputs: dict[str, Any]) -> Any:
        if mode is EmbedMode.IMAGE:
            return model.get_image_features(**inputs)
        if mode is EmbedMode.TEXT:
            return model.get_text_features(**inputs)
        raise ValueError(mode)

    def postprocess(self, outputs):
        outputs = torch.nn.functional.normalize(outputs, dim=-1)
        return outputs.detach().cpu()
