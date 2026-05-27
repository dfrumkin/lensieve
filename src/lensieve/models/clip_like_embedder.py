from enum import Enum, auto
from typing import Any

from lensieve.models.embedder import Embedder
from lensieve.models.model_manager import ModelRole


class EmbedMode(Enum):
    IMAGE = auto()
    TEXT = auto()


class ClipLikeEmbedder(Embedder):
    def __init__(self, **kwargs):
        super().__init__(model_kind=ModelRole.CLIP_LIKE, **kwargs)

    def preprocess(self, processor: Any, *, images=None, texts=None):
        if (images is None) == (texts is None):
            raise ValueError("Pass exactly one of images or texts.")

        if images is not None:
            return EmbedMode.IMAGE, processor(images=images, return_tensors="pt", padding=True)
        return EmbedMode.TEXT, processor(text=texts, return_tensors="pt", padding="max_length", truncation=True)

    def call_model(self, model: Any, mode: EmbedMode, inputs: dict[str, Any]) -> Any:
        output = model.get_image_features(**inputs) if mode is EmbedMode.IMAGE else model.get_text_features(**inputs)
        return output.pooler_output
