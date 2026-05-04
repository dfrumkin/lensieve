from typing import Any

import torch

from lensieve.models.inference_model import InferenceModel


class VisionEmbedder(InferenceModel):
    def preprocess(self, processor: Any, *, images: Any) -> tuple[None, dict[str, Any]]:
        return None, processor(images=images, return_tensors="pt")

    def call_model(self, model: Any, _: None, inputs: dict[str, Any]) -> torch.Tensor:
        return model(**inputs).pooler_output

    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(outputs, dim=-1).detach().cpu()
