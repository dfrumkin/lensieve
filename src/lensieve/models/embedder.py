import torch

from lensieve.models.inference_model import InferenceModel
from lensieve.models.model_manager import ModelManager


class Embedder(InferenceModel):
    def __init__(self, manager: ModelManager, model_name: str, embedding_dim: int):
        super().__init__(manager, model_name)
        self.embedding_dim = embedding_dim

    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(outputs, dim=-1).detach().cpu()
