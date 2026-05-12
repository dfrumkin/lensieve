import torch

from lensieve.models.inference_model import InferenceModel


class Embedder(InferenceModel):
    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(outputs, dim=-1).detach().cpu()
