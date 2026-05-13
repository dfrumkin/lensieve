import numpy as np
import torch

from lensieve.models.inference_model import InferenceModel


class Embedder(InferenceModel):
    def postprocess(self, outputs: torch.Tensor) -> np.ndarray:
        return torch.nn.functional.normalize(outputs, dim=-1).detach().cpu().numpy().astype(np.float32)
