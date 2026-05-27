from abc import ABC, abstractmethod
from typing import Any

import torch

from lensieve.models.model_manager import ModelManager, ModelRole


class InferenceModel(ABC):
    def __init__(self, manager: ModelManager, model_kind: ModelRole):
        self.manager = manager
        self.model_kind = model_kind

    @property
    def model_name(self) -> str:
        return self.manager.get_model_name(self.model_kind)

    def run(self, **kwargs) -> Any:
        model, processor = self.manager.load_encoder(self.model_kind)
        params, inputs = self.preprocess(processor, **kwargs)
        inputs = self.manager.move_to_device(inputs)

        with torch.inference_mode():
            outputs = self.call_model(model, params, inputs)

        return self.postprocess(outputs)

    @abstractmethod
    def preprocess(self, processor: Any, **kwargs) -> tuple[Any, dict[str, Any]]: ...

    @abstractmethod
    def call_model(self, model: Any, params: Any, inputs: dict[str, Any]) -> Any: ...

    @abstractmethod
    def postprocess(self, outputs: Any) -> Any: ...
