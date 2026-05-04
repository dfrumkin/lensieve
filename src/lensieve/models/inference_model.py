from abc import ABC, abstractmethod
from typing import Any

import torch


class InferenceModel(ABC):
    def __init__(self, manager, model_name: str):
        self.manager = manager
        self.model_name = model_name

    def run(self, **kwargs):
        model, processor = self.manager.load(self.model_name)

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
