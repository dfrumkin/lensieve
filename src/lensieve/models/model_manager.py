import gc
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor


class ModelManager:
    def __init__(self, device: str | None = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.current_name = None
        self.model = None
        self.processor = None

    def unload(self):
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.current_name = None
        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def load(self, model_name: str):
        if self.current_name == model_name and self.model is not None:
            return self.model, self.processor

        self.unload()

        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device in {"cuda", "mps"} else torch.float32,
        )
        self.model.eval()
        self.model.to(self.device)

        self.current_name = model_name
        return self.model, self.processor

    def move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
