import gc
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from llama_cpp import Llama
from omegaconf import DictConfig
from transformers import AutoModel, AutoProcessor


@dataclass(frozen=True, slots=True)
class ModelInfo:
    llm_repo_id: str
    llm_ctx: int
    llm: str
    clip_like: str
    vision: str


class ModelKind(StrEnum):
    LLM = "llm"
    CLIP_LIKE = "clip_like"
    VISION = "vision"


class DeviceType(StrEnum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

    @staticmethod
    def detect() -> "DeviceType":
        return (
            DeviceType.CUDA
            if torch.cuda.is_available()
            else DeviceType.MPS
            if torch.backends.mps.is_available()
            else DeviceType.CPU
        )


class ModelManager:
    def __init__(self, model_info: ModelInfo, device: DeviceType | None = None):
        self.device = device or DeviceType.detect()
        self.model_name: str | None = None
        self.model: torch.nn.Module | Llama | None = None
        self.processor: Any | None = None
        self.llm_repo_id = model_info.llm_repo_id
        self.llm_ctx = model_info.llm_ctx
        self.llm_path = None
        self.model_names: dict[ModelKind, str] = {
            ModelKind.LLM: model_info.llm,
            ModelKind.CLIP_LIKE: model_info.clip_like,
            ModelKind.VISION: model_info.vision,
        }

    def unload(self) -> None:
        is_pytorch = isinstance(self.model, torch.nn.Module)

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.model_name = None
        gc.collect()

        if is_pytorch:
            if self.device == DeviceType.CUDA:
                torch.cuda.empty_cache()
            elif self.device == DeviceType.MPS:
                torch.mps.empty_cache()

    def get_model_name(self, model_kind: ModelKind) -> str:
        return self.model_names[model_kind]

    def load_hf(self, model_kind: ModelKind) -> tuple[torch.nn.Module, Any]:
        if model_kind not in (ModelKind.CLIP_LIKE, ModelKind.VISION):
            raise ValueError(f"load_hf cannot load {model_kind}.  Use a different loader.")

        model_name = self.get_model_name(model_kind)
        if self.model_name == model_name and self.model is not None:
            return self.model, self.processor  # type: ignore

        self.unload()

        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        # Note: torch_dtype is deprecated in favor of dtype.
        model: torch.nn.Module = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32 if self.device == DeviceType.CPU else torch.float16,
            local_files_only=True,
        )
        model.eval()
        model.to(torch.device(self.device.value))
        self.model = model

        self.model_name = model_name
        return self.model, self.processor

    def _get_llm_path(self) -> str:
        if self.llm_path is not None:
            return self.llm_path

        filename = self.get_model_name(ModelKind.LLM)
        repo_id = self.llm_repo_id
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_files_only=True,
            )
        except LocalEntryNotFoundError as e:
            raise FileNotFoundError(
                "LLM GGUF file is not available in the local Hugging Face cache. "
                f"Download it first: repo_id={repo_id}, "
                f"filename={filename}"
            ) from e

        self.llm_path = path
        return self.llm_path

    def load_llm(self) -> Llama:
        model_name = self.get_model_name(ModelKind.LLM)
        if self.model_name == model_name and self.model is not None:
            return self.model  # type: ignore

        self.unload()

        self.processor = None
        self.model = Llama(
            model_path=self._get_llm_path(),
            n_ctx=self.llm_ctx,
            n_gpu_layers=0 if self.device == DeviceType.CPU else -1,
            verbose=False,
        )

        self.model_name = model_name
        return self.model

    def move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(torch.device(self.device.value)) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }


def get_model_manager(cfg: DictConfig, device: DeviceType | None = None) -> ModelManager:
    cf = cfg.models
    model_info = ModelInfo(
        llm=cf.llm.name,
        llm_repo_id=cf.llm.repo_id,
        llm_ctx=cf.llm.n_ctx,
        clip_like=cf.clip_like.name,
        vision=cf.vision.name,
    )
    return ModelManager(model_info=model_info, device=device)
