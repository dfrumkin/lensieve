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
    name: str


@dataclass(frozen=True, slots=True)
class LLMInfo(ModelInfo):
    repo_id: str
    n_ctx: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class EncoderInfo(ModelInfo):
    batch_size: int
    embedding_dim: int


class ModelRole(StrEnum):
    ROUTER = "router"
    IMAGE_RETRIEVAL = "image_retrieval"
    IMAGE_RETRIEVAL_REPAIR = "image_retrieval_repair"
    METADATA_QUERY = "metadata_query"
    METADATA_QUERY_REPAIR = "metadata_query_repair"
    METADATA_INTERPRETER = "metadata_interpreter"
    CLIP_LIKE = "clip_like"
    VISION = "vision"


type ModelInfoByRole = dict[ModelRole, ModelInfo]


class ModelSource(StrEnum):
    LLAMA_CPP_HF = "llama_cpp_hf"
    TRANSFORMERS = "transformers"


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
    def __init__(self, model_info_by_role: ModelInfoByRole, device: DeviceType) -> None:
        self.device = device
        self.model_info_by_role = model_info_by_role
        self.llm_paths: dict[LLMInfo, str] = {}
        self.loaded_model_info: ModelInfo | None = None
        self.model: torch.nn.Module | Llama | None = None
        self.processor: Any | None = None

    def get_model_name(self, model_role: ModelRole) -> str:
        return self.model_info_by_role[model_role].name

    @property
    def unique_models(self) -> frozenset[ModelInfo]:
        return frozenset(self.model_info_by_role.values())

    def unload_model(self) -> None:
        is_pytorch = isinstance(self.model, torch.nn.Module)

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.loaded_model_info = None
        gc.collect()

        if is_pytorch:
            if self.device == DeviceType.CUDA:
                torch.cuda.empty_cache()
            elif self.device == DeviceType.MPS:
                torch.mps.empty_cache()

    def load_encoder(self, model_role: ModelRole) -> tuple[torch.nn.Module, Any]:
        model_info = self.model_info_by_role[model_role]
        if not isinstance(model_info, EncoderInfo):
            raise ValueError(f"Model role {model_role.value} is not an encoder model.")

        if self.loaded_model_info == model_info and self.model is not None and self.processor is not None:
            return self.model, self.processor  # type: ignore

        self.unload_model()

        model_name = model_info.name
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
        self.loaded_model_info = model_info

        return self.model, self.processor

    def _get_llm_path(self, llm_info: LLMInfo) -> str:
        filename = llm_info.name
        repo_id = llm_info.repo_id

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

        return path

    def get_llm_info(self, model_role: ModelRole) -> LLMInfo:
        model_info = self.model_info_by_role[model_role]
        if not isinstance(model_info, LLMInfo):
            raise ValueError(f"Model role {model_role.value} is not an LLM model.")
        return model_info

    def load_llm(self, model_role: ModelRole) -> Llama:
        model_info = self.get_llm_info(model_role)
        if self.loaded_model_info == model_info and self.model is not None:
            return self.model  # type: ignore
        self.unload_model()

        llm_path = self.llm_paths.get(model_info)
        if llm_path is None:
            llm_path = self._get_llm_path(model_info)
            self.llm_paths[model_info] = llm_path

        model = Llama(
            model_path=llm_path,
            n_ctx=model_info.n_ctx,
            n_gpu_layers=0 if self.device == DeviceType.CPU else -1,
            verbose=False,
        )

        self.model = model
        self.loaded_model_info = model_info

        return model

    def move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(torch.device(self.device.value)) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }


def _get_model_info(models_cfg: DictConfig, role_cfg: DictConfig) -> ModelInfo:
    model_cfg = models_cfg[role_cfg.model]
    source = model_cfg.source

    match source:
        case ModelSource.LLAMA_CPP_HF.value:
            return LLMInfo(
                repo_id=model_cfg.repo_id,
                name=model_cfg.name,
                n_ctx=model_cfg.n_ctx,
                max_tokens=role_cfg.max_tokens,
            )
        case ModelSource.TRANSFORMERS.value:
            return EncoderInfo(
                name=model_cfg.name,
                batch_size=model_cfg.batch_size,
                embedding_dim=model_cfg.embedding_dim,
            )
        case _:
            raise ValueError(f"Unknown model source: {source}")


def get_model_manager(cfg: DictConfig) -> ModelManager:
    roles = cfg.model_roles
    model_info_by_role: ModelInfoByRole = {
        ModelRole(role): _get_model_info(cfg.models, role_cfg) for role, role_cfg in roles.items()
    }
    device = DeviceType.detect() if cfg.device is None else DeviceType(cfg.device)
    return ModelManager(model_info_by_role=model_info_by_role, device=device)
