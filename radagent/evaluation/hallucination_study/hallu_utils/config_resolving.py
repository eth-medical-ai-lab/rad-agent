"""
Resolve YAML configuration files into typed objects used by hint-labeling jobs.

The dataclasses in this module define the runtime, vLLM, and model settings
required by the judge processors, while the loader validates that the selected
model exists in the supplied config file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass(frozen=True)
class RunConfig:
    selected_model: str
    error_threshold: int
    timestamp_format: str
    batch_size: int
    rag_device: Optional[int] = None
    judge_type: Optional[str] = None


@dataclass(frozen=True)
class SamplingParamsConfig:
    max_tokens: int
    logprobs: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


@dataclass(frozen=True)
class VLLMConfig:
    sampling_params_config: SamplingParamsConfig


@dataclass(frozen=True)
class ModelConfig:
    path: str
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    enable_thinking: bool = False
    base_url: Optional[str] = None
    quantization: Optional[str] = None
    language_model_only: Optional[bool] = None
    trust_remote_code: Optional[bool] = None


@dataclass(frozen=True)
class EvalLabelingConfig:
    run_config: RunConfig
    vllm_config: VLLMConfig
    models: Dict[str, ModelConfig]


def load_eval_label_config(path: str | Path, judge_type: Optional[str] = None) -> EvalLabelingConfig:
    run_config, vllm_config, models = resolve_sub_cfgs(path=path, judge_type=judge_type)
    return EvalLabelingConfig(
        run_config=run_config,
        vllm_config=vllm_config,
        models=models,
    )


def resolve_sub_cfgs(path: str | Path, judge_type: Optional[str]) -> tuple[RunConfig, VLLMConfig, Dict[str, ModelConfig]]:
    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    run_config = RunConfig(**raw_config["run"], judge_type=judge_type)
    sampling_params_config = SamplingParamsConfig(**raw_config["vllm"]["sampling_params"])
    vllm_config = VLLMConfig(sampling_params_config=sampling_params_config)

    models = {
        model_name: ModelConfig(**model_config)
        for model_name, model_config in raw_config["models"].items()
    }

    if run_config.selected_model not in models:
        raise ValueError(
            f"Selected model '{run_config.selected_model}' not found in config. "
            f"Available models: {list(models)}"
        )

    return run_config, vllm_config, models
