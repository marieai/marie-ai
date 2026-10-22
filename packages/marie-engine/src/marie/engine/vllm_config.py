from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from importlib.util import find_spec
from inspect import signature
from typing import Any, Callable, get_args

from marie.registry.model_config import ResolvedModelConfig

try:
    from vllm import LLM
    from vllm.config.model import ModelDType
    from vllm.engine.arg_utils import EngineArgs
except ImportError as exc:
    raise ImportError(
        "vLLM is required to run this module. Install it using: uv add vllm"
    ) from exc

_LLM_PARAMETERS = set(signature(LLM.__init__).parameters)


def build_llm_kwargs(
    config: ResolvedModelConfig,
    *,
    dependency_available: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Adapt resolved loading/runtime settings without model-specific defaults."""
    loading = deepcopy(config.loading)
    requires = loading.pop("requires", None)
    fallback = loading.pop("fallback", None)
    if requires is not None:
        available = (
            dependency_available(requires)
            if dependency_available
            else find_spec(requires) is not None
        )
        if not available:
            if fallback is None:
                raise ImportError(
                    f"Model '{config.name}' requires {requires}; configure a fallback in {config.source_path}"
                )
            loading.update(fallback)
    kwargs = {**loading, **deepcopy(config.runtime)}
    kwargs["model"] = config.name
    kwargs["mm_processor_kwargs"] = deepcopy(config.processor)
    if config.revision is not None:
        kwargs["revision"] = config.revision
        kwargs["tokenizer_revision"] = config.revision
    return kwargs


def create_llm_instance(config: ResolvedModelConfig) -> LLM:
    """Initialize vLLM from registry-resolved settings and validate runtime support."""
    if not config.enabled:
        raise ValueError(
            f"vLLM is disabled for '{config.name}' in {config.source_path}"
        )
    kwargs = build_llm_kwargs(config)
    allowed = _LLM_PARAMETERS | {field.name for field in fields(EngineArgs)}
    unsupported = set(kwargs) - allowed
    location = f"{config.source_path} (profile {config.profile or 'defaults'})"
    if unsupported:
        raise ValueError(
            f"Unsupported vLLM settings {sorted(unsupported)} in {location}"
        )
    dtype = kwargs.get("dtype")
    if isinstance(dtype, str) and dtype not in get_args(ModelDType):
        raise ValueError(
            f"Unsupported vLLM dtype '{dtype}' in {location}; quantization and model dtype are separate settings"
        )
    return LLM(**kwargs)
