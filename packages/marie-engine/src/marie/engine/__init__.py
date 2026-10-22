from __future__ import annotations

from typing import TYPE_CHECKING

from marie.registry.model_catalog import load_model_catalog

if TYPE_CHECKING:
    from marie.engine.base import EngineLM

# Supported LLM providers
# Note: Claude is accessed via 'openai' provider with LiteLLM base_url
SUPPORTED_PROVIDERS = ["vllm", "openai"]
__all__ = [
    "EngineLM",
    "SUPPORTED_PROVIDERS",
    "MODEL_NAME_MAP",
    "REVERSE_MODEL_NAME_MAP",
    "__MULTIMODAL_MODELS__",
    "check_if_multimodal",
    "validate_multimodal_engine",
    "get_engine",
]


def __getattr__(name: str) -> type[EngineLM] | dict[str, str] | list[str]:
    if name == "EngineLM":
        from marie.engine.base import EngineLM

        return EngineLM
    if name in {"MODEL_NAME_MAP", "REVERSE_MODEL_NAME_MAP", "__MULTIMODAL_MODELS__"}:
        catalog = load_model_catalog()
        if name == "MODEL_NAME_MAP":
            return catalog.name_map
        if name == "REVERSE_MODEL_NAME_MAP":
            return catalog.reverse_name_map
        return catalog.multimodal_models
    raise AttributeError(name)


def check_if_multimodal(engine_name: str) -> bool:
    return load_model_catalog().is_multimodal(engine_name)


def validate_multimodal_engine(engine: EngineLM) -> None:
    if not check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine '{engine.model_string}' is not multimodal. "
            "Configure a model with image support in the external model catalog."
        )


def get_engine(engine_name: str, provider: str = "vllm", **kwargs) -> EngineLM:
    """
    Get the engine based on the engine name and provider.

    Args:
        engine_name: The engine/model name to use for the LLM call.
        provider: The provider to use for the LLM call.
            Supported: 'vllm', 'openai'
        **kwargs: Additional arguments passed to the engine constructor.
            For 'openai' provider, pass base_url to use LiteLLM proxy for
            other providers (Claude, Gemini, etc.)

    Returns:
        The engine to use for the LLM call.

    Example:
        ```python
        # VLLM local model
        engine = get_engine("qwen2_5_vl_7b", provider="vllm")

        # OpenAI API direct
        engine = get_engine("gpt-4o", provider="openai")

        # Claude via LiteLLM proxy
        engine = get_engine(
            "claude/claude-sonnet-4-20250514",
            provider="openai",
            base_url="http://localhost:4000",  # LiteLLM server
        )

        # Any provider via LiteLLM
        engine = get_engine(
            "anthropic/claude-3-opus",
            provider="openai",
            base_url="http://localhost:4000",
        )
        ```
    """
    if provider == "vllm":
        from .vllm_engine import VLLMEngine

        return VLLMEngine(model_name=engine_name, **kwargs)
    elif provider == "openai":
        from .openai_engine import OpenAIEngine

        return OpenAIEngine(
            model_name=engine_name,
            is_multimodal=True,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Provider '{provider}' not supported. "
            f"Supported providers: {SUPPORTED_PROVIDERS}"
        )
