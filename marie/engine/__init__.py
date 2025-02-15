__ENGINE_NAME_SHORTCUTS__ = {
    "together-llama-3-70b": "together-meta-llama/Llama-3-70b-chat-hf",
    "vllm-llama-3-8b": "vllm-meta-llama/Meta-Llama-3-8B-Instruct",
}

# Any better way to do this?
__MULTIMODAL_ENGINES__ = [
    "gpt-4-turbo",
    "gpt-4o",
]

from marie.engine.base import EngineLM


def _check_if_multimodal(engine_name: str):
    return any([name == engine_name for name in __MULTIMODAL_ENGINES__])


def validate_multimodal_engine(engine):
    if not _check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_ENGINES__}"
        )


def get_engine(engine_name: str, **kwargs) -> EngineLM:
    if engine_name in __ENGINE_NAME_SHORTCUTS__:
        engine_name = __ENGINE_NAME_SHORTCUTS__[engine_name]

    if (
        "seed" in kwargs
        and "gpt-4" not in engine_name
        and "gpt-3.5" not in engine_name
        and "gpt-35" not in engine_name
    ):
        raise ValueError(
            f"Seed is currently supported only for OpenAI engines, not {engine_name}"
        )

    if "cache" in kwargs and "experimental" not in engine_name:
        raise ValueError(
            f"Cache is currently supported only for LiteLLM engines, not {engine_name}"
        )

    # check if engine_name starts with "experimental:"
    if engine_name.startswith("experimental:"):
        engine_name = engine_name.split("experimental:")[1]
        raise ValueError(
            f"Experimental engines are not supported in this version of Marie. Please use a stable engine instead."
        )
    elif "vllm" in engine_name:
        from .vllm import ChatVLLM

        engine_name = engine_name.replace("vllm-", "")
        return ChatVLLM(model_string=engine_name, **kwargs)
    else:
        raise ValueError(f"Engine {engine_name} not supported")
