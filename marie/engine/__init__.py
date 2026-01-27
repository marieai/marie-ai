from marie.engine.base import EngineLM

# TODO add  Gemma 3 (Based on v4.49.0)
# https://github.com/huggingface/transformers/releases/tag/v4.49.0-Gemma-3
MODEL_NAME_MAP = {
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_7b_awq": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "qwen3_vl_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_vl_4b_merged_lora": "qwen3_vl_4b_merged_lora",
    "qwen2_5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2_5_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_14b": "Qwen/Qwen2.5-14B-Instruct",
    "meta_llama_11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "phi3_5_vl": "microsoft/Phi-3.5-vision-instruct",
    "pixtral_12b": "mistralai/Pixtral-12B-2409",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "opt_125m": "facebook/opt-125m",
    "phi4": "microsoft/phi-4",
    "llava_mistral_7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "deepseek_r1_qwen_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

# Create a reverse lookup dictionary
REVERSE_MODEL_NAME_MAP = {v: k for k, v in MODEL_NAME_MAP.items()}

# the multimodal models list
__MULTIMODAL_MODELS__ = [
    MODEL_NAME_MAP["qwen2_5_vl_3b"],
    MODEL_NAME_MAP["qwen2_5_vl_7b"],
    MODEL_NAME_MAP["qwen2_5_vl_7b_awq"],
    MODEL_NAME_MAP["qwen3_vl_4b"],
    MODEL_NAME_MAP["qwen3_vl_4b_merged_lora"],
    MODEL_NAME_MAP["meta_llama_11b"],
    MODEL_NAME_MAP["phi3_5_vl"],
    MODEL_NAME_MAP["pixtral_12b"],
    MODEL_NAME_MAP["llava_mistral_7b"],
]


def check_if_multimodal(engine_name: str):
    mapped_name = MODEL_NAME_MAP.get(engine_name)
    return engine_name in __MULTIMODAL_MODELS__ or mapped_name in __MULTIMODAL_MODELS__


def validate_multimodal_engine(engine):
    if not check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_MODELS__}"
        )


def get_engine(engine_name: str, provider: str = 'vllm', **kwargs) -> EngineLM:
    """
    Get the engine based on the engine name and provider.
    :param engine_name: The engine name to use for the LLM call.
    :param provider: The provider to use for the LLM call. Currently only vllm is supported.
    :param kwargs:
    :return: The engine to use for the LLM call.
    """
    if "vllm" == provider:
        from .vllm_engine import VLLMEngine

        return VLLMEngine(
            system_prompt="You are a helpful assistant for processing documents.",
            model_name=engine_name,
            is_multimodal=check_if_multimodal(engine_name),
            cache=False,
            processor_kwargs={  # All parameters will be passed dynamically to the processor
                'min_pixels': 1 * 28 * 28,
                'max_pixels': 1280 * 28 * 28,  # 800, 1000, 1280
            },
            **kwargs,
        )
    else:
        raise ValueError(f"Engine {engine_name} not supported")
