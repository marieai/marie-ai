from marie.engine import MODEL_NAME_MAP

# Ensure vLLM is installed before proceeding
try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise ImportError(
        "vLLM is required to run this module. Install it using:\n"
        "  pip install vllm\n"
        "or\n"
        "  pip install marie[vllm]"
    ) from e

# Ensure flash-attention is installed before proceeding
try:
    import flash_attn
except ImportError as e:
    raise ImportError(
        "flash-attention 2 is required to run this module. Install it using:\n"
        "  pip install flash-attn"
    )


def load_format_and_quantization(
    model_name: str,
    supports_quantization: bool = True,
    quantization_method: str = "bitsandbytes",
):
    """Determines the appropriate model loading format, quantization method, and dtype."""
    quantization = None
    load_format = "auto"
    dtype = "bfloat16"

    if supports_quantization:
        if quantization_method == "bitsandbytes":
            try:
                import bitsandbytes

                load_format = "bitsandbytes"
                quantization = "bitsandbytes"
                dtype = "bfloat16"
            except ImportError:
                print(
                    "❌ BitsAndBytes is NOT installed. Install it using:\n  pip install bitsandbytes"
                )
        elif quantization_method == "fp8":
            quantization = "fp8"
            dtype = "fp8"

    return load_format, quantization, dtype


def create_llm_instance(
    model_name: str,
    max_model_len: int = 4096,
    supports_quantization: bool = True,
    quantization_method: str = "bitsandbytes",
    mm_processor_kwargs=None,
    **kwargs
):
    """Creates an instance of the LLM with standardized settings."""
    load_format, quantization, dtype = load_format_and_quantization(
        model_name, supports_quantization, quantization_method
    )
    # Remove 'dtype' from kwargs if it exists, to avoid duplicate keyword argument errors
    _dtype = kwargs.pop("dtype", dtype)
    return LLM(
        model=model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        disable_mm_preprocessor_cache=True,
        quantization=quantization if supports_quantization else None,
        load_format=load_format,
        enforce_eager=False,
        dtype=_dtype,
        mm_processor_kwargs=mm_processor_kwargs if mm_processor_kwargs else {},
        **kwargs
    )


# ---- Model Configurations ----
def config_qwen2_5_vl(model_name: str, modality: str = "image"):
    """Configures Qwen2.5-VL model."""
    assert modality == "image"

    mm_processor_kwargs = {
        "min_pixels": 1 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
        "fps": 1,
    }

    llm = create_llm_instance(
        model_name,
        supports_quantization=False,
        dtype="bfloat16",
        mm_processor_kwargs=mm_processor_kwargs,
    )

    prompt = "<|im_start|>system\nSYSTEM_PROMPT_PLACEHOLDER<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>QUESTION_PLACEHOLDER<|im_end|>\n<|im_start|>assistant\n"

    return llm, prompt, None


def config_mistral(model_name: str, modality: str = "text"):
    """Configures Mistral-7B model."""
    assert modality == "text"

    llm = create_llm_instance(
        model_name, supports_quantization=True, quantization_method="fp8", dtype="fp8"
    )

    return llm, None, None


def config_opt125(model_name: str, modality: str = "text"):
    """Configures OPT-125M model."""
    assert modality == "text"

    llm = create_llm_instance(
        model_name,
        supports_quantization=True,
        quantization_method="fp8",
        dtype="fp8",
        max_model_len=2048,
    )

    return llm, "QUESTION_PLACEHOLDER", None


def config_phi4(model_name: str, modality: str = "text"):
    """Configures Phi-4 model."""
    assert modality == "text"

    llm = create_llm_instance(
        model_name, supports_quantization=True, quantization_method="fp8", dtype="fp8"
    )

    return llm, None, None


def config_phi3v(model_name: str, modality: str):
    """Configures Phi-3 Vision model."""
    assert modality == "image"

    mm_processor_kwargs = {"num_crops": 16}

    llm = create_llm_instance(
        model_name,
        supports_quantization=False,
        dtype="bfloat16",
        trust_remote_code=True,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    prompt = "<|user|>\n<|image_1|>\nQUESTION_PLACEHOLDER<|end|>\n<|assistant|>\n"

    return llm, prompt, None


def config_llava_next(model_name: str, modality: str):
    """Configures LLaVA-1.6 / LLaVA-NeXT models."""
    assert modality == "image"

    mm_processor_kwargs = {"fps": 1}

    llm = create_llm_instance(
        model_name,
        supports_quantization=True,
        quantization_method="fp8",
        dtype="fp8",
        mm_processor_kwargs=mm_processor_kwargs,
    )

    prompt = "[INST] <image>\nQUESTION_PLACEHOLDER [/INST]"

    return llm, prompt, None


def config_mllama(model_name: str, modality: str):
    """Configures Meta-LLaMA 3.2 model."""
    assert modality == "image"

    llm = create_llm_instance(model_name, supports_quantization=False, dtype="bfloat16")

    return llm, None, None


def config_qwen2_5(model_name: str, modality: str = "text"):
    """Configures Qwen2.5 model."""
    assert modality == "text"

    llm = create_llm_instance(
        model_name, supports_quantization=True, quantization_method="bitsandbytes"
    )

    return llm, None, None


VLLM_MODEL_MAP = {
    MODEL_NAME_MAP["qwen2_5_vl_3b"]: config_qwen2_5_vl,
    MODEL_NAME_MAP["qwen2_5_vl_7b"]: config_qwen2_5_vl,
    MODEL_NAME_MAP["qwen2_5_7b"]: config_qwen2_5,
    MODEL_NAME_MAP["qwen2_5_3b"]: config_qwen2_5,
    MODEL_NAME_MAP["meta_llama_11b"]: config_mllama,
    MODEL_NAME_MAP["phi3_5_vl"]: config_phi3v,
    MODEL_NAME_MAP["pixtral_12b"]: None,  # Not supported due to OOM
    MODEL_NAME_MAP["mistral_7b"]: config_mistral,
    MODEL_NAME_MAP["opt_125m"]: config_opt125,
    MODEL_NAME_MAP["phi4"]: config_phi4,
    MODEL_NAME_MAP["llava_mistral_7b"]: config_llava_next,
}
