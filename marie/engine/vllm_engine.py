import os.path
import time
from typing import Dict, List, Union

import diskcache as dc
import torch
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer

from marie.engine import MODEL_NAME_MAP, REVERSE_MODEL_NAME_MAP, _check_if_multimodal
from marie.engine.base import EngineLM, cached
from marie.engine.engine_utils import (
    as_bytes,
    convert_openai_to_transformers_format,
    extract_text_info,
    is_batched_request,
    open_ai_like_formatting,
    process_vision_info,
)
from marie.engine.guided import GuidedMode
from marie.engine.vllm_config import VLLM_MODEL_MAP as MODEL_MAP
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings

# Ensures vLLM and flash-attention are installed
try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError as e:
    raise ImportError(
        "vLLM is required to run this module. Install it using:\n"
        "  pip install vllm\n"
        "or\n"
        "  pip install marie[vllm]"
    ) from e

try:
    import flash_attn
except ImportError as e:
    raise ImportError(
        "flash-attention 2 is required to run this module. Install it using:\n"
        "  pip install flash-attn"
    )


class VLLMEngine(EngineLM):
    """
    VLLM Engine for Efficient LLM Inference.
    Supports both multimodal and text-based models.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = True,
        cache: Union[dc.Cache, bool] = False,
        processor_kwargs: Dict = None,
        **kwargs,
    ):
        super().__init__(
            model_string=model_name,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache,
        )
        self.logger = MarieLogger(self.__class__.__name__).logger
        processor_kwargs = processor_kwargs or {}

        # Device settings
        devices = ['cuda'] if torch.cuda.is_available() else ['cpu']
        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=True, multi_gpu=False
        )

        self.device = resolved_devices[0]
        self.device_str = self.device.type
        self.system_prompt = system_prompt

        if model_name not in MODEL_NAME_MAP:
            raise ValueError(
                f"❌ Model '{model_name}' is not supported. Supported models {MODEL_NAME_MAP}"
            )

        model_name = MODEL_NAME_MAP[
            model_name
        ]  # Returns: "Qwen/Qwen2.5-VL-3B-Instruct"
        engine_config = MODEL_MAP[model_name]  # Returns: config_qwen2_5_vl
        self.prompt = "HELLO WORLD"
        if True:
            self.llm, self.prompt, self.stop_token_ids = engine_config(
                model_name, "image" if is_multimodal else "text"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # @cached
    # @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_single_prompt(
        self,
        content: Union[str, List[str]],
        system_prompt: str = None,
        guided_mode: GuidedMode = None,
        guided_params: Union[List[str], str, Dict] = None,
        **kwargs,
    ):
        return self.vllm_generate(content, system_prompt, **kwargs)

    # @cached
    # @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_multiple_input(
        self,
        content: Union[
            List[List[Union[str, bytes, Image.Image]]],
            List[Union[str, bytes, Image.Image]],
        ],
        system_prompt=None,
        **kwargs,
    ):
        return self.vllm_generate(content, system_prompt, **kwargs)

    def __call__(
        self,
        content: Union[
            str,
            List[str],
            List[Union[Image.Image, bytes, str]],
            List[List[Union[Image.Image, bytes, str]]],
        ],
        **kwargs,
    ):
        return self.generate(content, **kwargs)

    def _generate_prompt(self, messages, system_prompt: str):
        """
        Generates a formatted prompt based on user input and model requirements.

        :param messages: List of messages in OpenAI-like format.
        :param system_prompt: System-level prompt override.
        :return: Formatted prompt string.
        """
        if self.prompt:
            text_info = extract_text_info(messages)
            user_text = text_info[0]["text"] if text_info else ""
            formatted_prompt = self.prompt.replace(
                "SYSTEM_PROMPT_PLACEHOLDER", system_prompt or ""
            )
            formatted_prompt = formatted_prompt.replace(
                "QUESTION_PLACEHOLDER", user_text
            )
            return formatted_prompt

        if not self.is_multimodal:
            messages = convert_openai_to_transformers_format(messages)

        return self.tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )

    def batch_generate(
        self,
        batch_content: Union[List[str], List[List[Union[Image.Image, bytes, str]]]],
        system_prompt=None,
        guided_mode: GuidedMode = None,
        guided_params: Union[List[str], str, Dict] = None,
        **kwargs,
    ) -> List[str]:
        """
        Performs batch inference for multiple inputs, supporting both text-only and multimodal content.

        This function processes input data, constructs formatted prompts, and executes batched inference
        with optimized error handling and logging.

        :param batch_content: A list of text prompts or multimodal inputs (image, text pairs).
        :param system_prompt: Optional system-level instructions for the model.
        :param guided_mode: Optional guided mode for inference.
        :param guided_params: Optional guided parameters for the guided mode.
        :param kwargs: Additional inference parameters.
        :return: A list of generated outputs corresponding to each input in batch_content.
        """
        system_prompt = system_prompt or self.system_prompt
        # params to add
        # json: Optional[Union[str, Dict]] = None
        # regex: Optional[str] = None
        # choice: Optional[List[str]] = None
        # grammar: Optional[str] = None
        # json_object: Optional[bool] = None
        # """These are other options that can be set"""
        # backend: Optional[str] = None
        # whitespace_pattern: Optional[str] = None

        # Format content appropriately based on modality
        if self.is_multimodal:
            batch_content = [
                open_ai_like_formatting(content) for content in batch_content
            ]

        # Construct message structures for inference
        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
            for content in batch_content
        ]

        prompts = [
            self._generate_prompt(messages, system_prompt) for messages in messages_list
        ]

        # https://docs.vllm.ai/en/latest/features/structured_outputs.html
        # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-api
        # https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/structured_outputs.py

        guided_decoding = self._get_guided_decoding_params(guided_mode, guided_params)
        print('guided_decoding', guided_decoding)
        print('guided_decoding.backend', guided_decoding.backend)

        # guided_decoding.backend = 'xgrammar' # 'outlines, 'lm-format-enforcer', 'xgrammar'

        sampling_params = SamplingParams(
            guided_decoding=guided_decoding,
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            max_tokens=kwargs.get("max_tokens", 512),
            stop_token_ids=None,  # No specific stop tokens enforced
        )

        batch_inputs = [
            {"prompt": prompt, "batch_id": idx} for idx, prompt in enumerate(prompts)
        ]

        if self.is_multimodal:
            for idx, messages in enumerate(messages_list):
                image_data, _ = process_vision_info(messages)
                if image_data:
                    self.logger.debug(
                        f"📷 Processed images for batch index {idx}: {image_data}"
                    )
                    batch_inputs[idx]["multi_modal_data"] = {"image": image_data}

        self.logger.info(
            f"🚀 Initiating batch inference with {len(batch_content)} requests."
        )

        start_time = time.time()
        try:
            batch_outputs = self.llm.generate(
                batch_inputs, sampling_params=sampling_params
            )
        except Exception as e:
            self.logger.error(f"❌ Batch inference failed: {e}")
            raise e
            return ["ERROR: Inference failed"] * len(batch_content)

        generated_texts = {
            output.request_id: output.outputs[0].text if output.outputs else ""
            for output in batch_outputs
        }

        ordered_outputs = [
            generated_texts.get(str(idx), "") for idx in range(len(batch_content))
        ]

        elapsed_time = time.time() - start_time
        total_tokens = sum(
            len(self.tokenizer.tokenize(text)) for text in ordered_outputs
        )
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

        stats = {
            "num_requests": len(batch_content),
            "total_tokens": total_tokens,
            "time_taken": round(elapsed_time, 2),
            "tokens_per_second": round(tokens_per_second, 2),
        }

        self.logger.info(f"✅ Batch inference completed in {elapsed_time:.2f} sec")
        self.logger.info(
            f"📊 Batch Stats: Requests={len(batch_content)}, Tokens={total_tokens}, TPS={tokens_per_second:.2f}"
        )

        return ordered_outputs

    def vllm_generate(
        self,
        content,
        system_prompt=None,
        guided_mode: GuidedMode = None,
        guided_params: Union[List[str], str, Dict] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate text using the VLLM model."""
        batched = is_batched_request(content)
        if not batched:
            content = [content]
        results = self.batch_generate(
            content, system_prompt, guided_mode, guided_params, **kwargs
        )
        if not batched:
            return results[0]
        return results

    def _get_guided_decoding_params(
        self,
        guided_mode: GuidedMode = None,
        guided_params: Union[List[str], str, Dict] = None,
    ) -> GuidedDecodingParams:
        """Constructs GuidedDecodingParams based on guided_mode."""
        # ref : vllm/model_executor/guided_decoding/__init__.py

        if not guided_mode or not guided_params:
            return None

        if guided_mode == GuidedMode.CHOICE:
            return GuidedDecodingParams(choice=guided_params)

        elif guided_mode == GuidedMode.REGEX:
            return GuidedDecodingParams(regex=guided_params)

        elif guided_mode == GuidedMode.JSON:
            return GuidedDecodingParams(json=guided_params)

        elif guided_mode == GuidedMode.GRAMMAR:
            return GuidedDecodingParams(grammar=guided_params)

        else:
            raise ValueError(f"Unsupported guided_mode: {guided_mode}")


if __name__ == "__main__":
    # install vllm from source or use the latest version from PyPI
    # pip install --upgrade vllm
    # pip install --upgrade mistral_common

    model_name = REVERSE_MODEL_NAME_MAP.get("Qwen/Qwen2.5-VL-3B-Instruct")
    # model_name = REVERSE_MODEL_NAME_MAP.get("Qwen/Qwen2.5-VL-7B-Instruct")
    # model_name = REVERSE_MODEL_NAME_MAP.get("Qwen/Qwen2.5-VL-3B-Instruct")
    # model_name = REVERSE_MODEL_NAME_MAP.get("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # model_name = REVERSE_MODEL_NAME_MAP.get("microsoft/Phi-3.5-vision-instruct")
    # model_name = REVERSE_MODEL_NAME_MAP.get("Qwen/Qwen2.5-7B-Instruct")
    # model_name = REVERSE_MODEL_NAME_MAP.get("Qwen/Qwen2.5-3B-Instruct")
    # model_name = REVERSE_MODEL_NAME_MAP.get("mistralai/Pixtral-12B-2409")  # OOM
    # model_name = REVERSE_MODEL_NAME_MAP.get("facebook/opt-125m") # poor
    # model_name = REVERSE_MODEL_NAME_MAP.get("mistralai/Mistral-7B-Instruct-v0.2")
    # model_name = REVERSE_MODEL_NAME_MAP.get("microsoft/phi-4")
    # model_name = REVERSE_MODEL_NAME_MAP.get("llava-hf/llava-v1.6-mistral-7b-hf")

    # force_download(model_name_or_path)
    is_multimodal = _check_if_multimodal(model_name)

    engine = VLLMEngine(
        system_prompt="You are a helpful assistant for processing EOB documents.",
        model_name=model_name,
        is_multimodal=is_multimodal,
        cache=False,
        processor_kwargs={  # All parameters will be passed dynamically to the processor
            'min_pixels': 1 * 28 * 28,
            'max_pixels': 1280 * 28 * 28,  # 800, 1000, 1280
        },
    )

    image_path = os.path.expanduser("~/tmp/demo/158986821_1.png")
    image = as_bytes(image_path)
    image = Image.open(image_path).convert("RGB")

    # some sample text
    document_context = ""
    if not is_multimodal:
        document_context = """
        ### **Input Text:**
        Patient Greg Bugaj, born 12/31/1980
        Policy Number: 123456789
        Claim Number: 987654321
        """

    prompt = f"""
### Task: Extract Key-Value Pairs

Extract structured key-value pairs from the given text while maintaining accuracy and formatting.

### **Rules:**
1 **Extract only key-value pairs** — Do not include explanations, summaries, or extra text.  
2 **Preserve key names exactly as they appear** — No modifications, abbreviations, or rewording.  
3 **Ensure values are extracted accurately** — No truncation or paraphrasing.  
4 **If a key has no value, return:** `KeyName: [MISSING]`  
5 **If no key-value pairs are found, return exactly:** `"No key-value pairs found."`  

### **Strict Output Format:**
Key1: Value1;
Key2: Value2;
Key3: Value3;
...

Your response **must contain only** the extracted key-value pairs in the format above. No additional text.

{document_context}
"""
    # Text only
    content1 = prompt
    content2 = [content1, content1]  # batched request

    # multimodal
    content3 = [image, prompt]
    content4 = [
        [image, prompt],
        [image, prompt],
        [image, prompt],
        [image, prompt],
        [as_bytes(image), prompt],
    ]  # Batched request

    for i in range(1):
        print(f"Iteration {i + 1}")
        result = engine(
            content4,
            system_prompt="You are a helpful assistant for processing EOB documents.",
        )
        print(result)
