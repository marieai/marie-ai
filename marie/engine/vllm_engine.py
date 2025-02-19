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
    open_ai_like_formatting,
    process_vision_info,
)
from marie.engine.vllm_config import VLLM_MODEL_MAP as MODEL_MAP
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings

# Ensures vLLM and flash-attention are installed
try:
    from vllm import LLM, SamplingParams
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

        print(engine_config)
        self.llm, self.prompt, self.stop_token_ids = engine_config(
            model_name, "image" if is_multimodal else "text"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_single_prompt(
        self, content: str, system_prompt: str = None, **kwargs
    ):
        return self.vllm_generate(content, system_prompt, **kwargs)

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes, Image.Image]],
        system_prompt=None,
        **kwargs,
    ):
        formatted_content = open_ai_like_formatting(content)
        return self.vllm_generate(formatted_content, system_prompt, **kwargs)

    def __call__(self, content, **kwargs):
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
        batch_content: List[Union[str, List[Union[str, bytes, Image.Image]]]],
        system_prompt=None,
        **kwargs,
    ) -> List[str]:
        """Batch inference function for multiple inputs with improved structure and error handling."""
        system_prompt = system_prompt or self.system_prompt

        # Ensure batch_content is always a list of lists for consistency
        if isinstance(batch_content[0], (str, bytes)):
            batch_content = [[content] for content in batch_content]

        # Convert each input into OpenAI-like format
        formatted_contents = [
            open_ai_like_formatting(content) for content in batch_content
        ]

        # Prepare multiple messages for batch inference
        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
            for content in formatted_contents
        ]

        # Generate prompts for all batch inputs
        prompts = [
            self._generate_prompt(messages, system_prompt) for messages in messages_list
        ]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            max_tokens=kwargs.get("max_tokens", 512),
            stop_token_ids=self.stop_token_ids,
        )

        # Ensure correct batch structure for VLLM
        batch_inputs = [
            {"prompt": prompt, "batch_id": idx} for idx, prompt in enumerate(prompts)
        ]

        # Process multimodal inputs if necessary
        if self.is_multimodal:
            for idx, messages in enumerate(messages_list):
                image_data = process_vision_info(messages)
                if image_data:
                    batch_inputs[idx]["multi_modal_data"] = {"image": image_data[0]}

        self.logger.debug(f"🚀 Batch Input Data: {batch_inputs}")

        # Start batch inference timing
        start_time = time.time()
        try:
            batch_outputs = self.llm.generate(
                batch_inputs, sampling_params=sampling_params
            )
        except Exception as e:
            self.logger.error(f"❌ Error during batch inference: {e}")
            return ["ERROR: Inference failed"] * len(batch_content)

        # Ensure correct parsing of batch output
        generated_texts = {
            output.request_id: output.outputs[0].text if output.outputs else ""
            for output in batch_outputs
        }

        # Sort outputs by batch_id to maintain original order
        ordered_outputs = [
            generated_texts.get(idx, "") for idx in range(len(batch_content))
        ]

        elapsed_time = time.time() - start_time
        self.logger.info(f"Batch Generation Time: {elapsed_time:.2f} sec")

        # Token statistics
        token_counts = [len(self.tokenizer.tokenize(text)) for text in ordered_outputs]
        total_tokens = sum(token_counts)
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

        stats = {
            "num_requests": len(batch_content),
            "total_tokens": total_tokens,
            "time_taken": round(elapsed_time, 2),
            "tokens_per_second": round(tokens_per_second, 2),
        }
        self.logger.info(f"Batch Processing Stats: {stats}")

        return ordered_outputs

    @torch.inference_mode()
    def vllm_generate(self, content, system_prompt=None, **kwargs) -> str:
        """Generate text using the VLLM model."""
        if not self.is_multimodal and isinstance(content, list):
            content = [
                node
                for node in content
                if isinstance(node, dict) and node.get("type") == "text"
            ]
        system_prompt = system_prompt or self.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        prompt_template = self._generate_prompt(messages, system_prompt)
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            max_tokens=2048,  # kwargs.get("max_tokens", 2048),
            stop_token_ids=self.stop_token_ids,
        )

        inputs = {"prompt": prompt_template}
        if self.is_multimodal:
            inputs["multi_modal_data"] = {"image": process_vision_info(messages)[0]}

        self.logger.debug(f"🚀 Input Data: {inputs}")

        # Inference timing
        start_time = time.time()
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        print('outputs len:', len(outputs))
        for output in outputs:
            print('output:', output.outputs[0].text)

        generated_text = (
            outputs[0].outputs[0].text if outputs else ""
        )  # Extract generated text

        elapsed_time = time.time() - start_time
        self.logger.info(f"Generation Time: {elapsed_time:.2f} sec")

        # Tokenize the generated text to determine token count
        if generated_text:
            num_tokens_generated = len(self.tokenizer.tokenize(generated_text))
        else:
            num_tokens_generated = 0
        tokens_per_second = (
            num_tokens_generated / elapsed_time if elapsed_time > 0 else 0
        )
        stats = {
            "num_tokens": num_tokens_generated,
            "time_taken": round(elapsed_time, 2),
            "tokens_per_second": round(tokens_per_second, 2),
        }
        self.logger.info(f"Processing stats : {stats}")

        return generated_text


def is_batched_request(
    content: Union[
        str,
        List[str],
        List[Union[Image.Image, bytes, str]],
        List[List[Union[Image.Image, bytes, str]]],
    ]
) -> bool:
    """
    Determines whether the input content is a batched request.

    :param content: The input content, which can be:
                    - A single string (text prompt)
                    - A list of strings (batched text requests)
                    - A multimodal input ([image, text])
                    - A list of multimodal inputs ([[image, text], [image, text]])
    :return: True if the request is batched, False otherwise.
    """

    # **Case 1: Single Text Request (not batched)**
    if isinstance(content, str):
        return False  # ❌ Single text prompt

    # **Case 2: Batched Text Requests (list of strings)**
    if isinstance(content, list) and all(isinstance(item, str) for item in content):
        return True  # ✅ ["Prompt 1", "Prompt 2"] (Batched text)

    # **Case 3: Single Multimodal Request ([image, text])**
    if isinstance(content, list):
        contains_image = any(isinstance(item, (Image.Image, bytes)) for item in content)
        contains_text = any(isinstance(item, str) for item in content)

        if (
            contains_image
            and contains_text
            and not any(isinstance(sublist, list) for sublist in content)
        ):
            return False  # ❌ Single multimodal request

    # **Case 4: Batched Multimodal Requests ([[image, text], [image, text]])**
    if isinstance(content, list) and all(
        isinstance(sublist, list)
        and any(isinstance(el, (Image.Image, bytes)) for el in sublist)
        and any(isinstance(el, str) for el in sublist)
        for sublist in content
    ):
        return True  # ✅ [[image, prompt], [image, prompt], [as_bytes(image), prompt]] (Batched)

    return False  # Default: Single request


def test_batching():
    image = Image.new('RGB', (100, 100))  # Mock image for testing
    image_bytes = as_bytes(image)  # Convert image to bytes

    # **Text only**
    content1: str = "Sample prompt"  # ❌ Single text
    content2: List[str] = ["Sample prompt 1", "Sample prompt 2"]  # ✅ Batched text

    # **Multimodal**
    content3: List[Union[Image.Image, str]] = [
        image,
        "Sample prompt",
    ]  # ❌ Single multimodal
    content4: List[List[Union[Image.Image, str]]] = [
        [image, "Sample prompt"],
        [image, "Sample prompt"],
        [image_bytes, "Sample prompt"],
    ]  # ✅ Batched multimodal

    # **Invalid Cases (Edge Testing)**
    content5: List[List[str]] = [
        ["Sample prompt"],
        ["Another prompt"],
    ]  # ❌ Incorrect format
    content6: List[Union[List[Union[Image.Image, str]], str]] = [
        [image, "Sample prompt"],
        "Another prompt",
    ]  # ❌ Mixed formats
    content7: List[List[Union[Image.Image, List[str]]]] = [
        [image, ["Nested prompt"]]
    ]  # ❌ Incorrect nesting

    # **Testing**
    print(is_batched_request(content1))  # False
    print(is_batched_request(content2))  # True
    print(is_batched_request(content3))  # False
    print(is_batched_request(content4))  # True
    print(is_batched_request(content5))  # False
    print(is_batched_request(content6))  # False
    print(is_batched_request(content7))  # False


if __name__ == "__main__":

    os.exit(1)
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
    content1 = "Sample promp"
    content2 = ["Sample promp 1", "Sample promp 2"]  # batched request

    # multimodal
    content3 = [image, prompt]
    content4 = [
        [image, prompt],
        [image, prompt],
        [as_bytes(image), prompt],
    ]  # Batched request

    for i in range(1):
        print(f"Iteration {i + 1}")
        result = engine(
            [image, prompt],
            system_prompt="You are a helpful assistant for processing EOB documents.",
        )
        print(result)
