import time
from typing import Dict, List, Optional, Union

import diskcache as dc
import torch
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer

from marie.engine import MODEL_NAME_MAP
from marie.engine.base import EngineLM
from marie.engine.engine_utils import (
    convert_openai_to_transformers_format,
    extract_text_info,
    is_batched_request,
    open_ai_like_formatting,
    process_vision_info,
)
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
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        **kwargs,
    ):
        return self.vllm_generate(
            content,
            system_prompt,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            **kwargs,
        )

    # @cached
    # @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_multiple_input(
        self,
        content: Union[
            List[List[Union[str, bytes, Image.Image]]],
            List[Union[str, bytes, Image.Image]],
        ],
        system_prompt=None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        **kwargs,
    ):
        return self.vllm_generate(
            content,
            system_prompt,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            **kwargs,
        )

    def __call__(
        self,
        content: Union[
            str,
            List[str],
            List[Union[Image.Image, bytes, str]],
            List[List[Union[Image.Image, bytes, str]]],
        ],
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        **kwargs,
    ):
        return self.generate(
            content,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            **kwargs,
        )

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
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Performs batch inference for multiple inputs, supporting both text-only and multimodal content.

        This function processes input data, constructs formatted prompts, and executes batched inference
        with optimized error handling and logging.

        :param batch_content: A list of text prompts or multimodal inputs (image, text pairs).
        :param system_prompt: Optional system-level instructions for the model.
        :param kwargs: Additional inference parameters.
        :return: A list of generated outputs corresponding to each input in batch_content.
        """
        system_prompt = system_prompt or self.system_prompt
        if self.is_multimodal:
            batch_content = [
                open_ai_like_formatting(content) for content in batch_content
            ]

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

        guided_decoding = self._get_guided_decoding_params(
            guided_json,
            guided_regex,
            guided_choice,
            guided_grammar,
            guided_json_object,
            guided_backend,
            guided_whitespace_pattern,
        )
        # guided_decoding.backend = 'xgrammar' # 'outlines, 'lm-format-enforcer', 'xgrammar'
        # https://github.com/vllm-project/vllm/issues/7592
        # https://github.com/vllm-project/vllm/issues/542
        # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationMixin.generate

        # the output is not equals compare to huggingface . #1128
        # https://github.com/vllm-project/vllm/pull/1424

        # Modified vllm/model_executor/models/qwen2_5_vl.py

        sampling_params = SamplingParams(
            guided_decoding=guided_decoding,
            temperature=kwargs.get("temperature", 0.1),  # 0 = GREEDY
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_tokens", 4096),
            stop_token_ids=None,  # No specific stop tokens enforced
            # repetition_penalty=1.2,
        )
        return_stats = kwargs.get("return_stats", False)

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
            return ["ERROR: Inference failed"] * len(batch_content)

        ordered_outputs = [
            output.outputs[0].text if output.outputs else "" for output in batch_outputs
        ]

        print(ordered_outputs)

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
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate text using the VLLM model."""
        batched = is_batched_request(content)
        if not batched:
            content = [content]
        results = self.batch_generate(
            content,
            system_prompt,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            **kwargs,
        )
        if not batched:
            return results[0]
        return results

    def _get_guided_decoding_params(
        self,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
    ) -> GuidedDecodingParams:
        """Constructs GuidedDecodingParams based on guided_mode."""
        # ref : vllm/model_executor/guided_decoding/__init__.py
        print(f'guided_json  {guided_json}')
        print(f'guided_regex  {guided_regex}')
        print(f'guided_choice {guided_choice}')
        print(f'guided_grammar {guided_grammar}')
        print(f'guided_json_object {guided_json_object}')
        print(f'guided_backend {guided_backend}')
        print(f'guided_whitespace_pattern {guided_whitespace_pattern}')

        return GuidedDecodingParams.from_optional(
            json=guided_json,
            regex=guided_regex,
            choice=guided_choice,
            grammar=guided_grammar,
            json_object=guided_json_object,
            backend=guided_backend,
            whitespace_pattern=guided_whitespace_pattern,
        )
