import logging
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import diskcache as dc
import torch
from marie.engine.base import EngineLM
from marie.engine.device import initialize_device_settings
from marie.engine.engine_utils import (
    convert_openai_to_transformers_format,
    extract_text_info,
    is_batched_request,
    open_ai_like_formatting,
    process_vision_info,
    resize_image,
)
from marie.engine.vllm_config import create_llm_instance
from marie.registry.model_config import ModelConfigOverrides
from marie.registry.model_registry import ModelRegistry
from PIL import Image
from pydantic import BaseModel

# Ensure vLLM is installed.
try:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
except ImportError as e:
    raise ImportError(
        "vLLM is required to run this module. Install it using:\n  uv add vllm"
    ) from e


class VLLMEngine(EngineLM):
    """
    VLLM Engine for Efficient LLM Inference.
    Supports both multimodal and text-based models.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        is_multimodal: bool | None = None,
        cache: dc.Cache | bool | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        runtime_kwargs: dict[str, Any] | None = None,
        loading_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        generation = dict(generation_kwargs or {})
        runtime = dict(runtime_kwargs or {})
        for key, value in kwargs.items():
            if key in SamplingParams.__struct_fields__:
                generation[key] = value
            else:
                runtime[key] = value
        override_data: dict[str, Any] = {
            "loading": loading_kwargs or {},
            "runtime": runtime,
            "processor": processor_kwargs or {},
            "generation": generation,
        }
        if system_prompt is not None:
            override_data["prompt"] = {"system": system_prompt}
        if is_multimodal is not None:
            override_data["multimodal"] = is_multimodal
        if isinstance(cache, bool):
            override_data["cache"] = cache
        config = ModelRegistry.resolve_config(
            model_name, "vllm", overrides=ModelConfigOverrides(**override_data)
        )
        if not config.enabled:
            raise ValueError(
                f"vLLM is disabled for '{config.name}' in {config.source_path}"
            )
        generation = deepcopy(config.generation)
        structured_options = generation.pop("structured_outputs", None)
        try:
            if structured_options is not None:
                generation["structured_outputs"] = StructuredOutputsParams(
                    **structured_options
                )
            SamplingParams(**generation)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid vLLM generation settings for '{config.name}' in {config.source_path}: {exc}"
            ) from exc
        self.model_config = config
        selected_system_prompt = (
            config.prompt.system if config.prompt.system is not None else ""
        )
        super().__init__(
            model_string=config.name,
            system_prompt=selected_system_prompt,
            is_multimodal=config.multimodal,
            cache=config.cache if cache is None else cache,
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Device settings
        devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]
        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=True, multi_gpu=False
        )

        self.device = resolved_devices[0]
        self.device_str = self.device.type
        self.llm = create_llm_instance(config)
        self.prompt = config.prompt.template
        self.generation_defaults = deepcopy(config.generation)
        self.stop_token_ids = self.generation_defaults.get("stop_token_ids")
        self.default_mm_processor_kwargs = deepcopy(config.image)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_tokens = self.generation_defaults.get("max_tokens")

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
            # Use mm_processor_kwargs from kwargs if provided, otherwise fall back to default
            # mm_processor_kwargs should be a dict with keys like 'min_pixels', 'max_pixels', `# 'fps', etc.
            mm_kwargs = {
                **self.default_mm_processor_kwargs,
                **kwargs.get("mm_processor_kwargs", {}),
            }
            batch_content = [
                open_ai_like_formatting(content, use_defaults=False, **mm_kwargs)
                for content in batch_content
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

        structured_outputs = self._get_structured_outputs_params(
            guided_json,
            guided_regex,
            guided_choice,
            guided_grammar,
            guided_json_object,
            guided_backend,
            guided_whitespace_pattern,
        )
        sampling = deepcopy(self.generation_defaults)
        sampling.pop("structured_outputs", None)
        sampling.update(
            {
                key: value
                for key, value in kwargs.items()
                if key in SamplingParams.__struct_fields__
                and key != "structured_outputs"
            }
        )
        sampling["structured_outputs"] = structured_outputs
        sampling_params = SamplingParams(**sampling)
        batch_inputs = [{"prompt": prompt} for prompt in prompts]

        if self.is_multimodal:
            for idx, messages in enumerate(messages_list):
                image_data, _ = process_vision_info(messages)
                if image_data:
                    self.logger.debug(
                        f"📷 Processed images for batch index {idx}: {image_data}"
                    )
                    batch_inputs[idx]["multi_modal_data"] = {
                        "image": [
                            resize_image(
                                image,
                                min_pixels=mm_kwargs.get("min_pixels"),
                                max_pixels=mm_kwargs.get("max_pixels"),
                            )
                            for image in image_data
                        ]
                    }
                    batch_inputs[idx]["mm_processor_kwargs"] = {
                        **self.model_config.processor,
                        **kwargs.get("mm_processor_kwargs", {}),
                    }

        self.logger.info(
            f"🚀 Initiating batch inference with {len(batch_content)} requests."
        )
        start_time = time.time()
        try:
            batch_outputs = self.llm.generate(
                batch_inputs, sampling_params=sampling_params
            )
        except Exception:
            self.logger.exception("❌ Batch inference failed")
            return ["ERROR: Batch Inference failed"] * len(batch_content)

        ordered_outputs = [
            output.outputs[0].text if output.outputs else "" for output in batch_outputs
        ]

        elapsed_time = time.time() - start_time
        total_tokens = sum(
            len(self.tokenizer.tokenize(text)) for text in ordered_outputs
        )
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

        self.logger.info(f"Batch inference completed in {elapsed_time:.2f} sec")
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

    def _get_structured_outputs_params(
        self,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
    ) -> Optional[StructuredOutputsParams]:
        options = deepcopy(self.generation_defaults.get("structured_outputs") or {})
        for key, value in (
            ("json", guided_json),
            ("regex", guided_regex),
            ("choice", guided_choice),
            ("grammar", guided_grammar),
            ("json_object", guided_json_object),
            ("whitespace_pattern", guided_whitespace_pattern),
        ):
            if value is not None:
                options[key] = value
        if not any(
            options.get(key) is not None
            for key in ("json", "regex", "choice", "grammar", "json_object")
        ):
            return None
        if guided_backend is not None:
            options["disable_any_whitespace"] = (
                "disable-any-whitespace" in guided_backend
            )
        return StructuredOutputsParams(**options)
