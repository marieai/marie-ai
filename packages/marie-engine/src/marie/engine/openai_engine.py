import logging
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import diskcache as dc
from marie.engine.base import EngineLM
from marie.engine.batch_processor import BatchProcessor
from marie.engine.completion_contract import (
    CompletionCallParams,
    PreparedCalls,
    RequestContext,
    build_completion_call,
)
from marie.engine.completion_observer import CompletionAttemptObserver
from marie.engine.engine_utils import (
    check_image_preparation_budget,
    convert_openai_to_transformers_format,
    extract_text_info,
    is_batched_request,
    open_ai_like_formatting,
)
from marie.engine.llm_queue.config import LlmQueueProducerConfig
from marie.engine.openai_compat import build_async_openai_client
from PIL import Image
from pydantic import BaseModel


class OpenAIEngine(EngineLM):
    """
    OpenAIEngine is a wrapper around the OpenAI API.
    Supports both multimodal and text-based models.

    api_key is supplied by the caller. Direct execution requires a nonempty key;
    queued engines retain an optional key without creating a provider client.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = True,
        cache: Union[dc.Cache, bool] = False,
        processor_kwargs: Dict = None,
        base_url: str = None,
        max_concurrency: Optional[int] = None,
        batch_timeout: Optional[float] = None,
        *,
        api_key: str | None = None,
        queue_config: LlmQueueProducerConfig | None = None,
        **kwargs,
    ):
        if any(
            kwargs.get(name) is not None
            for name in (
                "pool_id",
                "queue_pool_id",
                "llm_queue_pool_id",
                "queue_contract_version",
                "llm_queue_contract_version",
                "queue_producer_id",
                "queue_client",
            )
        ):
            raise ValueError("caller_pool_forbidden")
        queue_options = {
            "enabled": kwargs.get("queue_enabled"),
            "queue_url": kwargs.get("queue_url"),
            "valkey_url": kwargs.get("queue_valkey_url"),
            "fabric_group_id": kwargs.get("queue_fabric_group_id"),
        }
        if queue_config is None:
            queue_config = LlmQueueProducerConfig.from_env(**queue_options)
        elif any(value is not None for value in queue_options.values()):
            raise ValueError("queue_configuration_conflict")
        self.api_key = api_key
        if not queue_config.enabled:
            self.validate()
        super().__init__(
            model_string=model_name,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        processor_kwargs = processor_kwargs or {}
        self.client = None
        if not queue_config.enabled:
            self.client = build_async_openai_client(
                api_key=self.api_key, base_url=base_url
            )

        # Derive backend address for the circuit breaker key
        backend_address = base_url or "https://api.openai.com"

        self.model_string = model_name

        self.batch_processor = BatchProcessor(
            self.client,
            self.model_string,
            logger=self.logger,
            max_concurrency=max_concurrency,
            batch_timeout=batch_timeout,
            backend_address=backend_address,
            queue_config=queue_config,
        )

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("api_key is required for direct execution")

    def build_queue_dispatcher(self):
        raise RuntimeError("LLM queue dispatchers are gateway-owned")

    def close(self) -> None:
        if self.batch_processor is not None:
            self.batch_processor.close()
        self.client = None

    @property
    def uses_queued_dispatch(self) -> bool:
        """Whether submissions use Gateway dispatch."""
        return self.batch_processor.uses_queued_dispatch

    @property
    def batch_timeout(self) -> float:
        """The existing batch execution budget in seconds."""
        return self.batch_processor.batch_timeout

    @property
    def max_concurrency(self) -> int:
        """The existing per-engine request limit."""
        return self.batch_processor.max_concurrency

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        **kwargs,
    ):
        return self._generate_from_multiple_input(
            content,
            system_prompt,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            on_result=on_result,
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
        if self.batch_processor.uses_queued_dispatch:
            kwargs.setdefault(
                "queue_deadline", time.monotonic() + self.batch_processor.batch_timeout
            )
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

    def complete_many(
        self,
        calls: list[CompletionCallParams] | PreparedCalls,
        *,
        request_id: str | None = None,
        on_result: Callable[[str, str | None], None] | None = None,
        metadata: dict[str, Any] | None = None,
        queue_deadline: float | None = None,
        cancellation: threading.Event | None = None,
        attempt_observer: CompletionAttemptObserver | None = None,
    ) -> list[str | None]:
        """Submit canonical calls through the owned processor and return ordered text.

        Args:
            calls: Canonical calls, or lazy calls for queued dispatch only.
            request_id: Batch correlation ID; generated by the processor if absent.
            on_result: Incremental callback with task ID and response text.
            metadata: Metadata passed to the processor for tracing and execution.
            queue_deadline: Absolute monotonic deadline for queued execution.
            cancellation: Caller-owned cancellation event for queued execution.
            attempt_observer: Observer using the existing mode's attempt delivery.

        Returns:
            Text slots in input order, retaining existing nullable results.

        Raises:
            ValueError: Lazy calls were supplied in direct mode.
            BatchExecutionError: Tasks failed; the exception retains successful
                and failed result records.
        """
        if (
            isinstance(calls, PreparedCalls)
            and not self.batch_processor.uses_queued_dispatch
        ):
            raise ValueError("Lazy image preparation requires queued dispatch")
        return self.batch_processor.batch_generate_calls(
            calls=calls,
            request_id=request_id,
            on_result=on_result,
            metadata=metadata,
            queue_deadline=queue_deadline,
            cancellation=cancellation,
            attempt_observer=attempt_observer,
        )

    def _generate_from_multiple_input(
        self,
        content: Union[
            str,
            List[str],
            List[Union[Image.Image, bytes, str]],
            List[List[Union[Image.Image, bytes, str]]],
        ],
        system_prompt=None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        attempt_observer: Optional[CompletionAttemptObserver] = None,
        **kwargs,
    ) -> str | None | list[str | None]:
        """Prepare prompt/image inputs and submit one canonical batch."""
        batched = is_batched_request(content)
        batch_content = content if batched else [content]
        queue_deadline = kwargs.pop("queue_deadline", None)
        cancellation = kwargs.pop("cancellation", None)
        if self.batch_processor.uses_queued_dispatch:
            queue_deadline = (
                queue_deadline or time.monotonic() + self.batch_processor.batch_timeout
            )
            contexts = kwargs.get("request_contexts")
            if contexts is not None and len(contexts) != len(batch_content):
                contexts = None
            item_kwargs = {
                key: value for key, value in kwargs.items() if key != "request_contexts"
            }

            if self.is_multimodal:
                for item_content in batch_content:
                    check_image_preparation_budget(
                        [item for item in item_content if isinstance(item, Image.Image)]
                    )

            def prepare(index: int) -> CompletionCallParams:
                return self.build_completion_calls(
                    batch_content=[batch_content[index]],
                    system_prompt=system_prompt,
                    guided_json=guided_json,
                    request_contexts=(
                        [contexts[index]] if contexts is not None else None
                    ),
                    **item_kwargs,
                )[0]

            calls = PreparedCalls(
                len(batch_content),
                prepare,
                build_completion_call(
                    model=self.model_string,
                    messages=[],
                    default_completion_params=self.batch_processor.default_completion_params,
                    completion_params=kwargs.get("completion_params"),
                    guided_json=guided_json,
                    stream=False,
                ),
            )
        else:
            calls = self.build_completion_calls(
                batch_content=batch_content,
                system_prompt=system_prompt,
                guided_json=guided_json,
                **kwargs,
            )

        self.logger.info(f"Initiating batch inference with {len(calls)} requests.")
        start_time = time.time()
        try:
            bp_kwargs = {}
            if self.batch_processor.uses_queued_dispatch:
                bp_kwargs.update(
                    queue_deadline=queue_deadline, cancellation=cancellation
                )
            if "metadata" in kwargs:
                bp_kwargs["metadata"] = kwargs["metadata"]

            ordered_outputs = self.complete_many(
                calls=calls,
                on_result=on_result,
                attempt_observer=attempt_observer,
                **bp_kwargs,
            )
        except Exception:
            self.logger.error("Batch inference failed:\n%s", traceback.format_exc())
            raise  # Propagate to caller

        elapsed_time = time.time() - start_time
        self.logger.info(f"Batch inference completed in {elapsed_time:.2f} sec")

        return ordered_outputs if batched else ordered_outputs[0]

    def build_completion_calls(
        self,
        *,
        batch_content: Union[List[str], List[List[Union[Image.Image, bytes, str]]]],
        system_prompt: str | None = None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        **kwargs: Any,
    ) -> list[CompletionCallParams]:
        """Build canonical calls from prompt/image inputs and per-item context.

        Args:
            batch_content: Text prompts or multimodal input batches.
            system_prompt: Optional system instruction override.
            guided_json: Optional guided generation schema.
            **kwargs: Existing completion, image-formatting and context options.

        Returns:
            One canonical completion call per input item.
        """
        effective_system_prompt = system_prompt or self.system_prompt

        if self.is_multimodal:
            default_options = {
                "min_pixels": 512 * 28 * 28,
                "max_pixels": 2560 * 28 * 28,
            }
            mm_kwargs = kwargs.get("mm_processor_kwargs", default_options)
            batch_content = [
                open_ai_like_formatting(content, True, **mm_kwargs)
                for content in batch_content
            ]

        reasoning_model = kwargs.get("reasoning_model", False)
        completion_params = kwargs.get("completion_params")
        request_contexts: Optional[List[RequestContext | None]] = kwargs.get(
            "request_contexts"
        )

        def transform_prompt_for_reasoning(reasoning_enabled: bool, prompt):
            if not reasoning_enabled:
                return prompt
            return f"""{prompt}

                ## Response Format

                Reply with JSON object ONLY."""

        messages_list = [
            [
                {"role": "system", "content": effective_system_prompt},
                {
                    "role": "user",
                    "content": transform_prompt_for_reasoning(reasoning_model, content),
                },
            ]
            for content in batch_content
        ]

        if request_contexts is not None and len(request_contexts) != len(messages_list):
            self.logger.warning(
                "Ignoring request_contexts length mismatch: expected=%s actual=%s",
                len(messages_list),
                len(request_contexts),
            )
            request_contexts = None

        return [
            build_completion_call(
                model=self.model_string,
                messages=messages,
                default_completion_params=self.batch_processor.default_completion_params,
                completion_params=completion_params,
                guided_json=guided_json,
                max_tokens=4096 * 4,
                stop=[],
                n=1,
                stream=False,
                context=(
                    request_contexts[index] if request_contexts is not None else None
                ),
            )
            for index, messages in enumerate(messages_list)
        ]
