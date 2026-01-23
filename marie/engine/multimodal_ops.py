import asyncio
import functools
from typing import Callable, Dict, List, Optional, Union

from PIL import Image
from pydantic import BaseModel

from marie.engine import EngineLM, get_engine
from marie.engine.config import validate_engine_or_get_default
from marie.engine.function import Function, FunctionReturnType
from marie.logging_core.predefined import default_logger as logger


class MultimodalLLMCall(Function):
    def __init__(self, engine: Union[str, EngineLM], system_prompt: str = None):
        """The MultiModalLM call function. This function will call the LLM with the input (image) and return the response.

        :param engine: engine to use for the LLM call
        :type engine: EngineLM
        :param system_prompt: system prompt to use for the LLM call, default depends on the engine.
        :type system_prompt: Variable, optional
        """
        super().__init__()
        self.engine = validate_engine_or_get_default(engine)
        self.system_prompt = system_prompt

    def forward(
        self,
        inputs: Union[
            List[List[Union[str, bytes, Image.Image]]],
            List[Union[str, bytes, Image.Image]],
        ],
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        **kwargs,
    ) -> FunctionReturnType:
        """
        The LLM call. This function will call the LLM with the input and return the response.

        :param inputs: list of input variables to the multimodal LLM call. One is an image and the second one is text
        :param guided_json: guided parameters to use for the LLM call
        :param guided_regex: guided regex pattern to use for the LLM call
        :param guided_choice: guided choice to use for the LLM call
        :param guided_grammar: guided grammar to use for the LLM call
        :param guided_json_object: guided JSON object to use for the LLM call
        :param guided_backend: guided backend to use for the LLM call
        :param guided_whitespace_pattern: guided whitespace pattern to use for the LLM call
        :param on_result: Optional callback invoked when each task completes.
                         Signature: (task_id: str, response: Optional[str]) -> None
                         This enables incremental result processing.
        :return: response sampled from the LLM

        :example:
        >>> engine = get_engine("qwen_vl_3b")
        >>> target_image = "A byte representation of the image"
        >>> prompt = "What is the capital of France?"
        >>> response = MultimodalLLMCall(engine)([target_image, prompt])
        """

        def validate_input(input_items: List[Union[str, bytes, Image.Image]]):
            for variable in input_items:
                if not isinstance(variable, (str, bytes, Image.Image)):
                    raise ValueError(
                        f"MultimodalLLMCall only accepts str, bytes or PIL Image, got {type(variable)}"
                    )

        if isinstance(inputs[0], list):
            for sublist in inputs:
                validate_input(sublist)
        else:
            validate_input(inputs)

        system_prompt_value = self.system_prompt

        response_text = self.engine(
            inputs,
            system_prompt=system_prompt_value,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            on_result=on_result,
            **kwargs,
        )

        logger.info(
            f"MultimodalLLMCall function forward",
            extra={
                "text": f"System:{system_prompt_value}\nQuery: {inputs}\nResponse: {response_text}"
            },
        )

        return response_text

    async def aforward(
        self,
        inputs: Union[
            List[List[Union[str, bytes, Image.Image]]],
            List[Union[str, bytes, Image.Image]],
        ],
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        **kwargs,
    ) -> FunctionReturnType:
        """
        Async version of forward. Calls the multimodal LLM with the input and returns the response.

        :param inputs: list of input variables to the multimodal LLM call.
        :param guided_json: guided parameters to use for the LLM call
        :param guided_regex: guided regex pattern to use for the LLM call
        :param guided_choice: guided choice to use for the LLM call
        :param guided_grammar: guided grammar to use for the LLM call
        :param guided_json_object: guided JSON object to use for the LLM call
        :param guided_backend: guided backend to use for the LLM call
        :param guided_whitespace_pattern: guided whitespace pattern to use for the LLM call
        :param on_result: Optional callback invoked when each task completes.
                         Signature: (task_id: str, response: Optional[str]) -> None
                         This enables incremental result processing.
        :return: response sampled from the LLM
        """

        def validate_input(input_items: List[Union[str, bytes, Image.Image]]):
            for variable in input_items:
                if not isinstance(variable, (str, bytes, Image.Image)):
                    raise ValueError(
                        f"MultimodalLLMCall only accepts str, bytes or PIL Image, got {type(variable)}"
                    )

        if isinstance(inputs[0], list):
            for sublist in inputs:
                validate_input(sublist)
        else:
            validate_input(inputs)

        system_prompt_value = self.system_prompt

        if hasattr(self.engine, "__call__") and asyncio.iscoroutinefunction(
            self.engine
        ):
            response_text = await self.engine(
                inputs,
                system_prompt=system_prompt_value,
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
        else:
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None,
                functools.partial(
                    self.forward,
                    inputs,
                    guided_json=guided_json,
                    guided_regex=guided_regex,
                    guided_choice=guided_choice,
                    guided_grammar=guided_grammar,
                    guided_json_object=guided_json_object,
                    guided_backend=guided_backend,
                    guided_whitespace_pattern=guided_whitespace_pattern,
                    on_result=on_result,
                    **kwargs,
                ),
            )

        logger.info(
            "MultimodalLLMCall function aforward",
            extra={
                "text": f"System:{system_prompt_value}\nQuery: {inputs}\nResponse: {response_text}"
            },
        )

        return response_text
