from openai.types.chat import ChatCompletion

from marie.engine.output_parser import parse_json_markdown

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    raise ImportError(
        "If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables."
    )
import asyncio
import os
import time
import uuid
from typing import Dict, List, Optional, Union

import diskcache as dc
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from marie.engine.base import EngineLM
from marie.engine.engine_utils import (
    convert_openai_to_transformers_format,
    extract_text_info,
    is_batched_request,
)
from marie.logging_core.logger import MarieLogger


class BatchProcessor:
    def __init__(self, client, model_string, logger: MarieLogger):
        self.client = client
        self.model_string = model_string
        self.logger = logger
        if not isinstance(self.client, AsyncOpenAI):
            raise ValueError(
                "Client must be an instance of OpenAI API client for async operations."
            )

    def extract_text_from_response(self, response: ChatCompletion) -> str:
        """
        Extract text from the OpenAI API response.

        Args:
            response: The response object from OpenAI API.

        Returns:
            str: The extracted text.
        """

        if (
            not response.choices
            or not hasattr(response.choices[0].message, "content")
            or not response.choices[0].message.content
        ):
            self.logger.warning("No text extracted from Response.")
            raise ValueError("No text extracted from response.")

        extracted_text = response.choices[0].message.content.strip()
        self.logger.info(f"Extracted text length: {len(extracted_text)} characters.")
        return extracted_text

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    async def completion_non_streaming(
        self,
        messages,
        task_id,
        request_id,
        guided_json: Optional[Union[Dict, BaseModel, str]],
    ):
        """
        Asynchronously performs inference for a single request.
        Each request has a unique task_id for tracking.
        """
        try:
            start = time.time()
            self.logger.info(
                f"Request {request_id} - Task {task_id} - Starting inference."
            )
            print(f'guided_json = {guided_json}')
            # The results are being returned in reasoning_content and not in content
            # [BUG] DeepSeek V3 Does Not Support Structured Output in LangChain with ChatOpenAI() #302
            # Until this is address we will parse the results from reasoning_content manually.
            # https://github.com/deepseek-ai/DeepSeek-V3/issues/302
            # https://docs.vllm.ai/en/latest/features/structured_outputs.html#experimental-automatic-parsing-openai-api
            # examples/online_serving/openai_chat_completion_structured_outputs_with_reasoning.py
            completion = await self.client.chat.completions.create(
                model=self.model_string,
                messages=messages,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False,
                temperature=0.1,
                max_tokens=4096 * 2,
                top_p=0.95,
                n=1,
                # extra_body={"guided_json": guided_json} # this will cause issues with DeepSeek and reasoning
            )

            print(
                "reasoning_content: ", completion.choices[0].message.reasoning_content
            )
            print("content: ", completion.choices[0].message.content)
            total_time = time.time() - start
            self.logger.info(
                f"Request {request_id} - Task {task_id} - Task completed in {total_time:.2f} sec."
            )

            content = self.extract_text_from_response(completion)
            return task_id, content
        except ValueError as e:
            self.logger.error(
                f"Request {request_id} - Task {task_id} - Error in completion_non_streaming(Retrying): {e}"
            )
            raise e
        except Exception as e:
            self.logger.error(
                f"Request {request_id} - Task {task_id} - Error in completion_non_streaming: {e}"
            )
            return task_id, None

    async def load_batched_request(
        self,
        messages_list,
        request_id,
        guided_json: Optional[Union[Dict, BaseModel, str]],
    ):
        """
        Processes the batch of requests, assigning a unique task_id to each request.
        """
        tasks = [
            self.completion_non_streaming(
                messages, f"{request_id}_task_{i}", request_id, guided_json
            )
            for i, messages in enumerate(messages_list)
        ]

        results = await asyncio.gather(*tasks)
        successful_completions = [
            response for task_id, response in results if response is not None
        ]
        return successful_completions, results

    def batch_generate(
        self,
        messages_list: Union[List[str], List[List[Union[Image.Image, bytes, str]]]],
        system_prompt=None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Performs batch inference for multiple inputs, supporting both text-only and multimodal content.
        """
        request_id = str(uuid.uuid4())
        system_prompt = system_prompt or "Default system prompt"

        self.logger.info(
            f"Request {request_id} - Initiating batch inference with {len(messages_list)} requests."
        )
        start_time = time.time()

        try:
            batch_outputs, task_results = asyncio.run(
                self.load_batched_request(messages_list, request_id, guided_json)
            )
        except Exception as e:
            self.logger.error(f"❌ Request {request_id} - Batch inference failed: {e}")
            return ["ERROR: Inference failed"] * len(messages_list)

        for task_id, response in task_results:
            if response:
                self.logger.info(
                    f"Request {request_id} - Task {task_id} - Response received."
                )
            else:
                self.logger.error(
                    f"Request {request_id} - Task {task_id} - Response failed."
                )
                self.logger.error(response)

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"✅ Request {request_id} - Batch inference completed in {elapsed_time:.2f} sec"
        )

        return batch_outputs


class OpenAIEngine(EngineLM):
    """
    OpenAIEngine is a wrapper around the OpenAI API.
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
        base_url: str = None,
        **kwargs,
    ):
        self.validate()
        super().__init__(
            model_string=model_name,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache,
        )
        self.logger = MarieLogger(self.__class__.__name__).logger
        processor_kwargs = processor_kwargs or {}
        api_key = os.getenv("OPENAI_API_KEY")
        if not base_url:
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        models = self.client.models.list()
        print('self.model_string : ', models)
        self.model_string = model_name

        self.batch_processor = BatchProcessor(
            self.client, self.model_string, logger=self.logger
        )

    def validate(self) -> None:
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                "Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models."
            )

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
        return self.openai_generate(
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
        return self.openai_generate(
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
            raise NotImplemented('Implement multimodal inference on first use.')

        reasoning_model = True

        # https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724
        # https://github.com/trustsight-io/deepseek-go/issues/2
        def transform_prompt_for_reasoning(reasoning_model: bool, prompt: str):
            if reasoning_model:
                return f"""{prompt}

                ## Response Format

                Reply with JSON object ONLY."""
            else:
                return prompt

        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": transform_prompt_for_reasoning(reasoning_model, content),
                },
            ]
            for content in batch_content
        ]

        return_stats = kwargs.get("return_stats", False)

        if self.is_multimodal:
            raise NotImplemented('Implement multimodal inference on first use.')

        self.logger.info(
            f"🚀 Initiating batch inference with {len(batch_content)} requests."
        )
        start_time = time.time()
        messages = [
            {"role": "system", "content": "You are a creative assistants"},
            {
                "role": "user",
                "content": f"Tell me an animal fact, for animal from Poland. {uuid.uuid4()}",
            },
        ]

        print("Batched request have completed")

        try:
            ordered_outputs = self.batch_processor.batch_generate(
                messages_list, guided_json=guided_json
            )
        except Exception as e:
            self.logger.error(f"❌ Batch inference failed: {e}")
            return ["ERROR: Inference failed"] * len(batch_content)

        elapsed_time = time.time() - start_time
        self.logger.info(f"✅ Batch inference completed in {elapsed_time:.2f} sec")

        return ordered_outputs

    def openai_generate(
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
        """Generate text using the  model."""
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


if __name__ == "__main__":
    # https://github.com/novex-ai/parallel-parrot/blob/main/parallel_parrot/openai_api.py
    # https://github.com/openai/openai-cookbook/blob/main/examples/Structured_Outputs_Intro.ipynb
    # https://github.com/openai/openai-cookbook/blob/main/examples/Unit_test_writing_using_a_multi-step_prompt.ipynb
    # https://github.com/openai/openai-cookbook/blob/main/examples/Named_Entity_Recognition_to_enrich_text.ipynb
    # https://github.com/openai/openai-cookbook/blob/main/examples/Parse_PDF_docs_for_RAG.ipynb
    # https://docs.boundaryml.com/examples/prompt-engineering/chain-of-thought
    os.environ['OPENAI_API_KEY'] = 'EMPTY'
    engine = OpenAIEngine(
        model_name='deepseek-r1',
        system_prompt="You are a helpfull assistant",
        is_multimodal=False,
        cache=False,
        processor_kwargs=None,
        base_url="http://184.105.87.211:8000/v1",
        # base_url="http://localhost:8090/v1",
    )

    class Animal(BaseModel):
        name: str
        fact: str
        city: str

    print(engine)
    prompts = []
    for i in range(1, 100):
        prompt = f"Tell me an animal fact, for animal from Poland  for random city # {i} : JSON Schema : {Animal.model_json_schema()}"
        prompts.append(prompt)

    results = engine.generate(prompts)
    print(results)

    for result in results:
        json_dict = parse_json_markdown(result)
        animal = Animal.model_validate(json_dict)
        print(animal)
