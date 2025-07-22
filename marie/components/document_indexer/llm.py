import os
from typing import Any, List, Optional, Tuple, Union

import torch
from docarray import DocList

from marie.api.docs import DOC_KEY_INDEXER, MarieDoc
from marie.components.document_indexer.base import BaseDocumentIndexer
from marie.components.document_indexer.llm_task import (
    PROMPT_STRATEGIES,
    LLMConfig,
    initialize_tasks,
    md_wrap,
    modify_outputs,
    parse_task_output,
)
from marie.constants import __model_path__
from marie.engine import check_if_multimodal, get_engine
from marie.engine.multimodal_ops import MultimodalLLMCall
from marie.logging_core.logger import MarieLogger
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import convert_frames, frames_from_docs
from marie.utils.json import load_json_file


class MMLLMDocumentIndexer(BaseDocumentIndexer):
    """
    Multi-Modal LLM based model for image-text -> text output.
    """

    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        task: str = "llm-document-indexer",
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        """
        Initializes the llm interface for Image-Prompt.

        :param model_path:  The name or path of the model to be used.
        :param use_gpu: Whether to use GPU for processing. Defaults to True.
        :param task: The task to be performed. Defaults to "llm-document-indexer".
        :param batch_size: The size of the batch to be processed. Defaults to 16.
        :param use_auth_token: The authentication token to be used. Defaults to None.
        :param devices: The devices to be used for processing. Defaults to None.
        :param show_error: Whether to show errors. Defaults to True.
        :param ocr_engine: The OCR engine to be used. Defaults to None, in which case the default OCR engine is used.
        :param kwargs: Additional keyword arguments.
        :returns: None
        """

        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.task = task
        self.logger.info(f"Document indexer Multi-Modal-LLM: {model_path}")

        registry_kwargs = {
            "__model_path__": __model_path__,
            "use_auth_token": use_auth_token,
        }
        self.model_path = model_path
        model_path = ModelRegistry.get(
            model_path,
            version=None,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )
        assert os.path.exists(model_path)
        self.logger.info(f"Resolved model : {model_path}")

        # TODO: config could be loaded from a file or passed as a parameter
        config_path = os.path.join(model_path, "marie.json")
        # config_path = os.path.join(model_path, "marie.simple.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "Expected config 'marie.json' not found in model directory"
            )
        self.logger.info(f"Specifications loading from : {config_path}")
        config: LLMConfig = LLMConfig(**load_json_file(config_path))
        self.engine_provider = config.engine_provider
        self.model_name = config.name_or_path
        if not check_if_multimodal(self.model_name):
            raise ValueError(f"The engine requested is not multimodal.")

        engine_instance = get_engine(self.model_name, self.engine_provider)
        self.model_inference = MultimodalLLMCall(engine_instance, system_prompt=None)

        self.tasks = config.tasks
        initialize_tasks(self.tasks, model_path)
        self.logger.info(f"Task Initialization complete")
        self.task_map = {task.name: task for task in self.tasks}

    def predict(
        self,
        documents: DocList[MarieDoc],
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
        batch_size: Optional[int] = None,
        lines: List[List[int]] = None,
        **kwargs,
    ) -> DocList[MarieDoc]:

        if batch_size is None:
            batch_size = self.batch_size

        if len(documents) == 0:
            return documents

        task_outputs = {}
        # Execute tasks sequentially based on the order defined in config
        task_request = kwargs.get("tasks")
        tasks = (
            self.resolve_task_graph(task_request)
            if task_request is not None
            else self.tasks
        )
        for task in tasks:
            task_name = task.name

            chained_inputs = []
            if task.chained_tasks:
                for chained_task_name in task.chained_tasks:
                    chained_task = self.task_map[chained_task_name]
                    chained_input = task_outputs.get(chained_task_name, "")
                    chained_input = md_wrap(chained_input, chained_task.output_type)
                    chained_inputs.append(chained_input)

            prompt_strategy_name = task.prompt_mod_strategy
            prompt_strategy_fn = PROMPT_STRATEGIES[prompt_strategy_name]

            # Build prompts using the chosen strategy and chained inputs if available
            prompts = [
                prompt_strategy_fn(
                    task.prompt,
                    document=doc,
                    words=words[i],
                    lines=lines[i],
                    append_text="\n".join(chained_inputs),
                    **kwargs,
                )
                for i, doc in enumerate(documents)
            ]

            frames = frames_from_docs(documents)
            frames = convert_frames(frames, img_format="pil")
            batch = list(list(mm_prompt) for mm_prompt in zip(frames, prompts))

            # TODO: Add support for system prompts
            self.logger.info(
                f"Running '{task_name}' with strategy '{prompt_strategy_name}'"
            )
            model_output = self.model_inference(
                batch, guided_json=task.guided_json_schema
            )
            parsed_output = parse_task_output(model_output, task.output_type)
            task_outputs[task_name] = modify_outputs(parsed_output, task.output_mod)

        for document in documents:
            document.tags[DOC_KEY_INDEXER] = dict()

        for task_name, model_output in task_outputs.items():
            if not self.task_map[task_name].store_results:
                continue
            for document, prediction in zip(documents, model_output):
                document.tags[DOC_KEY_INDEXER][task_name] = prediction

        return documents

    def preprocess(self, data, *args, **kwargs):
        """Preprocess the input data for inference. This method is called by the predict method."""
        return data

    @torch.no_grad()
    def inference(
        self,
        image: Any,
        words: List[Any],
        boxes: List[Any],
        text: str,
        labels: List[str],
        threshold: float,
    ) -> Tuple[List, List, List]:
        """Run Inference

        :param image: The image to be processed. This can be a PIL.Image or numpy.
        :param words: The words to be processed.
        :param boxes: The boxes to be processed, in the format (x, y, w, h). Boxes should be normalized to 1000
        :param text: The text to be processed(prompt for LLM).
        :param labels: The labels to be used for inference.
        :param threshold: The threshold to be used for filtering the results.
        :returns: The predictions, boxes (normalized), and scores.
        """
        raise NotImplementedError()

    def postprocess(
        self,
        data,
    ):
        """Postprocess the results of the inference."""

        return data

    def resolve_task_graph(self, task_requests):

        requested_tasks = []
        for task_name in task_requests:
            if task_name not in self.task_map:
                raise ValueError(
                    f"Undefined task requested {task_name}. Available tasks: {list(self.task_map.keys())}"
                )
            requested_tasks.append(self.task_map[task_name])

        chained_tasks = []
        for task in requested_tasks:
            if task.chained_tasks:
                chained_tasks.extend(
                    [self.task_map[task_name] for task_name in task.chained_tasks]
                )

        return chained_tasks + requested_tasks
