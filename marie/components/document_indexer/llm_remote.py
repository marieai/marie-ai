import asyncio
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from docarray import DocList

from marie.api.docs import DOC_KEY_INDEXER, MarieDoc
from marie.common.file_io import get_cache_dir
from marie.components.document_indexer.base import BaseDocumentIndexer
from marie.components.document_indexer.llm_task import (
    PROMPT_STRATEGIES,
    LLMConfig,
    LLMTask,
    filter_pages,
    initialize_tasks,
    md_wrap,
    modify_outputs,
)
from marie.constants import __model_path__
from marie.engine import EngineLM
from marie.extract.annotators.util import process_batch, route_llm_engine
from marie.helper import run_async
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import convert_frames, frames_from_docs
from marie.utils.image_utils import hash_frames_fast
from marie.utils.json import load_json_file
from marie.utils.utils import batchify


class RemoteLLMDocumentIndexer(BaseDocumentIndexer):
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
        self.multimodal = config.multimodal
        if not self.multimodal:
            raise ValueError(f"Model {self.model_name} is not a multimodal model")

        self.engine = route_llm_engine(self.model_name, self.multimodal)

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

        if len(documents) == 0:
            return documents

        root_asset_path = kwargs.get("root_asset_path", None)
        if root_asset_path is None:
            frames = frames_from_docs(documents)
            root_asset_path = os.path.join(
                get_cache_dir(),
                "agent-cache",
                self.task,
                hash_frames_fast(frames=frames),
            )

        doc_map = {i: doc for i, doc in enumerate(documents)}
        task_outputs = defaultdict(dict)  # {"task_name": {0: "page_output", ...}, ...}
        # Execute tasks sequentially based on the order defined in config
        task_request = kwargs.get("tasks")
        tasks = (
            self.resolve_task_graph(task_request)
            if task_request is not None
            else self.tasks
        )
        for task in tasks:
            task_name = task.name
            filtered_pages_idx = sorted(doc_map.keys())

            chained_inputs = {}
            for chained_task_name in task.chained_tasks:
                chained_task = self.task_map[chained_task_name]
                prior_outputs = task_outputs.get(chained_task_name, {})
                for page_idx, page_output in prior_outputs.items():
                    if isinstance(page_output, tuple) and len(page_output) == 2:
                        page_output, _ = page_output
                    if page_idx not in chained_inputs:
                        chained_inputs[page_idx] = ""
                    chained_inputs[page_idx] += "\n" + md_wrap(
                        page_output, chained_task.output_type
                    )

            if task.chained_tasks:
                filtered_pages_idx = sorted(chained_inputs.keys())
            if task.page_filter:
                pf = task.page_filter
                filtered_pages_idx = filter_pages(
                    pf.pattern, task_outputs[pf.task], page_subset=filtered_pages_idx
                )

            task_outputs[task_name] = run_async(
                allm_task_inference(
                    documents=documents,
                    words=words,
                    lines=lines,
                    page_idxs=filtered_pages_idx,
                    engine=self.engine,
                    task=task,
                    batch_size=batch_size or self.batch_size,
                    root_output_path=root_asset_path,
                    chained_inputs=chained_inputs,
                    is_multimodal=self.multimodal,
                    **kwargs,
                )
            )

        for document in documents:
            document.tags[DOC_KEY_INDEXER] = dict()

        for task_name, model_output in task_outputs.items():
            if not self.task_map[task_name].store_results:
                continue
            for i, prediction in model_output.items():
                documents[i].tags[DOC_KEY_INDEXER][task_name] = prediction

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

    def postprocess(self, data):
        """Postprocess the results of the inference."""
        return data

    def resolve_task_graph(self, task_requests) -> List[LLMTask]:
        """
        Resolves a directed task dependency graph and returns a list of requested tasks
        including their dependent chained tasks, ensuring proper task order.

        Raises:
            ValueError: If a task name in the input is not defined in the task map.

        Args:
            task_requests (List[str]): A list of task names requested for resolution.

        Returns:
            List[LLMTask]: A list of resolved tasks including any dependent chained tasks.
        """
        requested_tasks = []
        for task_name in task_requests:
            if task_name not in self.task_map:
                raise ValueError(
                    f"Undefined task requested {task_name}. Available tasks: {list(self.task_map.keys())}"
                )
            task = self.task_map[task_name]
            if task.chained_tasks:
                chained_tasks = self.resolve_task_graph(task.chained_tasks)
                requested_tasks.extend(
                    [ct for ct in chained_tasks if ct not in requested_tasks]
                )
            if task not in requested_tasks:
                requested_tasks.append(task)

        return requested_tasks


async def allm_task_inference(
    documents: DocList[MarieDoc],
    words: List[List[str]],
    lines: List[List[int]],
    page_idxs: List[int],
    engine: EngineLM,
    task: LLMTask,
    batch_size: int,
    root_output_path: str,
    chained_inputs: Optional[Dict[int, str]] = None,
    is_multimodal: bool = True,
    **kwargs,
) -> Dict[int, str]:

    task_name = task.name
    indexed_outputs = dict()
    batches = list(batchify(page_idxs, batch_size))
    prompt_strategy_name = task.prompt_mod_strategy
    prompt_strategy_fn = PROMPT_STRATEGIES[prompt_strategy_name]
    output_path = os.path.join(root_output_path, "agent-output", task_name)
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Running '{task_name}' with strategy '{prompt_strategy_name}'")
    logger.info(
        f"Batching {len(page_idxs)} into {batches} batches of size {batch_size}"
    )

    async def _worker(batch) -> Tuple[Optional[List], List[int]]:
        docs, prompts, image_paths = [], [], []
        for i in batch:
            image_paths.append(os.path.join(output_path, f"{i + 1:05}.png"))
            docs.append(documents[i])
            # Build prompts using the chosen strategy and chained inputs if available
            prompts.append(
                prompt_strategy_fn(
                    task.prompt,
                    document=documents[i],
                    words=words[i],
                    lines=lines[i],
                    append_text=chained_inputs.get(i, None),
                    **kwargs,
                )
            )

        docs = DocList(docs)
        images = convert_frames(frames_from_docs(docs), img_format="pil")
        # TODO : resizing should be configured and done here (min/max image size)
        inp = list(list(mm_prompt) for mm_prompt in zip(images, prompts, image_paths))

        if len(inp) == 0:
            logger.warning(f"No pages to process for task {task_name}")
            return None, batch

        batch_results = await process_batch(
            inp,
            engine,
            output_path,
            is_multimodal=is_multimodal,
            expect_output=task.output_type,
        )
        return batch_results, batch

    workers = [asyncio.create_task(_worker(batch)) for batch in batches]
    error_file_path = os.path.join("/tmp/marie/llm-engine", task_name, "error.log")
    if not os.path.exists(os.path.dirname(error_file_path)):
        os.makedirs(os.path.dirname(error_file_path), exist_ok=True)

    try:
        results = await asyncio.gather(*workers, return_exceptions=True)
        for r in results:
            if isinstance(r, asyncio.CancelledError):
                logger.warning(f"One of the tasks was cancelled : {r}")
            elif isinstance(r, Exception):
                logger.error("Task failed:", exc_info=r)
            else:
                model_output, batch = r
                logger.info(f"Processing completed Batch: pages - {batch}")
                logger.info(f"Results - {model_output}")
                mod_output = modify_outputs(model_output, task.output_mod)
                indexed_outputs.update(
                    {i: page_output for i, page_output in zip(batch, mod_output)}
                )

    except asyncio.CancelledError as cancel_error:
        logger.warning("One or more tasks were cancelled.")
        # DUMP FOR DEBUGGING
        with open(error_file_path, "a", encoding="utf-8") as error_file:
            error_message = f"Task(s) cancelled due to: {repr(cancel_error)}\n"
            error_file.write(error_message)

    return indexed_outputs
