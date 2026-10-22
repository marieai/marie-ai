from typing import Any, Optional, Union

import torch
from docarray import DocList
from marie.engine.llm_queue.config import llm_queue_enabled

from marie.api.docs import AssetKeyDoc
from marie.executor.extract.document_annotator_executor import (
    DocumentAnnotatorExecutor,
)
from marie.extract.annotators.llm_table_annotator import LLMTableAnnotator
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.runtime import requests


class DocumentAnnotatorTableLLMExecutor(DocumentAnnotatorExecutor):
    """Executor for document annotation"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        kwargs['storage'] = storage
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger
        queue_enabled = llm_queue_enabled()
        submission_mode = "queued-dispatch" if queue_enabled else "direct-batch"
        self.logger.info(
            f"LLM request submission mode: {submission_mode} "
            f"(LLM_QUEUE_ENABLED={str(queue_enabled).lower()})"
        )

        logger.info(f"Started executor : {self.__class__.__name__}")

    def deployment_status_details(self) -> dict[str, Any]:
        enabled = llm_queue_enabled()
        return {
            "llm_dispatch": {
                "enabled": enabled,
                "mode": "queued-dispatch" if enabled else "direct-batch",
            }
        }

    @requests(on="/annotator/table-llm")
    async def annotator_table_llm(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Document table annotator executor
        Much of this is hardcoded in here and need to be moved into proper pipeline.

        EXAMPLE USAGE

            As Executor

            .. code-block:: python

                exec = AnnotatorExecutor()

        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """

        return await self._process_annotation_request(
            docs, parameters, LLMTableAnnotator, *args, **kwargs
        )
