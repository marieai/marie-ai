from typing import Optional, Union

import torch
from docarray import DocList

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.extract import DocumentAnnotatorExecutor
from marie.extract.annotators.llm_table_annotator import LLMTableAnnotator
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger


class DocumentAnnotatorTableLLMExecutor(DocumentAnnotatorExecutor):
    """Executor for document annotation"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        kwargs['storage'] = storage
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        logger.info(f"Started executor : {self.__class__.__name__}")

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
