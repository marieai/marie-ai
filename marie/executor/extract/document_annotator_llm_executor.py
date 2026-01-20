from typing import Optional, Union

import torch
from docarray import DocList

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.extract import DocumentAnnotatorExecutor
from marie.extract.annotators.llm_annotator import LLMAnnotator
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger


class DocumentAnnotatorLLMExecutor(DocumentAnnotatorExecutor):
    """Executor for document annotation"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        llm_tracking: dict[str, any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ) -> None:

        kwargs['storage'] = storage
        kwargs['llm_tracking'] = llm_tracking
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        logger.info(f"Started executor : {self.__class__.__name__}")

    @requests(on="/annotator/llm")
    async def annotator_llm(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Document annotator executor
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
        try:
            await self._process_annotation_request(
                docs, parameters, LLMAnnotator, *args, **kwargs
            )
            # TODO: Handle the response properly
            # we should be returning a Document Response here, but for now we just return a success message
            return {'status': 'success', 'message': 'Documents annotated successfully'}

        except Exception as e:
            # Sinc we are not throwing an exception the job will be marked as successful
            # we log the error and return an error message, later we can improve this to mark the job as failed
            # by introducing an improved error handling mechanism
            self.logger.error(f"Error during annotation: {e}")
            return {'status': 'error', 'message': str(e)}
