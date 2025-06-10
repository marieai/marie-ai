from typing import List, Optional, Union

import torch
from docarray import DocList

from marie import requests
from marie.api import AssetKeyDoc
from marie.executor.pipeline.document_pipeline_executor import DocumentPipelineExecutor
from marie.logging_core.predefined import default_logger as logger
from marie.pipe.llm_pipeline import LLMPipeline


class DocumentLLMPipelineExecutor(DocumentPipelineExecutor):
    """Executor for pipeline document proccessing"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        pipelines: List[dict[str, any]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        super().__init__(
            name, device, num_worker_preprocess, storage, pipelines, dtype, **kwargs
        )
        logger.info(f"Starting Pipeline Setup")
        has_cuda = True if self.device.type.startswith("cuda") else False
        self.pipeline = LLMPipeline(pipelines_config=pipelines, cuda=has_cuda)

    @requests(on="/document/classify")
    def handle_classify(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        self.run_pipeline(docs, parameters, *args, **kwargs)

    @requests(on="/document/index")
    def handle_index(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        self.run_pipeline(docs, parameters, *args, **kwargs)
