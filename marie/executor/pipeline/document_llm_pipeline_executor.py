from typing import List, Optional

from docarray import DocList

from marie import requests
from marie.api import AssetKeyDoc
from marie.executor.pipeline.document_pipeline_executor import PipelineExecutor
from marie.logging_core.predefined import default_logger as logger
from marie.pipe.llm_pipeline import LLMPipeline


class DocumentLLMPipelineExecutor(PipelineExecutor):
    """Executor for pipeline document proccessing"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        pipelines: List[dict[str, any]] = None,
        **kwargs,
    ):
        super().__init__(name, device, num_worker_preprocess, storage, **kwargs)
        logger.info(f"Starting Pipeline Setup")
        logger.info(f"Pipelines config: {pipelines}")
        has_cuda = True if self.device.type.startswith("cuda") else False
        self.pipeline = LLMPipeline(pipelines_config=pipelines, cuda=has_cuda)

    @requests(on="/document/classify")
    def handle_classify(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        return self.run_pipeline(self.pipeline, docs, parameters, *args, **kwargs)

    @requests(on="/document/index")
    def handle_index(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        return self.run_pipeline(self.pipeline, docs, parameters, *args, **kwargs)
