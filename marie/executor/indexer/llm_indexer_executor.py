from typing import Any, List, Optional

from docarray import DocList

from marie import requests
from marie.api import AssetKeyDoc
from marie.executor.marie_pipeline_executor import PipelineExecutor
from marie.logging_core.predefined import default_logger as logger
from marie.pipe.llm_pipeline import LLMPipeline


class LLMIndexerExecutor(PipelineExecutor):
    """Executor for pipeline document processing"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        pipelines: List[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(name, device, num_worker_preprocess, storage, **kwargs)
        logger.info(f"Starting Pipeline Setup")
        logger.info(f"Pipelines config: {pipelines}")
        has_cuda = True if self.device.type.startswith("cuda") else False
        self.pipeline = LLMPipeline(pipelines_config=pipelines, cuda=has_cuda)

    @requests(on="/document/index")
    def handle_index(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Handles index requests and executes the document indexing pipeline. Extracts runtime
        configuration, validates input parameters, processes document frames, executes the
        pipeline with required settings, and persists the resultant metadata.

        Parameters:
            docs (DocList[AssetKeyDoc]): A list of document assets to be processed.
            parameters (dict): A dictionary containing request parameters.
        """
        self.parse_params_and_execute(
            docs, parameters, self.pipeline, default_ref_type="index"
        )
