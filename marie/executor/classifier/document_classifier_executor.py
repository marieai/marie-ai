from typing import Any, List, Optional, Union

import torch
from docarray import DocList

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.marie_pipeline_executor import PipelineExecutor
from marie.logging_core.predefined import default_logger as logger
from marie.pipe.classification_pipeline import ClassificationPipeline


class DocumentClassificationExecutor(PipelineExecutor):
    """
    Provides functionality for executing document classification pipelines.

    This class is designed to facilitate the classification of documents through a
    configurable pipeline. It supports the initialization of pipeline configurations
    and manages the execution flow for classifying large volumes of documents. The
    use of multiple workers for preprocessing and optional GPU acceleration are also
    supported.
    """

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        pipelines: List[dict[str, Any]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        super().__init__(name, device, num_worker_preprocess, storage, **kwargs)
        logger.info(f"Starting Pipeline Setup")
        logger.info(f"Pipelines config: {pipelines}")
        has_cuda = True if self.device.type.startswith("cuda") else False
        self.pipeline = ClassificationPipeline(
            pipelines_config=pipelines, cuda=has_cuda
        )

    @requests(on="/document/classify")
    # @safely_encoded # BREAKS WITH docarray 0.39 as it turns this into a LegacyDocument which is not supported
    def classify(self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs):
        """
        Executes the classification pipeline for documents.

        This function initiates the processing of the provided documents with the specified
        parameters. It leverages the existing pipeline instance to classify the input
        documents while allowing for additional configuration through parameters.

        Parameters:
            docs (DocList[AssetKeyDoc]): A list of AssetKeyDoc objects to be classified.
            parameters (dict): Additional parameters to configure the pipeline.

        Raises:
            Any exception encountered during pipeline execution will propagate.
        """

        self.parse_params_and_execute(
            docs, parameters, self.pipeline, default_ref_type="classify"
        )
