from typing import List, Optional

from docarray import DocList

from marie import requests, safely_encoded
from marie.api import AssetKeyDoc, value_from_payload_or_args
from marie.boxes import PSMode
from marie.executor.pipeline.document_pipeline_executor import (
    PipelineExecutor,
    create_working_dir,
)
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import torch_gc
from marie.ocr import CoordinateFormat
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
        return self.run_llm_pipeline(docs, parameters)

    @requests(on="/document/index")
    def handle_index(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        return self.run_llm_pipeline(docs, parameters)

    def run_llm_pipeline(self, docs: DocList[AssetKeyDoc], parameters: dict):
        job_id, ref_id, ref_type, queue_id, payload = self.extract_base_parameters(
            parameters
        )

        # due to compatibility issues with other frameworks we allow passing same arguments in the 'args' object
        pms_mode = PSMode.from_value(
            value_from_payload_or_args(payload, "mode", default=str(PSMode.SPARSE))
        )
        coordinate_format = CoordinateFormat.from_value(
            value_from_payload_or_args(
                payload, "format", default=str(CoordinateFormat.XYWH)
            )
        )

        if payload.get("regions", []):
            raise NotImplementedError("Regions is not implemented yet")
        if pms_mode is not PSMode.SPARSE:
            raise NotImplementedError(f"PMS mode `{pms_mode}` is not implemented yet")
        if coordinate_format is not CoordinateFormat.XYWH:
            raise NotImplementedError(
                f"Coordinate format `{coordinate_format}` is not implemented yet"
            )

        self.logger.info("Extracting Runtime Config from features list")
        runtime_conf = {}
        pipeline_names = [
            conf["pipeline"]["name"] for conf in self.pipeline.pipelines_config
        ]
        for feature in payload.get("features", []):
            if feature.get("type") != "pipeline":
                continue
            name = feature.get("name")
            if name and any(name == p_name for p_name in pipeline_names):
                runtime_conf = feature
        self.logger.debug(f"Resolved Runtime Config: {runtime_conf}")

        frames = self.get_frames_from_docs(docs, runtime_conf.get("page_limit"))
        root_asset_dir = create_working_dir(frames)
        try:
            metadata = self.pipeline.execute_frames_pipeline(
                ref_id=ref_id,
                ref_type=ref_type,
                frames=frames,
                root_asset_dir=root_asset_dir,
                job_id=job_id,
                runtime_conf=runtime_conf,
            )
            if metadata is None:
                self.logger.error(f"Metadata is None, this should not happen")
                raise ValueError("Pipeline Execution Error: Metadata is None")

            include_ocr = value_from_payload_or_args(
                payload, "return_ocr", default=False
            )
            # strip out ocr results from metadata
            if not include_ocr and "ocr" in metadata:
                del metadata["ocr"]

            response = {
                "status": "success",
                "runtime_info": self.runtime_info,
                "metadata": metadata,
            }
            converted = safely_encoded(lambda x: x)(response)
            return converted

        except BaseException as error:
            self.logger.error(f"Pipeline error : {error}", exc_info=True)
            raise error

        finally:
            del frames
            torch_gc()
            MDC.remove("request_id")
