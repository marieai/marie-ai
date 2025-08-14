import os
from typing import Any, List, Optional, Union

import numpy as np
from black.trans import defaultdict
from PIL import Image

from marie.boxes import PSMode
from marie.common.file_io import get_cache_dir
from marie.excepts import BadConfigSource
from marie.logging_core.profile import TimeContext
from marie.models.utils import initialize_device_settings
from marie.ocr import CoordinateFormat
from marie.ocr.util import get_known_ocr_engines
from marie.pipe.base_pipeline import BasePipeline
from marie.pipe.components import (
    burst_frames,
    load_pipeline,
    ocr_frames,
    restore_assets,
    store_assets,
)
from marie.pipe.llm_indexer import LLMIndexerPipelineComponent
from marie.utils.image_utils import hash_frames_fast
from marie.utils.utils import ensure_exists


class LLMPipeline(BasePipeline):
    """
    Multi-Modal LLM based pipeline for documents.

    The pipeline will perform the following operations on the document:
    - Burst the document, if it is a multi-page document into individual pages
    - Get OCR if possible (can use if configured to do so)
    - Perform a web of Indexing tasks on the document pages
    - Store results

    TODO: add correct Example usage
    Example usage:
        .. code-block:: python

            pipeline_config = load_yaml(
                os.path.join(
                    __config_dir__, "tests-integration", "pipeline-integration.partial.yml"
                )
            )
            pipeline = IndexingPipeline(pipeline_config=pipeline_config["pipeline"])

            with TimeContext(f"### ExtractPipeline info"):
                results = pipeline.execute(
                    ref_id=filename, ref_type="pid", frames=frames_from_file(img_path)
                )
    """

    def __init__(
        self,
        pipelines_config: List[dict[str, Any]] = None,
        device: Optional[str] = "cuda",
        silence_exceptions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(silence_exceptions, **kwargs)
        self.pipelines_config = pipelines_config
        self.default_pipeline_config = None

        for conf in pipelines_config:
            conf = conf["pipeline"]
            if conf.get("default", False):
                if self.default_pipeline_config is not None:
                    raise BadConfigSource(
                        "Invalid pipeline configuration, multiple defaults found"
                    )
                self.default_pipeline_config = conf

        if self.default_pipeline_config is None:
            raise BadConfigSource("Invalid pipeline configuration, default not found")

        # sometimes we have CUDA/GPU support but want to only use CPU
        resolved_devices, _ = initialize_device_settings(
            devices=[device], use_cuda=True, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]
        # self.has_cuda = True if self.device.type.startswith("cuda") else False

        self.ocr_engines = get_known_ocr_engines(self.device.type, "default")
        # self.ocr_engines = {"default": None}

        (
            self.pipeline_name,
            self.classifier_groups,
            self.indexer_groups,
        ) = load_pipeline(self.default_pipeline_config, self.ocr_engines["default"])

    def execute_frames_pipeline(
        self,
        ref_id: str,
        ref_type: str,
        frames: List[np.ndarray],
        root_asset_dir: str,
        job_id: str,
        runtime_conf: Optional[dict[str, any]] = None,
    ) -> dict[str, any]:
        if ref_type is None or ref_id is None:
            raise ValueError("Invalid reference type or id")

        self.logger.info(
            f"Executing pipeline for document : {ref_id}, {ref_type} > {root_asset_dir}"
        )
        self.logger.info(f"Executing pipeline runtime_conf : {runtime_conf}")

        # check if the current pipeline name is the default pipeline name
        if "name" in runtime_conf:
            expected_pipeline_name = runtime_conf["name"]
            if expected_pipeline_name != self.pipeline_name:
                self.logger.warning(
                    f"pipeline name : {expected_pipeline_name}, expected : {self.pipeline_name} , reloading pipeline"
                )
                (
                    self.pipeline_name,
                    self.classifier_groups,
                    self.indexer_groups,
                ) = self.reload_pipeline(expected_pipeline_name, self.pipelines_config)

            for group, indexer in self.indexer_groups.items():
                self.logger.info(f"Loaded indexers : {group}, {len(indexer)}")

        metadata = {
            "ref_id": ref_id,
            "ref_type": ref_type,
            "job_id": job_id,
            "pipeline": self.pipeline_name,
            "pages": f"{len(frames)}",
        }

        restore_assets(
            ref_id, ref_type, root_asset_dir, full_restore=True, overwrite=True
        )
        burst_frames(ref_id, frames, root_asset_dir)

        # Load Available OCR data if possible
        ocr_results = ocr_frames(self.ocr_engines, ref_id, frames, root_asset_dir)
        metadata["ocr"] = ocr_results

        # Track pipline execution time for metrics
        with TimeContext(f"### {self.pipeline_name} LLMPipeline info") as tc:
            self.execute_llm_pipeline(frames, metadata, ocr_results, runtime_conf)
            metadata[f"delta_time_{self.pipeline_name}"] = tc.now()
        self.store_metadata(
            ref_id, ref_type, root_asset_dir, metadata, pipeline_name=self.pipeline_name
        )
        store_assets(ref_id, ref_type, root_asset_dir, match_wildcard="*.json")
        del metadata["ocr"]

        return metadata

    def execute_llm_pipeline(self, frames, metadata, ocr_results, runtime_conf: dict):
        """
        Executes the LLM pipeline for processing frames and OCR results while updating the metadata with indexing results.

        Parameters:
            frames: List
                A collection of data frames to be processed through the pipeline.
            metadata: dict
                Dictionary containing metadata information for the pipeline. It will be updated with indexing results.
            ocr_results: List
                Results from OCR processing required for indexing and further tasks.
            runtime_conf: dict
                Runtime configuration used to customize the pipeline execution, including task-specific settings.

        Returns:
            None

        Raises:
            Any exceptions raised internally during pipeline execution will propagate.
        """
        if self.indexer_groups:
            if "indexers" not in metadata:
                metadata["indexes"] = []

        processing_group_pipeline = defaultdict(list)

        llm_task_config = runtime_conf.get("llm_tasks", {})
        for group, indexer_group in self.indexer_groups.items():
            self.logger.info(
                f"Processing llm pipeline/group :  {self.pipeline_name}, {group}"
            )
            processing_group_pipeline[group].append(
                LLMIndexerPipelineComponent(
                    name="mmllm_pipeline_component",
                    document_indexers=indexer_group["indexers"],
                    llm_tasks=[
                        task
                        for task in indexer_group.get("llm_tasks", [])
                        if llm_task_config.get(task, {"enabled": True}).get(
                            "enabled", True
                        )
                    ],
                )
            )

        for group, processing_pipeline in processing_group_pipeline.items():
            results = self.execute_pipeline(
                processing_pipeline,
                {},  # sub_classifiers,
                frames,
                ocr_results,
                "indexing_pipeline",
                include_ocr_lines=True,
            )
            if "indexes" in results:
                for task_name, index in results["indexes"].items():
                    metadata["indexes"].append(
                        {
                            "group": group,
                            "task": task_name,
                            "index": index,
                        }
                    )

    def execute(
        self,
        ref_id: str,
        ref_type: str,
        frames: Union[List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        regions: List = None,
        queue_id: str = None,
        job_id: str = None,
        runtime_conf: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute the pipeline for the document with the given frames.If regions are specified,
        then only the specified regions will be extracted from the document with the rest of the steps being skipped.

        By default, this will perform the following steps

        1. Burst the document
        2. Perform OCR on the document
        3. Classify the document
        4. Index the document
        5. Store the results in the backend store(s3 , redis, etc.)

        :param ref_id:  reference id of the document (e.g. file name)
        :param ref_type: reference type of the document (e.g. invoice, receipt, etc)
        :param frames: frames to process for the document
        :param pms_mode:  Page segmentation mode for OCR default is SPARSE
        :param coordinate_format: coordinate format for OCR default is XYWH
        :param queue_id:  queue id to associate with the document
        :param job_id: job id to associate with the document
        :param runtime_conf: runtime configuration for the pipeline (e.g. which steps to execute) default is None.
        :return:  metadata for the document (e.g. OCR results, classification results, etc)
        """

        if regions:
            raise NotImplementedError("Regions is not implemented yet")
        if pms_mode is not PSMode.SPARSE:
            raise NotImplementedError(f"PMS mode `{pms_mode}` is not implemented yet")
        if coordinate_format is not CoordinateFormat.XYWH:
            raise NotImplementedError(
                f"Coordinate format `{coordinate_format}` is not implemented yet"
            )

        # create local asset directory
        frame_checksum = hash_frames_fast(frames=frames)
        cache_dir = get_cache_dir()
        generators_dir = os.path.join(cache_dir, "generators")
        root_asset_dir = ensure_exists(os.path.join(generators_dir, frame_checksum))
        self.logger.info(f"Root asset dir {ref_id}, {ref_type} : {root_asset_dir}")

        if runtime_conf is None:
            self.logger.warning("runtime_conf is None, using default config")
            runtime_conf = {}
        self.logger.info(f"runtime_conf args : {runtime_conf}")

        return self.execute_frames_pipeline(
            ref_id, ref_type, frames, root_asset_dir, job_id, runtime_conf
        )
