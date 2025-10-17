import os
import shutil
from datetime import datetime
from typing import Any, List, Optional, Union

import numpy as np
from PIL import Image

from marie.boxes import PSMode
from marie.common.file_io import get_cache_dir
from marie.excepts import BadConfigSource
from marie.models.utils import initialize_device_settings
from marie.ocr import CoordinateFormat
from marie.ocr.util import get_known_ocr_engines
from marie.pipe.base_pipeline import BasePipeline
from marie.pipe.components import (
    burst_frames,
    get_config_from_name,
    is_component_enabled,
    load_pipeline,
    ocr_frames,
    restore_assets,
    store_assets,
)
from marie.utils.image_utils import hash_frames_fast
from marie.utils.utils import ensure_exists


class ClassificationPipeline(BasePipeline):
    """
    Classification pipeline for documents.

    The pipeline will perform the following operations on the document:
    - Burst the document, if it is a multi-page document into individual pages
    - Perform OCR on the document pages
    - Classify the document pages

    Example usage:
        .. code-block:: python

            pipeline_config = load_yaml(
                os.path.join(
                    __config_dir__, "tests-integration", "pipeline-integration.partial.yml"
                )
            )
            pipeline = ClassificationPipeline(pipeline_config=pipeline_config["pipeline"])

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
        has_cuda = True if self.device.type.startswith("cuda") else False

        if self.default_pipeline_config.get("ocr", {}).get("enabled", True):
            self.ocr_engines = get_known_ocr_engines(self.device.type, "default")
        else:
            self.logger.warning("Disabling OCR capabilities as configured.")
            self.ocr_engines = {"default": None}

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
        runtime_conf: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
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

        pipline_config = get_config_from_name(self.pipeline_name, self.pipelines_config)

        page_classifier_enabled = is_component_enabled(
            "page_classifier",
            pipeline_config=pipline_config,
            runtime_config=runtime_conf,
            default=True,
        )
        page_indexer_enabled = is_component_enabled(
            "page_indexer",
            pipeline_config=pipline_config,
            runtime_config=runtime_conf,
            default=True,
        )

        self.logger.info(
            f"Feature : page classifier enabled : {page_classifier_enabled}"
        )
        self.logger.info(f"Feature : page indexer enabled : {page_indexer_enabled}")

        for group, classifiers in self.classifier_groups.items():
            self.logger.info(f"Loaded classifiers : {group}, {len(classifiers)}")

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

        ocr_results = ocr_frames(self.ocr_engines, ref_id, frames, root_asset_dir)
        metadata["ocr"] = ocr_results

        if not page_classifier_enabled:
            self.logger.warning("Page classifier(s) are disabled")
        else:
            self.execute_classifier_and_indexer_pipeline(
                frames,
                metadata,
                metadata["ocr"],
                self.pipeline_name,
                self.classifier_groups,
                self.indexer_groups,
                page_indexer_enabled,
            )

        self.store_metadata(
            ref_id, ref_type, root_asset_dir, metadata, pipeline_name=self.pipeline_name
        )
        store_assets(ref_id, ref_type, root_asset_dir, match_wildcard="*.json")
        del metadata["ocr"]

        return metadata

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

        2. Burst the document
        3. Perform OCR on the document
        5. Classify the document
        6. Store the results in the backend store(s3 , redis, etc.)

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

        # create local asset directory
        frame_checksum = hash_frames_fast(frames=frames)
        cache_dir = get_cache_dir()
        generators_dir = os.path.join(cache_dir, "generators")

        # create backup name by appending a timestamp
        # TODO : Need to refactor this
        if False:  # os.path.exists(os.path.join("/tmp/generators", frame_checksum)):
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(
                os.path.join("/tmp/generators", frame_checksum),
                os.path.join("/tmp/generators", f"{frame_checksum}-{ts}"),
            )

        root_asset_dir = ensure_exists(os.path.join(generators_dir, frame_checksum))

        self.logger.info(f"Root asset dir {ref_id}, {ref_type} : {root_asset_dir}")
        self.logger.info(f"runtime_conf args : {runtime_conf}")

        if runtime_conf is None:
            self.logger.warning("runtime_conf is None, using default config")
            runtime_conf = {}

        return self.execute_frames_pipeline(
            ref_id, ref_type, frames, root_asset_dir, job_id, runtime_conf
        )
