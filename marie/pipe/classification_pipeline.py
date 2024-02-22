import os
import shutil
import types
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import torch
from docarray import DocList
from PIL import Image

from marie.boxes import PSMode
from marie.excepts import BadConfigSource
from marie.logging.logger import MarieLogger
from marie.logging.profile import TimeContext
from marie.ocr import CoordinateFormat
from marie.ocr.util import get_words_and_boxes
from marie.pipe import ClassifierPipelineComponent, PipelineComponent, PipelineContext
from marie.pipe.components import (
    burst_frames,
    get_known_ocr_engines,
    load_pipeline,
    ocr_frames,
    reload_pipeline,
    restore_assets,
    setup_classifiers,
    setup_indexers,
    split_filename,
    store_assets,
    store_metadata,
)
from marie.utils.docs import docs_from_image
from marie.utils.image_utils import hash_frames_fast
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists


class ClassificationPipeline:
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
        pipelines_config: List[dict[str, any]] = None,
        **kwargs,
    ) -> None:
        # super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        self.logger = MarieLogger(context=self.__class__.__name__)
        self.load_pipeline = types.MethodType(load_pipeline, self)

        self.pipelines_config = pipelines_config
        self.reload_pipeline = types.MethodType(reload_pipeline, self)
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

        (
            self.pipeline_name,
            self.classifier_groups,
            self.document_indexers,
        ) = self.load_pipeline(self.default_pipeline_config)

        # sometimes we have CUDA/GPU support but want to only use CPU
        # TODO : REFINE THIS
        use_cuda = torch.cuda.is_available()
        device = "cpu" if not use_cuda else "cuda"
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        if device == "cuda" and not use_cuda:
            device = "cpu"

        self.ocr_engines = get_known_ocr_engines(device=device, engine="default")

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

        page_classifier_enabled = runtime_conf.get("page_classifier", {}).get(
            "enabled", True
        )

        # check if the current pipeline name is the default pipeline name
        if "name" in runtime_conf:
            expected_pipeline_name = runtime_conf["name"]
            if expected_pipeline_name != self.pipeline_name:
                self.logger.warning(
                    f"pipeline name : {expected_pipeline_name}, expected : {self.pipeline_name} , reloading pipeline"
                )
                self.reload_pipeline(expected_pipeline_name)

        page_indexer_enabled = runtime_conf.get("page_indexer", {}).get("enabled", True)

        self.logger.info(
            f"Feature : page classifier enabled : {page_classifier_enabled}"
        )
        self.logger.info(f"Feature : page indexer enabled : {page_indexer_enabled}")

        for group, classifiers in self.classifier_groups.items():
            self.logger.info(f"Loaded classifiers : {group}, {len(classifiers)}")

        # if page_indexer_enabled:
        #     processing_pipeline.append(
        #         NamedEntityPipelineComponent(
        #             name="ner_pipeline_component",
        #             document_indexers=self.document_indexers,
        #         )
        #     )

        metadata = {
            "ref_id": ref_id,
            "ref_type": ref_type,
            "job_id": job_id,
            "pipeline": self.pipeline_name,
            "pages": f"{len(frames)}",
        }

        restore_assets(
            ref_id, ref_type, root_asset_dir, full_restore=False, overwrite=True
        )
        burst_frames(ref_id, frames, root_asset_dir)
        ocr_results = ocr_frames(self.ocr_engines, ref_id, frames, root_asset_dir)

        metadata["ocr"] = ocr_results
        metadata["classifications"] = []

        # TODO : Need to refactor this
        for group, classifier_group in self.classifier_groups.items():
            self.logger.info(
                f"Processing classifier pipeline/group :  {self.pipeline_name}, {group}"
            )
            document_classifiers = classifier_group["classifiers"]
            sub_classifiers = classifier_group["sub_classifiers"]

            processing_pipeline = [
                ClassifierPipelineComponent(
                    name="classifier_pipeline",
                    document_classifiers=document_classifiers,
                )
            ]

            results = self.execute_pipeline(
                processing_pipeline, sub_classifiers, frames, ocr_results
            )
            metadata["classifications"].append(
                {"group": group, "classification": results}
            )

        store_metadata(ref_id, ref_type, root_asset_dir, metadata)
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
        runtime_conf: Optional[dict[str, any]] = None,
    ) -> dict[str, any]:
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
        # create backup name by appending a timestamp
        if os.path.exists(os.path.join("/tmp/generators", frame_checksum)):
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(
                os.path.join("/tmp/generators", frame_checksum),
                os.path.join("/tmp/generators", f"{frame_checksum}-{ts}"),
            )

        root_asset_dir = ensure_exists(os.path.join("/tmp/generators", frame_checksum))

        self.logger.info(f"Root asset dir {ref_id}, {ref_type} : {root_asset_dir}")
        self.logger.info(f"runtime_conf args : {runtime_conf}")

        if runtime_conf is None:
            self.logger.warning("runtime_conf is None, using default config")
            runtime_conf = {}

        return self.execute_frames_pipeline(
            ref_id, ref_type, frames, root_asset_dir, job_id, runtime_conf
        )

    def execute_pipeline(
        self,
        processing_pipeline: List[PipelineComponent],
        sub_classifiers: dict[str, any],
        frames: List,
        ocr_results: dict,
    ) -> dict[str, any]:
        """Execute processing pipeline"""

        words = []
        boxes = []
        documents = docs_from_image(frames)
        assert len(documents) == len(frames)

        for page_idx in range(len(frames)):
            page_words, page_boxes = get_words_and_boxes(ocr_results, page_idx)
            words.append(page_words)
            boxes.append(page_boxes)

        assert len(words) == len(boxes)

        context = PipelineContext(pipeline_id="classification_pipeline")
        context["metadata"] = {}

        for pipe in processing_pipeline:
            try:
                # create a PipelineContext and pass it to the component
                pipe_results = pipe.run(documents, context, words=words, boxes=boxes)
                if pipe_results.state is not None:
                    if not isinstance(pipe_results.state, DocList):
                        raise ValueError(
                            f"Invalid state type : {type(pipe_results.state)}"
                        )
                    documents = pipe_results.state
            except Exception as e:
                self.logger.error(f"Error executing pipe : {e}")

        # TODO : This is temporary, we need to make this configurable
        self.logger.info("### ClassificationPipeline results")
        self.logger.info(context["metadata"]["page_classifier"])

        page_classifier = context["metadata"]["page_classifier"]

        for idx, page_classifier_result in enumerate(page_classifier):
            for detail in page_classifier_result["details"]:
                page = int(detail["page"])
                classification = detail["classification"]
                filtered_classifiers = {}

                for key, val in sub_classifiers.items():
                    fileter_config = val["filter"]
                    filter_type = fileter_config["type"]
                    filter_pattern = fileter_config["pattern"]

                    if filter_type == "exact" and classification == filter_pattern:
                        self.logger.info(f"Adding sub-classifier : {key}")
                        filtered_classifiers[key] = val

                if filtered_classifiers:
                    self.logger.info(
                        f"Filtered classifiers : {filtered_classifiers.keys()}"
                    )
                    sub_classifier_pipeline = ClassifierPipelineComponent(
                        name="sub_classifier_pipeline",
                        document_classifiers=filtered_classifiers,
                    )

                    ctx = PipelineContext(pipeline_id="sub_classification_pipeline")
                    ctx["metadata"] = {}
                    pipe_results = sub_classifier_pipeline.run(
                        documents[page : page + 1],
                        ctx,
                        words=[words[page]],
                        boxes=[boxes[page]],
                    )
                    detail["sub_classifier"] = ctx["metadata"]["page_classifier"]

        # Pivot the results to make it easier to work with by page
        class_by_page = {}
        for idx, page_classifier_result in enumerate(page_classifier):
            classifier = page_classifier_result["classifier"]
            for detail in page_classifier_result["details"]:
                page = int(detail["page"])
                if page not in class_by_page:
                    class_by_page[page] = []
                detail["classifier"] = classifier
                class_by_page[page].append(detail)

        # calculate max score for each page by max score
        score_by_page = {}
        for page, details in class_by_page.items():
            max_score = 0.0
            for detail in details:
                if page not in score_by_page:
                    score_by_page[page] = {}

                score = float(detail["score"])
                if score >= max_score:
                    max_score = score
                    score_by_page[page] = detail
                    del detail["page"]

        results = {
            "strategy": "max_score",
            "pages": {},
        }

        for page in list(class_by_page.keys()):
            results["pages"][page] = {
                "details": class_by_page[page],
                "best": score_by_page[page],
            }

        return results
