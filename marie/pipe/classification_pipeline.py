import os
import shutil
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
from marie.models.utils import initialize_device_settings
from marie.ocr import CoordinateFormat, OcrEngine
from marie.ocr.util import get_known_ocr_engines, get_words_and_boxes
from marie.pipe import (
    ClassifierPipelineComponent,
    NamedEntityPipelineComponent,
    PipelineComponent,
    PipelineContext,
)
from marie.pipe.components import (
    burst_frames,
    ocr_frames,
    restore_assets,
    setup_classifiers,
    setup_indexers,
    split_filename,
    store_assets,
)
from marie.pipe.voting import ClassificationResult, get_voting_strategy
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
        device: Optional[str] = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        self.logger = MarieLogger(context=self.__class__.__name__)

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

        self.ocr_engines = get_known_ocr_engines(
            device=self.device.type, engine="default"
        )
        (
            self.pipeline_name,
            self.classifier_groups,
            self.document_indexers,
        ) = self.load_pipeline(
            self.default_pipeline_config, self.ocr_engines["default"]
        )

    def load_pipeline(
        self, pipeline_config: dict[str, any], ocr_engine: Optional[OcrEngine] = None
    ) -> tuple[str, dict[str, any], dict[str, any]]:

        # TODO : Need to refactor this (use the caller to get the device and then fallback to the pipeline config)
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        device = pipeline_config.get("device", "cpu" if not use_cuda else "cuda")
        if device == "cuda" and not use_cuda:
            device = "cpu"

        if "name" not in pipeline_config:
            raise BadConfigSource("Invalid pipeline config, missing name field")

        pipeline_name = pipeline_config["name"]
        document_classifiers = setup_classifiers(
            pipeline_config, key="page_classifier", device=device, ocr_engine=ocr_engine
        )

        document_sub_classifiers = setup_classifiers(
            pipeline_config, key="sub_classifier", device=device, ocr_engine=ocr_engine
        )

        classifier_groups = dict()
        for classifier_group, classifiers in document_classifiers.items():
            sub_classifiers = document_sub_classifiers.get(classifier_group, {})
            classifier_groups[classifier_group] = {
                "group": classifier_group,
                "classifiers": classifiers,
                "sub_classifiers": sub_classifiers,
            }

        document_indexers = setup_indexers(
            pipeline_config, key="page_indexer", device=device, ocr_engine=ocr_engine
        )
        # dump information about the loaded classifiers that are grouped by the classifier group
        for classifier_group, classifiers in document_classifiers.items():
            self.logger.info(
                f"Loaded classifiers :{classifier_group},  {len(classifiers)},  {classifiers.keys()}"
            )
        for classifier_group, classifiers in document_sub_classifiers.items():
            self.logger.info(
                f"Loaded sub-classifiers : {classifier_group}, {len(classifiers)},  {classifiers.keys()}"
            )
        self.logger.info(
            f"Loaded indexers : {len(document_indexers)},  {document_indexers.keys()}"
        )

        return pipeline_name, classifier_groups, document_indexers

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

            if page_indexer_enabled:
                processing_pipeline.append(
                    NamedEntityPipelineComponent(
                        name="ner_pipeline_component",
                        document_indexers=self.document_indexers,
                    )
                )

            results = self.execute_pipeline(
                processing_pipeline, sub_classifiers, frames, ocr_results
            )

            metadata["classifications"].append(
                {"group": group, "classification": results}
            )

        self.store_metadata(ref_id, ref_type, root_asset_dir, metadata)
        store_assets(ref_id, ref_type, root_asset_dir, match_wildcard="*.json")
        del metadata["ocr"]

        return metadata

    def store_metadata(
        self,
        ref_id: str,
        ref_type: str,
        root_asset_dir: str,
        metadata: dict[str, any],
        infix: str = "meta",
    ) -> None:
        """
        Store current metadata for the document. Format is {ref_id}.meta.json in the root asset directory
        :param ref_id: reference id of the document
        :param ref_type: reference type of the document
        :param root_asset_dir: root asset directory
        :param metadata: metadata to store
        :param infix: infix to use for the metadata file, default is "meta" e.g. {ref_id}.meta.json
        :return: None
        """
        filename, prefix, suffix = split_filename(ref_id)
        metadata_path = os.path.join(root_asset_dir, f"{filename}.{infix}.json")
        self.logger.info(f"Storing metadata : {metadata_path}")
        store_json_object(metadata, metadata_path)

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
        if False:  # os.path.exists(os.path.join("/tmp/generators", frame_checksum)):
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
        self.logger.info(context["metadata"])

        page_indexer = (
            context["metadata"]["page_indexer"]
            if "page_indexer" in context["metadata"]
            else []
        )
        page_classifier = (
            context["metadata"]["page_classifier"]
            if "page_classifier" in context["metadata"]
            else []
        )

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

        prediction_agent = "majority"
        voter = get_voting_strategy(prediction_agent, "abstain", max_diff=0.25)

        # Classification strategy: max_score, max_votes, max_score_with_diff
        # calculate max score for each page by max score
        score_by_page = {}
        for page, details in class_by_page.items():
            classification_results = [ClassificationResult(**x) for x in details]
            result = voter(classification_results)
            score_by_page[page] = result

        results = {
            "strategy": prediction_agent,
            "pages": {},
        }

        for page in list(class_by_page.keys()):
            results["pages"][page] = {
                "details": class_by_page[page],
                "best": score_by_page[page],
            }

        return results

    def reload_pipeline(self, pipeline_name) -> None:
        with TimeContext(f"### Reloading pipeline : {pipeline_name}", self.logger):
            try:
                self.logger.info(f"Reloading pipeline : {pipeline_name}")
                if self.pipelines_config is None:
                    raise BadConfigSource(
                        "Invalid pipeline configuration, no pipelines found"
                    )

                pipeline_config = None
                for conf in self.pipelines_config:
                    conf = conf["pipeline"]
                    if conf.get("name") == pipeline_name:
                        pipeline_config = conf
                        break

                if pipeline_config is None:
                    raise BadConfigSource(
                        f"Invalid pipeline configuration, pipeline not found : {pipeline_name}"
                    )

                (
                    self.pipeline_name,
                    self.classifier_groups,
                    self.document_indexers,
                ) = self.load_pipeline(pipeline_config)
                self.logger.info(f"Reloaded successfully pipeline : {pipeline_name} ")
            except Exception as e:
                self.logger.error(f"Error reloading pipeline : {e}")
                raise e
