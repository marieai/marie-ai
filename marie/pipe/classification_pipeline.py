import os
import shutil
from abc import ABC
from datetime import datetime
from typing import Union, List, Optional

import numpy as np
import torch
from PIL import Image
from docarray import DocList

from marie.api.docs import MarieDoc
from marie.boxes import PSMode
from marie.logging.logger import MarieLogger
from marie.ocr import CoordinateFormat
from marie.ocr.util import get_words_and_boxes
from marie.pipe import ClassifierPipelineComponent, PipelineResult
from marie.pipe import PipelineComponent, PipelineContext
from marie.pipe.components import (
    setup_classifiers,
    split_filename,
    ocr_frames,
    store_assets,
    burst_frames,
    restore_assets,
    get_known_ocr_engines,
)
from marie.utils.docs import docs_from_image
from marie.utils.image_utils import hash_frames_fast
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists


class ClassifierPipelineScoringComponent(PipelineComponent, ABC):
    def __init__(self, name: str, strategy: str, logger: MarieLogger = None) -> None:
        """
        :param name: Will be passed to base class
        """
        super().__init__(name, logger=logger)
        self.strategy = strategy

    def predict(
        self,
        documents: DocList[MarieDoc],
        context: Optional[PipelineContext] = None,
        *,  # force users to use keyword arguments
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
    ) -> PipelineResult:
        context["metadata"]["document_classification"] = self.classify(
            documents, words, boxes
        )

        return PipelineResult(documents)

    def classify(
        self,
        documents: DocList[MarieDoc],
        words: List[List[str]],
        boxes: List[List[List[int]]],
    ):
        """
        Classify document at by aggregating page level classification results

        :param documents: documents to classify
        :param words: words
        :param boxes: boxes
        :return: classification results
        """

        document_meta = []
        return document_meta


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
        pipeline_config: dict[str, any] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)

        self.ocr_engines = get_known_ocr_engines(use_cuda=use_cuda)
        self.document_classifiers = setup_classifiers(pipeline_config)

        self.logger.info(
            f"Loaded classifiers : {len(self.document_classifiers)},  {self.document_classifiers.keys()}"
        )

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

        self.logger.info(f"page classifier enabled : {page_classifier_enabled}")
        processing_pipeline = [
            ClassifierPipelineComponent(
                name="classifier_pipeline_component",
                document_classifiers=self.document_classifiers,
            ),
            ClassifierPipelineScoringComponent(
                "scoring_pipeline_component", strategy="max"
            ),
        ]

        metadata = {
            "ref_id": ref_id,
            "ref_type": ref_type,
            "job_id": job_id,
            "pages": f"{len(frames)}",
        }

        # check if we have already processed this document and restore assets
        restore_assets(
            ref_id, ref_type, root_asset_dir, full_restore=False, overwrite=True
        )

        # burst frames into individual images
        burst_frames(ref_id, frames, root_asset_dir)
        ocr_results = ocr_frames(self.ocr_engines, ref_id, frames, root_asset_dir)
        metadata["ocr"] = ocr_results

        self.execute_pipeline(processing_pipeline, frames, ocr_results, metadata)

        self.store_metadata(ref_id, ref_type, root_asset_dir, metadata)
        store_assets(ref_id, ref_type, root_asset_dir)

        return metadata

    def store_metadata(
        self, ref_id: str, ref_type: str, root_asset_dir: str, metadata: dict[str, any]
    ) -> None:
        """
        Store current metadata for the document. Format is {ref_id}.meta.json in the root asset directory
        """
        filename, prefix, suffix = split_filename(ref_id)
        metadata_path = os.path.join(root_asset_dir, f"{filename}.meta.json")
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
        frames: List,
        ocr_results: dict,
        metadata: dict,
    ) -> DocList[MarieDoc]:
        """Execute the post processing pipeline
        TODO : This is temporary, we need to make this configurable
        """
        self.logger.info(
            f"Executing document processing pipeline : {processing_pipeline}"
        )

        words = []
        boxes = []
        documents = docs_from_image(frames)

        assert len(documents) == len(frames)

        for page_idx in range(len(frames)):
            page_words, page_boxes = get_words_and_boxes(ocr_results, page_idx)
            words.append(page_words)
            boxes.append(page_boxes)

        assert len(words) == len(boxes)

        context = PipelineContext(pipeline_id="post_processing_pipeline")
        context["metadata"] = metadata

        for pipe in processing_pipeline:
            try:
                # create a PipelineContext and pass it to the component
                self.logger.info(f"Executing component : {pipe}")
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

        return documents