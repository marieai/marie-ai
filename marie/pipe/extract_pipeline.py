import glob
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import torch
from PIL import Image
from docarray import DocList

from marie.boxes import PSMode
from marie.common.file_io import get_file_count
from marie.logging.logger import MarieLogger
from marie.ocr import CoordinateFormat, DefaultOcrEngine, VotingOcrEngine
from marie.ocr.mock_ocr_engine import MockOcrEngine
from marie.ocr.util import get_words_and_boxes
from marie.overlay.overlay import OverlayProcessor
from marie.pipe import ClassifierPipelineComponent
from marie.pipe import NamedEntityPipelineComponent
from marie.pipe import PipelineComponent, PipelineContext
from marie.pipe.components import (
    setup_classifiers,
    setup_indexers,
    split_filename,
    s3_asset_path,
    ocr_frames,
    store_assets,
    burst_frames,
    restore_assets,
    get_known_ocr_engines,
)
from marie.renderer import TextRenderer, PdfRenderer
from marie.renderer.adlib_renderer import AdlibRenderer
from marie.renderer.blob_renderer import BlobRenderer
from marie.utils.docs import docs_from_image
from marie.utils.docs import frames_from_file
from marie.utils.image_utils import hash_frames_fast
from marie.utils.json import store_json_object
from marie.utils.tiff_ops import merge_tiff, save_frame_as_tiff_g4
from marie.utils.utils import ensure_exists
from marie.utils.zip_ops import merge_zip


class ExtractPipeline:
    """
    Extract pipeline for documents.

    The pipeline will perform the following operations on the document:
    - Burst the document, if it is a multi-page document into individual pages
    - Segment the document (document cleaning)
    - Perform OCR on the document pages or regions
    - Extract the regions from the document pages
    - Classify the document pages
    - Index the document pages (Named Entity Recognition)
    - Store the results in the backend store(s3 , redis, etc.)

    Example usage:
        .. code-block:: python

            pipeline_config = load_yaml(
                os.path.join(
                    __config_dir__, "tests-integration", "pipeline-integration.partial.yml"
                )
            )
            pipeline = ExtractPipeline(pipeline_config=pipeline_config["pipeline"])

            with TimeContext(f"### ExtractPipeline info"):
                results = pipeline.execute(
                    ref_id=filename, ref_type="pid", frames=frames_from_file(img_path)
                )
    """

    def __init__(
        self,
        # models_dir: str = os.path.join(__model_path__),
        pipeline_config: dict[str, any] = None,
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False

        device = pipeline_config.get("device", "cpu" if not use_cuda else "cuda")
        if device == "cuda" and not use_cuda:
            device = "cpu"

        self.logger = MarieLogger(context=self.__class__.__name__)

        self.overlay_processor = OverlayProcessor(
            work_dir=ensure_exists("/tmp/form-segmentation"), cuda=use_cuda
        )

        self.ocr_engines = get_known_ocr_engines(device=device)
        self.document_classifiers = setup_classifiers(pipeline_config)
        self.document_indexers = setup_indexers(pipeline_config)

        self.logger.info(
            f"Loaded classifiers : {len(self.document_classifiers)},  {self.document_classifiers.keys()}"
        )

        self.logger.info(
            f"Loaded indexers : {len(self.document_indexers)},  {self.document_indexers.keys()}"
        )

    def segment(
        self,
        ref_id: str,
        frames: Union[list[np.ndarray], list[Image.Image]],
        root_asset_dir: str,
        force: bool = False,
    ) -> list[np.ndarray]:
        """
        Segment the frames and return the segmented frames

        :param ref_id: reference id of the document
        :param frames: frames to segment
        :param root_asset_dir:  root directory to store the segmented frames
        :param force: force segmentation
        :return:
        """

        output_dir = ensure_exists(os.path.join(root_asset_dir, "clean"))
        filename, prefix, suffix = split_filename(ref_id)
        file_count = get_file_count(output_dir)
        clean_frames = []

        if force or file_count != len(frames):
            self.logger.info(f"Segmenting frames for {ref_id}")

            for i, frame in enumerate(frames):
                try:
                    doc_id = f"{prefix}_{i}"
                    real, mask, clean = self.overlay_processor.segment_frame(
                        doc_id, frame
                    )
                    clean_frames.append(clean)

                    # save real cleaned image
                    save_path = os.path.join(output_dir, f"{i}.tif")
                    save_frame_as_tiff_g4(clean, save_path)
                    # imwrite(save_path, clean, dpi=(300, 300))

                    self.logger.info(f"Saved clean img : {save_path}")
                except Exception as e:
                    self.logger.warning(f"Unable to segment document : {e}")
        else:
            # load frames from output directory and compare with frames
            # if they are the same, skip processing
            if file_count == len(frames):
                self.logger.info(f"Skipping segmentation for {ref_id}")
                for _path in sorted(
                    glob.glob(os.path.join(output_dir, "*.*")),
                    key=lambda name: int(name.split("/")[-1].split(".")[0]),
                ):
                    frame = frames_from_file(_path)[0]
                    clean_frames.append(frame)

        # validate asset count
        file_count = get_file_count(output_dir)
        if file_count != len(frames):
            self.logger.warning(
                f"File count mismatch[segmenter] : {file_count} != {len(frames)}"
            )

        return clean_frames

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

        page_indexer_enabled = runtime_conf.get("page_indexer", {}).get("enabled", True)

        self.logger.info(f"page classifier enabled : {page_classifier_enabled}")
        self.logger.info(f"page indexer enabled : {page_indexer_enabled}")
        post_processing_pipeline = []

        post_processing_pipeline.append(
            ClassifierPipelineComponent(
                name="classifier_pipeline_component",
                document_classifiers=self.document_classifiers,
            )
        )

        post_processing_pipeline.append(
            NamedEntityPipelineComponent(
                name="ner_pipeline_component",
                document_indexers=self.document_indexers,
            )
        )

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

        clean_frames = self.segment(ref_id, frames, root_asset_dir)
        ocr_results = ocr_frames(self.ocr_engines, ref_id, clean_frames, root_asset_dir)
        metadata["ocr"] = ocr_results

        self.execute_pipeline(post_processing_pipeline, frames, ocr_results, metadata)

        # TODO : Convert to execution pipeline
        self.render_pdf(ref_id, frames, ocr_results, root_asset_dir)
        self.render_blobs(ref_id, frames, ocr_results, root_asset_dir)
        self.render_adlib(ref_id, frames, ocr_results, root_asset_dir)

        self.pack_assets(ref_id, ref_type, root_asset_dir, metadata)
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

    def execute_regions_pipeline(
        self,
        ref_id: str,
        ref_type: str,
        frames: List,
        regions: List,
        root_asset_dir: str,
        ps_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        job_id: str = None,
        runtime_conf: Optional[dict[str, any]] = None,
    ) -> dict[str, any]:
        self.logger.info(
            f"Executing pipeline : {ref_id}, {ref_type} with regions : {regions}"
        )
        # check if we have already processed this document and restore assets
        # self.restore_assets(
        #     ref_id, ref_type, root_asset_dir, full_restore=False, overwrite=True
        # )

        # make sure we have clean image
        # clean_frames = self.segment(ref_id, frames, root_asset_dir)
        clean_frames = frames

        results = ocr_frames(
            self.ocr_engines,
            ref_id,
            clean_frames,
            root_asset_dir,
            force=True,
            regions=regions,
            ps_mode=ps_mode,
            coord_format=coordinate_format,
        )

        # self.store_assets(ref_id, ref_type, root_asset_dir)
        # document metadata
        metadata = {
            "ref_id": ref_id,
            "ref_type": ref_type,
            "job_id": job_id,
            "pages": f"{len(frames)}",
            "ocr": results,
        }

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

        1. Segment the document
        2. Burst the document
        3. Perform OCR on the document
        4. Extract the regions from the document
        5. Classify the document
        6. Store the results in the backend store(s3 , redis, etc.)

        :param ref_id:  reference id of the document (e.g. file name)
        :param ref_type: reference type of the document (e.g. invoice, receipt, etc)
        :param frames: frames to process for the document
        :param pms_mode:  Page segmentation mode for OCR default is SPARSE
        :param coordinate_format: coordinate format for OCR default is XYWH
        :param regions:  regions to extract from the document pages
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

        if regions and len(regions) > 0:
            return self.execute_regions_pipeline(
                ref_id,
                ref_type,
                frames,
                regions,
                root_asset_dir,
                pms_mode,
                coordinate_format,
                job_id,
                runtime_conf,
            )
        else:
            return self.execute_frames_pipeline(
                ref_id, ref_type, frames, root_asset_dir, job_id, runtime_conf
            )

    def render_text(self, frames, results, root_asset_dir) -> None:
        renderer = TextRenderer(config={"preserve_interword_spaces": True})
        renderer.render(
            frames,
            results,
            output_filename=os.path.join(root_asset_dir, "results.txt"),
        )

    def render_pdf(self, ref_id: str, frames, results, root_asset_dir) -> None:
        output_dir = ensure_exists(os.path.join(root_asset_dir, "pdf"))
        renderer = PdfRenderer(config={})
        # generating two pdfs one with overlay and one without
        renderer.render(
            frames,
            results,
            output_filename=os.path.join(output_dir, "results.pdf"),
            **{
                "overlay": True,
            },
        )

        renderer.render(
            frames,
            results,
            output_filename=os.path.join(output_dir, "results_clean.pdf"),
            **{
                "overlay": False,
            },
        )

    def render_blobs(self, ref_id: str, frames, results, root_asset_dir):
        blobs_dir = ensure_exists(os.path.join(root_asset_dir, "blobs"))
        filename, prefix, suffix = split_filename(ref_id)

        renderer = BlobRenderer(config={})
        renderer.render(
            frames,
            results,
            blobs_dir,
            filename_generator=lambda x: f"{prefix}_{x}.BLOBS.XML",
        )

    def render_adlib(self, ref_id: str, frames, results, root_asset_dir):
        adlib_dir = ensure_exists(os.path.join(root_asset_dir, "adlib"))
        self.logger.info(f"Rendering adlib : {adlib_dir}")

        filename, prefix, suffix = split_filename(ref_id)
        renderer = AdlibRenderer(summary_filename=f"{filename}.xml", config={})

        renderer.render(
            frames,
            results,
            adlib_dir,
            filename_generator=lambda x: f"{prefix}_{x}.{suffix}.xml",
        )

    def pack_assets(
        self, ref_id: str, ref_type: str, root_asset_dir, metadata: dict[str, any]
    ):
        # create assets
        assets_dir = ensure_exists(os.path.join(root_asset_dir, "assets"))
        blob_dir = os.path.join(root_asset_dir, "blobs")
        pdf_dir = os.path.join(root_asset_dir, "pdf")
        adlib_dir = ensure_exists(os.path.join(root_asset_dir, "adlib"))
        clean_dir = ensure_exists(os.path.join(root_asset_dir, "clean"))

        filename, prefix, suffix = split_filename(ref_id)

        merge_zip(adlib_dir, os.path.join(assets_dir, f"{prefix}.ocr.zip"))
        merge_zip(blob_dir, os.path.join(assets_dir, f"{prefix}.blobs.xml.zip"))

        # convert multiple to G4 standard (mulitpage) TIFF
        clean_filename = os.path.join(assets_dir, f"{prefix}.clean.tif")
        clean_filename_corrected = os.path.join(assets_dir, f"{prefix}.tif.clean")

        merge_tiff(
            clean_dir,
            clean_filename,
            sort_key=lambda name: int(name.split("/")[-1].split(".")[0]),
        )

        # rename .clean.tif to .tif.clean
        shutil.move(clean_filename, clean_filename_corrected)

        # copy PDF to assets
        shutil.copy(
            os.path.join(pdf_dir, "results.pdf"),
            os.path.join(assets_dir, f"{prefix}.pdf"),
        )

        remote_path = s3_asset_path(
            ref_id, ref_type, include_prefix=False, include_filename=False
        )
        resolved_paths = []

        for path in Path(root_asset_dir).rglob("*"):
            if not path.is_file():
                continue
            resolved_path = path.relative_to(root_asset_dir)
            resolved_path = os.path.join(remote_path, resolved_path)
            resolved_paths.append(resolved_path)
        metadata["assets"] = resolved_paths

        return metadata

    def execute_pipeline(
        self,
        processing_pipeline: List[PipelineComponent],
        frames: List,
        ocr_results: dict,
        metadata: dict,
    ):
        """Execute the post processing pipeline
        TODO : This is temporary, we need to make this configurable
        """
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
                pipe_results = pipe.run(documents, context, words=words, boxes=boxes)
                if pipe_results.state is not None:
                    if not isinstance(pipe_results.state, DocList):
                        raise ValueError(
                            f"Invalid state type : {type(pipe_results.state)}"
                        )
                    documents = pipe_results.state
            except Exception as e:
                self.logger.error(f"Error executing pipe : {e}")
