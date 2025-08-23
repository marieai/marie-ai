import glob
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from marie.boxes import PSMode
from marie.common.file_io import get_cache_dir, get_file_count
from marie.components.document_registration.datamodel import DocumentBoundaryPrediction
from marie.components.template_matching.document_matched import (
    load_template_matching_definitions,
    match_templates,
)
from marie.ocr import CoordinateFormat
from marie.ocr.util import get_known_ocr_engines
from marie.pipe.base_pipeline import BasePipeline
from marie.pipe.components import (
    burst_frames,
    is_component_enabled,
    load_pipeline,
    ocr_frames,
    restore_assets,
    s3_asset_path,
    setup_document_boundary,
    setup_overlay,
    setup_template_matching,
    split_filename,
    store_assets,
)
from marie.renderer import PdfRenderer, TextRenderer
from marie.renderer.adlib_renderer import AdlibRenderer
from marie.renderer.blob_renderer import BlobRenderer
from marie.utils.docs import docs_from_image, frames_from_file
from marie.utils.image_utils import hash_frames_fast
from marie.utils.json import store_json_object
from marie.utils.tiff_ops import merge_tiff, save_frame_as_tiff_g4
from marie.utils.utils import ensure_exists
from marie.utils.zip_ops import merge_zip


class ExtractPipeline(BasePipeline):
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
        pipeline_config: dict[str, Any] = None,
        cuda: bool = True,
        silence_exceptions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(silence_exceptions, **kwargs)
        self.default_pipeline_config = pipeline_config
        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False

        device = pipeline_config.get("device", "cpu" if not use_cuda else "cuda")
        if device == "cuda" and not use_cuda:
            device = "cpu"

        self.overlay_processor = setup_overlay(pipeline_config)
        self.boundary_processor = setup_document_boundary(pipeline_config)
        self.engine_name = "default"

        self.ocr_engines = get_known_ocr_engines(device=device, engine=self.engine_name)
        (
            self.pipeline_name,
            self.classifier_groups,
            self.indexer_groups,
        ) = load_pipeline(pipeline_config, self.ocr_engines[self.engine_name])

        # TODO : Refactor this to use the pipeline_config instead of the some of the hardcoded values
        self.matcher, self.template_matching_definitions = setup_template_matching(
            device=device, pipeline_config=pipeline_config
        )

    def segment(
        self,
        ref_id: str,
        frames: Union[list[np.ndarray], list[Image.Image]],
        root_asset_dir: str,
        force: bool = False,
        enabled: bool = True,
    ) -> list[np.ndarray]:
        """
        Segment the frames and return the segmented frames

        :param ref_id: reference id of the document
        :param frames: frames to segment
        :param root_asset_dir:  root directory to store the segmented frames
        :param force: force segmentation
        :param enabled: enable/disable segmentation (default is True), this does not prevent TIFFs from being generated
        :return:
        """
        self.logger.info(f"Segmenting [{len(frames)}] frames for {ref_id}")
        if len(frames) == 0:
            self.logger.warning(f"No frames to segment for {ref_id}")
            raise ValueError("No frames to segment")

        output_dir = ensure_exists(os.path.join(root_asset_dir, "clean"))
        filename, prefix, suffix = split_filename(ref_id)
        file_count = get_file_count(output_dir)
        clean_frames = []

        if force or file_count != len(frames):
            for i, frame in enumerate(frames):
                try:
                    doc_id = f"{prefix}_{i}"
                    try:
                        if enabled:
                            real, mask, clean = self.overlay_processor.segment_frame(
                                doc_id, frame
                            )
                        else:
                            self.logger.debug(f"Skipping segmentation for {ref_id}")
                            real, mask, clean = frame, None, frame
                    except Exception as e:
                        self.logger.warning(
                            f"Unable to segment document (using original frame) : {e}"
                        )
                        real, mask, clean = frame, None, frame

                    clean_frames.append(clean)
                    # save real cleaned image
                    save_path = os.path.join(output_dir, f"{i}.tif")
                    save_frame_as_tiff_g4(clean, save_path)
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

    def boundary(
        self,
        ref_id: str,
        frames: Union[list[np.ndarray], list[Image.Image]],
        root_asset_dir: str,
        enabled: bool = True,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        """
        Run boundary registration on the frames and return the registered frames

        :param ref_id: reference id of the document
        :param frames: frames to find the boundary for
        :param root_asset_dir: root directory to store the boundary frames
        :param enabled: enable/disable segmentation (default is True), this does not prevent TIFFs from being generated
        :return: registered frames
        """
        self.logger.info(f"Boundary detection:{enabled}, {ref_id}")
        documents = docs_from_image(frames)
        registered_frames = []
        metadata = []
        if enabled:
            try:
                output_dir = ensure_exists(os.path.join(root_asset_dir, "boundary"))
                results = self.boundary_processor.run(
                    documents, registration_method="fit_to_page"
                )
                for i, (frame, result) in enumerate(zip(frames, results)):
                    boundary: DocumentBoundaryPrediction = result.tags[
                        "document_boundary"
                    ]
                    meta = boundary.to_dict(include_images=False)
                    meta["page"] = i
                    metadata.append(meta)
                    self.logger.info(f"Boundary detected {i} : {boundary.detected}")
                    if boundary.detected:
                        frame = boundary.aligned_image
                        cv2.imwrite(
                            os.path.join(output_dir, f"boundary_{i}.png"),
                            boundary.aligned_image,
                        )
                    registered_frames.append(frame)
            except Exception as e:
                self.logger.warning(
                    f"Unable to perform document boundary (using original frame) : {e}"
                )
        else:
            registered_frames = frames

        assert len(registered_frames) == len(frames)
        return registered_frames, metadata

    def template_matching(
        self,
        definition_id: str,
        frames: Union[list[np.ndarray], list[Image.Image]],
        root_asset_dir: str,
        ocr_results: dict,
        enabled: bool = True,
    ) -> list[dict[str, Any]]:
        """ """

        self.logger.info(f"Template matching")
        if self.matcher is None:
            self.logger.warning("Template matcher is not configured")
            return []

        definition_file = os.path.join(
            self.template_matching_definitions, f"{definition_id}.definition.json"
        )
        if not os.path.exists(definition_file):
            self.logger.warning(
                f"Template definition file not found : {definition_file}"
            )
            return []

        definition = load_template_matching_definitions(definition_file)
        return match_templates(frames, definition, self.matcher, ocr_results)

    def execute_frames_pipeline(
        self,
        ref_id: str,
        ref_type: str,
        frames: List[np.ndarray],
        root_asset_dir: str,
        job_id: str,
        runtime_conf: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute the pipeline for the document with the given frames.
        :param ref_id: reference id of the document (e.g. file name)
        :param ref_type: reference type of the document (e.g. invoice, receipt, etc)
        :param frames: frames to process for the document (e.g. bursted pages)
        :param root_asset_dir: root asset directory to store the results
        :param job_id: job id to associate with the document
        :param runtime_conf: runtime configuration for the pipeline (e.g. which steps to execute) default is None.
        :return: metadata for the document (e.g. OCR results, classification results, etc)
        """
        if ref_type is None or ref_id is None:
            raise ValueError("Invalid reference type or id")

        self.logger.info(
            f"Executing pipeline for document : {ref_id}, {ref_type} > {root_asset_dir}"
        )
        self.logger.info(f"Executing pipeline runtime_conf : {runtime_conf}")

        # Default to defaults set in the default pipeline config
        page_classifier_enabled = runtime_conf.get("page_classifier", {}).get(
            "enabled",
            is_component_enabled(
                self.default_pipeline_config.get("page_classifier"), True
            ),
        )
        page_indexer_enabled = runtime_conf.get("page_indexer", {}).get(
            "enabled",
            is_component_enabled(
                self.default_pipeline_config.get("page_indexer"), True
            ),
        )
        page_cleaner_enabled = runtime_conf.get("page_cleaner", {}).get(
            "enabled",
            is_component_enabled(
                self.default_pipeline_config.get("page_cleaner"), False
            ),
        )  # default to False, client should enable
        page_boundary_enabled = runtime_conf.get("page_boundary", {}).get(
            "enabled",
            is_component_enabled(
                self.default_pipeline_config.get("page_boundary"), True
            ),
        )
        template_matching_enabled = runtime_conf.get("template_matching", {}).get(
            "enabled",
            is_component_enabled(
                self.default_pipeline_config.get("template_matching"), True
            ),
        )

        self.logger.info(f"Feature : classifier enabled : {page_classifier_enabled}")
        self.logger.info(f"Feature : indexer enabled : {page_indexer_enabled}")
        self.logger.info(f"Feature : cleaner enabled : {page_cleaner_enabled}")
        self.logger.info(f"Feature : boundary enabled : {page_boundary_enabled}")
        self.logger.info(
            f"Feature : template matching enabled : {template_matching_enabled}"
        )

        for group, classifiers in self.classifier_groups.items():
            self.logger.info(f"Loaded classifiers : {group}, {len(classifiers)}")

        metadata = {
            "ref_id": ref_id,
            "ref_type": ref_type,
            "job_id": job_id,
            "work_dir": root_asset_dir,
            "pipeline": self.pipeline_name,
            "pages": f"{len(frames)}",
        }

        # check if we have already processed this document and restore assets
        restore_assets(
            ref_id, ref_type, root_asset_dir, full_restore=False, overwrite=True
        )

        # remove old metadata results if any (for now)
        # Better option is to have client remove the old metadata

        page_boundary_enabled = False
        if page_boundary_enabled:
            filename, prefix, suffix = split_filename(ref_id)
            metadata_path = os.path.join(root_asset_dir, f"{filename}.meta.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

        # burst frames into individual images
        burst_frames(ref_id, frames, root_asset_dir)
        frames, boundary_meta = self.boundary(
            ref_id, frames, root_asset_dir, enabled=page_boundary_enabled
        )

        clean_frames = self.segment(
            ref_id, frames, root_asset_dir, enabled=page_cleaner_enabled
        )
        ocr_results = ocr_frames(
            self.ocr_engines,
            ref_id,
            clean_frames,
            root_asset_dir,
            engine_name=self.engine_name,
        )

        def_id = runtime_conf.get("template_matching", {}).get("definition_id", "0")
        template_matching_meta = self.template_matching(
            def_id,
            frames,
            root_asset_dir,
            ocr_results,
            enabled=template_matching_enabled,
        )

        metadata["ocr"] = ocr_results
        metadata["boundary"] = boundary_meta
        metadata["template_matching"] = template_matching_meta

        self.execute_classifier_and_indexer_pipeline(
            frames,
            metadata,
            metadata["ocr"],
            self.pipeline_name,
            self.classifier_groups,
            self.indexer_groups,
            page_indexer_enabled,
        )

        # TODO : Convert to execution pipeline
        self.render_pdf(ref_id, frames, ocr_results, root_asset_dir)
        self.render_blobs(ref_id, frames, ocr_results, root_asset_dir)
        self.render_adlib(ref_id, frames, ocr_results, root_asset_dir)

        self.pack_assets(ref_id, ref_type, root_asset_dir, metadata)
        self.store_metadata(ref_id, ref_type, root_asset_dir, metadata)
        store_assets(ref_id, ref_type, root_asset_dir)

        return metadata

    def store_metadata(
        self, ref_id: str, ref_type: str, root_asset_dir: str, metadata: dict[str, Any]
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
        runtime_conf: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
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
            "work_dir": root_asset_dir,
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
        runtime_conf: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
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

        cache_dir = get_cache_dir()
        generators_dir = os.path.join(cache_dir, "generators")

        # create backup name by appending a timestamp
        if False and os.path.exists(os.path.join(generators_dir, frame_checksum)):
            if True:
                self.logger.warning(
                    f"Asset dir already exists, moving to backup : {frame_checksum}"
                )
                return {}

            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(
                os.path.join(generators_dir, frame_checksum),
                os.path.join(generators_dir, f"{frame_checksum}-{ts}"),
            )

        root_asset_dir = ensure_exists(os.path.join(generators_dir, frame_checksum))
        self.logger.info(f"Root asset dir {ref_id}, {ref_type} : {root_asset_dir}")
        self.logger.info(f"runtime_conf args : {runtime_conf}")

        if runtime_conf is None:
            self.logger.warning("runtime_conf is None, using default " "config")
            runtime_conf = {}

        try:
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
        except Exception as e:
            self.logger.error(f"Error executing pipeline : {e}")
            raise e

    def render_text(self, frames, results, root_asset_dir) -> None:
        renderer = TextRenderer(config={"preserve_interword_spaces": True})
        renderer.render(
            frames,
            results,
            output_file_or_dir=os.path.join(root_asset_dir, "results.txt"),
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
        self, ref_id: str, ref_type: str, root_asset_dir, metadata: dict[str, Any]
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

        # make copy of clea.tiff as .tif
        shutil.copy(clean_filename, os.path.join(assets_dir, f"{prefix}.tif"))
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
