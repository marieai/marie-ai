import glob
import os
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import torch
from PIL import Image
from docarray import DocumentArray

from marie.boxes import PSMode
from marie.common.file_io import get_file_count
from marie.components import TransformersDocumentClassifier
from marie.excepts import BadConfigSource
from marie.executor.ner import NerExtractionExecutor
from marie.logging.logger import MarieLogger
from marie.ocr import CoordinateFormat, DefaultOcrEngine, VotingOcrEngine
from marie.ocr.mock_ocr_engine import MockOcrEngine
from marie.ocr.util import get_words_and_boxes
from marie.overlay.overlay import OverlayProcessor
from marie.pipe import PipelineComponent, PipelineContext
from marie.pipe import ClassifierPipelineComponent
from marie.pipe import NamedEntityPipelineComponent
from marie.renderer import TextRenderer, PdfRenderer
from marie.renderer.adlib_renderer import AdlibRenderer
from marie.renderer.blob_renderer import BlobRenderer
from marie.storage import StorageManager
from marie.utils.docs import docs_from_image
from marie.utils.docs import frames_from_file
from marie.utils.image_utils import hash_frames_fast
from marie.utils.json import store_json_object, load_json_file
from marie.utils.tiff_ops import burst_tiff_frames, merge_tiff, save_frame_as_tiff_g4
from marie.utils.utils import ensure_exists
from marie.utils.zip_ops import merge_zip


def split_filename(img_path: str) -> (str, str, str):
    filename = img_path.split("/")[-1]
    prefix = filename.split(".")[0]
    suffix = filename.split(".")[-1]

    return filename, prefix, suffix


def filename_supplier_page(
    filename: str, prefix: str, suffix: str, pagenumber: int
) -> str:
    return f"{prefix}_{pagenumber:05}.{suffix}"


def s3_asset_path(
    ref_id: str, ref_type: str, include_prefix=False, include_filename=False
) -> str:
    """
    Create a path to store the assets for a given ref_id and ref_type
    The path is of the form s3://marie/{ref_type}/{prefix} and can be used between different marie instances

    All paths are lowercased and ref_type is cleaned to avoid path traversal attacks by replacing "/" with "_",

    Following are equivalent:

    .. code-block:: text

        s3://marie/ocr/sample
        s3://marie/OCR/sample
        s3://marie/ocr/SAMPLE
        s3://marie/OCR/SAMPLE


    Example usage:

    .. code-block:: python

        # this will return s3://marie/ocr/sample
        path = s3_asset_path(ref_id="sample.tif", ref_type="ocr")

        # this will return s3://marie/ocr/sample/sample
        path = s3_asset_path(ref_id="sample.tif", ref_type="ocr", include_prefix=True)

        # this will return s3://marie/ocr/sample/SAMple.tif
        path = s3_asset_path(ref_id="SAMple.tif", ref_type="ocr", include_filename=True)

    :param ref_type: type of the reference document
    :param ref_id:  id of the reference document
    :param include_prefix: include the filename prefix in the path(name of the file without extension)
    :param include_filename: include the filename in the path (name of the file with extension)
    :return: s3 path to store the assets
    """
    # prefix and filename need to be exclusive of each other
    assert not (include_prefix and include_filename)

    filename, prefix, suffix = split_filename(ref_id)
    # clean ref_type to avoid path traversal attacks
    ref_type = ref_type.replace("/", "_").lower()
    marie_bucket = os.environ.get("MARIE_S3_BUCKET", "marie")

    ret_path = f"s3://{marie_bucket}/{ref_type.lower()}/{prefix.lower()}"
    if include_prefix:
        ret_path = f"s3://{marie_bucket}/{ref_type.lower()}/{prefix.lower()}/{prefix}"

    if include_filename:
        ret_path = f"s3://{marie_bucket}/{ref_type.lower()}/{prefix.lower()}/{filename}"

    return ret_path


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
        self.logger = MarieLogger(context=self.__class__.__name__)

        # TODO : add support for dependency injection
        mock_ocr = False
        self.ocr_engines = dict()
        if mock_ocr:
            self.ocr_engines["document"] = MockOcrEngine(cuda=use_cuda)
            self.ocr_engines["region"] = MockOcrEngine(cuda=use_cuda)
        else:
            self.ocr_engines["document"] = DefaultOcrEngine(cuda=use_cuda)
            self.ocr_engines["region"] = VotingOcrEngine(cuda=use_cuda)

        self.overlay_processor = OverlayProcessor(
            work_dir=ensure_exists("/tmp/form-segmentation"), cuda=use_cuda
        )

        if pipeline_config is None:
            self.logger.warning("Pipeline config is None, using default config")
            pipeline_config = {}

        self.document_classifiers = self.setup_classifiers(pipeline_config)
        self.document_indexers = self.setup_indexers(pipeline_config)

        self.logger.info(
            f"Loaded classifiers : {len(self.document_classifiers)},  {self.document_classifiers.keys()}"
        )

        self.logger.info(
            f"Loaded indexers : {len(self.document_indexers)},  {self.document_indexers.keys()}"
        )

    def setup_indexers(self, pipeline_config) -> dict[str, any]:
        """
        Setup the document indexers(Named Entity Recognition)
        :param pipeline_config: pipeline configuration
        :return: document indexers
        """
        document_indexers = dict()
        configs = (
            pipeline_config["page_indexer"] if "page_indexer" in pipeline_config else []
        )

        for config in configs:
            if "model_name_or_path" not in config:
                raise BadConfigSource(
                    f"Missing model_name_or_path in indexer config : {config}"
                )

            if not config.get("enabled", True):
                self.logger.warning(
                    f"Skipping indexer : {config['model_name_or_path']}"
                )
                continue

            model_name_or_path = config["model_name_or_path"]
            device = config["device"] if "device" in config else "cpu"
            name = config["name"] if "name" in config else config["model_name_or_path"]
            model_type = config["type"] if "type" in config else "transformers"
            self.logger.info(f"Using model : {model_name_or_path} on device : {device}")

            if name in document_indexers:
                raise BadConfigSource(f"Duplicate indexer name : {name}")

            model_filter = config["filter"] if "filter" in config else {}
            # FIXME : we should not be using NerExtractionExecutor directly here
            if model_type == "transformers":
                document_indexers[name] = {
                    "indexer": NerExtractionExecutor(
                        model_name_or_path=model_name_or_path
                    ),
                    "filter": model_filter,
                }
            else:
                raise ValueError(f"Invalid indexer type : {model_type}")

        return document_indexers

    def setup_classifiers(self, pipeline_config: dict) -> dict[str, any]:
        """
        Setup the document classifiers
        :param pipeline_config:
        :return:
        """
        document_classifiers = dict()

        configs = (
            pipeline_config["page_classifier"]
            if "page_classifier" in pipeline_config
            else []
        )
        # classifier_configs =  pipeline_config["page_classifier"]
        for config in configs:
            if "model_name_or_path" not in config:
                raise BadConfigSource(
                    f"Missing model_name_or_path in classifier config : {config}"
                )

            if not config.get("enabled", True):
                self.logger.warning(
                    f"Skipping classifier : {config['model_name_or_path']}"
                )
                continue

            model_name_or_path = config["model_name_or_path"]
            device = config["device"] if "device" in config else "cpu"
            name = config["name"] if "name" in config else config["model_name_or_path"]
            model_type = config["type"] if "type" in config else "transformers"
            self.logger.info(f"Using model : {model_name_or_path} on device : {device}")

            if name in document_classifiers:
                raise BadConfigSource(f"Duplicate classifier name : {name}")

            if model_type == "transformers":
                document_classifiers[name] = TransformersDocumentClassifier(
                    model_name_or_path=model_name_or_path,
                    batch_size=1,
                    use_gpu=True,
                )
            else:
                raise ValueError(f"Invalid classifier type : {model_type}")

        return document_classifiers

    def segment(
        self,
        ref_id: str,
        frames: Union[np.ndarray, List[Image.Image]],
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

    def burst_frames(
        self,
        ref_id: str,
        frames: Union[np.ndarray, List[Image.Image]],
        root_asset_dir: str,
        force: bool = False,
    ) -> None:
        """
        Burst the frames and save them to the output directory
        :param ref_id:  reference id of the document
        :param frames:  frames to burst
        :param root_asset_dir:  root directory to store the burst frames
        :param force: force bursting
        :return:
        """
        output_dir = ensure_exists(os.path.join(root_asset_dir, "burst"))
        filename, prefix, suffix = split_filename(ref_id)
        filename_generator = partial(filename_supplier_page, filename, prefix, suffix)

        file_count = get_file_count(output_dir)
        self.logger.debug(
            f"Bursting filename : {filename}, prefix : {prefix}, suffix : {suffix}"
        )
        if force or file_count != len(frames):
            self.logger.info(f"Bursting frames for {ref_id}")
            burst_tiff_frames(frames, output_dir, filename_generator=filename_generator)
        else:
            self.logger.info(f"Skipping bursting for {ref_id}")

        # validate asset count
        file_count = get_file_count(output_dir)
        if file_count != len(frames):
            self.logger.warning(
                f"File count mismatch [burst] : {file_count} != {len(frames)}"
            )

    def ocr_frames(
        self,
        ref_id: str,
        frames: Union[List[np.ndarray], List[Image.Image]],
        root_asset_dir: str,
        force: bool = False,
        ps_mode: PSMode = PSMode.SPARSE,
        coord_format: CoordinateFormat = CoordinateFormat.XYWH,
        regions: [] = None,
        runtime_conf: Optional[dict[str, any]] = None,
    ) -> dict:
        """
        Perform OCR on the frames and return the results
        :param ref_id:  reference id of the document
        :param frames:  frames to perform OCR on
        :param root_asset_dir:  root directory to store the OCR results
        :param force:  force OCR (default: False)
        :param ps_mode:  page segmentation mode(default: Sparse)
        :param coord_format: coordinate format(default: XYWH)
        :param regions: regions to perform OCR on (default: None)
        :param runtime_conf: runtime configuration for the pipeline (e.g. which steps to execute) default is None.
        :return:  OCR results

        Example runtime_conf payload:

            .. code-block:: json
                  "features": [
                    {
                      "type": "pipeline",
                      "name": "default",
                      "ocr": {
                        "document": {
                          "engine": "default"
                        },
                        "region": {
                          "engine": "best"
                        }
                      }
                    }
                ]
        """

        output_dir = ensure_exists(os.path.join(root_asset_dir, "results"))
        filename, prefix, suffix = split_filename(ref_id)

        if runtime_conf is not None:
            ocr_runtime_config = runtime_conf.get("ocr", {})
            if "document" in ocr_runtime_config:
                self.logger.info(
                    f"Using document OCR engine : {ocr_runtime_config['document']['engine']}"
                )

        if regions and len(regions) > 0:
            json_path = os.path.join(output_dir, f"{prefix}.regions.json")
        else:
            json_path = os.path.join(output_dir, f"{prefix}.json")
        force = True
        if force or not os.path.exists(json_path):
            self.logger.debug(f"Performing OCR : {json_path}")
            ocr_engine = self.ocr_engines["document"]
            if regions and len(regions) > 0:
                ocr_engine = self.ocr_engines["region"]

            results = ocr_engine.extract(frames, ps_mode, coord_format, regions)
            store_json_object(results, json_path)
        else:
            self.logger.debug(f"Skipping OCR : {json_path}")
            results = load_json_file(json_path)

        return results

    def execute_frames_pipeline(
        self,
        ref_id: str,
        ref_type: str,
        frames: List,
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

        if False:
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
        self.restore_assets(
            ref_id, ref_type, root_asset_dir, full_restore=False, overwrite=True
        )

        # burst frames into individual images
        self.burst_frames(ref_id, frames, root_asset_dir)

        # make sure we have clean image clean frames are used for OCR and to generate clean document
        clean_frames = self.segment(ref_id, frames, root_asset_dir)
        ocr_results = self.ocr_frames(ref_id, clean_frames, root_asset_dir)

        metadata["ocr"] = ocr_results

        self.execute_pipeline(post_processing_pipeline, frames, ocr_results, metadata)

        # TODO : Convert to execution pipeline
        self.render_pdf(ref_id, frames, ocr_results, root_asset_dir)
        self.render_blobs(ref_id, frames, ocr_results, root_asset_dir)
        self.render_adlib(ref_id, frames, ocr_results, root_asset_dir)

        self.pack_assets(ref_id, ref_type, root_asset_dir, metadata)
        self.store_metadata(ref_id, ref_type, root_asset_dir, metadata)
        self.store_assets(ref_id, ref_type, root_asset_dir)

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

        results = self.ocr_frames(
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
        metadata = {}

        metadata["ref_id"] = ref_id
        metadata["ref_type"] = ref_type
        metadata["job_id"] = job_id
        metadata[
            "pages"
        ] = f"{len(frames)}"  # Using string to avoid type conversion issues
        metadata["ocr"] = results

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

    def store_assets(
        self, ref_id: str, ref_type: str, root_asset_dir: str
    ) -> List[str]:
        """
        Store assets in primary storage (S3)

        :param ref_id:  document reference id (e.g. filename)
        :param ref_type: document reference type (e.g. document, page, process)
        :param root_asset_dir: root asset directory where all assets are stored
        :return:
        """

        try:
            s3_asset_base = s3_asset_path(ref_id, ref_type)

            connected = StorageManager.ensure_connection(
                "s3://", silence_exceptions=True
            )
            if not connected:
                self.logger.error(f"Error storing assets : Could not connect to S3")
                return [s3_asset_base]

            # copy the files to s3
            StorageManager.copy_dir(
                root_asset_dir,
                s3_asset_base,
                relative_to_dir=root_asset_dir,
                match_wildcard="*",
            )

            return StorageManager.list(s3_asset_base, return_full_path=True)
        except Exception as e:
            self.logger.error(f"Error storing assets : {e}")

    def restore_assets(
        self,
        ref_id: str,
        ref_type: str,
        root_asset_dir: str,
        full_restore=False,
        overwrite=False,
    ) -> str or None:
        """
        Restore assets from primary storage (S3) into root asset directory. This restores
        the assets from the last run of the extrac pipeline.

        :param ref_id: document reference id (e.g. filename)
        :param ref_type: document reference type(e.g. document, page, process)
        :param root_asset_dir: root asset directory
        :param full_restore: if True, restore all assets, otherwise only restore subset of assets (clean, results, pdf)
        that are required for the extract pipeline.
        :param overwrite: if True, overwrite existing assets in root asset directory
        :return:
        """

        s3_root_path = s3_asset_path(ref_id, ref_type)
        connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
        if not connected:
            self.logger.error(f"Error restoring assets : Could not connect to S3")
            return None

        self.logger.info(f"Restoring assets from {s3_root_path} to {root_asset_dir}")

        if full_restore:
            try:
                StorageManager.copy_remote(
                    s3_root_path,
                    root_asset_dir,
                    match_wildcard="*",
                    overwrite=overwrite,
                )
            except Exception as e:
                self.logger.error(f"Error restoring assets : {e}")
        else:
            dirs_to_restore = ["clean", "results", "pdf"]
            for dir_to_restore in dirs_to_restore:
                try:
                    StorageManager.copy_remote(
                        s3_root_path,
                        root_asset_dir,
                        match_wildcard=f"*/{dir_to_restore}/*",
                        overwrite=overwrite,
                    )
                except Exception as e:
                    self.logger.error(f"Error restoring assets {dir_to_restore} : {e}")
        return s3_root_path

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
        context['metadata'] = metadata

        for pipe in processing_pipeline:
            try:
                # create a PipelineContext and pass it to the component
                self.logger.info(f"Executing component : {pipe}")
                pipe_results = pipe.run(documents, context, words=words, boxes=boxes)
                if pipe_results.state is not None:
                    if not isinstance(pipe_results.state, DocumentArray):
                        raise ValueError(
                            f"Invalid state type : {type(pipe_results.state)}"
                        )
                    documents = pipe_results.state
            except Exception as e:
                self.logger.error(f"Error executing pipe : {e}")
