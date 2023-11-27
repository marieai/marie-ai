import os
from functools import partial
from typing import Optional, Union, List

import numpy as np
import torch
from PIL import Image

from marie.boxes import PSMode, BoxProcessorUlimDit
from marie.common.file_io import get_file_count
from marie.components import TransformersDocumentClassifier
from marie.document import TrOcrProcessor
from marie.excepts import BadConfigSource
from marie.executor.ner import NerExtractionExecutor
from marie.executor.util import setup_cache
from marie.logging.predefined import default_logger as logger
from marie.ocr import CoordinateFormat, MockOcrEngine, DefaultOcrEngine, VotingOcrEngine
from marie.storage import StorageManager
from marie.utils.json import store_json_object, load_json_file
from marie.utils.tiff_ops import burst_tiff_frames
from marie.utils.types import strtobool
from marie.utils.utils import ensure_exists


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


def get_known_ocr_engines(device: str = "cuda", engine: str = None) -> dict[str, any]:
    """
    Get the known OCR engines
    mock : Mock OCR engine, returns dummy results
    default : Default OCR engine, uses the best OCR engine available on the system
    best : Voting OCR engine, uses ensemble of OCR engines to perform OCR on the document

    Most GPU will not have enough memory to run multiple OCR engines in parallel and hence it is recommended to use
    the default OCR engine on GPU. If you have a large GPU with enough memory, you can use the best OCR engine.

    :param device: device to use for OCR (cpu or cuda)
    :param engine: engine to use for OCR (mock, default, best)
    :return: OCR engines
    """

    use_cuda = False
    if device == "cuda":
        use_cuda = True

    logger.info(f"Getting OCR engine using engine : {engine}, device : {device}")
    setup_cache(list_of_models=None)

    box_processor = BoxProcessorUlimDit(
        work_dir=ensure_exists("/tmp/boxes"),
        cuda=use_cuda,
    )

    trocr_processor = TrOcrProcessor(work_dir=ensure_exists("/tmp/icr"), cuda=use_cuda)

    ocr_engines = dict()

    if engine is None:
        ocr_engines["mock"] = MockOcrEngine(cuda=use_cuda, box_processor=box_processor)
        ocr_engines["default"] = DefaultOcrEngine(
            cuda=use_cuda,
            box_processor=box_processor,
            default_ocr_processor=trocr_processor,
        )
        ocr_engines["best"] = VotingOcrEngine(
            cuda=use_cuda,
            box_processor=box_processor,
            default_ocr_processor=trocr_processor,
        )
    elif engine == "mock":
        ocr_engines["mock"] = MockOcrEngine(cuda=use_cuda, box_processor=box_processor)
    elif engine == "default":
        ocr_engines["default"] = DefaultOcrEngine(
            cuda=use_cuda,
            box_processor=box_processor,
            default_ocr_processor=trocr_processor,
        )
    elif engine == "best":
        ocr_engines["best"] = VotingOcrEngine(
            cuda=use_cuda,
            box_processor=box_processor,
            default_ocr_processor=trocr_processor,
        )
    else:
        raise ValueError(f"Invalid OCR engine : {engine}")

    return ocr_engines


def setup_classifiers(
    pipeline_config: Optional[dict] = None,
    key: str = "page_classifier",
    device: str = "cuda",
) -> dict[str, any]:
    """
    Setup the document classifiers (Document Classification) for the pipeline
    :param pipeline_config: pipeline configuration
    :param key: key to use in the pipeline config
    :param device: device to use for classification (cpu or cuda)
    :return: document classifiers
    """
    use_cuda = True if device == "cuda" and torch.cuda.is_available() else False

    if pipeline_config is None:
        logger.warning("Pipeline config is None, using default config")
        pipeline_config = {}

    document_classifiers = dict()

    configs = pipeline_config[key] if key in pipeline_config else []
    for config in configs:
        if "model_name_or_path" not in config:
            raise BadConfigSource(
                f"Missing model_name_or_path in classifier config : {config}"
            )

        if not config.get("enabled", True):
            logger.warning(f"Skipping classifier : {config['model_name_or_path']}")
            continue

        id2label = (
            pipeline_config["id2label"] if "id2label" in pipeline_config else None
        )

        model_name_or_path = config["model_name_or_path"]
        device = config["device"] if "device" in config else "cpu"
        name = config["name"] if "name" in config else config["model_name_or_path"]
        model_type = config["type"] if "type" in config else "transformers"
        task = config["task"] if "task" in config else "text-classification"
        batch_size = config["batch_size"] if "batch_size" in config else 1
        id2label = (
            config["id2label"] if "id2label" in config else id2label
        )  # Override id2label if provided in config

        logger.info(f"Using model : {model_name_or_path} on device : {device}")

        if name in document_classifiers:
            raise BadConfigSource(f"Duplicate classifier name : {name}")

        if model_type == "transformers":
            document_classifiers[name] = TransformersDocumentClassifier(
                model_name_or_path=model_name_or_path,
                batch_size=batch_size,
                use_gpu=use_cuda,
                task=task,
                id2label=id2label,
            )
        else:
            raise ValueError(f"Invalid classifier type : {model_type}")

    return document_classifiers


def setup_indexers(pipeline_config: Optional[dict] = None) -> dict[str, any]:
    """
    Setup the document indexers(Named Entity Recognition)
    :param pipeline_config: pipeline configuration
    :return: document indexers
    """

    if pipeline_config is None:
        logger.warning("Pipeline config is None, using default config")
        pipeline_config = {}

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
            logger.warning(f"Skipping indexer : {config['model_name_or_path']}")
            continue

        model_name_or_path = config["model_name_or_path"]
        device = config["device"] if "device" in config else "cpu"
        name = config["name"] if "name" in config else config["model_name_or_path"]
        model_type = config["type"] if "type" in config else "transformers"
        logger.info(f"Using model : {model_name_or_path} on device : {device}")

        if name in document_indexers:
            raise BadConfigSource(f"Duplicate indexer name : {name}")

        model_filter = config["filter"] if "filter" in config else {}
        # FIXME : we should not be using NerExtractionExecutor directly here
        if model_type == "transformers":
            document_indexers[name] = {
                "indexer": NerExtractionExecutor(model_name_or_path=model_name_or_path),
                "filter": model_filter,
            }
        else:
            raise ValueError(f"Invalid indexer type : {model_type}")

    return document_indexers


def restore_assets(
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
        logger.error(f"Error restoring assets : Could not connect to S3")
        return None

    logger.info(f"Restoring assets from {s3_root_path} to {root_asset_dir}")

    if full_restore:
        try:
            StorageManager.copy_remote(
                s3_root_path,
                root_asset_dir,
                match_wildcard="*",
                overwrite=overwrite,
            )
        except Exception as e:
            logger.error(f"Error restoring assets : {e}")
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
                logger.error(f"Error restoring assets {dir_to_restore} : {e}")
    return s3_root_path


def store_assets(
    ref_id: str, ref_type: str, root_asset_dir: str, match_wildcard: Optional[str] = "*"
) -> List[str]:
    """
    Store assets in primary storage (S3)

    :param ref_id:  document reference id (e.g. filename)
    :param ref_type: document reference type (e.g. document, page, process)
    :param root_asset_dir: root asset directory where all assets are stored
    :param match_wildcard: wildcard to match files to store
    :return:
    """

    try:
        s3_asset_base = s3_asset_path(ref_id, ref_type)
        connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
        if not connected:
            logger.error(f"Error storing assets : Could not connect to S3")
            return [s3_asset_base]

        # copy the files to s3
        StorageManager.copy_dir(
            root_asset_dir,
            s3_asset_base,
            relative_to_dir=root_asset_dir,
            match_wildcard=match_wildcard,
        )

        return StorageManager.list(s3_asset_base, return_full_path=True)
    except Exception as e:
        logger.error(f"Error storing assets : {e}")


def burst_frames(
    ref_id: str,
    frames: List[np.ndarray],
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
    logger.debug(
        f"Bursting filename : {filename}, prefix : {prefix}, suffix : {suffix}"
    )
    if force or file_count != len(frames):
        logger.info(f"Bursting frames for {ref_id}")
        burst_tiff_frames(frames, output_dir, filename_generator=filename_generator)
    else:
        logger.info(f"Skipping bursting for {ref_id}")

    # validate asset count
    file_count = get_file_count(output_dir)
    if file_count != len(frames):
        logger.warning(f"File count mismatch [burst] : {file_count} != {len(frames)}")


def ocr_frames(
    ocr_engines: dict[str, any],
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
    :param ocr_engines: OCR engines to use
    :param ref_id:  reference id of the document
    :param frames:  frames to perform OCR on
    :param root_asset_dir:  root directory to store the OCR results
    :param force:  force OCR (default: False)
    :param ps_mode:  page segmentation mode(default: Sparse)
    :param coord_format: coordinate format(default: XYWH)
    :param regions: regions to perform OCR on (default: None)
    :param runtime_conf: runtime configuration for the pipeline (e.g. which steps to execute) default is None.
    :return: OCR results

    Example runtime_conf payload:

        .. code-block:: json
              "features": [
                {
                  "type": "pipeline",
                  "name": "default",
                  "ocr": {
                    "document": {
                      "engine": "default|best|google|amazon|azure|mock"
                      "force": true,
                    },
                    "region": {
                      "engine": "default|best|google|amazon|azure|mock",
                      "force": true,
                    }
                  }
                }
            ]
    """

    output_dir = ensure_exists(os.path.join(root_asset_dir, "results"))
    filename, prefix, suffix = split_filename(ref_id)

    engine = ocr_engines["default"]
    if regions and len(regions) > 0:
        engine = ocr_engines["best"]

    if runtime_conf is not None:
        ocr_runtime_config = runtime_conf.get("ocr", {})

        node = "document"
        if "document" in ocr_runtime_config:
            node = "document"
        elif "region" in ocr_runtime_config:
            node = "region"

        if node in ocr_runtime_config:
            if "engine" in ocr_runtime_config[node]:
                engine_name = ocr_runtime_config[node]["engine"]
                if engine_name in ocr_engines:
                    engine = ocr_engines[engine_name]
                else:
                    logger.warning(f"Invalid OCR engine : {engine_name}, using default")
            # check if we need to force OCR
            if "force" in ocr_runtime_config[node]:
                force = strtobool(ocr_runtime_config[node]["force"])

    if regions and len(regions) > 0:
        json_path = os.path.join(output_dir, f"{prefix}.regions.json")
    else:
        json_path = os.path.join(output_dir, f"{prefix}.json")

    if force or not os.path.exists(json_path):
        logger.debug(f"Performing OCR : {json_path}")
        results = engine.extract(frames, ps_mode, coord_format, regions)
        store_json_object(results, json_path)
    else:
        logger.debug(f"Skipping OCR : {json_path}")
        results = load_json_file(json_path)

    return results
