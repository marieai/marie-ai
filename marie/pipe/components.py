import os
from collections import defaultdict
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from marie.boxes import PSMode
from marie.common.file_io import get_file_count
from marie.components.document_classifier import (
    TransformersDocumentLevelClassifier,
    TransformersPageLevelClassifier,
)
from marie.components.document_classifier.transformers import (
    TransformersSplittingClassifier,
)
from marie.components.document_indexer import TransformersDocumentIndexer
from marie.components.document_indexer.llm import MMLLMDocumentIndexer
from marie.components.document_registration.unilm_dit import (
    NoopDocumentBoundaryRegistration,
    UnilmDocumentBoundaryRegistration,
)
from marie.components.template_matching import (
    CompositeTemplateMatcher,
    MetaTemplateMatcher,
    VQNNFTemplateMatcher,
)
from marie.constants import __config_dir__, __model_path__
from marie.excepts import BadConfigSource
from marie.logging_core.predefined import default_logger as logger
from marie.ocr import CoordinateFormat, OcrEngine
from marie.overlay.overlay import NoopOverlayProcessor, OverlayProcessor
from marie.utils.asset_util import filename_supplier_page, split_filename
from marie.utils.image_utils import detect_page_rotation, ensure_max_page_size
from marie.utils.json import load_json_file, store_json_object
from marie.utils.tiff_ops import burst_tiff_frames
from marie.utils.types import strtobool
from marie.utils.utils import ensure_exists


def setup_overlay(
    pipeline_config: Optional[dict] = None,
    key: str = "page_overlay",
    device: str = "cuda",
) -> Union[OverlayProcessor, NoopOverlayProcessor]:
    """
    Setup the document overlay (Document cleanup) for the pipeline
    :param pipeline_config: pipeline configuration
    :param key: key to use in the pipeline config
    :param device: device to use for overlay (cpu or cuda)
    :return: document overlay processor or NoopOverlayProcessor if not enabled
    """
    use_cuda = True if device == "cuda" and torch.cuda.is_available() else False

    if pipeline_config is None:
        logger.warning("Pipeline config is None, using default config")
        pipeline_config = {}

    if key not in pipeline_config:
        logger.warning(f"Missing {key} in pipeline config, using default config")
        return OverlayProcessor(
            work_dir=ensure_exists("/tmp/form-segmentation"), cuda=use_cuda
        )

    config = pipeline_config[key] if key in pipeline_config else {}

    if "model_name_or_path" not in config:
        raise BadConfigSource(
            f"Missing model_name_or_path in page overlay config : {config}"
        )

    if not config.get("enabled", True):
        logger.warning(
            f"Page Overlay disabled (using NOOP): {config['model_name_or_path']}"
        )
        return NoopOverlayProcessor(
            work_dir=ensure_exists("/tmp/form-segmentation"), cuda=use_cuda
        )

    return OverlayProcessor(
        work_dir=ensure_exists("/tmp/form-segmentation"), cuda=use_cuda
    )


def setup_classifiers(
    pipeline_config: Optional[dict] = None,
    key: str = "page_classifier",
    device: str = "cuda",
    ocr_engine: Optional[OcrEngine] = None,
) -> dict[str, any]:
    """
    Setup the document classifiers (Document Classification) for the pipeline
    :param pipeline_config: pipeline configuration
    :param key: key to use in the pipeline config
    :param device: device to use for classification (cpu or cuda)
    :param ocr_engine: OCR engine to use for the pipeline (default: None)
    :return: map of document classifiers with their names as keys and classifier instances as values
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
        model_filter = config["filter"] if "filter" in config else {}
        max_pages = config["max_pages"] if "max_pages" in config else None

        if "group" not in config:
            raise BadConfigSource(f"Missing group in classifier config : {config}")

        group = config["group"] if "group" in config else "default"

        id2label = (
            config["id2label"] if "id2label" in config else id2label
        )  # Override id2label if provided in config

        logger.info(f"Using model : {model_name_or_path} on device : {device}")

        if group not in document_classifiers:
            document_classifiers[group] = dict()

        if name in document_classifiers[group]:
            raise BadConfigSource(
                f"Duplicate classifier name : {name} in group : {group}"
            )

        if model_type == "transformers":
            classifier = TransformersPageLevelClassifier(
                model_name_or_path=model_name_or_path,
                batch_size=batch_size,
                use_gpu=use_cuda,
                task=task,
                id2label=id2label,
                max_pages=max_pages,
            )
        elif model_type == "transformers_document_level":
            classifier = TransformersDocumentLevelClassifier(
                model_name_or_path=model_name_or_path,
                batch_size=batch_size,
                use_gpu=use_cuda,
                task=task,
                id2label=id2label,
                max_pages=max_pages,
            )
        elif model_type == "splitter":
            classifier = TransformersSplittingClassifier(
                model_name_or_path=model_name_or_path,
                batch_size=batch_size,
                use_gpu=use_cuda,
                task=task,
                id2label=id2label,
                max_pages=max_pages,
            )
        else:
            raise ValueError(f"Invalid classifier type : {model_type}")

        document_classifiers[group][name] = {
            "classifier": classifier,
            "group": group,
            "filter": model_filter,
            # "key": key
        }
    return document_classifiers


def setup_indexers(
    pipeline_config: Optional[dict] = None,
    key: str = "page_indexer",
    device: str = "cuda",
    ocr_engine: Optional[OcrEngine] = None,
) -> dict[str, any]:
    """
    Setup the document indexers(Named Entity Recognition) for the pipeline
    :param pipeline_config: pipeline configuration
    :param key: key to use in the pipeline config
    :param device: device to use for classification (cpu or cuda)
    :param ocr_engine: OCR engine to use for the pipeline (default: None)
    :return: document classifiers grouped by their group names and indexed by their names
    """

    if pipeline_config is None:
        logger.warning("Pipeline config is None, using default config")
        pipeline_config = {}

    document_indexers = dict()
    configs = pipeline_config[key] if key in pipeline_config else []

    for config in configs:
        if "model_name_or_path" not in config:
            raise BadConfigSource(f"Missing model_name_or_path in config : {config}")

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

        if "group" not in config:
            raise BadConfigSource(f"Missing group in indexer config : {config}")

        group = config["group"] if "group" in config else "default"

        if group not in document_indexers:
            document_indexers[group] = dict()

        model_filter = config["filter"] if "filter" in config else {}
        # TODO: Add support for other indexer types
        if model_type == "transformers":
            document_indexers[group][name] = {
                "indexer": TransformersDocumentIndexer(
                    model_name_or_path=model_name_or_path,
                    devices=[device],
                    ocr_engine=ocr_engine,
                ),
                "filter": model_filter,
                "group": group,
            }
        elif model_type == "mmllm":
            document_indexers[group][name] = {
                "indexer": MMLLMDocumentIndexer(
                    model_path=model_name_or_path,
                    devices=[device],
                ),
                "group": group,
            }
        else:
            raise ValueError(f"Invalid indexer type : {model_type}")

    return document_indexers


def setup_document_boundary(
    pipeline_config: Optional[dict] = None,
    key: str = "page_boundary",
    device: str = "cuda",
) -> Union[UnilmDocumentBoundaryRegistration, NoopDocumentBoundaryRegistration]:
    use_cuda = True if device == "cuda" and torch.cuda.is_available() else False

    if pipeline_config is None:
        logger.warning("Pipeline config is None, using default config")
        pipeline_config = {}

    if key not in pipeline_config:
        logger.warning(f"Missing {key} in pipeline config, using default config")
        return NoopDocumentBoundaryRegistration()

    config = pipeline_config[key] if key in pipeline_config else {}

    if "model_name_or_path" not in config:
        raise BadConfigSource(
            f"Missing model_name_or_path in document boundary config : {config}"
        )

    if not config.get("enabled", True):
        logger.warning(
            f"Page boundary disabled (using NOOP): {config['model_name_or_path']}"
        )
        return NoopDocumentBoundaryRegistration()

    return UnilmDocumentBoundaryRegistration(
        model_name_or_path=config["model_name_or_path"],
        use_gpu=use_cuda,
    )


def setup_template_matching(
    pipeline_config: Optional[dict] = None,
    key: str = "template_matcher",
    device: str = "cuda",
):
    if pipeline_config is None:
        logger.warning("Pipeline config is None, using default config")
        pipeline_config = {}

    if key not in pipeline_config:
        logger.warning(f"Missing {key} in pipeline config, using default config")
        return None, None

    config = pipeline_config[key] if key in pipeline_config else {}

    # if "model_name_or_path" not in config:
    #     raise BadConfigSource(
    #         f"Missing model_name_or_path in document template matching config : {config}"
    #     )

    if "definitions_path" not in config:
        raise BadConfigSource(
            f"Missing definitions_path in document template matching config : {config}"
        )
    resolved_definitions_path = os.path.join(__model_path__, config["definitions_path"])
    if not os.path.exists(resolved_definitions_path):
        raise BadConfigSource(
            f"Invalid definitions_path in document template matching config : {config}"
        )

    if not config.get("enabled", True):
        logger.warning(
            f"Template matching disabled (using NOOP): {config['definitions_path']}"
        )
        return None, None

    matcher_vqnnft = VQNNFTemplateMatcher(model_name_or_path="NONE")
    matcher_meta = MetaTemplateMatcher(model_name_or_path="NONE")
    matcher = CompositeTemplateMatcher(
        matchers=[matcher_meta, matcher_vqnnft], break_on_match=True
    )

    logger.info(
        f"Loaded template matching definitions from {resolved_definitions_path}"
    )
    return matcher, resolved_definitions_path


def ocr_frames(
    ocr_engines: dict[str, any],
    ref_id: str,
    frames: Union[List[np.ndarray], List[Image.Image]],
    root_asset_dir: str,
    queue_id: str = None,
    force: bool = False,
    ps_mode: PSMode = PSMode.SPARSE,
    coord_format: CoordinateFormat = CoordinateFormat.XYWH,
    regions: [] = None,
    runtime_conf: Optional[dict[str, any]] = None,
    engine_name: str = "default",
) -> dict:
    """
    Perform OCR on the frames and return the results
    :param ocr_engines: OCR engines to use
    :param ref_id:  reference id of the document
    :param frames:  frames to perform OCR on
    :param root_asset_dir:  root directory to store the OCR results
    :param queue_id: queue identifier used to isolate OCR engine temp/debug paths
    :param force:  force OCR (default: False)
    :param ps_mode:  page segmentation mode(default: Sparse)
    :param coord_format: coordinate format(default: XYWH)
    :param regions: regions to perform OCR on (default: None)
    :param runtime_conf: runtime configuration for the pipeline (e.g. which steps to execute) default is None.
    :param engine_name: OCR engine to use (default: default)
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

    engine = ocr_engines[engine_name]
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
        logger.info(f"Performing OCR : {json_path}")
        results = engine.extract(
            frames,
            ps_mode,
            coord_format,
            regions,
            queue_id=queue_id,
        )
        store_json_object(results, json_path)
    else:
        logger.debug(f"Skipping OCR : {json_path}")
        results = load_json_file(json_path)

    return results


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
    # NOTE: filename = prefix = refId when refId is not a filename
    filename, prefix, suffix = split_filename(ref_id)
    suffix = suffix or "tif"

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


def rotate_frames(
    ref_id: str,
    frames: List[np.ndarray],
    root_asset_dir: str,
    force: bool = False,
) -> tuple[dict, bool]:
    """Rotate frames in-place based on detected orientation"""
    output_dir = ensure_exists(os.path.join(root_asset_dir, "rotation"))
    _, prefix, _ = split_filename(ref_id)
    json_path = os.path.join(output_dir, f"{prefix}_rotation.json")

    any_rotated = False
    if force or not os.path.exists(json_path):
        page_rotation = dict()
        for i, frame in enumerate(frames):
            results = detect_page_rotation(frame)
            if rotation := results.get("rotate", 0):
                logger.info(f"Rotating frame {i} by {rotation} degrees")
                frames[i] = np.rot90(frame, k=-rotation // 90)
                changed, resized = ensure_max_page_size([frames[i]])
                if changed:
                    logger.info(f"Resized frame {i} after rotation")
                    frames[i] = resized[0]
                any_rotated = True
            page_rotation[i] = results
        results = {"any_rotated": any_rotated, "pages": page_rotation}
        store_json_object(results, json_path)
    else:
        # TODO we only rotate the burst tiff not the multipage tiff, so we need to call with force
        logger.info(f"Skipping Rotation : {json_path}")
        results = load_json_file(json_path)

    return results, any_rotated


def update_existing_meta(existing_meta: dict, metadata: dict):
    if not existing_meta:
        return metadata
    if not metadata:
        return existing_meta

    # List elements are overridden on dict.update, so they need to be merged independently
    # New metadata lists take priority when handling duplicate elements with the same identifier value
    meta_lists = [("classifications", "group"), ("indexers", "group")]
    merged_meta_lists = dict()
    for category, identifier in meta_lists:
        existing_list = existing_meta.get(category, [])
        new_list = metadata.get(category, [])

        # Determine if existing keys are stale
        existing_keys = {unit[identifier]: False for unit in existing_list}
        for unit in new_list:
            existing_keys[unit[identifier]] = unit[identifier] in existing_keys
        # Filter out stale categories
        merged_list = [
            unit for unit in existing_list if not existing_keys[unit[identifier]]
        ]
        merged_list.extend(new_list)

        if merged_list:
            merged_meta_lists[category] = merged_list

    # Merge metas (prioritize new metadata)
    existing_meta.update(metadata)
    existing_meta.update(merged_meta_lists)

    return existing_meta


def setup_llm_tasks(pipeline_config, document_indexers):
    """
    LLM Tasks are the overall objectives of each LLM indexer. Each indexer is responsible for HOW
    it will accomplish each task listed in the pipeline config, but at minimum each indexer must
    define a task with for each 'llm_task' in the  pipeline_config.
    """
    if "llm_tasks" not in pipeline_config:
        return dict()

    tasks = pipeline_config["llm_tasks"]

    document_llm_tasks = defaultdict(list)
    for task in tasks:
        if "name" not in task:
            raise BadConfigSource(f"Missing name in llm_tasks config : {task}")

        name = task["name"]
        group = task.get("group", "default")

        if group not in document_indexers:
            raise BadConfigSource(f"Unknown Group: {group}")

        for indexer_name, indexer_def in document_indexers[group].items():
            indexer = indexer_def["indexer"]
            if not isinstance(indexer, MMLLMDocumentIndexer):
                continue

            if name not in indexer.task_map:
                raise BadConfigSource(
                    f"Indexer: '{indexer_name}' does not have llm_task: '{name}' defined. "
                    f"Model path: {indexer.model_path}"
                )

        document_llm_tasks[group].append(name)

    return document_llm_tasks


def load_pipeline(
    pipeline_config: dict[str, any], ocr_engine: Optional[OcrEngine] = None
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

    document_llm_tasks = setup_llm_tasks(pipeline_config, document_indexers)

    indexer_groups = dict()
    for group, indexers in document_indexers.items():
        indexer_groups[group] = {
            "group": group,
            "indexers": indexers,
        }
        if group in document_llm_tasks:
            indexer_groups[group]["llm_tasks"] = document_llm_tasks[group]

    # dump information about the loaded classifiers that are grouped by the classifier group
    for classifier_group, classifiers in document_classifiers.items():
        logger.info(
            f"Loaded classifiers :{classifier_group},  {len(classifiers)},  {classifiers.keys()}"
        )
    for classifier_group, classifiers in document_sub_classifiers.items():
        logger.info(
            f"Loaded sub-classifiers : {classifier_group}, {len(classifiers)},  {classifiers.keys()}"
        )

    for indexer_group, indexers in document_indexers.items():
        logger.info(
            f"Loaded indexers : {indexer_group}, {len(indexers)},  {indexers.keys()}"
        )
    for llm_group, tasks in document_llm_tasks.items():
        logger.info(f"Registered LLM tasks : {llm_group}, {len(tasks)}, {tasks}")

    return pipeline_name, classifier_groups, indexer_groups
