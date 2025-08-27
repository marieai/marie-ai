import os
import time
import urllib.error
from pathlib import Path

import numpy as np
import torch
from docarray import DocList

from marie.api import AssetKeyDoc
from marie.constants import __model_path__
from marie.excepts import RuntimeFailToStart
from marie.importer import ImportExtensions
from marie.logging_core.predefined import default_logger as logger
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.image_utils import ensure_max_page_size
from marie.utils.types import strtobool


def setup_cache(list_of_models: list[dict[str, str]] = None) -> None:
    """Setup cathe and download known models for faster startup"""
    logger.info("Setting up cache for models, this may take a while.")
    if strtobool(os.environ.get("MARIE_CACHE_SKIP_LOAD", False)):
        logger.warning("Skipping cache setup.")
        return

    MARIE_CACHE_LOCK_TIMEOUT = os.environ.get("MARIE_CACHE_LOCK_TIMEOUT", None)

    try:
        with ImportExtensions(
            required=True,
            help_text=f"FileLock is needed to guarantee single initialization.",
        ):
            import filelock

            # this is needed to guarantee single initialization of as the models accessed by multiple processes
            locks_root = Path(os.path.join(__model_path__, "cache"))
            lock_file = locks_root.joinpath(f"models.lock")

            if lock_file.exists():
                logger.warning(
                    f"Lock file exists, checking if it is stale : {lock_file}"
                )
                last_modification = lock_file.stat().st_mtime
                stale_time_in_sec = 15 * 60

                if (time.time() - last_modification) > stale_time_in_sec:
                    logger.info("Removing stale lock file")
                    lock_file.unlink()

            max_time_to_wait = 5 * 60
            if MARIE_CACHE_LOCK_TIMEOUT:
                max_time_to_wait = int(MARIE_CACHE_LOCK_TIMEOUT)

            file_lock = filelock.FileLock(lock_file, timeout=max_time_to_wait)
            with file_lock:
                from torch.hub import _get_torch_home

                torch_home = os.environ.get("TORCH_HOME", None)
                torch.hub.set_dir(torch_home)
                torch_cache_home = _get_torch_home()
                logger.info(f"Setting up cache for models as {torch_cache_home}")

                if list_of_models is None:
                    list_of_models = [
                        {"name": "pytorch/fairseq:main", "model": "roberta.large"},
                        {"name": "pytorch/fairseq:main", "model": "roberta.base"},
                    ]

                for item in list_of_models:
                    logger.info(f"Caching model {item['name']} > {item['model']}")
                    torch.hub.load(item["name"], item["model"])

            # https://github.com/tox-dev/py-filelock/issues/31
            if lock_file.exists():
                logger.info(f"Removing lock file : {lock_file}")
                lock_file.unlink()

    except urllib.error.HTTPError as err:
        logger.error(f"Unable to download model : {err.code}")
        raise RuntimeFailToStart(f"Unable to download model : {err}")
    except filelock.Timeout as err:
        logger.error(f"Unable to acquire lock on {lock_file}")
    except Exception as e:
        logger.error(f"Error setting up cache : {e}")


def parse_parameters(parameters: dict, strict: bool = True) -> tuple:
    """
    Parses a dictionary of parameters and extracts relevant information such as job_id, ref_id, ref_type, queue_id,
    and payload. This function includes error handling for missing values based on the `strict` flag.

    Args:
        parameters (dict): A dictionary of parameters containing keys like 'job_id', 'ref_id', 'ref_type',
                           'queue_id', and 'payload'.
        strict (bool, optional): If set to True, raises a ValueError when required parameters are missing.
                                 Defaults to True.

    Returns:
        tuple: A tuple containing extracted values in the following order:
               (job_id, ref_id, ref_type, queue_id, payload).

    Raises:
        ValueError: If a required parameter is missing and the `strict` flag is set to True.
    """
    if parameters is None or "job_id" not in parameters:
        logger.error(f"Job ID is not present in parameters")
        if strict:
            raise ValueError("Job ID is not present in parameters")

    job_id = parameters.get("job_id", "0000-0000-0000-0000")

    logger.debug("Parsing Parameters")
    for key, value in parameters.items():
        logger.debug("The value of {} is {}".format(key, value))

    ref_id = parameters.get("ref_id")
    if ref_id is None and strict:
        raise ValueError("ref_id is not present in parameters")
    ref_type = parameters.get("ref_type", "not_defined")
    queue_id: str = parameters.get("queue_id", "0000-0000-0000-0000")

    payload = parameters.get("payload")
    if payload is None:
        logger.error("Empty Payload")
        if strict:
            raise ValueError("Empty Payload")

    return job_id, ref_id, ref_type, queue_id, payload


def get_frames_from_docs(
    docs: DocList[AssetKeyDoc], pages: list[int] = None
) -> list[np.ndarray]:
    """
    Extracts and processes frames from a single document.

    This function is responsible for extracting frames from a provided document, ensuring
    that the document adheres to specific constraints (only a single document is supported).
    It checks and retrieves the document's frames from specific pages if specified, or all pages
    if none are given. Additionally, it ensures that the frames comply with a maximum page size
    constraint, adjusting their size as necessary and logging relevant warnings.

    Parameters:
        docs (DocList[AssetKeyDoc]): A list containing a single document from which to extract frames.
        pages (list[int], optional): A list of page indices to extract frames from. If not provided,
                                     frames from all pages of the document will be extracted.

    Returns:
        list[numpy.ndarray]: A list of processed frames extracted from the specified pages of the
                             document.

    Raises:
        ValueError: If no documents are found in the input, or if multiple documents are provided.
    """
    if len(docs) == 0:
        raise ValueError("Expected single document. No documents found")
    if len(docs) > 1:
        raise ValueError("Expected single document. Multiple documents found.")

    doc: AssetKeyDoc = docs[0]
    logger.debug(f"Document asset key: {doc.asset_key}")
    pages = doc.pages if pages is None else pages
    docs = docs_from_asset(doc.asset_key, pages)
    src_frames = frames_from_docs(docs)
    changed, frames = ensure_max_page_size(src_frames)
    if changed:
        logger.warning(f"Page size of frames was changed ")
        for i, (s, f) in enumerate(zip(src_frames, frames)):
            logger.warning(f"Frame[{i}] changed : {s.shape} -> {f.shape}")

    return frames


def get_payload_features(
    payload,
    name=None,
    f_type=None,
) -> list:
    if "features" not in payload:
        return []

    features = []
    for feature in payload["features"]:
        if not isinstance(feature, dict):
            continue
        if name and feature.get("name") != name:
            continue
        if f_type and feature.get("type") != f_type:
            continue
        features.append(feature)
    return features
