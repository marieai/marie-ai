import os
import time
import urllib.error
from pathlib import Path

import torch

from marie.constants import __model_path__
from marie.excepts import RuntimeFailToStart
from marie.importer import ImportExtensions
from marie.logging_core.predefined import default_logger as logger
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
