import os
from pathlib import Path

import torch

from marie.constants import (
    __model_path__,
)
from marie.excepts import RuntimeFailToStart
from marie.importer import ImportExtensions
from marie.logging.predefined import default_logger as logger
from marie.utils.types import strtobool


def setup_cache(list_of_models: list[dict[str, str]] = None) -> None:
    """Setup cathe and download known models for faster startup"""
    import urllib.error

    if strtobool(os.environ.get("MARIE_SKIP_MODEL_CACHE", False)):
        logger.warning("Skipping cache setup.")
        return

    try:
        with ImportExtensions(
            required=True,
            help_text=f"FileLock is needed to guarantee single initialization.",
        ):
            import filelock

            # this is needed to guarantee single initialization of as the models accessed by multiple processes
            locks_root = Path(os.path.join(__model_path__, "cache"))
            lock_file = locks_root.joinpath(f"models.lock")
            file_lock = filelock.FileLock(lock_file, timeout=-1)
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
    except urllib.error.HTTPError as err:
        logger.error(f"Unable to download model : {err.code}")
        raise RuntimeFailToStart(f"Unable to download model : {err}")
    except Exception as e:
        logger.error(f"Error setting up cache : {e}")
