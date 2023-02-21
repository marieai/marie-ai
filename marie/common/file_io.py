import os
from os import PathLike
from pathlib import Path

import portalocker

__all__ = ["get_cache_dir", "file_lock", "get_file_count"]

from typing import Optional, Union

StrOrBytesPath = Union[str, Path, PathLike]


def get_cache_dir(cache_dir: Optional[StrOrBytesPath] = None) -> StrOrBytesPath:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $MARIE_CACHE, if set
        2) MARIE_CACHE ~/.marie
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.getenv("MARIE_CACHE", "~/.marie"))
    return cache_dir


def file_lock(path: StrOrBytesPath):  # type: ignore
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.

    This is useful to make sure workers don't cache files to the same location.

    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`

    Examples:

        filename = "/path/to/file"
        with file_lock(filename):
            if not os.path.isfile(filename):
                do_create_file()
    """
    dirname = os.path.dirname(path)
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        # makedir is not atomic. Exceptions can happen when multiple workers try
        # to create the same dir, despite exist_ok=True.
        # When this happens, we assume the dir is created and proceed to creating
        # the lock. If failed to create the directory, the next line will raise
        # exceptions.
        pass
    return portalocker.Lock(path + ".lock", timeout=1800)  # type: ignore


def get_file_count(path: str) -> int:
    return len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    )


#
#
# # Install default handlers
# PathManager.register_handler(HTTPURLHandler())
