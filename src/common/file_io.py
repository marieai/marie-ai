# Implementation based on fvcore

import errno
import logging
import os
import shutil
from urllib.parse import urlparse

from collections import OrderedDict
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Union,
)

from common.download import download

__all__ = ["PathManager", "get_cache_dir", "file_lock", "VolumeHandler"]


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
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


def file_lock(path: str):  # type: ignore
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


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    _strict_kwargs_check = True

    def _check_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.

        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError("Unused arguments: {}".format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning("{}={} argument ignored".format(k, v))

    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, this function is meant to be
        used with read-only resources.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def _copy(
        self,
        src_path: str,
        dst_path: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        self._check_kwargs(kwargs)
        return path

    def _exists(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.exists(path)


class HTTPURLHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    def __init__(self) -> None:
        self.cache_map: Dict[str, str] = {}

    def _get_supported_prefixes(self) -> List[str]:
        return ["http://", "https://", "ftp://"]

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        self._check_kwargs(kwargs)
        if path not in self.cache_map or not os.path.exists(self.cache_map[path]):
            logger = logging.getLogger(__name__)
            parsed_url = urlparse(path)
            dirname = os.path.join(get_cache_dir(), os.path.dirname(parsed_url.path.lstrip("/")))
            filename = path.split("/")[-1]
            cached = os.path.join(dirname, filename)
            with file_lock(cached):
                if not os.path.isfile(cached):
                    logger.info("Downloading {} ...".format(path))
                    cached = download(path, dirname, filename=filename)
            logger.info("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        return self.cache_map[path]


class PathManager:
    """
    A class for users to open generic paths or translate generic paths to file names.
    """

    _PATH_HANDLERS: MutableMapping[str, PathHandler] = OrderedDict()
    _NATIVE_PATH_HANDLER = NativePathHandler()

    @staticmethod
    def __get_path_handler(path: str) -> PathHandler:
        """
        Finds a PathHandler that supports the given path. Falls back to the native
        PathHandler if no other handler is found.

        Args:
            path (str): URI path to resource

        Returns:
            handler (PathHandler)
        """
        for p in PathManager._PATH_HANDLERS.keys():
            if path.startswith(p):
                return PathManager._PATH_HANDLERS[p]
        return PathManager._NATIVE_PATH_HANDLER

    @staticmethod
    def register_handler(handler: PathHandler) -> None:
        """
        Register a path handler associated with `handler._get_supported_prefixes`
        URI prefixes.

        Args:
            handler (PathHandler)
        """
        assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            assert prefix not in PathManager._PATH_HANDLERS
            PathManager._PATH_HANDLERS[prefix] = handler

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        PathManager._PATH_HANDLERS = OrderedDict(
            sorted(
                PathManager._PATH_HANDLERS.items(),
                key=lambda t: t[0],
                reverse=True,
            )
        )

    @staticmethod
    def set_strict_kwargs_checking(enable: bool) -> None:
        """
        Toggles strict kwargs checking. If enabled, a ValueError is thrown if any
        unused parameters are passed to a PathHandler function. If disabled, only
        a warning is given.

        With a centralized file API, there's a tradeoff of convenience and
        correctness delegating arguments to the proper I/O layers. An underlying
        `PathHandler` may support custom arguments which should not be statically
        exposed on the `PathManager` function. For example, a custom `HTTPURLHandler`
        may want to expose a `cache_timeout` argument for `open()` which specifies
        how old a locally cached resource can be before it's refetched from the
        remote server. This argument would not make sense for a `NativePathHandler`.
        If strict kwargs checking is disabled, `cache_timeout` can be passed to
        `PathManager.open` which will forward the arguments to the underlying
        handler. By default, checking is enabled since it is innately unsafe:
        multiple `PathHandler`s could reuse arguments with different semantic
        meanings or types.

        Args:
            enable (bool)
        """
        PathManager._NATIVE_PATH_HANDLER._strict_kwargs_check = enable
        for handler in PathManager._PATH_HANDLERS.values():
            handler._strict_kwargs_check = enable

    @staticmethod
    def get_local_path(path: str, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        return PathManager.__get_path_handler(path)._get_local_path(path, **kwargs)  # type: ignore

    @staticmethod
    def exists(path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        return PathManager.__get_path_handler(path)._exists(path, **kwargs)  # type: ignore


PathManager.register_handler(HTTPURLHandler())
