from typing import MutableMapping, OrderedDict, Dict, Any, List, Optional
import logging
import os

from marie import __model_path__

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


class ModelRegistryHandler:
    """
    ModelRegistryHandler is a base class that defines common functionality for a URI protocol.
    It routes I/O for a generic URI which may look like "protocol://*"
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
            List[str]: the list of URI prefixes this ModelRegistryHandler can support
        """
        raise NotImplementedError()

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Check if the model exists locally
        Returns:
            bool: true if the model exists, false otherwise
        """
        raise NotImplementedError()

    def _get_local_path(self, _name_or_path: str, **kwargs: Any) -> str:
        """
        Get the local path for the provider name or path
        Returns:
            str: the local path to the model directory
        """
        raise NotImplementedError()

    def _discover(self) -> Dict[str, str]:
        """
        Discover models that this provider can handle
        Returns:
            Dict[str,str]: Dictionary of model name and location path
        """
        raise NotImplementedError()


class NativeModelRegistryHandler(ModelRegistryHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.

    This is the default model zoo handler and we will
    """

    def _discover(self) -> Dict[str, str]:
        print(__model_path__)

        for root, dir, files in os.walk(__model_path__):
            basepath = os.path.basename(root)
            path = os.path.relpath(root, basepath).split(os.sep)
            print((len(path) - 1) * "---", os.path.basename(root))
            for file in files:
                print(len(path) * "---", file)
                if file == "marie.json":
                    print(f"Found : {file}")

    def _get_supported_prefixes(self) -> List[str]:
        return ["file://"]

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        self._check_kwargs(kwargs)
        return path

    def _exists(self, _name_or_path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.exists(_name_or_path)


class ModelRegistry:
    """
    A class for users to access models from specific providers

        Handlers:
            zoo  : Local model zoo(native file access)
            git  : Git
            S3   : Amazon S3
            mflow: Mflow support
            transformers: Transformers model
    """

    _PATH_HANDLERS: MutableMapping[str, ModelRegistryHandler] = OrderedDict()
    _NATIVE_PATH_HANDLER = NativeModelRegistryHandler()

    @staticmethod
    def __get_path_handler(_name_or_path: str) -> ModelRegistryHandler:
        """
        Finds a ModelRegistryHandler that supports the given protocol path. Falls back to the native
        ModelRegistryHandler if no other handler is found.

        Args:
            _name_or_path (str): URI path to resource

        Returns:
            handler (PathHandler)
        """
        for p in ModelRegistry._PATH_HANDLERS.keys():
            if _name_or_path.startswith(p):
                return ModelRegistry._PATH_HANDLERS[p]
        return ModelRegistry._NATIVE_PATH_HANDLER

    def __init__(self, **kwargs):
        pass

    def load_model(self):
        pass

    def discover(self):
        for p in ModelRegistry._PATH_HANDLERS.keys():
            handler = ModelRegistry._PATH_HANDLERS[p]
            discovered = handler._discover()
            print(discovered)

    @staticmethod
    def get_local_path(_name_or_path: str, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            _name_or_path (str): A URI supported by this ModelRegistryHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """

        return ModelRegistry.__get_path_handler(_name_or_path)._get_local_path(_name_or_path, **kwargs)  # type: ignore

    @staticmethod
    def register_handler(handler: ModelRegistryHandler) -> None:
        """
        Register a path handler associated with `handler._get_supported_prefixes`
        URI prefixes.

        Args:
            handler (PathHandler)
        """
        # assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            assert prefix not in ModelRegistry._PATH_HANDLERS
            ModelRegistry._PATH_HANDLERS[prefix] = handler

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        ModelRegistry._PATH_HANDLERS = OrderedDict(
            sorted(
                ModelRegistry._PATH_HANDLERS.items(), key=lambda t: t[0], reverse=True,
            )
        )
