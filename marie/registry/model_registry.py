import json
import logging
import os
from typing import Any, Dict, List, MutableMapping, Optional, OrderedDict, Tuple, Union

from marie.constants import __model_path__

logger = logging.getLogger(__name__)


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

    def _discover(self, **kwargs: Any) -> Dict[str, Union[str, os.PathLike]]:
        """
        Discover all models that this provider can locate
        Returns:
            Dict[str,str]: Dictionary of model name and location path
        """
        raise NotImplementedError()

    def _resolve(self, _name_or_path, **kwargs):
        """
        Attempt to resolve specific model by name
        """
        raise NotImplementedError()


class NativeModelRegistryHandler(ModelRegistryHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.

    This is the default model zoo handler and we will default to this handler when protocol can not be determined or when
    loading it from the model zoo with `model://name`
    """

    def __init__(self):
        self.resolved_models = {}
        self.discovered = False

    def _resolve(
        self, _name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[os.PathLike, Dict[str, str]]:

        if os.path.isdir(_name_or_path):
            config = os.path.join(_name_or_path, "marie.json")
        elif os.path.isfile(_name_or_path):
            config = _name_or_path
        else:
            model_root = (
                kwargs.pop("__model_path__")
                if "__model_path__" in kwargs
                else __model_path__
            )
            config = os.path.join(model_root, _name_or_path, "marie.json")
            if not os.path.exists(config):
                raise RuntimeError(f"Invalid resolution source : {_name_or_path}")

        logger.info(f"Found model definition in {config}")

        with open(config, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            return os.path.dirname(config), data

    def _discover(self, **kwargs: Any) -> Dict[str, Union[str, os.PathLike]]:
        model_root = (
            kwargs.pop("__model_path__")
            if "__model_path__" in kwargs
            else __model_path__
        )

        self._check_kwargs(kwargs)
        logger.info(f"Resolving native model from : {model_root}")
        resolved = {}
        for root_dir, dir, files in os.walk(model_root):
            name_key = "_name_or_path"
            for file in files:
                if file == "marie.json":
                    config_path, data = self._resolve(root_dir)
                    if name_key not in data:
                        logger.warning(
                            f"Key '{name_key}' not found in discovered config"
                        )
                        continue
                    name = data[name_key]
                    if name in resolved:
                        resolve_val = resolved[name]
                        raise ValueError(
                            f"Model name '{name_key}' already registered from : {resolve_val}"
                        )

                    resolved[name] = config_path
        self.resolved_models = resolved
        self.discovered = True
        return self.resolved_models

    def _get_supported_prefixes(self) -> List[str]:
        return ["file://", "model://"]

    def _get_local_path(self, _name_or_path: str, **kwargs: Any) -> Union[str, None]:
        if not self.discovered:
            self._discover(**kwargs)
        if _name_or_path in self.resolved_models:
            return self.resolved_models[_name_or_path]
        else:
            config_dir, config_data = self._resolve(_name_or_path, **kwargs)
            if config_dir is not None:
                self.resolved_models[_name_or_path] = config_dir
            return config_dir

    def _exists(self, _name_or_path: str, **kwargs: Any) -> bool:
        if not self.discovered:
            self._discover(**kwargs)
        return _name_or_path in self.resolved_models


class ModelRegistry:
    """
    A class for users to access models from specific providers

        Handlers:
            zoo    : Local model zoo(native file access)
            git    : Git
            S3     : Amazon S3
            gdrive : Google Drive
            dvc    : Data Version Control
            mflow  : Mflow support
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
            handler (ModelRegistryHandler)
        """
        for p in ModelRegistry._PATH_HANDLERS.keys():
            if _name_or_path.startswith(p):
                return ModelRegistry._PATH_HANDLERS[p]
        return ModelRegistry._NATIVE_PATH_HANDLER

    def __init__(self, **kwargs):
        pass

    def load_model(self):
        pass

    @staticmethod
    def get_local_path(_name_or_path: str, **kwargs: Any) -> Union[str, None]:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            _name_or_path (str): A URI supported by this ModelRegistryHandler

        Returns:
            local_path (str): a file path which exists on the local file system,
                              or None if the model could not be located
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
                ModelRegistry._PATH_HANDLERS.items(),
                key=lambda t: t[0],
                reverse=True,
            )
        )

    @staticmethod
    def discover(**kwargs) -> Dict[str, Union[str, os.PathLike]]:
        """
        Discover all models from registered handlers
        """
        resolved = {}
        handlers = [
            ModelRegistry._PATH_HANDLERS[p] for p in ModelRegistry._PATH_HANDLERS.keys()
        ]
        handlers.append(ModelRegistry._NATIVE_PATH_HANDLER)
        for handler in handlers:
            try:
                discovered = handler._discover(**kwargs)
                resolved = {**resolved, **discovered}
            except Exception as e:
                logger.error(f"Handler {handler} failed during discovery", e)
        return resolved
