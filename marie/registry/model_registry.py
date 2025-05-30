import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union

from marie.constants import __model_path__

logger = logging.getLogger(__name__)


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

    @staticmethod
    def strip_prefix(_name_or_path: str, prefixes: List[str]) -> str:
        for prefix in prefixes:
            if _name_or_path.startswith(prefix):
                return _name_or_path[len(prefix) :]
        return _name_or_path

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

    def get_local_path(
        self,
        _name_or_path: str,
        version: str = None,
        raise_exceptions_for_missing_entries: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Get the local path for the provider name or path

        str: the local path to the model directory
        """
        raise NotImplementedError()

    def discover(self, **kwargs: Any) -> Dict[str, Union[str, os.PathLike]]:
        """
        Discover all models that this provider can locate
        Returns:
            Dict[str,str]: Dictionary of model name and location path
        """
        raise NotImplementedError()

    def _resolve(
        self, _name_or_path, raise_exceptions_for_missing_entries: bool = True, **kwargs
    ):
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
        self,
        _name_or_path: Union[str, os.PathLike],
        raise_exceptions_for_missing_entries: bool = True,
        **kwargs,
    ) -> Tuple[Union[os.PathLike, None], Union[Dict[str, Any], None]]:

        full_filename = "marie.json"
        if os.path.isdir(_name_or_path):
            config = os.path.join(_name_or_path, full_filename)
        elif os.path.isfile(_name_or_path):
            config = _name_or_path
        else:
            model_root = (
                kwargs.pop("__model_path__")
                if "__model_path__" in kwargs
                else __model_path__
            )
            config = os.path.join(model_root, _name_or_path, full_filename)
            if not os.path.exists(config):
                if raise_exceptions_for_missing_entries:
                    raise EnvironmentError(
                        f"{_name_or_path} does not appear to have a file named {full_filename} in root path : "
                        f"{model_root}"
                    )
                else:
                    return None, None

        logger.info(f"Found model definition in {config}")
        with open(config, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            return os.path.dirname(config), data

    def discover(self, **kwargs: Any) -> Dict[str, Union[str, os.PathLike]]:
        model_root = (
            kwargs.pop("__model_path__")
            if "__model_path__" in kwargs
            else __model_path__
        )
        logger.info(f"Resolving native model from : {model_root}")

        # NOOP
        use_auth_token = (
            kwargs.pop("use_auth_token") if "use_auth_token" in kwargs else None
        )

        self._check_kwargs(kwargs)
        resolved = {}

        for root_dir, file_dir, files in os.walk(model_root):
            name_key = "_name_or_path"
            for file in files:
                if file == "marie.json":
                    config_path, data = self._resolve(
                        root_dir, raise_exceptions_for_missing_entries=False
                    )
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
        return ["file://", "model://", "zoo://"]

    def get_local_path(
        self,
        _name_or_path: str,
        version: str = None,
        raise_exceptions_for_missing_entries: bool = True,
        **kwargs: Any,
    ) -> Union[str | os.PathLike, None]:
        model_name_or_path = ModelRegistryHandler.strip_prefix(
            _name_or_path, self._get_supported_prefixes()
        )

        if not self.discovered:
            self.discover(**kwargs)
        if model_name_or_path in self.resolved_models:
            return self.resolved_models[model_name_or_path]
        else:
            config_dir, config_data = self._resolve(
                model_name_or_path, raise_exceptions_for_missing_entries, **kwargs
            )
            if config_dir is not None:
                self.resolved_models[model_name_or_path] = config_dir
            return config_dir

    def _exists(self, _name_or_path: str, **kwargs: Any) -> bool:
        if not self.discovered:
            self.discover(**kwargs)
        return _name_or_path in self.resolved_models


class HuggingFaceModelRegistry(ModelRegistryHandler):
    def _get_supported_prefixes(self) -> List[str]:
        return ["hf://", "transformer://"]

    def get_local_path(
        self,
        _name_or_path: str,
        version: str = None,
        raise_exceptions_for_missing_entries: bool = True,
        **kwargs: Any,
    ) -> Union[str | os.PathLike, None]:
        model_name_or_path = ModelRegistryHandler.strip_prefix(
            _name_or_path, self._get_supported_prefixes()
        )
        # https://huggingface.co/docs/huggingface_hub/guides/download
        model_path = None
        throwable = None

        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=model_name_or_path, revision=version, token=None
            )
        except Exception as e:
            logger.error(f"Error downloading model from HuggingFace: {e}")
            throwable = e
        except ModuleNotFoundError as e:
            logger.error(f"HuggingFace is not installed: {e}")
            throwable = e

        if raise_exceptions_for_missing_entries and model_path is None:
            raise EnvironmentError(
                f"{model_name_or_path} does not appear to have a valid HuggingFace model",
                throwable,
            )

        return model_path


class ModelRegistry:
    """
    A class for users to access models from specific providers

        Handlers:
            zoo          : Local model zoo(native file access) - default handler with model://
            git          : Git
            S3           : S3 storage (AWS, Minio, etc.)
            gdrive       : Google Drive
            dvc          : Data Version Control
            mflow        : Mflow support
            transformers : HuggingFace Transformers models
            azure        : Azure storage
    """

    _PATH_HANDLERS: MutableMapping[str, ModelRegistryHandler] = OrderedDict()
    _NATIVE_PATH_HANDLER = NativeModelRegistryHandler()

    @staticmethod
    def get_handler(_name_or_path: str) -> ModelRegistryHandler:
        """
        Finds a ModelRegistryHandler that supports the given protocol path. Falls back to the native
        ModelRegistryHandler if no other handler is found.

        Args:
            _name_or_path (str): URI path to resource

        Returns:
            handler (ModelRegistryHandler)
        """
        if "://" not in _name_or_path:
            return ModelRegistry._NATIVE_PATH_HANDLER
        scheme = _name_or_path[: _name_or_path.index("://") + 3]

        for p in ModelRegistry._PATH_HANDLERS.keys():
            if scheme == p:
                return ModelRegistry._PATH_HANDLERS[p]

        raise ValueError(f"Unsupported protocol '{scheme}' for model : {_name_or_path}")

    @staticmethod
    def get(
        name_or_path: str,
        version: Optional[str] = None,
        raise_exceptions_for_missing_entries: Optional[bool] = True,
        **kwargs: Any,
    ) -> Union[str, None]:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk. Depending on the protocol, this cache may be shared with other providers.

        :param name_or_path: URI path to resource
        :param version: Optional version of the resource
        :param raise_exceptions_for_missing_entries: If True, raise an exception if the resource is not found.

        :return: Local folder path (string) of repo if found, else None
        """

        handler = ModelRegistry.get_handler(name_or_path)

        return handler.get_local_path(
            name_or_path, version, raise_exceptions_for_missing_entries, **kwargs
        )  # type: ignore

    @staticmethod
    def config(
        name_or_path: str,
        version: Optional[str] = None,
        raise_exceptions_for_missing_entries: Optional[bool] = True,
        **kwargs: Any,
    ) -> Union[str, None]:
        """
        Get a configuration for the model.

        :param name_or_path: URI path to resource
        :param version: Optional version of the resource
        :param raise_exceptions_for_missing_entries: If True, raise an exception if the resource is not found.
        """

        resolved_path = ModelRegistry.get(
            name_or_path, version, raise_exceptions_for_missing_entries, **kwargs
        )

        if resolved_path is not None:
            full_filename = "marie.json"
            if os.path.isdir(resolved_path):
                config = os.path.join(resolved_path, full_filename)
            elif os.path.isfile(resolved_path):
                config = resolved_path
            else:
                raise EnvironmentError(
                    f"{resolved_path} does not appear to have a file named {full_filename}"
                )
            with open(config, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                return data

    @staticmethod
    def checkpoint(
        name_or_path: str,
        version: Optional[str] = None,
        raise_exceptions_for_missing_entries: Optional[bool] = True,
        checkpoint: str = "pytorch_model.bin",
        **kwargs: Any,
    ) -> Union[str, None]:
        """
        Load a checkpoint for the model. This is a file that contains the model weights. Defaults to `pytorch_model.bin`.

        :param name_or_path: URI path to resource
        :param version: Optional version of the resource
        :param raise_exceptions_for_missing_entries: If True, raise an exception if the resource is not found.
        :param checkpoint: Name of the checkpoint file to load (default: `pytorch_model.bin`)
        """
        resolved_path = ModelRegistry.get(
            name_or_path, version, raise_exceptions_for_missing_entries, **kwargs
        )

        if resolved_path is not None:
            if checkpoint is not None:
                full_filename = checkpoint
            else:
                full_filename = "pytorch_model.bin"

            if os.path.isdir(resolved_path):
                checkpoint = os.path.join(resolved_path, full_filename)
            elif os.path.isfile(resolved_path):
                checkpoint = resolved_path
            else:
                raise EnvironmentError(
                    f"{resolved_path} does not appear to have a file named {full_filename}"
                )
            return checkpoint

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
                discovered = handler.discover(**kwargs)
                resolved = {**resolved, **discovered}
            except Exception as e:
                logger.error(f"Handler {handler} failed during discovery", e)
        return resolved


ModelRegistry.register_handler(NativeModelRegistryHandler())
ModelRegistry.register_handler(HuggingFaceModelRegistry())
