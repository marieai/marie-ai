import fnmatch
from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from pathlib import Path
from typing import MutableMapping, OrderedDict, Any, List, Optional, Union, Dict
import multiprocessing as mp
import os

from marie.excepts import BadConfigSource
from marie.logging.predefined import default_logger as logger

import io

StrOrBytesPath = Union[str, Path, PathLike]


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
        local_path: str,
        remote_path: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> bool:
        """
        Copies a source path to a destination path.

        :param local_path:
        :param remote_path:
        :param overwrite: flag for forcing overwrite of existing file or directory
        :param handler:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    # read  to a string or bytes
    def _read_str(self, path: StrOrBytesPath, **kwargs: Any) -> str:
        """
        Read a text file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            str: the content of the file
        """
        raise NotImplementedError()

    def _write(
        self,
        src_path: StrOrBytesPath,
        dst_path: str,
        overwrite: bool = False,
        handler: Optional["PathHandler"] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file
            handler (Optional[PathHandler]): A PathHandler to use for the dst_path
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

    def _list(self, path: str, return_full_path=False, **kwargs: Any) -> List[str]:
        """
        list resources at the given URI.
        """
        raise NotImplementedError()

    def _mkdir(self, path: str, **kwargs: Any) -> None:
        """
        mkdir at the given URI.
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

    def ensure_connection(self) -> None:
        """
        Checks if the resource at the given URI is online.
        """
        raise NotImplementedError()

    def _read(
        self,
        path: str,
        suppress_errors: bool = False,
        **kwargs: Any,
    ) -> bytes:
        """
        Reads the resource at the given URI and returns the contents as bytes.
        This is not optimal for large files.

        :param path:  A URI supported by this PathHandler
        :param suppress_errors:  Bool flag for suppressing errors
        :param kwargs:  Additional arguments
        :return:  bytes: the contents of the resource
        """

        raise NotImplementedError()

    def _read_to_file(
        self,
        path: str,
        dst_path: str | os.PathLike | io.BytesIO,
        overwrite=False,
        **kwargs: Any,
    ) -> None:
        """
        Read resource data synchronously at the given URI and writes the contents to the given file.
        :param path:    A URI supported by this PathHandler
        :param dst_path:   A file path to write to
        :param overwrite:  Bool flag for forcing overwrite of existing file
        :param kwargs:
        :return:    None
        """

        raise NotImplementedError()


class StorageManager:
    """
    PathManager is helper interface for downloading & uploading files to supported remote storage
    Support remote servers: http(s)/S3/GS/Azure/File-System-Folder
    """

    _PATH_HANDLERS: MutableMapping[str, PathHandler] = OrderedDict()  # type: ignore
    _NATIVE_PATH_HANDLER = None  # type: Optional[PathHandler]

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
        for p in StorageManager._PATH_HANDLERS.keys():
            if path.startswith(p):
                return StorageManager._PATH_HANDLERS[p]
        return StorageManager._NATIVE_PATH_HANDLER

    @staticmethod
    def register_handler(handler: PathHandler, native: Optional[bool] = False) -> None:
        """
        Register a path handler associated with one or more ``URI prefixes``.

        :param handler: PathHandler to register
        :param native:  If True, this handler will be used as the default handler for all paths.
        :return:
        """

        assert isinstance(handler, PathHandler), handler

        if native:
            if StorageManager._NATIVE_PATH_HANDLER is None:
                StorageManager._NATIVE_PATH_HANDLER = handler
            else:
                raise BadConfigSource(
                    f"Native handler already registered as : {StorageManager._NATIVE_PATH_HANDLER}"
                )
        else:
            # assert isinstance(handler, PathHandler), handler
            for prefix in handler._get_supported_prefixes():
                assert prefix not in StorageManager._PATH_HANDLERS
                StorageManager._PATH_HANDLERS[prefix] = handler

            # Sort path handlers in reverse order so longer prefixes take priority,
            # eg: http://foo/bar before http://foo
            StorageManager._PATH_HANDLERS = OrderedDict(
                sorted(
                    StorageManager._PATH_HANDLERS.items(),
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
        StorageManager._NATIVE_PATH_HANDLER._strict_kwargs_check = enable
        for handler in StorageManager._PATH_HANDLERS.values():
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
        return StorageManager.__get_path_handler(path)._get_local_path(path, **kwargs)  # type: ignore

    @staticmethod
    def exists(path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        return StorageManager.__get_path_handler(path)._exists(path, **kwargs)  # type: ignore

    @staticmethod
    def list(path: str, return_full_path=False, **kwargs: Any) -> List[str]:
        """
        List the contents of a directory at the given URI.

        :param path:  A URI to a directory
        :param return_full_path: If True, return a list of full object paths, otherwise return a list of relative object paths (default False)

        :param kwargs: Additional arguments to pass to the underlying PathHandler
        :return: The paths of all the objects the storage base path under prefix. None in case of list operation is not supported (http and https protocols for example)

        """
        return StorageManager.__get_path_handler(path)._list(path, return_full_path, **kwargs)  # type: ignore

    @staticmethod
    def mkdir(path: str, **kwargs: Any) -> List[str]:
        return StorageManager.__get_path_handler(path)._mkdir(path, **kwargs)  # type: ignore

    @staticmethod
    def copy(
        src_path: str,
        dst_path: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> bool:

        src_handler = StorageManager.__get_path_handler(src_path)
        dst_handler = StorageManager.__get_path_handler(dst_path)

        if src_handler != dst_handler:
            print(
                "Copy between different PathHandlers: {} and {}".format(
                    src_handler, dst_handler
                )
            )
            return src_handler._copy(src_path, dst_path, overwrite, dst_handler, **kwargs)  # type: ignore

        return StorageManager.__get_path_handler(src_path)._copy(
            src_path, dst_path, overwrite, **kwargs
        )  # type: ignore

    @staticmethod
    def ensure_connection(
        path: Optional[str] = None, silence_exceptions=Optional[bool], **kwargs: Any
    ) -> bool:
        """
        Ensures that the connection to the given path is established.
        If no path is give, it will ensure that the connection to all paths is established.

        Examples:

        .. code-block:: python

            # Ensure connection to all paths (default)
            StorageManager.ensure_connection()
            StorageManager.ensure_connection("s3://")

            # Silence exceptions
            connected  = StorageManager.ensure_connection(silence_exceptions=True)
            connected  = StorageManager.ensure_connection("s3://", silence_exceptions=True)

        :param path: Optional path to ensure connection to.
        :param silence_exceptions:  If True, exceptions will be silenced and False will be returned instead.
        :param kwargs:
        :return:
        """

        try:
            if path is None:
                for handler in StorageManager._PATH_HANDLERS.values():
                    handler.ensure_connection()
            else:
                StorageManager.__get_path_handler(path).ensure_connection()
        except Exception as e:
            if silence_exceptions:
                return False
            else:
                raise e
        return True

    @staticmethod
    def read(path: str, **kwargs: Any) -> bytes:
        """
        Reads the resource at the given URI and returns the contents as bytes.
        This is not optimal for large files.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bytes: the contents of the resource
        """
        return StorageManager.__get_path_handler(path)._read(path, **kwargs)  # type: ignore

    @classmethod
    def write(
        cls,
        src_path: StrOrBytesPath,
        dst_path: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> bool:

        return StorageManager.__get_path_handler(dst_path)._write(
            src_path, dst_path, overwrite, **kwargs
        )  # type: ignore

    @staticmethod
    def read_to_file(
        path: str,
        dst_path_or_buffer: str | os.PathLike | io.BytesIO,
        overwrite=False,
        **kwargs: Any,
    ) -> None:
        """
        Read resource data synchronously at the given URI and writes the contents to the given file.

        EXAMPLE USAGE

            .. code-block:: python

                # Read from S3 and write to local file
                StorageManager.read_to_file("s3://bucket/key", "/tmp/file")

                # Read from S3 and write to BytesIO
                buffer = io.BytesIO()
                StorageManager.read_to_file("s3://bucket/key", buffer)

                # Read from S3 and write to BytesIO using a context manager
                with open("/tmp/sample.txt", "wb+") as temp_file:
                    StorageManager.read_to_file(location, temp_file, overwrite=True)

                # Read from S3 and write to BytesIO using a temporary file
                with tempfile.NamedTemporaryFile() as temp_file:
                    StorageManager.read_to_file(location, temp_file, overwrite=True)


        :param path:    A URI supported by this PathHandler
        :param dst_path_or_buffer:    A file path to write to
        :param overwrite: If True, overwrite the destination file if it exists.
        :param kwargs:
        :return:    None
        """
        return StorageManager.__get_path_handler(path)._read_to_file(path, dst_path_or_buffer, overwrite, **kwargs)  # type: ignore

    @classmethod
    def copy_dir(
        cls,
        local_path: str,
        remote_path: str,
        relative_to_dir: Optional[str] = "",
        match_wildcard=None,
    ) -> Optional[str]:
        """
        Copy local folder recursively to a remote storage, maintaining the sub folder structure
        in the remote storage.

        EXAMPLE USAGE:

            .. code-block:: python

                StorageManager.copy_dir(tmpdir, "s3://marie", relative_to_dir=tmpdir, match_wildcard="*")

        :param local_path:  The local directory to copy
        :param remote_path: The remote path to copy to
        :param relative_to_dir: If set, the relative path of the local directory will be used as the remote path
        :param match_wildcard:  If set, only files matching the wildcard will be copied
        :return:
        """
        if not Path(local_path).is_dir():
            logger.error("Local path '{}' does not exist".format(local_path))
            return

        futures = []
        with ThreadPoolExecutor(max_workers=mp.cpu_count() // 2) as executor:
            for path in Path(local_path).rglob(match_wildcard or "*"):
                if not path.is_file():
                    continue
                resolved_path = path
                if relative_to_dir:
                    resolved_path = path.relative_to(relative_to_dir)
                resolved_path = os.path.join(remote_path, resolved_path)

                futures.append(executor.submit(cls.write, path, resolved_path))

        success = 0
        failed = 0

        for future in futures:
            try:
                future.result()
                success += 1
            except Exception as e:
                print(e)
                failed += 1

        if failed == 0:
            return remote_path

        logger.error(f"Failed uploading {success}/{failed} files from {local_path}")

    @classmethod
    def copy_remote(
        cls,
        remote_path: str,
        local_path: str,
        match_wildcard=None,
        overwrite=False,
        silence_errors=False,
    ) -> Optional[str]:
        """
        Copy remote folder recursively to a local storage, maintaining the sub folder structure
        in the local storage.

        EXAMPLE USAGE:

        .. code-block:: python

            # Copy all files and folders
            StorageManager.copy_remote("s3://marie", tmpdir, match_wildcard="*")

            # Copy all files that end with .txt
            StorageManager.copy_remote("s3://marie", tmpdir, match_wildcard="*.txt")

            # Copy all files and folders in the subfolder
            StorageManager.copy_remote("s3://marie", tmpdir, match_wildcard="*/subfolder/*")

        :param remote_path: The remote directory to copy
        :param local_path: The local path to copy to
        :param match_wildcard: If set, only files and directories matching the wildcard will be copied
        :param overwrite: If True, overwrite the destination file if it exists.
        :param silence_errors: If True, errors will be silenced and None will be returned instead.
        :return:
        """
        if local_path:
            try:
                Path(local_path).mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                logger.error(
                    "Failed creating local folder '{}': {}".format(local_path, ex)
                )
                return

        # list the files
        items = cls.list(remote_path, return_full_path=True)

        if not items:
            logger.error("No files found at {}".format(remote_path))
            return

        for path in items:
            try:
                if match_wildcard and not fnmatch.fnmatch(path, match_wildcard):
                    logger.debug(
                        f"Skipping {path} as it does not match wildcard : {match_wildcard}"
                    )
                    continue
                relative = os.path.relpath(path, remote_path)

                dst_path = os.path.join(local_path, relative)
                cls.read_to_file(path, dst_path, overwrite=overwrite)
            except Exception as ex:
                if silence_errors:
                    logger.error(f"Failed copying {path}: {ex}")
                    continue
                else:
                    raise ex

    @classmethod
    def can_handle(
        cls,
        path: str,
        allow_native: bool = False,
    ) -> bool:
        """
        Returns True if there is a PathHandler that can handle the given URI
        :param path:  A URI to check
        :param allow_native:  If True, the native path handler will be considered as a valid handler
        :return:  True if there is a PathHandler that can handle the given URI
        """
        handler = StorageManager.__get_path_handler(path)
        if handler == StorageManager._NATIVE_PATH_HANDLER and not allow_native:
            return False
        return handler is not None
