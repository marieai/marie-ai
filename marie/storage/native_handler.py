import io
import os
import tempfile
from typing import Optional, Any, List

import shutil
from marie.storage import PathHandler


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    def _get_supported_prefixes(self) -> List[str]:
        return [None, "file://"]

    def _copy(
        self,
        src_path: str,
        dst_path: str,
        overwrite: bool = False,
        handler: Optional["PathHandler"] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Copies a source path to a destination path.
        :param src_path:    source path
        :param dst_path:     destination path this could be s3://, redis://, local path or any other handler
        :param overwrite:  Bool flag for forcing overwrite of existing file
        :param kwargs:  additional arguments
        """

        # if handler is present then use the handler to copy the file otherwise use the default copy
        if handler is None:
            os.shutil.copy(src_path, dst_path)  # type: ignore
        else:
            handler._copy(src_path, dst_path, overwrite=overwrite, **kwargs)  # type: ignore

        return True

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        self._check_kwargs(kwargs)
        return path

    def _exists(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        # check if path starts with file:// and remove it
        if path.startswith("file://"):
            path = path[7:]
        return os.path.exists(path)

    def _read_to_file(
        self,
        path: str,
        dst_path: str | os.PathLike | io.BytesIO,
        overwrite=False,
        **kwargs: Any,
    ) -> None:

        self._check_kwargs(kwargs)
        # check if path starts with file:// and remove it
        if path.startswith("file://"):
            path = path[7:]
        if isinstance(dst_path, (io.BytesIO, tempfile._TemporaryFileWrapper)):
            with open(path, "rb") as f:
                dst_path.write(f.read())
        else:
            shutil.copy(path, dst_path)
