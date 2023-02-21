import os
from typing import Optional, Any, List

from marie.storage import PathHandler


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    def _get_supported_prefixes(self) -> List[str]:
        return None

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

        print(src_path)
        print(dst_path)

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
        return os.path.exists(path)
