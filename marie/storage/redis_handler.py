import os
from typing import Any, Dict

from marie.storage import PathHandler


class RedisHandler(PathHandler):
    """
    Resolve URL like redis://.
    """

    PREFIX = "redis://"

    def __init__(self, volume_base_dir="/tmp") -> None:
        self.cache_map: Dict[str, str] = {}
        self.volume_base_dir = volume_base_dir

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        raise Exception("Operation not implemented")

    def _exists(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        resolved_path = self._get_local_path(path)
        return os.path.exists(resolved_path)

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        raise Exception("Operation not supported")

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        raise Exception("Operation not supported")
