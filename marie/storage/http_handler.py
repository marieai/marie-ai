import os
from typing import Dict, List, Any
from urllib.parse import urlparse

from marie import logging
from marie.common.download import download
from marie.common.file_io import get_cache_dir, file_lock
from marie.storage import PathHandler


class HTTPURLHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    def ensure_connection(self):
        pass

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
            dirname = os.path.join(
                get_cache_dir(), os.path.dirname(parsed_url.path.lstrip("/"))
            )
            filename = path.split("/")[-1]
            cached = os.path.join(dirname, filename)
            with file_lock(cached):
                if not os.path.isfile(cached):
                    logger.info("Downloading {} ...".format(path))
                    cached = download(path, dirname, filename=filename)
            logger.info("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        return self.cache_map[path]
