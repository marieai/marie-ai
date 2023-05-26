from typing import Dict, Any, Optional, List

from marie.logging.logger import MarieLogger
from marie_server.storage.storage_client import StorageArea


class InMemoryKV(StorageArea):
    """
    In-memory key-value store. This is useful for testing and development.
    """

    kv_store: Dict[bytes, Dict[bytes, bytes]] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.logger = MarieLogger("InMemoryKV")
        self.logger.info(f"config : {config}")

    async def internal_kv_get(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> Optional[Any]:
        self.logger.debug(f"internal_kv_get: {key!r}, {namespace!r}")
        if namespace is None:
            namespace = b"DEFAULT"
        if namespace not in self.kv_store:
            return None
        return self.kv_store[namespace][key]

    async def internal_kv_multi_get(
        self,
        keys: List[bytes],
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> Dict[bytes, bytes]:
        raise NotImplementedError

    async def internal_kv_put(
        self,
        key: bytes,
        value: bytes,
        overwrite: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        self.logger.debug(
            f"internal_kv_put: {key!r}, {namespace!r}, {overwrite}, {value!r}"
        )
        if namespace is None:
            namespace = b"DEFAULT"
        if namespace not in self.kv_store:
            self.kv_store[namespace] = {}
        if key in self.kv_store[namespace] and not overwrite:
            return 0
        self.kv_store[namespace][key] = value
        return 1

    async def internal_kv_del(
        self,
        key: bytes,
        del_by_prefix: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        self.logger.debug(f"internal_kv_del: {key!r}, {namespace!r}, {del_by_prefix}")
        if namespace is None:
            namespace = b"DEFAULT"
        if namespace not in self.kv_store:
            return 0
        if del_by_prefix:
            keys_to_delete = [
                k for k in self.kv_store[namespace].keys() if k.startswith(key)
            ]
            for k in keys_to_delete:
                del self.kv_store[namespace][k]
            return len(keys_to_delete)
        else:
            if key in self.kv_store[namespace]:
                del self.kv_store[namespace][key]
                return 1
            else:
                return 0

    async def internal_kv_exists(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> bool:
        self.logger.debug(f"internal_kv_exists: {key!r}, {namespace!r}")
        if namespace is None:
            namespace = b"DEFAULT"
        if namespace not in self.kv_store:
            return False
        return key in self.kv_store[namespace]

    async def internal_kv_keys(
        self, prefix: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> List[bytes | str]:
        self.logger.debug(f"internal_kv_keys: {prefix!r}, {namespace!r}")
        if namespace is None:
            namespace = b"DEFAULT"
        if namespace not in self.kv_store:
            return []
        return [k for k in self.kv_store[namespace].keys() if k.startswith(prefix)]

    def internal_kv_reset(self) -> None:
        self.logger.debug(f"internal_kv_reset")
        self.kv_store = {}

    def debug_info(self) -> str:
        return f"InMemoryKV: {self.kv_store}"
