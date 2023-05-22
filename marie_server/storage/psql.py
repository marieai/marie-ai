from typing import Dict, Any, Optional, List

from marie.logging.logger import MarieLogger
from marie_server.storage.storage_client import StorageArea


class PostgreSQLKV(StorageArea):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = MarieLogger("PostgreSQLKV")
        print("config", config)
        self.running = False
        # self._setup_storage(config)

    async def internal_kv_get(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> Optional[Any]:
        pass

    async def internal_kv_multi_get(
        self,
        keys: List[bytes],
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> Dict[bytes, bytes]:
        pass

    async def internal_kv_put(
        self,
        key: bytes,
        value: bytes,
        overwrite: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        pass

    async def internal_kv_del(
        self,
        key: bytes,
        del_by_prefix: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        pass

    async def internal_kv_exists(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> bool:
        pass

    async def internal_kv_keys(
        self, prefix: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> List[bytes]:
        pass

    def debug_info(self) -> str:
        pass
