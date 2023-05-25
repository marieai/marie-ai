from typing import Dict, Any, Optional, List

from uuid_extensions import uuid7str

from marie.logging.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin
from marie.utils import json
from marie.utils.json import to_json
from marie_server.storage.storage_client import StorageArea
from datetime import datetime


class PostgreSQLKV(PostgresqlMixin, StorageArea):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = MarieLogger("PostgreSQLKV")
        print("config", config)
        self.running = False
        self._setup_storage(config, create_table_callback=self.create_table_callback)

    def create_table_callback(self, table_name: str):
        self.logger.info(f"Creating table : {table_name}")

        self._execute_sql_gracefully(
            f"""
             CREATE TABLE IF NOT EXISTS {self.table} (
                 id UUID PRIMARY KEY,
                 namespace VARCHAR(1024) NULL,
                 key VARCHAR(1024) NOT NULL,                 
                 value JSONB NULL,
                 shard int,
                 created_at timestamp with time zone default current_timestamp,
                 updated_at timestamp with time zone default current_timestamp,
                 is_deleted BOOL DEFAULT FALSE
             );
--              CREATE INDEX index_queue_on_scheduled_for ON queue (scheduled_for);
--              CREATE INDEX index_queue_on_status ON queue (status);
             """,
        )

    async def internal_kv_get(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> Optional[Any]:
        raise NotImplementedError

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
        if key is None:
            raise ValueError("key cannot be None")
        if namespace is None:
            namespace = b"DEFAULT"

        uid = uuid7str()
        query = f"""
            INSERT INTO {self.table} (id, namespace, key, value, shard, created_at, updated_at) 
            VALUES (
                 '{uid}', 
                 '{namespace.decode()}',
                 '{key.decode()}', 
                 '{value.decode()}',
                 1, 
                 current_timestamp, 
                 current_timestamp
            )
            """

        self._execute_sql_gracefully(query)

    async def internal_kv_del(
        self,
        key: bytes,
        del_by_prefix: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        raise NotImplementedError

    async def internal_kv_exists(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> bool:
        raise NotImplementedError

    async def internal_kv_keys(
        self, prefix: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> List[bytes]:
        raise NotImplementedError

    def debug_info(self) -> str:
        return "PostgreSQLKV"
