from typing import Any, Dict, List, Optional

import psycopg2
from uuid_extensions import uuid7str

from marie.logging.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin
from marie.storage.kv.storage_client import StorageArea


class PostgreSQLKV(PostgresqlMixin, StorageArea):
    """
    PostgreSQLKV is a key-value store backed by PostgreSQL.
    Provides a simple key-value interface for storing and retrieving data from a PostgreSQL database utilizing the
    JSONB data type.
    """

    def __init__(self, config: Dict[str, Any], reset=True):
        super().__init__()
        self.logger = MarieLogger(self.__class__.__name__)
        self.running = False
        self._setup_storage(
            config,
            create_table_callback=self.create_table_callback,
            reset_table_callback=self.internal_kv_reset if reset else None,
        )

    def create_table_callback(self, table_name: str):
        """
        :param table_name: Name of the table to be created.
        :return: None
        """
        self.logger.info(f"Creating table : {table_name}")

        self._execute_sql_gracefully(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace VARCHAR(1024) NULL,
                key VARCHAR(1024) NOT NULL,
                value JSONB NULL,
                shard int DEFAULT 0,
                created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
                updated_at timestamp with time zone DEFAULT NULL,
                is_deleted BOOL DEFAULT FALSE
            );
            CREATE UNIQUE INDEX idx_{self.table}_ns_key ON {self.table} (namespace, key);

            CREATE TABLE IF NOT EXISTS {self.table}_history (
                history_id SERIAL PRIMARY KEY,
                id UUID,
                namespace VARCHAR(1024),
                key VARCHAR(1024),
                value JSONB,
                shard int,
                created_at timestamp with time zone,
                updated_at timestamp with time zone,
                is_deleted BOOL,
                change_time timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
                operation CHAR(1) CHECK (operation IN ('I', 'U', 'D'))
            );

            CREATE OR REPLACE FUNCTION log_changes_{self.table}() RETURNS TRIGGER AS $$
            BEGIN
                IF (TG_OP = 'INSERT') THEN
                    INSERT INTO {self.table}_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
                    VALUES (NEW.id, NEW.namespace, NEW.key, NEW.value, NEW.shard, NEW.created_at, NEW.updated_at, NEW.is_deleted, 'I');
                    RETURN NEW;
                ELSIF (TG_OP = 'UPDATE') THEN
                    INSERT INTO {self.table}_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
                    VALUES (NEW.id, NEW.namespace, NEW.key, NEW.value, NEW.shard, NEW.created_at, NEW.updated_at, NEW.is_deleted, 'U');
                    RETURN NEW;
                ELSIF (TG_OP = 'DELETE') THEN
                    INSERT INTO {self.table}_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
                    VALUES (OLD.id, OLD.namespace, OLD.key, OLD.value, OLD.shard, OLD.created_at, OLD.updated_at, OLD.is_deleted, 'D');
                    RETURN OLD;
                END IF;
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;

            CREATE TRIGGER log_changes_{self.table}_trigger
            AFTER INSERT OR UPDATE OR DELETE ON {self.table}
            FOR EACH ROW EXECUTE FUNCTION log_changes_{self.table}();
            """
        )

    async def internal_kv_get(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> Optional[Any]:
        if key is None:
            raise ValueError("key cannot be None")
        if namespace is None:
            namespace = b"DEFAULT"

        query = f"SELECT key, value FROM {self.table} WHERE key = '{key.decode()}'  AND namespace = '{namespace.decode()}' AND is_deleted = FALSE"
        cursor = self._execute_sql_gracefully(query, data=())

        result = cursor.fetchone()
        if result and (result[0] is not None):
            return result[1]
        return None

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
        self.logger.info(
            f"internal_kv_put: {key!r}, {namespace!r}, {overwrite}, {value!r}"
        )
        if key is None:
            raise ValueError("key cannot be None")
        if namespace is None:
            namespace = b"DEFAULT"

        uid = uuid7str()
        shard = 0

        insert_q = f"""
            INSERT INTO {self.table} (id, namespace, key, value, shard, created_at, updated_at) 
            VALUES ('{uid}', '{namespace.decode()}', '{key.decode()}', '{value.decode()}', {shard},current_timestamp,current_timestamp )
        """

        upsert_q = f"""
            ON CONFLICT (key, namespace) 
            DO 
            UPDATE SET value = '{value.decode()}', updated_at = current_timestamp
        """

        query = insert_q + upsert_q if overwrite else insert_q
        cursor = self._execute_sql_gracefully(query)
        if cursor is None:
            return 0
        return cursor.rowcount

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

        if del_by_prefix:
            raise NotImplementedError
        else:
            query = f"DELETE FROM {self.table} WHERE key = '{key.decode()}' AND namespace = '{namespace.decode()}'"
            cursor = self._execute_sql_gracefully(query, data=())
            if cursor is not None:
                return 1
        return 0

    async def internal_kv_exists(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> bool:
        raise NotImplementedError

    async def internal_kv_keys(
        self, prefix: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> List[bytes | str]:
        if namespace is None:
            namespace = b"DEFAULT"
        result = []
        with self:
            try:
                query = f"SELECT key  FROM {self.table} WHERE  namespace = '{namespace.decode()}' AND is_deleted = FALSE"
                for record in self._execute_sql_gracefully(query, data=()):
                    result.append(record[0])
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error executing sql statement: {error}")
        return result

    def internal_kv_reset(self) -> None:
        self.logger.info(f"internal_kv_reset : {self.table}")

        self._execute_sql_gracefully(f"DROP TABLE IF EXISTS {self.table}")
        self._execute_sql_gracefully(f"DROP TABLE IF EXISTS {self.table}_history")
        self._execute_sql_gracefully(
            f"DROP FUNCTION IF EXISTS log_changes_{self.table} CASCADE"
        )

    def debug_info(self) -> str:
        return "PostgreSQLKV"
