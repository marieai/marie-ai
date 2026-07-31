import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional, TypeVar

import psycopg
from uuid_extensions import uuid7str

from marie.constants import (
    JOB_INFO_KEY_PREFIX,
    JOB_STATUS_NOTIFICATION_CHANNEL,
    KV_NAMESPACE_JOB,
)
from marie.logging_core.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin
from marie.storage.database.postgres_pool import (
    AsyncPostgresConnection,
    AsyncPostgresConnectionPool,
)
from marie.storage.kv.storage_client import StorageArea
from marie.utils.scheduler_trace import scheduler_trace

_TERMINAL_JOB_STATUSES = frozenset({'FAILED', 'STOPPED', 'SUCCEEDED'})
_T = TypeVar('_T')


def _terminal_job_notification(
    namespace: str, key: str, value: str
) -> dict[str, str | None] | None:
    if namespace != KV_NAMESPACE_JOB.decode() or not key.startswith(
        JOB_INFO_KEY_PREFIX
    ):
        return None

    try:
        job_info = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(job_info, dict):
        return None

    status = job_info.get('status')
    job_id = key[len(JOB_INFO_KEY_PREFIX) :]
    if status not in _TERMINAL_JOB_STATUSES or not job_id:
        return None

    run_owner = job_info.get('run_owner')
    run_attempt_id = job_info.get('run_attempt_id')
    return {
        'job_id': job_id,
        'status': status,
        'run_owner': run_owner if isinstance(run_owner, str) else None,
        'run_attempt_id': (run_attempt_id if isinstance(run_attempt_id, str) else None),
    }


class PostgreSQLKV(PostgresqlMixin, StorageArea):
    """
    PostgreSQLKV is a key-value store backed by PostgreSQL.
    Provides a simple key-value interface for storing and retrieving data from a PostgreSQL database utilizing the
    JSONB data type.
    """

    def __init__(self, config: Dict[str, Any], reset=False):
        super().__init__()
        self.logger = MarieLogger(self.__class__.__name__)
        self.running = False
        self._config = dict(config)
        self._async_pool = AsyncPostgresConnectionPool()
        self._async_pool_lock = asyncio.Lock()
        self._async_pool_initialized = False
        self._closed = False
        self._setup_storage(
            config,
            create_table_callback=self.create_table_callback,
            reset_table_callback=self.internal_kv_reset if reset else None,
        )
        self.postgreSQL_pool.close()

    async def _ensure_async_pool(self) -> None:
        if self._closed:
            raise RuntimeError("PostgreSQLKV is closed")
        if self._async_pool_initialized:
            return
        async with self._async_pool_lock:
            if self._closed:
                raise RuntimeError("PostgreSQLKV is closed")
            if self._async_pool_initialized:
                return
            await self._async_pool.initialize(
                self._config,
                autocommit=True,
                trace_name="async_kv",
            )
            self._async_pool_initialized = True

    async def _run_db_operation(
        self,
        operation: str,
        function: Callable[[AsyncPostgresConnection], Awaitable[_T]],
        *,
        job_id: str | None = None,
    ) -> _T:
        await self._ensure_async_pool()
        started_at = time.perf_counter()
        acquired_at: float | None = None
        succeeded = False
        try:
            async with self._async_pool.acquire() as connection:
                acquired_at = time.perf_counter()
                result = await function(connection)
            succeeded = True
            return result
        finally:
            completed_at = time.perf_counter()
            scheduler_trace(
                'postgres_kv_operation_completed',
                mode='async',
                operation=operation,
                job_id=job_id,
                succeeded=succeeded,
                pool_wait_ms=(
                    (acquired_at - started_at) * 1000.0
                    if acquired_at is not None
                    else None
                ),
                database_operation_ms=(
                    (completed_at - acquired_at) * 1000.0
                    if acquired_at is not None
                    else None
                ),
                total_ms=(completed_at - started_at) * 1000.0,
            )

    def create_table_callback(self, table_name: str):
        """
        :param table_name: Name of the table to be created.
        :return: None
        """
        qualified = self.qualified_table
        # Use just table name for function/trigger names (no dots allowed)
        safe_name = self.table.replace(".", "_")
        self.logger.info(f"Creating table : {qualified}")

        self._execute_sql_gracefully(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace VARCHAR(1024) NULL,
                key VARCHAR(1024) NOT NULL,
                value JSONB NULL,
                shard int DEFAULT 0,
                created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
                updated_at timestamp with time zone DEFAULT NULL,
                is_deleted BOOL DEFAULT FALSE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{safe_name}_ns_key ON {qualified} (namespace, key);

            CREATE TABLE IF NOT EXISTS {qualified}_history (
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

            CREATE OR REPLACE FUNCTION log_changes_{safe_name}() RETURNS TRIGGER AS $$
            BEGIN
                IF (TG_OP = 'INSERT') THEN
                    INSERT INTO {qualified}_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
                    VALUES (NEW.id, NEW.namespace, NEW.key, NEW.value, NEW.shard, NEW.created_at, NEW.updated_at, NEW.is_deleted, 'I');
                    RETURN NEW;
                ELSIF (TG_OP = 'UPDATE') THEN
                    INSERT INTO {qualified}_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
                    VALUES (NEW.id, NEW.namespace, NEW.key, NEW.value, NEW.shard, NEW.created_at, NEW.updated_at, NEW.is_deleted, 'U');
                    RETURN NEW;
                ELSIF (TG_OP = 'DELETE') THEN
                    INSERT INTO {qualified}_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
                    VALUES (OLD.id, OLD.namespace, OLD.key, OLD.value, OLD.shard, OLD.created_at, OLD.updated_at, OLD.is_deleted, 'D');
                    RETURN OLD;
                END IF;
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;

            CREATE TRIGGER log_changes_{safe_name}_trigger
            AFTER INSERT OR UPDATE OR DELETE ON {qualified}
            FOR EACH ROW EXECUTE FUNCTION log_changes_{safe_name}();
            """
        )

    async def internal_kv_get(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> Optional[Any]:
        if key is None:
            raise ValueError("key cannot be None")
        if namespace is None:
            namespace = b"DEFAULT"

        async def get(connection: AsyncPostgresConnection) -> Any | None:
            row = await connection.fetchrow(
                f"SELECT value FROM {self.qualified_table} "
                "WHERE key = $1 AND namespace = $2 AND is_deleted = FALSE",
                key.decode(),
                namespace.decode(),
            )
            return row['value'] if row is not None else None

        key_text = key.decode()
        job_id = (
            key_text[len(JOB_INFO_KEY_PREFIX) :]
            if namespace == KV_NAMESPACE_JOB
            and key_text.startswith(JOB_INFO_KEY_PREFIX)
            else None
        )
        return await self._run_db_operation('get', get, job_id=job_id)

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

        async def put(connection: AsyncPostgresConnection) -> int:
            uid = uuid7str()
            shard = 0

            ns = namespace.decode()
            k = key.decode()
            v = value.decode()
            terminal_notification = _terminal_job_notification(ns, k, v)

            insert_q = f"""
                INSERT INTO {self.qualified_table} (id, namespace, key, value, shard, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, current_timestamp, current_timestamp)
            """
            params = [uid, ns, k, v, shard]

            if overwrite:
                insert_q += f"""
                    ON CONFLICT (namespace, key)
                    DO
                    UPDATE SET value = %s, updated_at = current_timestamp
                """
                params.append(v)
            else:
                insert_q += """
                    ON CONFLICT (namespace, key) DO NOTHING
                """

            insert_q += " RETURNING id"
            notify_started: float | None = None
            async with connection.transaction():
                row = await connection.fetchrow(insert_q, *params)
                if row is None:
                    return 0
                if terminal_notification is not None:
                    notify_started = time.perf_counter()
                    scheduler_trace(
                        'job_terminal_notification_emit_started',
                        **terminal_notification,
                    )
                    await connection.execute(
                        'SELECT pg_notify($1, $2)',
                        JOB_STATUS_NOTIFICATION_CHANNEL,
                        json.dumps(terminal_notification),
                    )
            if terminal_notification is not None and notify_started is not None:
                scheduler_trace(
                    'job_terminal_notification_emitted',
                    **terminal_notification,
                    elapsed_ms=(time.perf_counter() - notify_started) * 1000.0,
                )
            return 1

        key_text = key.decode()
        job_id = (
            key_text[len(JOB_INFO_KEY_PREFIX) :]
            if namespace == KV_NAMESPACE_JOB
            and key_text.startswith(JOB_INFO_KEY_PREFIX)
            else None
        )
        return await self._run_db_operation('put', put, job_id=job_id)

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

            async def delete(connection: AsyncPostgresConnection) -> int:
                row = await connection.fetchrow(
                    f"DELETE FROM {self.qualified_table} "
                    "WHERE key = $1 AND namespace = $2 RETURNING id",
                    key.decode(),
                    namespace.decode(),
                )
                return int(row is not None)

            return await self._run_db_operation('delete', delete)

    async def internal_kv_exists(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> bool:
        raise NotImplementedError

    async def internal_kv_keys(
        self, prefix: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> List[bytes | str]:
        if namespace is None:
            namespace = b"DEFAULT"

        async def keys(connection: AsyncPostgresConnection) -> List[str]:
            try:
                rows = await connection.fetch(
                    f"SELECT key FROM {self.qualified_table} "
                    "WHERE namespace = $1 AND is_deleted = FALSE",
                    namespace.decode(),
                )
            except psycopg.Error as error:
                self.logger.error(f"Error executing sql statement: {error}")
                return []
            return [row['key'] for row in rows]

        return await self._run_db_operation('keys', keys)

    def internal_kv_reset(self) -> None:
        qualified = self.qualified_table
        safe_name = self.table.replace(".", "_")
        self.logger.info(f"internal_kv_reset : {qualified}")
        statements = (
            f"DROP TABLE IF EXISTS {qualified}",
            f"DROP TABLE IF EXISTS {qualified}_history",
            f"DROP FUNCTION IF EXISTS log_changes_{safe_name} CASCADE",
        )
        if self.postgreSQL_pool.closed:
            with psycopg.connect(
                host=self._config.get('hostname', self._config.get('host')),
                port=int(self._config['port']),
                user=self._config.get('username', self._config.get('user')),
                password=self._config['password'],
                dbname=self._config['database'],
                options='-c timezone=UTC -c statement_timeout=120000',
                application_name=self._config.get(
                    'application_name', 'marie_scheduler'
                ),
            ) as connection:
                for statement in statements:
                    connection.execute(statement)
            return

        conn = None
        try:
            conn = self._get_connection()
            for statement in statements:
                self._execute_sql_gracefully(statement, connection=conn)
        finally:
            self._close_connection(conn)

    async def close(self) -> None:
        async with self._async_pool_lock:
            if self._closed:
                return
            self._closed = True
            if self._async_pool_initialized:
                await self._async_pool.close()
                self._async_pool_initialized = False
            if not self.postgreSQL_pool.closed:
                self.postgreSQL_pool.close()

    def debug_info(self) -> str:
        return "PostgreSQLKV"
