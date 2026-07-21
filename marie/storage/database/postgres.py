import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import psycopg
from psycopg import pq
from psycopg_pool import ConnectionPool, PoolTimeout

from marie.excepts import BadConfigSource
from marie.utils.scheduler_trace import scheduler_trace


class PostgresqlMixin:
    """Bind PostgreSQL database provider."""

    provider = "postgres"

    def _setup_storage(
        self,
        config: Dict[str, Any],
        create_table_callback: Optional[Callable] = None,
        reset_table_callback: Optional[Callable] = None,
        connection_only=False,
        pool: Optional[ConnectionPool] = None,
    ) -> None:
        """
        Setup PostgreSQL connection pool.

        @param config:
        @param create_table_callback: Create table if it doesn't exist.
        @param reset_table_callback:  Reset table if it exists.
        @param connection_only:       Only connect to the database.
        @return:
        """
        try:
            hostname = config["hostname"]
            port = int(config["port"])
            username = config["username"]
            password = config["password"]
            database = config["database"]
            max_connections = int(
                config.get("max_connections", config.get("max_pool_size", 10))
            )
            min_connections = int(
                config.get("min_connections", config.get("min_pool_size", 1))
            )
            application_name = config.get("application_name", "marie_scheduler")
            self._pg_pool_acquire_timeout_seconds = float(
                config.get("pool_acquire_timeout_seconds", 30.0)
            )
            self._pg_pool_acquire_warn_after_seconds = float(
                config.get("pool_acquire_warn_after_seconds", 1.0)
            )
            self._pg_pool_acquire_trace_after_seconds = float(
                config.get("pool_acquire_trace_after_seconds", 0.001)
            )

            self.postgreSQL_pool = pool or ConnectionPool(
                "",
                min_size=min_connections,
                max_size=max_connections,
                timeout=self._pg_pool_acquire_timeout_seconds,
                open=True,
                name=application_name,
                kwargs={
                    'user': username,
                    'password': password,
                    'dbname': database,
                    'host': hostname,
                    'port': port,
                    'options': '-c timezone=UTC -c statement_timeout=120000',
                    'application_name': application_name,
                    'connect_timeout': 10,
                    'keepalives': 1,
                    'keepalives_idle': 30,
                    'keepalives_interval': 10,
                    'keepalives_count': 3,
                },
            )
            if pool is None:
                self.postgreSQL_pool.wait(
                    timeout=float(config.get("pool_open_timeout_seconds", 10.0))
                )

            if connection_only:
                self.logger.info(f"Connected to postgresql database: {config}")
                return

            self.schema = config.get("schema")  # Optional schema name
            self.table = config["default_table"]
            self.logger.info(f"[DEBUG] PostgresqlMixin config: {config}")
            self.logger.info(
                f"[DEBUG] PostgresqlMixin schema={self.schema}, table={self.table}, qualified_table={self.schema}.{self.table if self.schema else self.table}"
            )
            if self.table is None or self.table == "":
                raise ValueError("default_table cannot be empty")

            # Create schema if specified and doesn't exist
            if self.schema:
                self._ensure_schema_exists()

            self._init_table(create_table_callback, reset_table_callback)

        except Exception as e:
            raise BadConfigSource(
                f"Cannot connect to postgresql database: {config}, {e}"
            )

    def _pool_counts(self) -> tuple[int | None, int | None]:
        pool = getattr(self, "postgreSQL_pool", None)
        if pool is None:
            return None, None
        stats = pool.get_stats()
        available = stats.get("pool_available")
        size = stats.get("pool_size")
        used = None
        if isinstance(available, int) and isinstance(size, int):
            used = max(0, size - available)
        return available, used

    def _close_connection(self, conn):
        """Close a connection"""
        if not conn:
            self.logger.debug(
                f"Connection is None or already closed, nothing to do, conn: {conn}"
            )
            return
        try:
            if not conn.closed:
                tx_status = conn.pgconn.transaction_status
                if tx_status != pq.TransactionStatus.IDLE:
                    stack_trace = "".join(traceback.format_stack())
                    self.logger.warning(
                        f"Returning connection to pool in non-idle state (status: {tx_status}). "
                        "Forcing rollback."
                    )
                    self.logger.warning(
                        f"Call stack leading to uncommitted transaction:\n {stack_trace}"
                    )
                    conn.rollback()
            self.postgreSQL_pool.putconn(conn)
        except (psycopg.OperationalError, psycopg.InterfaceError) as e:
            self.logger.warning(
                f"Handling connection error: {e}. Discarding invalid connection."
            )
            try:
                conn.close()
            finally:
                self.postgreSQL_pool.putconn(conn)
        except Exception as e:
            self.logger.error(f"Unexpected error closing connection: {e}")

    def _close_cursor(self, cursor):
        """Close a cursor"""
        try:
            if cursor and not cursor.closed:
                cursor.close()
        except Exception as e:
            self.logger.warning(f"Failed to close cursor: {e}")

    def _get_connection(self):
        """
        Get a connection from the pool with proper transaction state management.
        Ensures connection is in a clean state for new operations.
        """
        start = time.perf_counter()
        timeout = float(getattr(self, "_pg_pool_acquire_timeout_seconds", 30.0))
        connection = None

        try:
            connection = self.postgreSQL_pool.getconn(timeout=timeout)
        except PoolTimeout as error:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            available, used = self._pool_counts()
            scheduler_trace(
                "postgres_pool_acquire_timeout",
                max_connections=getattr(self.postgreSQL_pool, "max_size", None),
                available_connections=available,
                used_connections=used,
                elapsed_ms=elapsed_ms,
            )
            self.logger.error(
                "Timed out waiting for PostgreSQL connection pool capacity "
                f"after {elapsed_ms:.1f}ms "
                f"(available={available}, used={used}, "
                f"max={getattr(self.postgreSQL_pool, 'max_size', None)})"
            )
            raise PoolTimeout(
                "Timed out waiting for PostgreSQL connection pool capacity "
                f"after {elapsed_ms:.1f}ms"
            ) from error

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        available, used = self._pool_counts()
        if elapsed_ms >= (
            float(getattr(self, "_pg_pool_acquire_trace_after_seconds", 0.001)) * 1000.0
        ):
            scheduler_trace(
                "postgres_pool_acquire_wait_done",
                max_connections=getattr(self.postgreSQL_pool, "max_size", None),
                available_connections=available,
                used_connections=used,
                elapsed_ms=elapsed_ms,
            )

        warn_after_ms = (
            float(getattr(self, "_pg_pool_acquire_warn_after_seconds", 1.0)) * 1000.0
        )
        if elapsed_ms >= warn_after_ms:
            self.logger.warning(
                "Waited %.1fms for PostgreSQL connection pool capacity "
                "(available=%s, used=%s, max=%s)",
                elapsed_ms,
                available,
                used,
                getattr(self.postgreSQL_pool, "max_size", None),
            )

        try:
            if connection.closed:
                raise psycopg.OperationalError(
                    "PostgreSQL pool returned a closed connection"
                )

            tx_status = connection.pgconn.transaction_status
            if tx_status != pq.TransactionStatus.IDLE:
                stack_trace = "".join(traceback.format_stack())
                self.logger.warning(
                    f"Connection from pool has active transaction (status: {tx_status}). "
                    "Rolling back and cleaning up."
                )
                self.logger.warning(
                    f"Call stack leading to uncommitted transaction:\n {stack_trace}"
                )
                connection.rollback()

            connection.autocommit = False
            return connection
        except Exception:
            if connection is not None:
                self.postgreSQL_pool.putconn(connection)
            raise

    @contextmanager
    def _read_connection(self):
        """Borrow a pool connection for a read-only, autocommit operation."""
        connection = self._get_connection()
        try:
            connection.autocommit = True
            yield connection
        finally:
            self._close_connection(connection)

    @property
    def qualified_table(self) -> str:
        """Return the fully qualified table name (schema.table or just table)."""
        if hasattr(self, 'schema') and self.schema:
            return f"{self.schema}.{self.table}"
        return self.table

    def _ensure_schema_exists(self) -> None:
        """Create the schema if it doesn't exist."""
        if not hasattr(self, 'schema') or not self.schema:
            return
        self._execute_sql_gracefully(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        self.logger.info(f"Ensured schema exists: {self.schema}")

    def _init_table(
        self,
        create_table_callback: Optional[Callable] = None,
        reset_table_callback: Optional[Callable] = None,
    ) -> None:
        """
        Use table if exists or create one if it doesn't.
        """
        if reset_table_callback:
            self.logger.info(f"Resetting table : {self.qualified_table}")
            reset_table_callback()

        if self._table_exists():
            self.logger.info(f"Using existing table : {self.qualified_table}")
        else:
            self._create_table_with_callback(create_table_callback)

    def _create_table_with_callback(
        self, create_table_callback: Optional[Callable] = None
    ) -> None:
        """
        Create table if it doesn't exist.
        @param create_table_callback:
        @return:
        """

        if create_table_callback:
            create_table_callback(self.table)

    def diagnose_pool(self):
        """Debug connection pool status with transaction state details."""
        if hasattr(self, 'postgreSQL_pool'):
            pool = self.postgreSQL_pool
            print(f"Pool - Min: {pool.min_size}, Max: {pool.max_size}")
            for key, value in sorted(pool.get_stats().items()):
                print(f"{key}: {value}")
        else:
            print("No PostgreSQL pool found")

    def get_pool_status(self):
        """Get current pool status for programmatic use."""
        if hasattr(self, 'postgreSQL_pool'):
            pool = self.postgreSQL_pool
            return {
                'minconn': pool.min_size,
                'maxconn': pool.max_size,
                'closed': pool.closed,
                **pool.get_stats(),
            }
        return None

    def _table_exists(self) -> bool:
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            # Check with schema if specified, otherwise just table name
            if hasattr(self, 'schema') and self.schema:
                cursor = self._execute_sql_gracefully(
                    "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_schema=%s AND table_name=%s)",
                    (self.schema, self.table),
                    return_cursor=True,
                    connection=conn,
                )
            else:
                cursor = self._execute_sql_gracefully(
                    "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)",
                    (self.table,),
                    return_cursor=True,
                    connection=conn,
                )
            return cursor.fetchall()[0][0]
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def _execute_sql_gracefully(
        self,
        statement: object,
        data: object = tuple(),
        *,
        named_cursor_name: Optional[str] = None,
        itersize: Optional[int] = 10000,
        connection: Optional[psycopg.Connection] = None,
        max_retries: int = 3,
        return_cursor: bool = False,
        commit: bool = True,
    ) -> Any:
        # A cursor cannot be returned if this function is responsible for the connection,
        # as the connection would be closed in the 'finally' block, rendering the cursor useless.
        if return_cursor and connection is None:
            raise ValueError(
                "A connection must be provided when 'return_cursor' is True."
            )

        owns_connection = connection is None
        conn = connection or self._get_connection()
        operation_started = time.perf_counter()
        try:
            for attempt in range(max_retries):
                cursor = None
                try:
                    cursor = (
                        conn.cursor(named_cursor_name)
                        if named_cursor_name
                        else conn.cursor()
                    )
                    if named_cursor_name:
                        cursor.itersize = itersize

                    if data and data != statement:
                        cursor.execute(statement, data)
                    else:
                        cursor.execute(statement)
                    if commit:
                        conn.commit()

                    if return_cursor:
                        scheduler_trace(
                            "postgres_operation",
                            operation="execute_sql_gracefully",
                            statement_count=1,
                            rows_read=None,
                            rows_written=None,
                            elapsed_ms=(time.perf_counter() - operation_started)
                            * 1000.0,
                        )
                        return cursor
                    else:
                        # Get results and close cursor
                        try:
                            if cursor.description:
                                results = cursor.fetchall()
                            else:
                                results = cursor.rowcount
                            scheduler_trace(
                                "postgres_operation",
                                operation="execute_sql_gracefully",
                                statement_count=1,
                                rows_read=(
                                    len(results) if isinstance(results, list) else 0
                                ),
                                rows_written=(
                                    0 if cursor.description else max(0, cursor.rowcount)
                                ),
                                elapsed_ms=(time.perf_counter() - operation_started)
                                * 1000.0,
                            )
                            return results
                        finally:
                            cursor.close()

                except psycopg.InterfaceError as error:
                    self._close_cursor(cursor)

                    if "connection already closed" not in str(error):
                        self._safe_rollback(conn)
                        raise

                    # We can only retry if we own the connection.
                    if not owns_connection or attempt >= max_retries - 1:
                        raise  # Can't retry external connections or on the last attempt.

                    self.logger.warning(
                        f"Connection closed, retrying ({attempt + 1}/{max_retries})"
                    )
                    # discard the broken connection before acquiring a new one.
                    self._close_connection(conn)
                    conn = self._get_connection()

                except Exception as error:
                    self.logger.error(f"SQL error: {error}")
                    self._safe_rollback(conn)
                    self._close_cursor(cursor)
                    raise
            return None
        finally:
            # If we acquired the connection, we are responsible for closing it.
            if owns_connection:
                self._close_connection(conn)

    def _safe_rollback(self, conn):
        """Rollback without raising on closed connections."""
        try:
            conn.rollback()
        except psycopg.InterfaceError:
            pass  # Connection already closed
