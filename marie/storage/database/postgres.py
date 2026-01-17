import traceback
from typing import Dict, Any, Callable, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool  # noqa: F401

from marie.excepts import BadConfigSource
import traceback


class PostgresqlMixin:
    """Bind PostgreSQL database provider."""

    provider = "postgres"

    def _setup_storage(
            self,
            config: Dict[str, Any],
            create_table_callback: Optional[Callable] = None,
            reset_table_callback: Optional[Callable] = None,
            connection_only=False,
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
            max_connections = int(config.get("max_connections", 10))
            min_connections = int(config.get("min_connections", 1))
            application_name = config.get("application_name", "marie_scheduler")

            # ThreadedConnectionPool
            self.postgreSQL_pool = psycopg2.pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                user=username,
                password=password,
                database=database,
                host=hostname,
                port=port,
                # Add connection validation
                # HINT:  Available values: serializable, repeatable read, read committed, read uncommitted.
                **{
                    'options': '-c timezone=UTC',
                    'application_name': application_name
                }
            )

            if connection_only:
                self.logger.info(f"Connected to postgresql database: {config}")
                return

            self.schema = config.get("schema")  # Optional schema name
            self.table = config["default_table"]
            self.logger.info(f"[DEBUG] PostgresqlMixin config: {config}")
            self.logger.info(f"[DEBUG] PostgresqlMixin schema={self.schema}, table={self.table}, qualified_table={self.schema}.{self.table if self.schema else self.table}")
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

    def _close_connection(self, conn):
        """Close a connection"""
        if not conn or conn.closed:
            self.logger.debug(
                f"Connection is None or already closed, nothing to do, conn: {conn}"
            )
            return

        try:
            from psycopg2.extensions import STATUS_IN_TRANSACTION
            from psycopg2.extensions import STATUS_READY

            if conn.status != STATUS_IN_TRANSACTION:
                if conn.status == STATUS_READY:
                    self.logger.debug(
                        "Returning connection to pool in idle state (STATUS_READY)"
                    )
                else:
                    self.logger.warning(
                        f"Returning connection to pool in non-idle state (status: {conn.status})."
                    )
            else:
                # this is a problem, we have a transaction that is not committed
                # we should not be in this state, but if we are, we should roll back
                stack_trace = "".join(traceback.format_stack())
                self.logger.warning(
                    f"Returning connection to pool in non-idle state (status: {conn.status}). Forcing rollback."
                )
                self.logger.warning(
                    f"Call stack leading to uncommitted transaction:\n {stack_trace}"
                )
                conn.rollback()
            self.postgreSQL_pool.putconn(conn)
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            self.logger.warning(
                f"Handling connection error: {e}. Discarding invalid connection."
            )
            if self.postgreSQL_pool:
                self.postgreSQL_pool.putconn(conn, close=True)
        except psycopg2.pool.PoolError as e:
            self.logger.warning(f"Error returning connection to pool: {e}")
            self.diagnose_pool()
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
        connection = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                connection = self.postgreSQL_pool.getconn()
                if connection.closed:
                    self.postgreSQL_pool.putconn(connection, close=True)
                    continue
                tx_status = connection.get_transaction_status()

                if tx_status == psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                    # Connection is clean, safe to set autocommit
                    connection.autocommit = False
                    return connection

                elif tx_status in (
                        psycopg2.extensions.TRANSACTION_STATUS_INTRANS,
                        psycopg2.extensions.TRANSACTION_STATUS_INERROR
                ):
                    # Log warning - this shouldn't happen with proper pool management
                    self.logger.warning(
                        f"Connection from pool has active transaction (status: {tx_status}). "
                        f"Rolling back and cleaning up. Attempt {attempt + 1}/{max_retries}"
                    )

                    # Rollback and reset
                    connection.rollback()
                    connection.autocommit = False
                    return connection

                else:
                    # Unknown state, return connection to pool and get a new one
                    self.logger.error(f"Connection in unknown state: {tx_status}")
                    self.postgreSQL_pool.putconn(connection, close=True)
                    continue

            except psycopg2.OperationalError as e:
                self.logger.error(f"Connection error on attempt {attempt + 1}: {e}")
                if connection:
                    self.postgreSQL_pool.putconn(connection, close=True)
                if attempt == max_retries - 1:
                    raise
                continue

            except Exception as e:
                self.logger.error(f"Unexpected error getting connection: {e}")
                if connection:
                    self.postgreSQL_pool.putconn(connection, close=True)
                raise

        raise psycopg2.OperationalError("Failed to get clean connection after maximum retries")

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
        self._execute_sql_gracefully(
            f"CREATE SCHEMA IF NOT EXISTS {self.schema}"
        )
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

    def _create_table_with_callback(self, create_table_callback: Optional[Callable] = None) -> None:
        """
        Create table if it doesn't exist.
        @param create_table_callback:
        @return:
        """

        if create_table_callback:
            create_table_callback(self.table)

    def diagnose_pool(self):
        """Debug connection pool status with transaction state details."""
        # Transaction status mapping from psycopg2
        TRANSACTION_STATUS_NAMES = {
            0: "IDLE",
            1: "ACTIVE",
            2: "INTRANS",
            3: "INERROR",
            4: "UNKNOWN"
        }

        if hasattr(self, 'postgreSQL_pool'):
            pool = self.postgreSQL_pool
            print(f"Pool - Min: {pool.minconn}, Max: {pool.maxconn}")
            print(f"Available: {len(pool._pool)}, Used: {len(pool._used)}")

            # Check available connections
            print("\nAvailable connections:")
            for i, conn in enumerate(list(pool._pool)):
                status_code = getattr(conn, 'status', 4)
                status_name = TRANSACTION_STATUS_NAMES.get(status_code, f"UNKNOWN({status_code})")
                print(f"  Connection {i}: {status_name} (code: {status_code})")

            # Check used connections
            print("\nUsed connections:")
            for key, conn in list(pool._used.items()):
                status_code = getattr(conn, 'status', 4)
                status_name = TRANSACTION_STATUS_NAMES.get(status_code, f"UNKNOWN({status_code})")
                print(f"  Connection {key}: {status_name} (code: {status_code})")

                # Check for open cursors (if accessible)
                if hasattr(conn, '_cursor_count'):
                    print(f"    Open cursors: {conn._cursor_count}")

        else:
            print("No PostgreSQL pool found")

    def get_pool_status(self):
        """Get current pool status for programmatic use."""
        TRANSACTION_STATUS_NAMES = {
            0: "IDLE",
            1: "ACTIVE",
            2: "INTRANS",
            3: "INERROR",
            4: "UNKNOWN"
        }

        if hasattr(self, 'postgreSQL_pool'):
            pool = self.postgreSQL_pool

            available_connections = []
            for conn in pool._pool:
                status_code = getattr(conn, 'status', 4)
                available_connections.append({
                    'status_code': status_code,
                    'status_name': TRANSACTION_STATUS_NAMES.get(status_code, f"UNKNOWN({status_code})")
                })

            used_connections = []
            for key, conn in pool._used.items():
                status_code = getattr(conn, 'status', 4)
                used_connections.append({
                    'key': key,
                    'status_code': status_code,
                    'status_name': TRANSACTION_STATUS_NAMES.get(status_code, f"UNKNOWN({status_code})")
                })

            return {
                'minconn': pool.minconn,
                'maxconn': pool.maxconn,
                'closed': getattr(pool, 'closed', 'unknown'),
                'available_count': len(pool._pool),
                'used_count': len(pool._used),
                'available_connections': available_connections,
                'used_connections': used_connections
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
                    connection=conn
                )
            else:
                cursor = self._execute_sql_gracefully(
                    "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)",
                    (self.table,),
                    return_cursor=True,
                    connection=conn
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
            connection: Optional[psycopg2.extensions.connection] = None,
            max_retries: int = 3,
            return_cursor: bool = False,
    ) -> psycopg2.extras.DictCursor | list | int | None:
        # A cursor cannot be returned if this function is responsible for the connection,
        # as the connection would be closed in the 'finally' block, rendering the cursor useless.
        if return_cursor and connection is None:
            raise ValueError(
                "A connection must be provided when 'return_cursor' is True."
            )

        owns_connection = connection is None
        conn = connection or self._get_connection()
        try:
            for attempt in range(max_retries):
                cursor = None
                try:
                    cursor = conn.cursor(named_cursor_name) if named_cursor_name else conn.cursor()
                    if named_cursor_name:
                        cursor.itersize = itersize

                    if data and data != statement:
                        cursor.execute(statement, data)
                    else:
                        cursor.execute(statement)
                    conn.commit()

                    if return_cursor:
                        return cursor
                    else:
                        # Get results and close cursor
                        try:
                            if cursor.description:
                                results = cursor.fetchall()
                            else:
                                results = cursor.rowcount
                            return results
                        finally:
                            cursor.close()

                except psycopg2.InterfaceError as error:
                    self._close_cursor(cursor)

                    if "connection already closed" not in str(error):
                        self._safe_rollback(conn)
                        raise

                    # We can only retry if we own the connection.
                    if not owns_connection or attempt >= max_retries - 1:
                        raise  # Can't retry external connections or on the last attempt.

                    self.logger.warning(f"Connection closed, retrying ({attempt + 1}/{max_retries})")
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
        except psycopg2.InterfaceError:
            pass  # Connection already closed
