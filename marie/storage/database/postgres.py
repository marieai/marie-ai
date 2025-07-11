import traceback
from typing import Dict, Any, Callable, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool  # noqa: F401

from marie.excepts import BadConfigSource


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

            # ThreadedConnectionPool
            self.postgreSQL_pool = psycopg2.pool.ThreadedConnectionPool(
                1,
                max_connections,
                user=username,
                password=password,
                database=database,
                host=hostname,
                port=port,
                # Add connection validation
                # HINT:  Available values: serializable, repeatable read, read committed, read uncommitted.
                **{
                    # 'options': "-c default_transaction_isolation=read committed",
                    'application_name': 'marie_scheduler'
                }
            )

            if connection_only:
                self.logger.info(f"Connected to postgresql database: {config}")
                return

            self.table = config["default_table"]
            if self.table is None or self.table == "":
                raise ValueError("default_table cannot be empty")

            self._init_table(create_table_callback, reset_table_callback)

        except Exception as e:
            raise BadConfigSource(
                f"Cannot connect to postgresql database: {config}, {e}"
            )

    def __enter__(self):
        self.connection = self._get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self._close_connection(self.connection)

    def _close_connection(self, connection):
        """Return connection to pool """
        try:
            # restore it to the pool
            if connection:
                self.postgreSQL_pool.putconn(connection)
        except Exception as e:
            self.logger.warning(f"Error returning connection to pool: {e}")
            try:
                connection.close()
            except:
                pass  # Connection might already be closed

    def _close_cursor(self, cursor):
        """Close cursor gracefully."""
        try:
            if cursor:
                cursor.close()
        except Exception as e:
            pass

    def _get_connectionXX(self):
        # by default psycopg2 is not auto-committing
        # this means we can have rollbacks
        # and maintain ACID-ity
        connection = self.postgreSQL_pool.getconn()
        connection.autocommit = False
        return connection

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

    def _init_table(
            self,
            create_table_callback: Optional[Callable] = None,
            reset_table_callback: Optional[Callable] = None,
    ) -> None:
        """
        Use table if exists or create one if it doesn't.
        """
        with self:

            if reset_table_callback:
                self.logger.info(f"Resetting table : {self.table}")
                reset_table_callback()

            if self._table_exists():
                self.logger.info(f"Using existing table : {self.table}")
            else:
                self._create_table(create_table_callback)

    def _create_table(self, create_table_callback: Optional[Callable] = None) -> None:
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
            for i, conn in enumerate(pool._pool):
                status_code = getattr(conn, 'status', 4)
                status_name = TRANSACTION_STATUS_NAMES.get(status_code, f"UNKNOWN({status_code})")
                print(f"  Connection {i}: {status_name} (code: {status_code})")

            # Check used connections
            print("\nUsed connections:")
            for key, conn in pool._used.items():
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
        with self:
            cursor = None
            try:
                cursor = self._execute_sql_gracefully(
                    "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)",
                    (self.table,), return_cursor=True
                )
                return cursor.fetchall()[0][0]
            finally:
                self._close_cursor(cursor)

    def _execute_sql_gracefullyX(
            self,
            statement: object,
            data: object = tuple(),
            *,
            named_cursor_name: Optional[str] = None,
            itersize: Optional[int] = 10000,
            connection: Optional[psycopg2.extensions.connection] = None,
            max_retries: int = 3,
    ) -> psycopg2.extras.DictCursor | None:
        conn = connection or self.connection

        for attempt in range(max_retries):
            try:
                cursor = conn.cursor(named_cursor_name,
                                     cursor_factory=psycopg2.extras.RealDictCursor) if named_cursor_name else conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor)
                if named_cursor_name:
                    cursor.itersize = itersize
                cursor.execute(statement, data if data else statement)
                conn.commit()

                return cursor

            except psycopg2.InterfaceError as error:
                if "connection already closed" not in str(error):
                    self._safe_rollback(conn)
                    raise

                # Connection closed - try to get new one
                if connection is not None or attempt == max_retries - 1:
                    raise  # Can't retry external connections or last attempt

                self.logger.warning(f"Connection closed, retrying ({attempt + 1}/{max_retries})")
                conn = self._get_fresh_connection()

            except Exception as error:
                self.logger.error(f"SQL error: {error}")
                self._safe_rollback(conn)
                raise

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
        conn = connection or self.connection

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

                # Connection closed - try to get new one
                if connection is not None or attempt == max_retries - 1:
                    raise  # Can't retry external connections or last attempt

                self.logger.warning(f"Connection closed, retrying ({attempt + 1}/{max_retries})")
                conn = self._get_fresh_connection()
            except Exception as error:
                self.logger.error(f"SQL error: {error}")
                self._safe_rollback(conn)
                self._close_cursor(cursor)
                raise
        return None

    def _safe_rollback(self, conn):
        """Rollback without raising on closed connections."""
        try:
            conn.rollback()
        except psycopg2.InterfaceError:
            pass  # Connection already closed

    def _get_fresh_connection(self):
        """Get a new connection from pool and update self.connection."""
        self._close_connection(self.connection)
        self.connection = self._get_connection()
        return self.connection
