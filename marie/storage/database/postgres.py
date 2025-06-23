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
            self.postgreSQL_pool.putconn(connection)
        except Exception as e:
            self.logger.warning(f"Error returning connection to pool: {e}")
            try:
                connection.close()
            except:
                pass  # Connection might already be closed

    def _get_connection(self):
        # by default psycopg2 is not auto-committing
        # this means we can have rollbacks
        # and maintain ACID-ity
        connection = self.postgreSQL_pool.getconn()
        connection.autocommit = False
        return connection

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

    def _table_exists(self) -> bool:
        return self._execute_sql_gracefully(
            "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)",
            (self.table,),
        ).fetchall()[0][0]

    def _execute_sql_gracefully(
            self,
            statement: object,
            data: object = tuple(),
            *,
            named_cursor_name: Optional[str] = None,
            itersize: Optional[int] = 10000,
            connection: Optional[psycopg2.extensions.connection] = None,
            max_retries: int = 3,
    ) -> psycopg2.extras.DictCursor:
        conn = connection or self.connection

        for attempt in range(max_retries):
            try:
                cursor = conn.cursor(named_cursor_name) if named_cursor_name else conn.cursor()
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
