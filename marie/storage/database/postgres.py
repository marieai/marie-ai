from typing import Dict, Any, Callable, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool  # noqa: F401

from marie.excepts import BadConfigSource


class PostgresqlMixin:
    """Bind PostgreSQL database provider."""

    provider = 'postgres'

    def _setup_storage(self, config: Dict[str, Any], create_table_callback: Optional[Callable] = None,
                       reset_table_callback: Optional[Callable] = None) -> None:
        """
        Setup PostgreSQL connection pool.

        @param config:
        @param create_table_callback: Create table if it doesn't exist.
        @param reset_table_callback:  Reset table if it exists.
        @return:
        """
        try:
            hostname = config["hostname"]
            port = int(config["port"])
            username = config["username"]
            password = config["password"]
            database = config["database"]
            self.table = config["default_table"]

            if self.table is None or self.table == "":
                raise ValueError("default_table cannot be empty")

            max_connections = 10
            self.postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(
                1,
                max_connections,
                user=username,
                password=password,
                database=database,
                host=hostname,
                port=port,
            )
            self._init_table(create_table_callback, reset_table_callback)

        except Exception as e:
            raise BadConfigSource(
                f'Cannot connect to postgresql database: {config}, {e}'
            )

    def __enter__(self):
        self.connection = self._get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self._close_connection(self.connection)

    def _close_connection(self, connection):
        # restore it to the pool
        self.postgreSQL_pool.putconn(connection)

    def _get_connection(self):
        # by default psycopg2 is not auto-committing
        # this means we can have rollbacks
        # and maintain ACID-ity
        connection = self.postgreSQL_pool.getconn()
        connection.autocommit = False
        return connection

    def _init_table(self, create_table_callback: Optional[Callable] = None,
                    reset_table_callback: Optional[Callable] = None) -> None:
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

    def _execute_sql_gracefully(self, statement, data=tuple(), *,
                                named_cursor_name: Optional[str] = None,
                                itersize: Optional[int] = 10000) -> psycopg2.extras.DictCursor:
        try:
            if named_cursor_name:
                cursor = self.connection.cursor(named_cursor_name)
                cursor.itersize = itersize
            else:
                cursor = self.connection.cursor()

            if data:
                cursor.execute(statement, data)
            else:
                cursor.execute(statement)
        except psycopg2.errors.UniqueViolation as error:
            self.logger.debug(f"Error while executing {statement}: {error}.")

        self.connection.commit()
        return cursor
