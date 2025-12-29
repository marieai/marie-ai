"""
ClickHouse Client Manager - Connection management for ClickHouse.

Provides a singleton client for ClickHouse connections, ported from
Langfuse's TypeScript ClickHouseClientManager.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from marie.llm_tracking.config import get_settings

logger = logging.getLogger(__name__)


class ClickHouseClientManager:
    """
    Singleton ClickHouse client manager.

    Provides connection pooling and query execution for ClickHouse.
    Used by the ClickHouseWriter for batched inserts and by marie-studio
    for read queries.
    """

    _instance: Optional["ClickHouseClientManager"] = None

    def __new__(cls) -> "ClickHouseClientManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the client manager."""
        if getattr(self, "_initialized", False):
            return

        self._settings = get_settings()
        self._client: Optional[Any] = None
        self._started = False
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ClickHouseClientManager":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance is not None:
            cls._instance.stop()
            cls._instance = None

    def start(self) -> None:
        """Initialize the ClickHouse client."""
        if self._started:
            return

        try:
            from clickhouse_driver import Client

            self._client = Client(
                host=self._settings.CLICKHOUSE_HOST,
                port=self._settings.CLICKHOUSE_NATIVE_PORT,
                database=self._settings.CLICKHOUSE_DATABASE,
                user=self._settings.CLICKHOUSE_USER,
                password=self._settings.CLICKHOUSE_PASSWORD,
                settings={
                    "async_insert": 1,  # Enable async inserts
                    "wait_for_async_insert": 0,  # Don't wait for confirmation
                },
            )

            # Test connection
            result = self._client.execute("SELECT 1")
            if result != [(1,)]:
                raise RuntimeError("ClickHouse connection test failed")

            self._started = True
            logger.info(
                f"ClickHouse client started: "
                f"{self._settings.CLICKHOUSE_HOST}:{self._settings.CLICKHOUSE_NATIVE_PORT}/"
                f"{self._settings.CLICKHOUSE_DATABASE}"
            )
        except ImportError:
            raise ImportError(
                "clickhouse-driver is required for ClickHouse support. "
                "Install with: pip install clickhouse-driver"
            )
        except Exception as e:
            logger.error(f"Failed to start ClickHouse client: {e}")
            raise

    def stop(self) -> None:
        """Close the ClickHouse connection."""
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting ClickHouse client: {e}")
            finally:
                self._client = None
                self._started = False
        logger.debug("ClickHouse client stopped")

    def _ensure_started(self) -> None:
        """Ensure the client is started."""
        if not self._started:
            self.start()

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple]:
        """
        Execute a query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result tuples
        """
        self._ensure_started()
        assert self._client is not None

        try:
            if params:
                return self._client.execute(query, params)
            return self._client.execute(query)
        except Exception as e:
            logger.error(f"ClickHouse query failed: {e}")
            raise

    def insert(
        self,
        table: str,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> int:
        """
        Insert data into a table.

        Args:
            table: Table name
            data: List of row dictionaries
            columns: Column names (extracted from data if not provided)

        Returns:
            Number of rows inserted
        """
        if not data:
            return 0

        self._ensure_started()
        assert self._client is not None

        # Extract columns from first row if not provided
        if columns is None:
            columns = list(data[0].keys())

        # Convert dicts to tuples in column order
        rows = [tuple(row.get(col) for col in columns) for row in data]

        try:
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES"
            self._client.execute(query, rows)
            return len(rows)
        except Exception as e:
            logger.error(f"ClickHouse insert failed: {e}")
            raise

    def execute_with_progress(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple]:
        """
        Execute a query with progress tracking.

        Useful for long-running queries.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result tuples
        """
        self._ensure_started()
        assert self._client is not None

        try:
            return self._client.execute(
                query,
                params or {},
                with_column_types=False,
                external_tables=None,
                query_id=None,
                settings={"send_progress_in_http_headers": 1},
            )
        except Exception as e:
            logger.error(f"ClickHouse query failed: {e}")
            raise

    def ping(self) -> bool:
        """
        Check if the connection is alive.

        Returns:
            True if connection is healthy
        """
        try:
            self._ensure_started()
            result = self._client.execute("SELECT 1")
            return result == [(1,)]
        except Exception:
            return False

    def get_table_columns(self, table: str) -> List[str]:
        """
        Get column names for a table.

        Args:
            table: Table name

        Returns:
            List of column names
        """
        self._ensure_started()
        result = self.execute(
            "SELECT name FROM system.columns WHERE table = %(table)s",
            {"table": table},
        )
        return [row[0] for row in result]

    def table_exists(self, table: str) -> bool:
        """
        Check if a table exists.

        Args:
            table: Table name

        Returns:
            True if table exists
        """
        self._ensure_started()
        result = self.execute(
            "SELECT count() FROM system.tables WHERE name = %(table)s",
            {"table": table},
        )
        return result[0][0] > 0


# Convenience function
def get_clickhouse_client() -> ClickHouseClientManager:
    """Get the singleton ClickHouse client."""
    return ClickHouseClientManager.get_instance()
