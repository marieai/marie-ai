"""Async PostgreSQL connection pool using asyncpg.

This module provides a singleton asyncpg pool for the new database-backed agent tools,
keeping the existing psycopg2-based code unchanged.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import asyncpg

from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.storage.database.asyncpg_pool")


class AsyncPostgresPool:
    """Singleton asyncpg pool with lazy initialization.

    Usage:
        pool = AsyncPostgresPool.get_instance()
        await pool.initialize(config)

        # Use convenience methods
        rows = await pool.fetch("SELECT * FROM users WHERE id = $1", user_id)

        # Or acquire connection directly
        async with pool.acquire() as conn:
            await conn.execute("INSERT INTO ...")

        # Shutdown when done
        await AsyncPostgresPool.shutdown()
    """

    _instance: Optional[AsyncPostgresPool] = None
    _pool: Optional[asyncpg.Pool] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> AsyncPostgresPool:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> AsyncPostgresPool:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize pool with config matching PostgresqlMixin pattern.

        Args:
            config: Database configuration dictionary with keys:
                - hostname: Database host
                - port: Database port (int)
                - username: Database user
                - password: Database password
                - database: Database name
                - min_connections: Minimum pool size (default: 1)
                - max_connections: Maximum pool size (default: 10)
        """
        if self._pool is not None:
            return

        self._config = config
        logger.info(f"Initializing asyncpg pool for database: {config.get('database')}")

        self._pool = await asyncpg.create_pool(
            host=config["hostname"],
            port=int(config["port"]),
            user=config["username"],
            password=config["password"],
            database=config["database"],
            min_size=config.get("min_connections", 1),
            max_size=config.get("max_connections", 10),
            command_timeout=60,
        )
        logger.info("asyncpg pool initialized")

    @property
    def is_initialized(self) -> bool:
        """Check if the pool is initialized."""
        return self._pool is not None

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool.

        Usage:
            async with pool.acquire() as conn:
                await conn.execute("...")
        """
        if self._pool is None:
            raise RuntimeError("Pool not initialized. Call initialize() first.")
        async with self._pool.acquire() as conn:
            yield conn

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Execute a query and return all rows.

        Args:
            query: SQL query with $1, $2, ... placeholders
            *args: Query parameters

        Returns:
            List of Record objects
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Execute a query and return the first row.

        Args:
            query: SQL query with $1, $2, ... placeholders
            *args: Query parameters

        Returns:
            Single Record or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Execute a query and return a single value.

        Args:
            query: SQL query with $1, $2, ... placeholders
            *args: Query parameters

        Returns:
            Single value from first row, first column
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the status.

        Args:
            query: SQL query with $1, $2, ... placeholders
            *args: Query parameters

        Returns:
            Status string (e.g., "INSERT 0 1")
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def executemany(self, query: str, args: List[tuple]) -> None:
        """Execute a query multiple times with different parameters.

        Args:
            query: SQL query with $1, $2, ... placeholders
            args: List of parameter tuples
        """
        async with self.acquire() as conn:
            await conn.executemany(query, args)

    @classmethod
    async def shutdown(cls) -> None:
        """Close the pool and release all connections."""
        if cls._instance and cls._instance._pool:
            logger.info("Shutting down asyncpg pool")
            await cls._instance._pool.close()
            cls._instance._pool = None
            cls._instance._config = None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._pool = None
        cls._config = None
