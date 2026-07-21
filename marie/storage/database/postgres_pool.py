"""Psycopg 3 PostgreSQL pools for synchronous and asynchronous callers."""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import psycopg
from psycopg.rows import AsyncRowFactory, DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool, PoolTimeout

from marie.utils.scheduler_trace import scheduler_trace

_PARAMETER = re.compile(r"\$(\d+)")


def _connection_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "host": config.get("hostname", config.get("host")),
        "port": int(config["port"]),
        "user": config.get("username", config.get("user")),
        "password": config["password"],
        "dbname": config["database"],
        "application_name": config.get("application_name", "marie"),
        "options": config.get("options", "-c timezone=UTC"),
        "row_factory": dict_row,
    }


def _sizes(config: Mapping[str, Any]) -> tuple[int, int]:
    return (
        int(config.get("min_connections", config.get("min_pool_size", 1))),
        int(config.get("max_connections", config.get("max_pool_size", 10))),
    )


def _normalize(query: str, args: Sequence[Any]) -> tuple[str, tuple[Any, ...] | None]:
    if not args:
        return query, None

    indexes: list[int] = []

    def replace(match: re.Match[str]) -> str:
        indexes.append(int(match.group(1)) - 1)
        return "%s"

    normalized = _PARAMETER.sub(replace, query)
    if not indexes:
        return query, tuple(args)
    try:
        return normalized, tuple(args[index] for index in indexes)
    except IndexError as error:
        raise ValueError("SQL placeholder references a missing argument") from error


def _first_value(row: Any) -> Any:
    if row is None:
        return None
    if isinstance(row, Mapping):
        return next(iter(row.values()))
    return row[0]


class PostgresPool:
    """Native synchronous psycopg 3 connection pool."""

    def __init__(self) -> None:
        self._pool: ConnectionPool[Any] | None = None

    def initialize(self, config: Mapping[str, Any]) -> None:
        if self._pool is not None:
            return
        minimum, maximum = _sizes(config)
        pool = ConnectionPool(
            "",
            min_size=minimum,
            max_size=maximum,
            kwargs=_connection_kwargs(config),
            open=False,
            timeout=float(config.get("pool_acquire_timeout_seconds", 30.0)),
        )
        pool.open(
            wait=True, timeout=float(config.get("pool_open_timeout_seconds", 10.0))
        )
        self._pool = pool

    @contextmanager
    def acquire(self) -> Iterator[psycopg.Connection[Any]]:
        if self._pool is None:
            raise RuntimeError("Pool not initialized. Call initialize() first.")
        with self._pool.connection() as connection:
            yield connection

    def execute(self, query: str, *args: Any) -> str:
        query, params = _normalize(query, args)
        with self.acquire() as connection:
            return connection.execute(query, params).statusmessage or ""

    def fetch(self, query: str, *args: Any) -> list[DictRow]:
        query, params = _normalize(query, args)
        with self.acquire() as connection:
            return connection.execute(query, params).fetchall()

    def fetchrow(self, query: str, *args: Any) -> DictRow | None:
        query, params = _normalize(query, args)
        with self.acquire() as connection:
            return connection.execute(query, params).fetchone()

    def fetchval(self, query: str, *args: Any) -> Any:
        return _first_value(self.fetchrow(query, *args))

    def executemany(self, query: str, args: Sequence[Sequence[Any]]) -> None:
        rows = [tuple(row) for row in args]
        if not rows:
            return
        normalized_query, _ = _normalize(query, rows[0])
        normalized_rows = [_normalize(query, row)[1] for row in rows]
        with self.acquire() as connection:
            connection.executemany(normalized_query, normalized_rows)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None


class AsyncPostgresPool:
    """Singleton native asynchronous psycopg 3 connection pool."""

    _instance: AsyncPostgresPool | None = None

    def __new__(cls) -> AsyncPostgresPool:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pool = None
        return cls._instance

    @classmethod
    def get_instance(cls) -> AsyncPostgresPool:
        return cls()

    async def initialize(self, config: Mapping[str, Any]) -> None:
        if self._pool is not None:
            return
        minimum, maximum = _sizes(config)
        pool = AsyncConnectionPool(
            "",
            min_size=minimum,
            max_size=maximum,
            kwargs=_connection_kwargs(config),
            open=False,
            timeout=float(config.get("pool_acquire_timeout_seconds", 30.0)),
        )
        await pool.open(
            wait=True, timeout=float(config.get("pool_open_timeout_seconds", 10.0))
        )
        self._pool = pool

    @property
    def is_initialized(self) -> bool:
        return self._pool is not None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
        if self._pool is None:
            raise RuntimeError("Pool not initialized. Call initialize() first.")
        async with self._pool.connection() as connection:
            yield connection

    async def execute(self, query: str, *args: Any) -> str:
        query, params = _normalize(query, args)
        async with self.acquire() as connection:
            return (await connection.execute(query, params)).statusmessage or ""

    async def fetch(self, query: str, *args: Any) -> list[DictRow]:
        query, params = _normalize(query, args)
        async with self.acquire() as connection:
            return await (await connection.execute(query, params)).fetchall()

    async def fetchrow(self, query: str, *args: Any) -> DictRow | None:
        query, params = _normalize(query, args)
        async with self.acquire() as connection:
            return await (await connection.execute(query, params)).fetchone()

    async def fetchval(self, query: str, *args: Any) -> Any:
        return _first_value(await self.fetchrow(query, *args))

    async def executemany(self, query: str, args: Sequence[Sequence[Any]]) -> None:
        rows = [tuple(row) for row in args]
        if not rows:
            return
        normalized_query, _ = _normalize(query, rows[0])
        normalized_rows = [_normalize(query, row)[1] for row in rows]
        async with self.acquire() as connection:
            await connection.executemany(normalized_query, normalized_rows)

    @classmethod
    async def shutdown(cls) -> None:
        if cls._instance is not None and cls._instance._pool is not None:
            await cls._instance._pool.close()
            cls._instance._pool = None

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


class AsyncPostgresConnection:
    """Compatibility-shaped facade over a psycopg 3 async connection."""

    def __init__(self, connection: psycopg.AsyncConnection[Any]) -> None:
        self._connection = connection

    async def execute(self, query: str, *args: Any) -> str:
        started = time.perf_counter()
        query, params = _normalize(query, args)
        cursor = await self._connection.execute(query, params)
        scheduler_trace(
            "postgres_operation",
            operation="execute",
            statement_count=1,
            rows_read=0,
            rows_written=max(0, cursor.rowcount),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return cursor.statusmessage or ""

    async def fetch(self, query: str, *args: Any) -> list[DictRow]:
        started = time.perf_counter()
        query, params = _normalize(query, args)
        rows = await (await self._connection.execute(query, params)).fetchall()
        scheduler_trace(
            "postgres_operation",
            operation="fetch",
            statement_count=1,
            rows_read=len(rows),
            rows_written=0,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return rows

    async def fetchrow(self, query: str, *args: Any) -> DictRow | None:
        started = time.perf_counter()
        query, params = _normalize(query, args)
        row = await (await self._connection.execute(query, params)).fetchone()
        scheduler_trace(
            "postgres_operation",
            operation="fetchrow",
            statement_count=1,
            rows_read=1 if row is not None else 0,
            rows_written=0,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return row

    async def fetchval(self, query: str, *args: Any) -> Any:
        return _first_value(await self.fetchrow(query, *args))

    async def executemany(self, query: str, args: Sequence[Sequence[Any]]) -> None:
        started = time.perf_counter()
        rows = [tuple(row) for row in args]
        if rows:
            legacy_query = query
            query, _ = _normalize(legacy_query, rows[0])
            rows = [_normalize(legacy_query, row)[1] for row in rows]
        await self._connection.executemany(query, rows)
        scheduler_trace(
            "postgres_operation",
            operation="executemany",
            statement_count=len(rows),
            rows_read=0,
            rows_written=len(rows),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def transaction(self):
        return self._connection.transaction()


class AsyncPostgresConnectionPool(AsyncPostgresPool):
    """Non-singleton psycopg 3 pool for independently running event loops."""

    def __new__(cls) -> AsyncPostgresConnectionPool:
        instance = object.__new__(cls)
        instance._pool = None
        return instance

    async def initialize(
        self,
        config: Mapping[str, Any],
        *,
        row_factory: AsyncRowFactory[Any] = dict_row,
        autocommit: bool = False,
    ) -> None:
        if self._pool is not None:
            return
        minimum, maximum = _sizes(config)
        kwargs = _connection_kwargs(config)
        kwargs["row_factory"] = row_factory
        kwargs["autocommit"] = autocommit
        pool = AsyncConnectionPool(
            "",
            min_size=minimum,
            max_size=maximum,
            kwargs=kwargs,
            open=False,
            timeout=float(config.get("pool_acquire_timeout_seconds", 30.0)),
        )
        await pool.open(
            wait=True,
            timeout=float(config.get("pool_open_timeout_seconds", 10.0)),
        )
        self._pool = pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[AsyncPostgresConnection]:
        if self._pool is None:
            raise RuntimeError("Pool not initialized. Call initialize() first.")
        started = time.perf_counter()
        try:
            async with self._pool.connection() as connection:
                scheduler_trace(
                    "postgres_pool_acquire_wait_done",
                    pool="async_scheduler",
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
                yield AsyncPostgresConnection(connection)
        except PoolTimeout:
            scheduler_trace(
                "postgres_pool_acquire_timeout",
                pool="async_scheduler",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            raise

    async def execute(self, query: str, *args: Any) -> str:
        async with self.acquire() as connection:
            return await connection.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        async with self.acquire() as connection:
            return await connection.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        async with self.acquire() as connection:
            return await connection.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        async with self.acquire() as connection:
            return await connection.fetchval(query, *args)

    async def executemany(self, query: str, args: Sequence[Sequence[Any]]) -> None:
        async with self.acquire() as connection:
            await connection.executemany(query, args)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def initialize_dsn(
        self, dsn: str, *, min_size: int = 1, max_size: int = 10
    ) -> None:
        if self._pool is not None:
            return
        pool = AsyncConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        await pool.open(wait=True)
        self._pool = pool


__all__ = [
    "AsyncPostgresConnection",
    "AsyncPostgresConnectionPool",
    "AsyncPostgresPool",
    "PostgresPool",
]
