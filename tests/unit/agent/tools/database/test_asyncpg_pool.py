"""Tests for the psycopg 3 PostgreSQL pools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marie.storage.database.postgres_pool import AsyncPostgresPool, PostgresPool


class AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        return None


class SyncContext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *args):
        return None


@pytest.fixture(autouse=True)
def reset_pool():
    AsyncPostgresPool.reset()
    yield
    AsyncPostgresPool.reset()


@pytest.mark.asyncio
async def test_async_pool_uses_psycopg(db_config):
    driver_pool = MagicMock()
    driver_pool.open = AsyncMock()

    with patch(
        "marie.storage.database.postgres_pool.AsyncConnectionPool",
        return_value=driver_pool,
    ) as factory:
        pool = AsyncPostgresPool.get_instance()
        await pool.initialize(db_config)

    factory.assert_called_once()
    assert factory.call_args.kwargs["kwargs"]["dbname"] == "test_db"
    driver_pool.open.assert_awaited_once_with(wait=True, timeout=10.0)
    assert pool.is_initialized


@pytest.mark.asyncio
async def test_async_pool_translates_numbered_placeholders(db_config):
    cursor = MagicMock()
    cursor.fetchall = AsyncMock(return_value=[{"id": 1}])
    connection = MagicMock()
    connection.execute = AsyncMock(return_value=cursor)
    driver_pool = MagicMock()
    driver_pool.open = AsyncMock()
    driver_pool.connection.return_value = AsyncContext(connection)

    with patch(
        "marie.storage.database.postgres_pool.AsyncConnectionPool",
        return_value=driver_pool,
    ):
        pool = AsyncPostgresPool.get_instance()
        await pool.initialize(db_config)
        rows = await pool.fetch("SELECT * FROM t WHERE a = $1 OR b = $1", 7)

    connection.execute.assert_awaited_once_with(
        "SELECT * FROM t WHERE a = %s OR b = %s", (7, 7)
    )
    assert rows == [{"id": 1}]


def test_sync_pool_is_available_without_event_loop(db_config):
    cursor = MagicMock(statusmessage="SELECT 1")
    connection = MagicMock()
    connection.execute.return_value = cursor
    driver_pool = MagicMock()
    driver_pool.connection.return_value = SyncContext(connection)

    with patch(
        "marie.storage.database.postgres_pool.ConnectionPool",
        return_value=driver_pool,
    ):
        pool = PostgresPool()
        pool.initialize(db_config)
        assert pool.execute("SELECT 1") == "SELECT 1"
        pool.close()

    driver_pool.open.assert_called_once_with(wait=True, timeout=10.0)
    driver_pool.close.assert_called_once()
