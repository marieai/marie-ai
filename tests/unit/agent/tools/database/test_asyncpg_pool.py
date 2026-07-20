"""Tests for the psycopg 3 synchronous and asynchronous pools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marie.storage.database.postgres_pool import AsyncPostgresPool, PostgresPool


class AsyncContextManager:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        return None


class ContextManager:
    def __init__(self, value):
        self._value = value

    def __enter__(self):
        return self._value

    def __exit__(self, *args):
        return None


@pytest.fixture(autouse=True)
def reset_async_singleton():
    AsyncPostgresPool.reset()
    yield
    AsyncPostgresPool.reset()


def test_async_pool_is_singleton():
    assert AsyncPostgresPool.get_instance() is AsyncPostgresPool.get_instance()


@pytest.mark.asyncio
async def test_async_initialize_uses_psycopg_pool(db_config):
    driver_pool = MagicMock()
    driver_pool.open = AsyncMock()

    with patch(
        'marie.storage.database.postgres_pool.AsyncConnectionPool',
        return_value=driver_pool,
    ) as pool_factory:
        pool = AsyncPostgresPool.get_instance()
        await pool.initialize(db_config)
        await pool.initialize(db_config)

    pool_factory.assert_called_once()
    call = pool_factory.call_args
    assert call.kwargs['min_size'] == 1
    assert call.kwargs['max_size'] == 5
    assert call.kwargs['kwargs']['host'] == 'localhost'
    assert call.kwargs['kwargs']['user'] == 'test_user'
    assert call.kwargs['kwargs']['dbname'] == 'test_db'
    driver_pool.open.assert_awaited_once_with(wait=True, timeout=10.0)
    assert pool.is_initialized is True


@pytest.mark.asyncio
async def test_async_pool_requires_initialization():
    pool = AsyncPostgresPool.get_instance()

    with pytest.raises(RuntimeError, match='Pool not initialized'):
        async with pool.acquire():
            pass


@pytest.mark.asyncio
async def test_async_fetch_translates_numbered_placeholders(db_config):
    cursor = MagicMock()
    cursor.fetchall = AsyncMock(return_value=[{'id': 7}])
    connection = MagicMock()
    connection.execute = AsyncMock(return_value=cursor)
    driver_pool = MagicMock()
    driver_pool.open = AsyncMock()
    driver_pool.connection.return_value = AsyncContextManager(connection)

    with patch(
        'marie.storage.database.postgres_pool.AsyncConnectionPool',
        return_value=driver_pool,
    ):
        pool = AsyncPostgresPool.get_instance()
        await pool.initialize(db_config)
        rows = await pool.fetch(
            'SELECT * FROM notes WHERE user_id = $1 AND title ILIKE $2 OR body ILIKE $2',
            'user-1',
            '%term%',
        )

    connection.execute.assert_awaited_once_with(
        'SELECT * FROM notes WHERE user_id = %s AND title ILIKE %s OR body ILIKE %s',
        ('user-1', '%term%', '%term%'),
    )
    assert rows == [{'id': 7}]


@pytest.mark.asyncio
async def test_async_execute_returns_psycopg_status(db_config):
    cursor = MagicMock(statusmessage='UPDATE 1')
    connection = MagicMock()
    connection.execute = AsyncMock(return_value=cursor)
    driver_pool = MagicMock()
    driver_pool.open = AsyncMock()
    driver_pool.connection.return_value = AsyncContextManager(connection)

    with patch(
        'marie.storage.database.postgres_pool.AsyncConnectionPool',
        return_value=driver_pool,
    ):
        pool = AsyncPostgresPool.get_instance()
        await pool.initialize(db_config)
        status = await pool.execute('UPDATE notes SET title = %s', 'new title')

    assert status == 'UPDATE 1'


@pytest.mark.asyncio
async def test_async_shutdown_closes_pool(db_config):
    driver_pool = MagicMock()
    driver_pool.open = AsyncMock()
    driver_pool.close = AsyncMock()

    with patch(
        'marie.storage.database.postgres_pool.AsyncConnectionPool',
        return_value=driver_pool,
    ):
        pool = AsyncPostgresPool.get_instance()
        await pool.initialize(db_config)
        await AsyncPostgresPool.shutdown()

    driver_pool.close.assert_awaited_once()
    assert pool.is_initialized is False


def test_sync_pool_fetches_without_event_loop(db_config):
    cursor = MagicMock()
    cursor.fetchall.return_value = [{'id': 9}]
    connection = MagicMock()
    connection.execute.return_value = cursor
    driver_pool = MagicMock()
    driver_pool.connection.return_value = ContextManager(connection)

    with patch(
        'marie.storage.database.postgres_pool.ConnectionPool',
        return_value=driver_pool,
    ):
        pool = PostgresPool()
        pool.initialize(db_config)
        rows = pool.fetch('SELECT * FROM notes WHERE id = %s', 9)
        pool.close()

    driver_pool.open.assert_called_once_with(wait=True, timeout=10.0)
    connection.execute.assert_called_once_with(
        'SELECT * FROM notes WHERE id = %s', (9,)
    )
    driver_pool.close.assert_called_once()
    assert rows == [{'id': 9}]
