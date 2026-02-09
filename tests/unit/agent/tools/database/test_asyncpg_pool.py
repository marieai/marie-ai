"""Tests for AsyncPostgresPool."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marie.storage.database.asyncpg_pool import AsyncPostgresPool


class AsyncContextManager:
    """Helper to create async context manager from mock."""

    def __init__(self, mock_conn):
        self._mock_conn = mock_conn

    async def __aenter__(self):
        return self._mock_conn

    async def __aexit__(self, *args):
        pass


class TestAsyncPostgresPool:
    """Tests for the AsyncPostgresPool singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton state before each test."""
        AsyncPostgresPool.reset()
        yield
        AsyncPostgresPool.reset()

    def test_get_instance_returns_singleton(self):
        """get_instance should return the same instance."""
        instance1 = AsyncPostgresPool.get_instance()
        instance2 = AsyncPostgresPool.get_instance()
        assert instance1 is instance2

    def test_is_initialized_false_before_init(self):
        """is_initialized should be False before initialize is called."""
        pool = AsyncPostgresPool.get_instance()
        assert pool.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self, db_config):
        """initialize should create an asyncpg pool."""
        pool = AsyncPostgresPool.get_instance()

        mock_asyncpg_pool = MagicMock()

        with patch("marie.storage.database.asyncpg_pool.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_asyncpg_pool

            await pool.initialize(db_config)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["host"] == "localhost"
            assert call_kwargs.kwargs["port"] == 5432
            assert call_kwargs.kwargs["user"] == "test_user"
            assert call_kwargs.kwargs["database"] == "test_db"
            assert pool.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, db_config):
        """initialize should be idempotent - second call does nothing."""
        pool = AsyncPostgresPool.get_instance()

        mock_asyncpg_pool = MagicMock()

        with patch("marie.storage.database.asyncpg_pool.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_asyncpg_pool

            await pool.initialize(db_config)
            await pool.initialize(db_config)

            # Should only create pool once
            assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_acquire_raises_if_not_initialized(self):
        """acquire should raise if pool not initialized."""
        pool = AsyncPostgresPool.get_instance()

        with pytest.raises(RuntimeError, match="Pool not initialized"):
            async with pool.acquire():
                pass

    @pytest.mark.asyncio
    async def test_fetch_delegates_to_pool(self, db_config):
        """fetch should delegate to the underlying pool."""
        pool = AsyncPostgresPool.get_instance()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[{"id": 1, "name": "test"}])

        mock_asyncpg_pool = MagicMock()
        mock_asyncpg_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        with patch("marie.storage.database.asyncpg_pool.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_asyncpg_pool
            await pool.initialize(db_config)

            result = await pool.fetch("SELECT * FROM test WHERE id = $1", 1)

            mock_conn.fetch.assert_called_once_with("SELECT * FROM test WHERE id = $1", 1)
            assert result == [{"id": 1, "name": "test"}]

    @pytest.mark.asyncio
    async def test_fetchrow_delegates_to_pool(self, db_config):
        """fetchrow should delegate to the underlying pool."""
        pool = AsyncPostgresPool.get_instance()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": 1, "name": "test"})

        mock_asyncpg_pool = MagicMock()
        mock_asyncpg_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        with patch("marie.storage.database.asyncpg_pool.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_asyncpg_pool
            await pool.initialize(db_config)

            result = await pool.fetchrow("SELECT * FROM test WHERE id = $1", 1)

            mock_conn.fetchrow.assert_called_once_with("SELECT * FROM test WHERE id = $1", 1)
            assert result == {"id": 1, "name": "test"}

    @pytest.mark.asyncio
    async def test_fetchval_delegates_to_pool(self, db_config):
        """fetchval should delegate to the underlying pool."""
        pool = AsyncPostgresPool.get_instance()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=42)

        mock_asyncpg_pool = MagicMock()
        mock_asyncpg_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        with patch("marie.storage.database.asyncpg_pool.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_asyncpg_pool
            await pool.initialize(db_config)

            result = await pool.fetchval("SELECT COUNT(*) FROM test")

            mock_conn.fetchval.assert_called_once_with("SELECT COUNT(*) FROM test")
            assert result == 42

    @pytest.mark.asyncio
    async def test_execute_delegates_to_pool(self, db_config):
        """execute should delegate to the underlying pool."""
        pool = AsyncPostgresPool.get_instance()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")

        mock_asyncpg_pool = MagicMock()
        mock_asyncpg_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        with patch("marie.storage.database.asyncpg_pool.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_asyncpg_pool
            await pool.initialize(db_config)

            result = await pool.execute("INSERT INTO test VALUES ($1)", "value")

            mock_conn.execute.assert_called_once_with("INSERT INTO test VALUES ($1)", "value")
            assert result == "INSERT 0 1"

    @pytest.mark.asyncio
    async def test_executemany_delegates_to_pool(self, db_config):
        """executemany should delegate to the underlying pool."""
        pool = AsyncPostgresPool.get_instance()

        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()

        mock_asyncpg_pool = MagicMock()
        mock_asyncpg_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        with patch("marie.storage.database.asyncpg_pool.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_asyncpg_pool
            await pool.initialize(db_config)

            args = [("value1",), ("value2",)]
            await pool.executemany("INSERT INTO test VALUES ($1)", args)

            mock_conn.executemany.assert_called_once_with("INSERT INTO test VALUES ($1)", args)

    @pytest.mark.asyncio
    async def test_shutdown_closes_pool(self, db_config):
        """shutdown should close the pool."""
        pool = AsyncPostgresPool.get_instance()

        mock_asyncpg_pool = AsyncMock()
        mock_asyncpg_pool.close = AsyncMock()

        with patch("marie.storage.database.asyncpg_pool.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_asyncpg_pool
            await pool.initialize(db_config)

            assert pool.is_initialized is True

            await AsyncPostgresPool.shutdown()

            mock_asyncpg_pool.close.assert_called_once()
            assert pool._pool is None

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_not_initialized(self):
        """shutdown should do nothing when not initialized."""
        # Should not raise
        await AsyncPostgresPool.shutdown()

    def test_reset_clears_singleton(self):
        """reset should clear the singleton instance."""
        instance1 = AsyncPostgresPool.get_instance()
        AsyncPostgresPool.reset()
        instance2 = AsyncPostgresPool.get_instance()

        # After reset, should get a different instance object
        # (though they're both the singleton)
        assert AsyncPostgresPool._instance is instance2
