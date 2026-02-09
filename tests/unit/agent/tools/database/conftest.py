"""Fixtures for database tool unit tests."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockRecord(dict):
    """Mock asyncpg.Record that supports both dict and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class MockAsyncPostgresPool:
    """Mock AsyncPostgresPool for unit testing."""

    def __init__(self):
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized = False
        self._execute_log: List[tuple] = []

    async def initialize(self, config: Dict[str, Any]) -> None:
        self._initialized = True
        self._config = config

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @asynccontextmanager
    async def acquire(self):
        yield self

    async def fetch(self, query: str, *args) -> List[MockRecord]:
        self._execute_log.append(("fetch", query, args))
        return self._mock_fetch(query, args)

    async def fetchrow(self, query: str, *args) -> Optional[MockRecord]:
        self._execute_log.append(("fetchrow", query, args))
        results = self._mock_fetch(query, args)
        return results[0] if results else None

    async def fetchval(self, query: str, *args) -> Any:
        self._execute_log.append(("fetchval", query, args))
        results = self._mock_fetch(query, args)
        if results and results[0]:
            return list(results[0].values())[0]
        return None

    async def execute(self, query: str, *args) -> str:
        self._execute_log.append(("execute", query, args))
        return self._mock_execute(query, args)

    async def executemany(self, query: str, args: List[tuple]) -> None:
        self._execute_log.append(("executemany", query, args))

    def _mock_fetch(self, query: str, args: tuple) -> List[MockRecord]:
        """Override in specific tests or use set_mock_data."""
        return []

    def _mock_execute(self, query: str, args: tuple) -> str:
        """Override in specific tests."""
        if "INSERT" in query.upper():
            return "INSERT 0 1"
        elif "UPDATE" in query.upper():
            return "UPDATE 1"
        elif "DELETE" in query.upper():
            return "DELETE 1"
        elif "CREATE" in query.upper():
            return "CREATE TABLE"
        return "OK"

    def reset(self):
        """Reset mock state."""
        self._execute_log.clear()
        self._data.clear()


class InMemoryMockPool(MockAsyncPostgresPool):
    """In-memory mock pool that actually stores data for realistic testing."""

    def __init__(self):
        super().__init__()
        self._tables: Dict[str, List[Dict[str, Any]]] = {}

    def _get_table(self, table_name: str) -> List[Dict[str, Any]]:
        if table_name not in self._tables:
            self._tables[table_name] = []
        return self._tables[table_name]

    def _mock_execute(self, query: str, args: tuple) -> str:
        query_upper = query.upper()

        if "CREATE TABLE" in query_upper or "CREATE SCHEMA" in query_upper or "CREATE INDEX" in query_upper:
            return "CREATE"

        if "INSERT INTO" in query_upper:
            return "INSERT 0 1"

        if "UPDATE" in query_upper:
            return "UPDATE 1"

        if "DELETE" in query_upper:
            return "DELETE 1"

        return "OK"


@pytest.fixture
def mock_pool():
    """Provide a basic mock pool."""
    return MockAsyncPostgresPool()


@pytest.fixture
def in_memory_pool():
    """Provide an in-memory mock pool."""
    return InMemoryMockPool()


@pytest.fixture
def db_config():
    """Standard database configuration for testing."""
    return {
        "hostname": "localhost",
        "port": 5432,
        "username": "test_user",
        "password": "test_password",
        "database": "test_db",
        "min_connections": 1,
        "max_connections": 5,
    }


@pytest.fixture
def mock_pool_singleton(mock_pool):
    """Patch the AsyncPostgresPool singleton to use mock pool."""
    with patch("marie.storage.database.asyncpg_pool.AsyncPostgresPool.get_instance") as mock_get:
        mock_get.return_value = mock_pool
        yield mock_pool


@pytest.fixture
def sample_user_id():
    """Sample user ID for testing."""
    return "test_user_123"


@pytest.fixture
def sample_note_id():
    """Sample note UUID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_todo_id():
    """Sample todo UUID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_timestamp():
    """Sample timestamp for testing."""
    return datetime.now(timezone.utc)
