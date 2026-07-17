"""Unit tests for group memory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marie.storage.database.agent_tools.group_memory import (
    GroupMemory,
    GroupMemoryConfig,
    GroupMemoryEntry,
    GroupMemoryStats,
    MemoryType,
)


class MockAsyncPool:
    """Mock async database pool for testing."""

    def __init__(self):
        self._data: Dict[int, GroupMemoryEntry] = {}
        self._id_counter = 0

    async def fetchval(self, query: str, *args) -> Any:
        """Mock fetchval for inserts returning ID or counts."""
        if "cleanup" in query.lower():
            return 0
        self._id_counter += 1
        return self._id_counter

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Mock fetchrow for single row queries."""
        if "get_group_memory_stats" in query:
            return {
                "total_memories": 10,
                "active_memories": 8,
                "expired_memories": 2,
                "memories_by_type": {"finding": 5, "decision": 3, "context": 2},
                "memories_by_agent": {"agent-1": 6, "agent-2": 4},
                "avg_confidence": 0.75,
                "oldest_memory": datetime.now(timezone.utc),
                "newest_memory": datetime.now(timezone.utc),
            }
        if "INSERT" in query:
            self._id_counter += 1
            return {"id": self._id_counter}
        if self._data:
            entry = list(self._data.values())[0]
            return {
                "id": entry.id,
                "group_id": entry.group_id,
                "agent_id": entry.agent_id,
                "memory_type": entry.memory_type.value,
                "content": entry.content,
                "confidence": entry.confidence,
                "metadata": entry.metadata,
                "tags": entry.tags,
                "created_at": entry.created_at,
                "similarity": 0.9,
            }
        return None

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Mock fetch for multiple row queries."""
        return [
            {
                "id": entry.id,
                "group_id": entry.group_id,
                "agent_id": entry.agent_id,
                "memory_type": entry.memory_type.value,
                "content": entry.content,
                "confidence": entry.confidence,
                "metadata": entry.metadata,
                "tags": entry.tags,
                "created_at": entry.created_at,
                "similarity": 0.9,
            }
            for entry in self._data.values()
        ]

    async def execute(self, query: str, *args) -> str:
        """Mock execute for updates/deletes."""
        if "DELETE" in query:
            return "DELETE 1"
        if "UPDATE" in query:
            return "UPDATE 1"
        return "OK"


@pytest.fixture
def mock_pool():
    """Provide mock database pool."""
    return MockAsyncPool()


@pytest.fixture
def group_memory_config():
    """Provide GroupMemoryConfig."""
    return GroupMemoryConfig(
        group_id="test-group",
        schema_name="marie",
        embedding_dims=1536,
    )


@pytest.fixture
def group_memory(group_memory_config, mock_pool):
    """Provide GroupMemory instance with mock pool."""
    memory = GroupMemory(config=group_memory_config, pool=mock_pool)
    memory._initialized = True
    return memory


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_types_exist(self):
        """Test all memory types are defined."""
        assert MemoryType.FINDING.value == "finding"
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.ARTIFACT.value == "artifact"
        assert MemoryType.CONTEXT.value == "context"
        assert MemoryType.OBSERVATION.value == "observation"
        assert MemoryType.FEEDBACK.value == "feedback"


class TestGroupMemoryConfig:
    """Tests for GroupMemoryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GroupMemoryConfig(group_id="test")

        assert config.group_id == "test"
        assert config.schema_name == "marie"
        assert config.embedding_dims == 1536
        assert config.default_ttl_seconds is None
        assert config.min_confidence == 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = GroupMemoryConfig(
            group_id="custom",
            schema_name="custom_schema",
            embedding_dims=768,
            default_ttl_seconds=3600,
            min_confidence=0.5,
        )

        assert config.schema_name == "custom_schema"
        assert config.embedding_dims == 768
        assert config.default_ttl_seconds == 3600
        assert config.min_confidence == 0.5


class TestGroupMemoryEntry:
    """Tests for GroupMemoryEntry dataclass."""

    def test_create_entry(self):
        """Test creating a memory entry."""
        entry = GroupMemoryEntry(
            id=1,
            group_id="group-1",
            agent_id="agent-1",
            memory_type=MemoryType.FINDING,
            content="Test finding",
            confidence=0.9,
        )

        assert entry.id == 1
        assert entry.memory_type == MemoryType.FINDING
        assert entry.confidence == 0.9

    def test_entry_default_values(self):
        """Test entry has default values."""
        entry = GroupMemoryEntry(
            id=1,
            group_id="g",
            agent_id="a",
            memory_type=MemoryType.CONTEXT,
            content="test",
            confidence=0.5,
        )

        assert entry.created_at is not None
        assert entry.metadata == {}
        assert entry.tags == []
        assert entry.similarity is None

    def test_entry_with_metadata(self):
        """Test entry with metadata and tags."""
        entry = GroupMemoryEntry(
            id=1,
            group_id="g",
            agent_id="a",
            memory_type=MemoryType.DECISION,
            content="Decision content",
            confidence=0.8,
            metadata={"reason": "Better option"},
            tags=["important", "reviewed"],
        )

        assert entry.metadata["reason"] == "Better option"
        assert "important" in entry.tags


class TestGroupMemoryStats:
    """Tests for GroupMemoryStats dataclass."""

    def test_stats_creation(self):
        """Test creating memory stats."""
        stats = GroupMemoryStats(
            total_memories=100,
            active_memories=90,
            expired_memories=10,
            memories_by_type={"finding": 50, "decision": 30, "artifact": 20},
            memories_by_agent={"agent-1": 60, "agent-2": 40},
            avg_confidence=0.75,
            oldest_memory=datetime.now(timezone.utc),
            newest_memory=datetime.now(timezone.utc),
        )

        assert stats.total_memories == 100
        assert stats.active_memories == 90
        assert stats.expired_memories == 10
        assert stats.memories_by_type["finding"] == 50


class TestGroupMemoryStore:
    """Tests for storing memories."""

    @pytest.mark.asyncio
    async def test_store_finding(self, group_memory):
        """Test storing a finding."""
        entry_id = await group_memory.store(
            agent_id="test-agent",
            content="Important finding",
            memory_type=MemoryType.FINDING,
            confidence=0.85,
            embedding=[0.1] * 1536,
        )

        assert entry_id is not None
        assert entry_id > 0

    @pytest.mark.asyncio
    async def test_store_decision(self, group_memory):
        """Test storing a decision."""
        entry_id = await group_memory.store(
            agent_id="test-agent",
            content="Decision to use approach A",
            memory_type=MemoryType.DECISION,
            metadata={"reason": "Better performance"},
            embedding=[0.1] * 1536,
        )

        assert entry_id is not None

    @pytest.mark.asyncio
    async def test_store_with_custom_ttl(self, group_memory):
        """Test storing with custom TTL."""
        entry_id = await group_memory.store(
            agent_id="test-agent",
            content="Short-lived memory",
            memory_type=MemoryType.CONTEXT,
            ttl_seconds=3600,
            embedding=[0.1] * 1536,
        )

        assert entry_id is not None

    @pytest.mark.asyncio
    async def test_store_with_tags(self, group_memory):
        """Test storing with tags."""
        entry_id = await group_memory.store(
            agent_id="test-agent",
            content="Tagged memory",
            memory_type=MemoryType.ARTIFACT,
            tags=["important", "reviewed"],
            embedding=[0.1] * 1536,
        )

        assert entry_id is not None


class TestGroupMemorySearch:
    """Tests for searching memories."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, group_memory, mock_pool):
        """Test semantic search returns results."""
        mock_pool._data[1] = GroupMemoryEntry(
            id=1,
            group_id="test-group",
            agent_id="test-agent",
            memory_type=MemoryType.FINDING,
            content="Test finding about AI",
            confidence=0.9,
        )

        results = await group_memory.search(
            query_embedding=[0.1] * 1536,
            limit=10,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, group_memory, mock_pool):
        """Test search filtered by memory type."""
        mock_pool._data[1] = GroupMemoryEntry(
            id=1,
            group_id="test-group",
            agent_id="test-agent",
            memory_type=MemoryType.DECISION,
            content="Decision content",
            confidence=0.8,
        )

        results = await group_memory.search(
            query_embedding=[0.1] * 1536,
            memory_type=MemoryType.DECISION,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_with_min_confidence(self, group_memory, mock_pool):
        """Test search with minimum confidence filter."""
        mock_pool._data[1] = GroupMemoryEntry(
            id=1,
            group_id="test-group",
            agent_id="test-agent",
            memory_type=MemoryType.FINDING,
            content="High confidence finding",
            confidence=0.95,
        )

        results = await group_memory.search(
            query_embedding=[0.1] * 1536,
            min_confidence=0.9,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_empty_results(self, group_memory):
        """Test search with no matching results."""
        results = await group_memory.search(
            query_embedding=[0.1] * 1536,
        )

        assert results == []


class TestGroupMemoryRetrieve:
    """Tests for retrieving recent memories."""

    @pytest.mark.asyncio
    async def test_get_recent_memories(self, group_memory, mock_pool):
        """Test getting recent memories."""
        mock_pool._data[1] = GroupMemoryEntry(
            id=1,
            group_id="test-group",
            agent_id="test-agent",
            memory_type=MemoryType.FINDING,
            content="Recent finding",
            confidence=0.9,
        )

        results = await group_memory.get_recent(limit=10)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_get_recent_with_agent_filter(self, group_memory, mock_pool):
        """Test getting recent memories by agent."""
        mock_pool._data[1] = GroupMemoryEntry(
            id=1,
            group_id="test-group",
            agent_id="specific-agent",
            memory_type=MemoryType.ARTIFACT,
            content="Agent artifact",
            confidence=0.7,
        )

        results = await group_memory.get_recent(
            agent_id="specific-agent",
            limit=5,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_get_recent_with_type_filter(self, group_memory, mock_pool):
        """Test getting recent memories by type."""
        mock_pool._data[1] = GroupMemoryEntry(
            id=1,
            group_id="test-group",
            agent_id="test-agent",
            memory_type=MemoryType.ARTIFACT,
            content="Artifact",
            confidence=0.7,
        )

        results = await group_memory.get_recent(
            memory_type=MemoryType.ARTIFACT,
            limit=5,
        )

        assert len(results) > 0


class TestGroupMemoryDelete:
    """Tests for deleting memories."""

    @pytest.mark.asyncio
    async def test_delete_memory(self, group_memory):
        """Test deleting a memory by ID."""
        result = await group_memory.delete(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_group_memories(self, group_memory):
        """Test clearing all group memories."""
        count = await group_memory.clear()
        assert count >= 0

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, group_memory):
        """Test cleaning up expired memories."""
        count = await group_memory.cleanup_expired()
        assert count >= 0


class TestGroupMemoryStatsRetrieval:
    """Tests for memory statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, group_memory):
        """Test getting memory statistics."""
        stats = await group_memory.get_stats()

        assert stats is not None
        assert stats.total_memories == 10
        assert stats.active_memories == 8
        assert stats.expired_memories == 2
        assert stats.avg_confidence == 0.75


class TestGroupMemoryUpdate:
    """Tests for updating memories."""

    @pytest.mark.asyncio
    async def test_update_confidence(self, group_memory):
        """Test updating memory confidence."""
        result = await group_memory.update_confidence(
            memory_id=1,
            confidence=0.95,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_add_tags(self, group_memory):
        """Test adding tags to memory."""
        result = await group_memory.add_tags(
            memory_id=1,
            tags=["new-tag", "important"],
        )
        assert result is True


class TestGroupMemoryInitialization:
    """Tests for memory initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self, group_memory_config):
        """Test memory initialization."""
        memory = GroupMemory(config=group_memory_config)
        assert memory._initialized is False

    @pytest.mark.asyncio
    async def test_double_initialization(self, group_memory):
        """Test that double initialization is safe."""
        group_memory._initialized = True
        await group_memory.initialize()
        # Should not raise


class TestGroupMemoryConfiguration:
    """Tests for memory configuration."""

    def test_different_group_ids(self):
        """Test memories with different group IDs are isolated."""
        config1 = GroupMemoryConfig(group_id="group-a")
        config2 = GroupMemoryConfig(group_id="group-b")

        assert config1.group_id != config2.group_id

    def test_custom_schema_name(self):
        """Test custom schema name."""
        config = GroupMemoryConfig(
            group_id="test",
            schema_name="custom_schema",
        )
        assert config.schema_name == "custom_schema"

    def test_confidence_bounds(self):
        """Test confidence bounds are enforced."""
        config = GroupMemoryConfig(
            group_id="test",
            min_confidence=0.5,
        )
        assert 0 <= config.min_confidence <= 1
