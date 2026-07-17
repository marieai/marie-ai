"""Unit tests for GroupMemoryTool."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marie.storage.database.agent_tools.group_memory import (
    GroupMemoryEntry,
    GroupMemoryStats,
    MemoryType,
)
from marie.storage.database.agent_tools.group_memory_tool import (
    GroupMemoryAction,
    GroupMemoryTool,
)


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings provider."""
    embeddings = MagicMock()
    embeddings.aembed = AsyncMock(return_value=[[0.1] * 1536])
    return embeddings


@pytest.fixture
def group_memory_tool(mock_embeddings):
    """Create GroupMemoryTool with mocked dependencies."""
    with patch(
        "marie.storage.database.agent_tools.group_memory_tool.GroupMemory"
    ) as mock_memory_cls:
        mock_memory = MagicMock()
        mock_memory.store = AsyncMock(return_value=1)
        mock_memory.search = AsyncMock(return_value=[])
        mock_memory.get_recent = AsyncMock(return_value=[])
        mock_memory.get_stats = AsyncMock(
            return_value=GroupMemoryStats(
                total_memories=10,
                active_memories=8,
                expired_memories=2,
                memories_by_type={"finding": 5, "decision": 3},
                memories_by_agent={"agent1": 6, "agent2": 4},
                avg_confidence=0.75,
                oldest_memory=datetime(2024, 1, 1, tzinfo=timezone.utc),
                newest_memory=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )
        )
        mock_memory_cls.return_value = mock_memory

        tool = GroupMemoryTool(
            group_id="test-group",
            agent_id="test-agent",
            embeddings=mock_embeddings,
        )
        tool._memory = mock_memory
        yield tool


class TestGroupMemoryToolMetadata:
    """Tests for GroupMemoryTool metadata."""

    def test_tool_name(self, group_memory_tool):
        """Test tool name is correct."""
        assert group_memory_tool.name == "group_memory"

    def test_tool_description(self, group_memory_tool):
        """Test tool description is present."""
        assert "Shared memory" in group_memory_tool.description
        assert "store" in group_memory_tool.description
        assert "search" in group_memory_tool.description


class TestGroupMemoryToolStore:
    """Tests for store action."""

    @pytest.mark.asyncio
    async def test_store_success(self, group_memory_tool, mock_embeddings):
        """Test successful memory storage."""
        result = await group_memory_tool.acall(
            action=GroupMemoryAction.STORE,
            content="Test finding content",
            memory_type="finding",
            confidence=0.9,
        )

        assert not result.is_error
        assert '"stored": true' in result.content
        assert '"memory_id": 1' in result.content

        group_memory_tool._memory.store.assert_called_once()
        call_kwargs = group_memory_tool._memory.store.call_args.kwargs
        assert call_kwargs["agent_id"] == "test-agent"
        assert call_kwargs["content"] == "Test finding content"
        assert call_kwargs["memory_type"] == MemoryType.FINDING
        assert call_kwargs["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_store_without_content(self, group_memory_tool):
        """Test store fails without content."""
        result = await group_memory_tool.acall(
            action=GroupMemoryAction.STORE,
            content=None,
        )

        assert result.is_error
        assert "content is required" in result.content

    @pytest.mark.asyncio
    async def test_store_invalid_memory_type(self, group_memory_tool):
        """Test store fails with invalid memory type."""
        result = await group_memory_tool.acall(
            action=GroupMemoryAction.STORE,
            content="Test content",
            memory_type="invalid_type",
        )

        assert result.is_error
        assert "Invalid memory_type" in result.content

    @pytest.mark.asyncio
    async def test_store_with_tags(self, group_memory_tool):
        """Test store with tags."""
        result = await group_memory_tool.acall(
            action=GroupMemoryAction.STORE,
            content="Tagged content",
            memory_type="finding",
            tags=["important", "document-analysis"],
        )

        assert not result.is_error
        call_kwargs = group_memory_tool._memory.store.call_args.kwargs
        assert call_kwargs["tags"] == ["important", "document-analysis"]


class TestGroupMemoryToolSearch:
    """Tests for search action."""

    @pytest.mark.asyncio
    async def test_search_success(self, group_memory_tool):
        """Test successful search."""
        mock_entries = [
            GroupMemoryEntry(
                id=1,
                group_id="test-group",
                agent_id="agent1",
                content="Found relevant content",
                memory_type=MemoryType.FINDING,
                confidence=0.9,
                tags=["test"],
                created_at=datetime.now(timezone.utc),
                similarity=0.95,
            )
        ]
        group_memory_tool._memory.search.return_value = mock_entries

        result = await group_memory_tool.acall(
            action=GroupMemoryAction.SEARCH,
            query="relevant content",
        )

        assert not result.is_error
        assert '"count": 1' in result.content
        assert "Found relevant content" in result.content
        assert '"similarity": 0.95' in result.content

    @pytest.mark.asyncio
    async def test_search_without_query(self, group_memory_tool):
        """Test search fails without query."""
        result = await group_memory_tool.acall(
            action=GroupMemoryAction.SEARCH,
            query=None,
        )

        assert result.is_error
        assert "query is required" in result.content

    @pytest.mark.asyncio
    async def test_search_with_memory_type_filter(self, group_memory_tool):
        """Test search with memory type filter."""
        group_memory_tool._memory.search.return_value = []

        result = await group_memory_tool.acall(
            action=GroupMemoryAction.SEARCH,
            query="test query",
            memory_type="decision",
        )

        assert not result.is_error
        call_kwargs = group_memory_tool._memory.search.call_args.kwargs
        assert call_kwargs["memory_type"] == MemoryType.DECISION

    @pytest.mark.asyncio
    async def test_search_without_embeddings(self):
        """Test search fails without embeddings provider."""
        with patch(
            "marie.storage.database.agent_tools.group_memory_tool.GroupMemory"
        ):
            tool = GroupMemoryTool(
                group_id="test-group",
                agent_id="test-agent",
                embeddings=None,
            )

        result = await tool.acall(
            action=GroupMemoryAction.SEARCH,
            query="test query",
        )

        assert result.is_error
        assert "embeddings provider" in result.content


class TestGroupMemoryToolGetRecent:
    """Tests for get_recent action."""

    @pytest.mark.asyncio
    async def test_get_recent_success(self, group_memory_tool):
        """Test getting recent memories."""
        mock_entries = [
            GroupMemoryEntry(
                id=1,
                group_id="test-group",
                agent_id="agent1",
                content="Recent memory",
                memory_type=MemoryType.CONTEXT,
                confidence=0.8,
                tags=[],
                created_at=datetime.now(timezone.utc),
            )
        ]
        group_memory_tool._memory.get_recent.return_value = mock_entries

        result = await group_memory_tool.acall(
            action=GroupMemoryAction.GET_RECENT,
            limit=5,
        )

        assert not result.is_error
        assert '"count": 1' in result.content
        assert "Recent memory" in result.content

    @pytest.mark.asyncio
    async def test_get_recent_with_filters(self, group_memory_tool):
        """Test get_recent with agent and type filters."""
        group_memory_tool._memory.get_recent.return_value = []

        result = await group_memory_tool.acall(
            action=GroupMemoryAction.GET_RECENT,
            agent_id_filter="specific-agent",
            memory_type="finding",
        )

        assert not result.is_error
        call_kwargs = group_memory_tool._memory.get_recent.call_args.kwargs
        assert call_kwargs["agent_id"] == "specific-agent"
        assert call_kwargs["memory_type"] == MemoryType.FINDING


class TestGroupMemoryToolGetStats:
    """Tests for get_stats action."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, group_memory_tool):
        """Test getting memory statistics."""
        result = await group_memory_tool.acall(
            action=GroupMemoryAction.GET_STATS,
        )

        assert not result.is_error
        assert '"total_memories": 10' in result.content
        assert '"active_memories": 8' in result.content
        assert '"avg_confidence": 0.75' in result.content
        assert '"group_id": "test-group"' in result.content


class TestGroupMemoryToolActionParsing:
    """Tests for action string parsing."""

    @pytest.mark.asyncio
    async def test_action_as_string(self, group_memory_tool):
        """Test action can be passed as string."""
        result = await group_memory_tool.acall(
            action="get_stats",
        )

        assert not result.is_error

    @pytest.mark.asyncio
    async def test_invalid_action_string(self, group_memory_tool):
        """Test invalid action string returns error."""
        result = await group_memory_tool.acall(
            action="invalid_action",
        )

        assert result.is_error
        assert "Unknown action" in result.content


class TestGroupMemoryToolEmbeddings:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    async def test_store_generates_embedding(self, group_memory_tool, mock_embeddings):
        """Test embedding is generated when storing."""
        await group_memory_tool.acall(
            action=GroupMemoryAction.STORE,
            content="Content to embed",
        )

        mock_embeddings.aembed.assert_called_once_with(["Content to embed"])

        call_kwargs = group_memory_tool._memory.store.call_args.kwargs
        assert call_kwargs["embedding"] is not None
        assert len(call_kwargs["embedding"]) == 1536

    @pytest.mark.asyncio
    async def test_store_handles_embedding_failure(self, group_memory_tool, mock_embeddings):
        """Test store continues if embedding fails."""
        mock_embeddings.aembed.side_effect = Exception("Embedding service unavailable")

        result = await group_memory_tool.acall(
            action=GroupMemoryAction.STORE,
            content="Content to store",
        )

        assert not result.is_error
        call_kwargs = group_memory_tool._memory.store.call_args.kwargs
        assert call_kwargs["embedding"] is None
