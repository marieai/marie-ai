"""Group memory for coordinated agent execution.

This module provides shared memory capabilities for agent groups,
enabling agents to share findings, decisions, and context during
coordinated execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

import logging
from marie.storage.database.postgres_pool import AsyncPostgresPool

logger = logging.getLogger("marie.storage.database.agent_tools.group_memory")


class MemoryType(str, Enum):
    """Types of memories that can be stored in group memory."""

    FINDING = "finding"
    DECISION = "decision"
    ARTIFACT = "artifact"
    CONTEXT = "context"
    OBSERVATION = "observation"
    FEEDBACK = "feedback"


@dataclass
class GroupMemoryEntry:
    """A single entry in group memory."""

    id: int
    group_id: str
    agent_id: str
    content: str
    memory_type: MemoryType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    similarity: Optional[float] = None


@dataclass
class GroupMemoryStats:
    """Statistics about group memory usage."""

    total_memories: int
    active_memories: int
    expired_memories: int
    memories_by_type: Dict[str, int]
    memories_by_agent: Dict[str, int]
    avg_confidence: float
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]


class GroupMemoryConfig(BaseModel):
    """Configuration for group memory."""

    group_id: str = Field(..., description="Group identifier for memory scoping")
    schema_name: str = Field(
        default="marie",
        description="PostgreSQL schema name",
    )
    embedding_dims: int = Field(
        default=1536,
        description="Vector embedding dimensions (1536 for text-embedding-3-small)",
    )
    default_ttl_seconds: Optional[int] = Field(
        default=None,
        description="Default TTL for new memories (None = no expiration)",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for search results",
    )


class GroupMemory:
    """Shared memory store for coordinated agent groups.

    Provides semantic search capabilities using pgvector with HNSW indexing.
    Agents within the same group can share findings, decisions, and context.

    Example:
        ```python
        config = GroupMemoryConfig(group_id="document-processing")
        memory = GroupMemory(config, db_config)
        await memory.initialize()

        # Store a finding
        await memory.store(
            agent_id="analyzer_1",
            content="Document contains 3 tables with financial data",
            memory_type=MemoryType.FINDING,
            confidence=0.9,
            embedding=embedding_vector,
        )

        # Search for relevant memories
        results = await memory.search(
            query_embedding=query_vector,
            memory_type=MemoryType.FINDING,
            limit=5,
        )
        ```
    """

    def __init__(
        self,
        config: GroupMemoryConfig,
        db_config: Optional[Dict[str, Any]] = None,
        pool: Optional[AsyncPostgresPool] = None,
    ):
        self.config = config
        self._db_config = db_config or {}
        self._pool = pool
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection."""
        if self._initialized:
            return

        if self._pool is None:
            self._pool = AsyncPostgresPool.get_instance()
            await self._pool.initialize(self._db_config)

        self._initialized = True

    async def _get_pool(self) -> AsyncPostgresPool:
        """Get database pool, initializing if needed."""
        if not self._initialized:
            await self.initialize()
        return self._pool

    async def store(
        self,
        agent_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.CONTEXT,
        confidence: float = 0.5,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> int:
        """Store a new memory entry.

        Args:
            agent_id: Identifier of the contributing agent
            content: Memory content text
            memory_type: Type of memory
            confidence: Confidence score (0-1)
            embedding: Vector embedding for semantic search
            metadata: Additional metadata
            tags: Tags for categorization
            ttl_seconds: Time-to-live in seconds (None uses default)

        Returns:
            ID of the created memory entry
        """
        pool = await self._get_pool()
        schema = self.config.schema_name

        ttl = (
            ttl_seconds if ttl_seconds is not None else self.config.default_ttl_seconds
        )

        embedding_str = None
        if embedding:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        result = await pool.fetchrow(
            f"""
            INSERT INTO {schema}.agent_group_memory (
                group_id, agent_id, content, memory_type,
                embedding, confidence, metadata, tags, ttl_seconds
            ) VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9)
            RETURNING id
            """,
            self.config.group_id,
            agent_id,
            content,
            memory_type.value,
            embedding_str,
            confidence,
            metadata or {},
            tags or [],
            ttl,
        )

        return result["id"]

    async def search(
        self,
        query_embedding: List[float],
        memory_type: Optional[MemoryType] = None,
        min_confidence: Optional[float] = None,
        limit: int = 10,
    ) -> List[GroupMemoryEntry]:
        """Search memories using semantic similarity.

        Args:
            query_embedding: Query vector for similarity search
            memory_type: Filter by memory type
            min_confidence: Minimum confidence threshold
            limit: Maximum results to return

        Returns:
            List of matching memory entries with similarity scores
        """
        pool = await self._get_pool()
        schema = self.config.schema_name

        min_conf = (
            min_confidence if min_confidence is not None else self.config.min_confidence
        )
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        type_param = memory_type.value if memory_type else None

        rows = await pool.fetch(
            f"""
            SELECT * FROM {schema}.agent_group_semantic_search(
                $1, $2::vector, $3::{schema}.memory_type, $4, $5
            )
            """,
            self.config.group_id,
            embedding_str,
            type_param,
            min_conf,
            limit,
        )

        return [
            GroupMemoryEntry(
                id=row["id"],
                group_id=self.config.group_id,
                agent_id=row["agent_id"],
                content=row["content"],
                memory_type=MemoryType(row["memory_type"]),
                confidence=row["confidence"],
                metadata=row["metadata"] or {},
                tags=row["tags"] or [],
                created_at=row["created_at"],
                similarity=row["similarity"],
            )
            for row in rows
        ]

    async def get_recent(
        self,
        agent_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 20,
    ) -> List[GroupMemoryEntry]:
        """Get recent memories from the group.

        Args:
            agent_id: Filter by contributing agent
            memory_type: Filter by memory type
            limit: Maximum results to return

        Returns:
            List of recent memory entries
        """
        pool = await self._get_pool()
        schema = self.config.schema_name

        type_param = memory_type.value if memory_type else None

        rows = await pool.fetch(
            f"""
            SELECT * FROM {schema}.get_recent_group_memories($1, $2, $3::{schema}.memory_type, $4)
            """,
            self.config.group_id,
            agent_id,
            type_param,
            limit,
        )

        return [
            GroupMemoryEntry(
                id=row["id"],
                group_id=self.config.group_id,
                agent_id=row["agent_id"],
                content=row["content"],
                memory_type=MemoryType(row["memory_type"]),
                confidence=row["confidence"],
                metadata=row["metadata"] or {},
                tags=row["tags"] or [],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def get_stats(self) -> GroupMemoryStats:
        """Get statistics about group memory usage."""
        pool = await self._get_pool()
        schema = self.config.schema_name

        row = await pool.fetchrow(
            f"SELECT * FROM {schema}.get_group_memory_stats($1)",
            self.config.group_id,
        )

        return GroupMemoryStats(
            total_memories=row["total_memories"] or 0,
            active_memories=row["active_memories"] or 0,
            expired_memories=row["expired_memories"] or 0,
            memories_by_type=row["memories_by_type"] or {},
            memories_by_agent=row["memories_by_agent"] or {},
            avg_confidence=row["avg_confidence"] or 0.0,
            oldest_memory=row["oldest_memory"],
            newest_memory=row["newest_memory"],
        )

    async def delete(self, memory_id: int) -> bool:
        """Delete a specific memory entry.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deleted, False if not found
        """
        pool = await self._get_pool()
        schema = self.config.schema_name

        result = await pool.execute(
            f"""
            DELETE FROM {schema}.agent_group_memory
            WHERE id = $1 AND group_id = $2
            """,
            memory_id,
            self.config.group_id,
        )

        return "DELETE 1" in result

    async def clear(self) -> int:
        """Clear all memories for this group.

        Returns:
            Number of memories deleted
        """
        pool = await self._get_pool()
        schema = self.config.schema_name

        result = await pool.execute(
            f"DELETE FROM {schema}.agent_group_memory WHERE group_id = $1",
            self.config.group_id,
        )

        count = result.split(" ")[-1] if result else "0"
        return int(count)

    async def cleanup_expired(self) -> int:
        """Remove expired memories across all groups.

        Returns:
            Number of memories deleted
        """
        pool = await self._get_pool()
        schema = self.config.schema_name

        result = await pool.fetchval(
            f"SELECT {schema}.cleanup_expired_group_memories()"
        )

        return result or 0

    async def update_confidence(
        self,
        memory_id: int,
        confidence: float,
    ) -> bool:
        """Update the confidence score of a memory.

        Args:
            memory_id: ID of the memory to update
            confidence: New confidence score (0-1)

        Returns:
            True if updated, False if not found
        """
        pool = await self._get_pool()
        schema = self.config.schema_name

        result = await pool.execute(
            f"""
            UPDATE {schema}.agent_group_memory
            SET confidence = $1
            WHERE id = $2 AND group_id = $3
            """,
            confidence,
            memory_id,
            self.config.group_id,
        )

        return "UPDATE 1" in result

    async def add_tags(
        self,
        memory_id: int,
        tags: List[str],
    ) -> bool:
        """Add tags to a memory entry.

        Args:
            memory_id: ID of the memory to update
            tags: Tags to add

        Returns:
            True if updated, False if not found
        """
        pool = await self._get_pool()
        schema = self.config.schema_name

        result = await pool.execute(
            f"""
            UPDATE {schema}.agent_group_memory
            SET tags = array_cat(tags, $1)
            WHERE id = $2 AND group_id = $3
            """,
            tags,
            memory_id,
            self.config.group_id,
        )

        return "UPDATE 1" in result
