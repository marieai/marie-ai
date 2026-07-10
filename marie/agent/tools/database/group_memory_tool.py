"""Group memory tool for coordinated agent execution.

This tool enables agents to share findings, decisions, and context
through a shared memory store with semantic search capabilities.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.agent.memory.group_memory import (
    GroupMemory,
    GroupMemoryConfig,
    MemoryType,
)
from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.helper import run_async
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.embeddings.qwen import QwenVLEmbeddings

logger = MarieLogger("marie.agent.tools.group_memory")


class GroupMemoryAction(str, Enum):
    """Actions for GroupMemoryTool."""

    STORE = "store"
    SEARCH = "search"
    GET_RECENT = "get_recent"
    GET_STATS = "get_stats"


class GroupMemoryInput(BaseModel):
    """Input schema for GroupMemoryTool."""

    action: GroupMemoryAction = Field(
        ...,
        description="Action to perform: store, search, get_recent, or get_stats",
    )
    content: Optional[str] = Field(
        None,
        description="Content to store (required for store action)",
    )
    query: Optional[str] = Field(
        None,
        description="Search query (required for search action)",
    )
    memory_type: Optional[str] = Field(
        None,
        description="Type of memory: finding, decision, artifact, context, observation, feedback",
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for stored memories (0-1)",
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Tags for categorization",
    )
    limit: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum results to return",
    )
    agent_id_filter: Optional[str] = Field(
        None,
        description="Filter by agent ID (for get_recent)",
    )


class GroupMemoryTool(AgentTool):
    """Tool for agents to access shared group memory.

    Enables coordinated agents to:
    - Store findings, decisions, and context
    - Search memories semantically
    - Retrieve recent memories from the group
    - Check memory statistics

    Example:
        ```python
        from marie.agent.tools.database import GroupMemoryTool
        from marie.embeddings.qwen import QwenVLEmbeddings

        embeddings = QwenVLEmbeddings()
        tool = GroupMemoryTool(
            group_id="document-analysis-001",
            agent_id="analyzer",
            embeddings=embeddings,
        )

        # Store a finding
        result = tool.call(
            action="store",
            content="Document contains 3 tables with financial data",
            memory_type="finding",
            confidence=0.9,
        )

        # Search for related memories
        result = tool.call(
            action="search",
            query="financial data tables",
            limit=5,
        )
        ```
    """

    def __init__(
        self,
        group_id: str,
        agent_id: str,
        embeddings: Optional["QwenVLEmbeddings"] = None,
        db_config: Optional[Dict[str, Any]] = None,
        schema_name: str = "marie",
        default_ttl_seconds: Optional[int] = None,
    ):
        """Initialize GroupMemoryTool.

        Args:
            group_id: Group identifier for memory scoping
            agent_id: Agent identifier for this tool instance
            embeddings: Optional embeddings provider for semantic search
            db_config: Database configuration
            schema_name: PostgreSQL schema name
            default_ttl_seconds: Default TTL for memories
        """
        self._group_id = group_id
        self._agent_id = agent_id
        self._embeddings = embeddings
        self._db_config = db_config or {}

        config = GroupMemoryConfig(
            group_id=group_id,
            schema_name=schema_name,
            default_ttl_seconds=default_ttl_seconds,
        )
        self._memory = GroupMemory(config, db_config)

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="group_memory",
            description=(
                "Shared memory for coordinated agents. Actions: "
                "store (save findings/decisions), search (semantic search), "
                "get_recent (latest memories), get_stats (usage statistics). "
                "Use to share context with other agents in the group."
            ),
            fn_schema=GroupMemoryInput,
        )

    def call(self, **kwargs) -> ToolOutput:
        """Sync execution via run_async."""
        return run_async(self.acall(**kwargs))

    async def acall(
        self,
        action: GroupMemoryAction,
        content: Optional[str] = None,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        agent_id_filter: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        """Execute the requested action."""
        raw_input = {
            "action": action.value if isinstance(action, GroupMemoryAction) else action,
            "content": content,
            "query": query,
            "memory_type": memory_type,
            "confidence": confidence,
            "tags": tags,
            "limit": limit,
            "agent_id_filter": agent_id_filter,
        }

        # Handle string action conversion
        if isinstance(action, str):
            try:
                action = GroupMemoryAction(action)
            except ValueError:
                return self._error_output(raw_input, f"Unknown action: {action}")

        handlers = {
            GroupMemoryAction.STORE: lambda: self._store(
                content, memory_type, confidence, tags, raw_input
            ),
            GroupMemoryAction.SEARCH: lambda: self._search(
                query, memory_type, limit, raw_input
            ),
            GroupMemoryAction.GET_RECENT: lambda: self._get_recent(
                agent_id_filter, memory_type, limit, raw_input
            ),
            GroupMemoryAction.GET_STATS: lambda: self._get_stats(raw_input),
        }

        handler = handlers.get(action)
        if handler:
            return await handler()
        return self._error_output(raw_input, f"Unknown action: {action}")

    async def _store(
        self,
        content: Optional[str],
        memory_type: Optional[str],
        confidence: float,
        tags: Optional[List[str]],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Store a memory entry."""
        if not content:
            return self._error_output(raw_input, "content is required for store action")

        mem_type = MemoryType.CONTEXT
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                return self._error_output(
                    raw_input,
                    f"Invalid memory_type: {memory_type}. "
                    f"Valid types: {[t.value for t in MemoryType]}",
                )

        # Generate embedding if embeddings provider is available
        embedding = None
        if self._embeddings:
            try:
                embeddings_result = await self._embeddings.aembed([content])
                if embeddings_result and len(embeddings_result) > 0:
                    embedding = embeddings_result[0]
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        memory_id = await self._memory.store(
            agent_id=self._agent_id,
            content=content,
            memory_type=mem_type,
            confidence=confidence,
            embedding=embedding,
            tags=tags,
        )

        return self._success_output(
            raw_input,
            {
                "stored": True,
                "memory_id": memory_id,
                "agent_id": self._agent_id,
                "memory_type": mem_type.value,
                "has_embedding": embedding is not None,
            },
        )

    async def _search(
        self,
        query: Optional[str],
        memory_type: Optional[str],
        limit: int,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Search memories semantically."""
        if not query:
            return self._error_output(raw_input, "query is required for search action")

        if not self._embeddings:
            return self._error_output(
                raw_input,
                "Semantic search requires embeddings provider. Use get_recent instead.",
            )

        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                return self._error_output(
                    raw_input,
                    f"Invalid memory_type: {memory_type}",
                )

        try:
            query_embedding = await self._embeddings.aembed([query])
            if not query_embedding or len(query_embedding) == 0:
                return self._error_output(
                    raw_input, "Failed to generate query embedding"
                )
        except Exception as e:
            return self._error_output(raw_input, f"Embedding generation failed: {e}")

        results = await self._memory.search(
            query_embedding=query_embedding[0],
            memory_type=mem_type,
            limit=limit,
        )

        return self._success_output(
            raw_input,
            {
                "count": len(results),
                "memories": [
                    {
                        "id": m.id,
                        "agent_id": m.agent_id,
                        "content": m.content,
                        "memory_type": m.memory_type.value,
                        "confidence": m.confidence,
                        "tags": m.tags,
                        "similarity": m.similarity,
                        "created_at": m.created_at.isoformat(),
                    }
                    for m in results
                ],
            },
        )

    async def _get_recent(
        self,
        agent_id_filter: Optional[str],
        memory_type: Optional[str],
        limit: int,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Get recent memories."""
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                return self._error_output(
                    raw_input,
                    f"Invalid memory_type: {memory_type}",
                )

        results = await self._memory.get_recent(
            agent_id=agent_id_filter,
            memory_type=mem_type,
            limit=limit,
        )

        return self._success_output(
            raw_input,
            {
                "count": len(results),
                "memories": [
                    {
                        "id": m.id,
                        "agent_id": m.agent_id,
                        "content": m.content,
                        "memory_type": m.memory_type.value,
                        "confidence": m.confidence,
                        "tags": m.tags,
                        "created_at": m.created_at.isoformat(),
                    }
                    for m in results
                ],
            },
        )

    async def _get_stats(self, raw_input: Dict[str, Any]) -> ToolOutput:
        """Get memory statistics."""
        stats = await self._memory.get_stats()

        return self._success_output(
            raw_input,
            {
                "group_id": self._group_id,
                "total_memories": stats.total_memories,
                "active_memories": stats.active_memories,
                "expired_memories": stats.expired_memories,
                "memories_by_type": stats.memories_by_type,
                "memories_by_agent": stats.memories_by_agent,
                "avg_confidence": stats.avg_confidence,
                "oldest_memory": (
                    stats.oldest_memory.isoformat() if stats.oldest_memory else None
                ),
                "newest_memory": (
                    stats.newest_memory.isoformat() if stats.newest_memory else None
                ),
            },
        )

    def _success_output(
        self,
        raw_input: Dict[str, Any],
        result: Dict[str, Any],
    ) -> ToolOutput:
        """Create a success output."""
        return ToolOutput(
            content=json.dumps(result, default=str, ensure_ascii=False),
            tool_name=self.name,
            raw_input=raw_input,
            raw_output=result,
            is_error=False,
        )

    def _error_output(
        self,
        raw_input: Dict[str, Any],
        message: str,
    ) -> ToolOutput:
        """Create an error output."""
        return ToolOutput(
            content=message,
            tool_name=self.name,
            raw_input=raw_input,
            raw_output=None,
            is_error=True,
        )
