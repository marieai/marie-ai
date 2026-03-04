"""Memory integration for Marie agent framework.

This module provides memory capabilities for agents:
- Mem0 integration for individual agent memory
- Group memory for coordinated multi-agent execution
"""

from marie_mem0 import (
    AsyncMem0Memory,
    Mem0Config,
    Mem0EmbedderConfig,
    Mem0LLMConfig,
    Mem0Memory,
    Mem0Provider,
    Mem0VectorStoreConfig,
)

from marie.agent.memory.group_memory import (
    GroupMemory,
    GroupMemoryConfig,
    GroupMemoryEntry,
    GroupMemoryStats,
    MemoryType,
)

__all__ = [
    # Mem0 exports
    "Mem0Config",
    "Mem0EmbedderConfig",
    "Mem0LLMConfig",
    "Mem0VectorStoreConfig",
    "Mem0Memory",
    "AsyncMem0Memory",
    "Mem0Provider",
    # Group memory exports
    "GroupMemory",
    "GroupMemoryConfig",
    "GroupMemoryEntry",
    "GroupMemoryStats",
    "MemoryType",
]
