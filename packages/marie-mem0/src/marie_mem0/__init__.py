"""Mem0 memory integration for Marie AI agents.

This package provides Mem0 SDK integration for persistent agent memory,
allowing agents to store and retrieve memories across conversations.
"""

import importlib.metadata

from marie_mem0._config import (
    Mem0Config,
    Mem0EmbedderConfig,
    Mem0LLMConfig,
    Mem0VectorStoreConfig,
)
from marie_mem0._memory import AsyncMem0Memory, Mem0Memory
from marie_mem0._provider import Mem0Provider

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Mem0Config",
    "Mem0EmbedderConfig",
    "Mem0LLMConfig",
    "Mem0VectorStoreConfig",
    "Mem0Memory",
    "AsyncMem0Memory",
    "Mem0Provider",
    "__version__",
]
