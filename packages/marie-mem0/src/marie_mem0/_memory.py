"""Mem0 memory client wrapper for Marie AI agents.

This module provides both sync and async wrappers around the mem0ai SDK
for managing persistent agent memories with pgvector storage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from marie_mem0._config import Mem0Config

logger = logging.getLogger(__name__)


class Mem0Memory:
    """Wrapper around mem0ai SDK for Marie agent memory.

    Provides a simplified interface for adding, searching, and managing
    agent memories using the Mem0 SDK with pgvector storage.

    Example:
        ```python
        from marie_mem0 import Mem0Config, Mem0Memory

        config = Mem0Config(enabled=True)
        memory = Mem0Memory(config)

        # Add a memory
        memory.add(
            messages=[{"role": "user", "content": "My name is John"}],
            user_id="user-123",
        )

        # Search memories
        results = memory.search(
            query="What is my name?",
            user_id="user-123",
        )
        ```
    """

    def __init__(self, config: "Mem0Config"):
        """Initialize the Mem0 memory client.

        Args:
            config: Mem0 configuration with vector store, LLM, and embedder settings
        """
        self.config = config
        self._client = None

        if config.enabled:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the mem0 client with configuration."""
        try:
            from mem0 import Memory
        except ImportError as e:
            logger.error(
                "mem0ai package not installed. Install with: pip install mem0ai"
            )
            raise ImportError(
                "mem0ai package required for Mem0 memory integration. "
                "Install with: pip install marie-mem0"
            ) from e

        mem0_config = {
            "vector_store": self.config.vector_store.model_dump(),
            "llm": self.config.llm.model_dump(),
            "embedder": self.config.embedder.model_dump(),
        }

        logger.debug(f"Initializing Mem0 client with config: {mem0_config}")
        self._client = Memory.from_config(config_dict=mem0_config)
        logger.info("Mem0 memory client initialized successfully")

    @property
    def is_enabled(self) -> bool:
        """Check if memory integration is enabled and client is initialized."""
        return self.config.enabled and self._client is not None

    def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add messages to memory.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            user_id: User identifier for memory scoping
            agent_id: Optional agent identifier for per-agent memory isolation
            metadata: Optional metadata to attach to the memory

        Returns:
            Response from mem0 containing memory IDs and status
        """
        if not self._client:
            logger.debug("Mem0 client not initialized, skipping add")
            return {}

        try:
            result = self._client.add(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
            )
            logger.debug(f"Added memory for user={user_id}, agent={agent_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return {}

    def search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories by query.

        Args:
            query: Search query string
            user_id: User identifier for memory scoping
            agent_id: Optional agent identifier for per-agent memory search
            limit: Maximum number of results to return

        Returns:
            List of matching memory dicts with 'memory', 'score', etc.
        """
        if not self._client:
            logger.debug("Mem0 client not initialized, skipping search")
            return []

        try:
            results = self._client.search(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
            )
            logger.debug(
                f"Searched memories for user={user_id}, found {len(results)} results"
            )
            return results
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    def get_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all memories for a user.

        Args:
            user_id: User identifier for memory scoping
            agent_id: Optional agent identifier for per-agent memory retrieval

        Returns:
            List of all memory dicts for the user
        """
        if not self._client:
            logger.debug("Mem0 client not initialized, skipping get_all")
            return []

        try:
            kwargs: Dict[str, Any] = {"user_id": user_id}
            if agent_id:
                kwargs["agent_id"] = agent_id
            results = self._client.get_all(**kwargs)
            logger.debug(f"Retrieved {len(results)} memories for user={user_id}")
            return results
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory identifier

        Returns:
            Memory dict if found, None otherwise
        """
        if not self._client:
            logger.debug("Mem0 client not initialized, skipping get")
            return None

        try:
            return self._client.get(memory_id=memory_id)
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory.

        Args:
            memory_id: The memory identifier to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._client:
            logger.debug("Mem0 client not initialized, skipping delete")
            return False

        try:
            self._client.delete(memory_id=memory_id)
            logger.debug(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def delete_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Delete all memories for a user.

        Args:
            user_id: User identifier for memory scoping
            agent_id: Optional agent identifier for per-agent memory deletion

        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._client:
            logger.debug("Mem0 client not initialized, skipping delete_all")
            return False

        try:
            kwargs: Dict[str, Any] = {"user_id": user_id}
            if agent_id:
                kwargs["agent_id"] = agent_id
            self._client.delete_all(**kwargs)
            logger.debug(f"Deleted all memories for user={user_id}, agent={agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}")
            return False

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get the history/versions of a memory.

        Args:
            memory_id: The memory identifier

        Returns:
            List of historical versions of the memory
        """
        if not self._client:
            logger.debug("Mem0 client not initialized, skipping history")
            return []

        try:
            return self._client.history(memory_id=memory_id)
        except Exception as e:
            logger.error(f"Failed to get memory history {memory_id}: {e}")
            return []


class AsyncMem0Memory:
    """Async wrapper around mem0ai SDK for Marie agent memory.

    Provides an async interface for adding, searching, and managing
    agent memories using the Mem0 SDK with pgvector storage.
    """

    def __init__(self, config: "Mem0Config"):
        """Initialize the async Mem0 memory client.

        Args:
            config: Mem0 configuration with vector store, LLM, and embedder settings
        """
        self.config = config
        self._client = None

        if config.enabled:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the async mem0 client with configuration."""
        try:
            from mem0 import AsyncMemory
        except ImportError as e:
            logger.error(
                "mem0ai package not installed. Install with: pip install mem0ai"
            )
            raise ImportError(
                "mem0ai package required for Mem0 memory integration. "
                "Install with: pip install marie-mem0"
            ) from e

        mem0_config = {
            "vector_store": self.config.vector_store.model_dump(),
            "llm": self.config.llm.model_dump(),
            "embedder": self.config.embedder.model_dump(),
        }

        logger.debug(f"Initializing async Mem0 client with config: {mem0_config}")
        self._client = AsyncMemory.from_config(config_dict=mem0_config)
        logger.info("Async Mem0 memory client initialized successfully")

    @property
    def is_enabled(self) -> bool:
        """Check if memory integration is enabled and client is initialized."""
        return self.config.enabled and self._client is not None

    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add messages to memory asynchronously."""
        if not self._client:
            return {}

        try:
            result = await self._client.add(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return {}

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories by query asynchronously."""
        if not self._client:
            return []

        try:
            results = await self._client.search(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
            )
            return results
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    async def get_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all memories for a user asynchronously."""
        if not self._client:
            return []

        try:
            kwargs: Dict[str, Any] = {"user_id": user_id}
            if agent_id:
                kwargs["agent_id"] = agent_id
            return await self._client.get_all(**kwargs)
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    async def delete(self, memory_id: str) -> bool:
        """Delete a specific memory asynchronously."""
        if not self._client:
            return False

        try:
            await self._client.delete(memory_id=memory_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def delete_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Delete all memories for a user asynchronously."""
        if not self._client:
            return False

        try:
            kwargs: Dict[str, Any] = {"user_id": user_id}
            if agent_id:
                kwargs["agent_id"] = agent_id
            await self._client.delete_all(**kwargs)
            return True
        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}")
            return False
