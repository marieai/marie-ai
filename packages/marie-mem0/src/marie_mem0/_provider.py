"""Mem0 Context Provider for Marie AI agents.

This module provides a context provider pattern for integrating Mem0
memory into agent workflows, similar to the agent-framework pattern.
"""

from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from marie_mem0._config import Mem0Config

logger = logging.getLogger(__name__)


class Mem0Provider(AbstractAsyncContextManager):
    """Mem0 Context Provider for agent memory integration.

    Provides automatic memory storage and retrieval for agent conversations.
    Implements the async context manager pattern for clean resource management.

    Example:
        ```python
        from marie_mem0 import Mem0Config, Mem0Provider

        config = Mem0Config(enabled=True)

        async with Mem0Provider(config=config, user_id="user-123") as provider:
            # Get context before invoking agent
            context = await provider.get_context(messages)

            # Store interaction after agent response
            await provider.store_interaction(messages, response)
        ```
    """

    DEFAULT_CONTEXT_PROMPT = "Relevant memories from previous interactions:"

    def __init__(
        self,
        config: Optional["Mem0Config"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        context_prompt: str = DEFAULT_CONTEXT_PROMPT,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the Mem0 provider.

        Args:
            config: Mem0 configuration. If None, creates default config.
            user_id: User identifier for memory scoping.
            agent_id: Agent identifier for per-agent memory isolation.
            thread_id: Thread/conversation identifier.
            context_prompt: Prompt to prepend to retrieved memories.
            api_key: Optional API key (for Mem0 cloud service).
        """
        self._config = config
        self._client = None
        self._async_client = None
        self.user_id = user_id
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.context_prompt = context_prompt
        self.api_key = api_key

    async def __aenter__(self) -> "Mem0Provider":
        """Async context manager entry - initialize the client."""
        if self._config and self._config.enabled:
            await self._initialize_async_client()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - cleanup resources."""
        self._async_client = None

    async def _initialize_async_client(self) -> None:
        """Initialize the async mem0 client."""
        if not self._config:
            return

        try:
            from mem0 import AsyncMemory

            mem0_config = {
                "vector_store": self._config.vector_store.model_dump(),
                "llm": self._config.llm.model_dump(),
                "embedder": self._config.embedder.model_dump(),
            }

            self._async_client = AsyncMemory.from_config(config_dict=mem0_config)
            logger.debug("Async Mem0 client initialized in provider")
        except ImportError:
            logger.warning("mem0ai package not installed, provider disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 client: {e}")

    @property
    def is_enabled(self) -> bool:
        """Check if the provider is enabled and has a client."""
        return self._async_client is not None

    def _validate_filters(self) -> None:
        """Validate that at least one filter is provided."""
        if not self.agent_id and not self.user_id and not self.thread_id:
            raise ValueError(
                "At least one of agent_id, user_id, or thread_id is required."
            )

    async def get_context(
        self,
        messages: Sequence[Dict[str, str]],
        limit: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Get memory context for the given messages.

        Searches for relevant memories based on the last user message
        and returns formatted context.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            limit: Maximum number of memories to retrieve.

        Returns:
            Context dict with 'content' key containing formatted memories,
            or None if no relevant memories found.
        """
        if not self._async_client:
            return None

        self._validate_filters()

        # Extract query from last user message
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        if not query.strip():
            return None

        try:
            results = await self._async_client.search(
                query=query,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.thread_id,
                limit=limit,
            )

            # Handle different response formats
            memories: List[Dict[str, Any]] = []
            if isinstance(results, list):
                memories = results
            elif isinstance(results, dict) and "results" in results:
                memories = results["results"]

            if not memories:
                return None

            # Format memories
            memory_lines = [m.get("memory", "") for m in memories if m.get("memory")]

            if not memory_lines:
                return None

            formatted = "\n".join(f"- {line}" for line in memory_lines)
            content = f"{self.context_prompt}\n{formatted}"

            return {"role": "system", "content": content}

        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return None

    async def store_interaction(
        self,
        messages: Sequence[Dict[str, str]],
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store an interaction in memory.

        Args:
            messages: List of message dicts from the conversation.
            response: Optional assistant response to include.
            metadata: Optional metadata to attach.

        Returns:
            Response from mem0 with stored memory info.
        """
        if not self._async_client:
            return {}

        self._validate_filters()

        # Build message list (exclude system messages)
        msg_list = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        # Add response if provided
        if response:
            msg_list.append({"role": "assistant", "content": response})

        if not msg_list:
            return {}

        try:
            result = await self._async_client.add(
                messages=msg_list,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.thread_id,
                metadata=metadata,
            )
            logger.debug(f"Stored interaction for user={self.user_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            return {}

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories by query.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching memory dicts.
        """
        if not self._async_client:
            return []

        self._validate_filters()

        try:
            results = await self._async_client.search(
                query=query,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.thread_id,
                limit=limit,
            )

            if isinstance(results, list):
                return results
            elif isinstance(results, dict) and "results" in results:
                return results["results"]
            return []
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    async def get_all(self) -> List[Dict[str, Any]]:
        """Get all memories for the current user/agent scope.

        Returns:
            List of all memory dicts.
        """
        if not self._async_client:
            return []

        self._validate_filters()

        try:
            kwargs: Dict[str, Any] = {}
            if self.user_id:
                kwargs["user_id"] = self.user_id
            if self.agent_id:
                kwargs["agent_id"] = self.agent_id

            return await self._async_client.get_all(**kwargs)
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    async def delete_all(self) -> bool:
        """Delete all memories for the current user/agent scope.

        Returns:
            True if successful, False otherwise.
        """
        if not self._async_client:
            return False

        self._validate_filters()

        try:
            kwargs: Dict[str, Any] = {}
            if self.user_id:
                kwargs["user_id"] = self.user_id
            if self.agent_id:
                kwargs["agent_id"] = self.agent_id

            await self._async_client.delete_all(**kwargs)
            return True
        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}")
            return False
