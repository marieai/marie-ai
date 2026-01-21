"""Conversation state management for Marie agent framework.

This module provides conversation persistence and state management
for maintaining context across agent interactions.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from datetime import datetime
from threading import RLock
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from marie.agent.message import Message
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.state.conversation")


class ToolResultEntry(BaseModel):
    """Entry for a cached tool result with timestamp."""

    result: Any = Field(..., description="Tool result")
    timestamp: float = Field(default_factory=time.time, description="Cache timestamp")


class ConversationState(BaseModel):
    """State for a single conversation.

    Tracks messages, metadata, and timestamps for conversation management.
    Tool results are cached with TTL and size limits to prevent memory leaks.
    """

    conversation_id: str = Field(..., description="Unique conversation identifier")
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation messages",
    )
    tool_results: Dict[str, ToolResultEntry] = Field(
        default_factory=dict,
        description="Cached tool results by tool call ID",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional conversation metadata",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Last update timestamp",
    )

    # Configuration for tool result cache
    _max_tool_results: int = 100  # Maximum number of cached tool results
    _tool_result_ttl: float = 3600.0  # TTL in seconds (1 hour)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def message_count(self) -> int:
        """Get number of messages in conversation."""
        return len(self.messages)

    def add_message(self, message: Union[Message, Dict[str, Any]]) -> None:
        """Add a message to the conversation.

        Args:
            message: Message object or dict
        """
        if isinstance(message, Message):
            msg_dict = message.model_dump()
        else:
            msg_dict = message

        self.messages.append(msg_dict)
        self.updated_at = datetime.now()

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages as Message objects.

        Args:
            limit: Optional limit on number of recent messages

        Returns:
            List of Message objects
        """
        msgs = self.messages[-limit:] if limit else self.messages
        return [Message(**m) for m in msgs]

    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()
        self.updated_at = datetime.now()

    def add_tool_result(self, tool_call_id: str, result: Any) -> None:
        """Cache a tool result with TTL and size limits.

        Args:
            tool_call_id: Tool call identifier
            result: Tool result to cache
        """
        # Clean up expired entries first
        self._cleanup_tool_results()

        # Evict oldest entries if over limit
        while len(self.tool_results) >= self._max_tool_results:
            oldest_key = min(
                self.tool_results.keys(),
                key=lambda k: self.tool_results[k].timestamp,
            )
            del self.tool_results[oldest_key]

        # Add new entry
        self.tool_results[tool_call_id] = ToolResultEntry(result=result)
        self.updated_at = datetime.now()

    def get_tool_result(self, tool_call_id: str) -> Optional[Any]:
        """Get a cached tool result.

        Args:
            tool_call_id: Tool call identifier

        Returns:
            Cached result or None if not found or expired
        """
        entry = self.tool_results.get(tool_call_id)
        if entry is None:
            return None

        # Check if expired
        if time.time() - entry.timestamp > self._tool_result_ttl:
            del self.tool_results[tool_call_id]
            return None

        return entry.result

    def _cleanup_tool_results(self) -> int:
        """Remove expired tool results.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            k
            for k, v in self.tool_results.items()
            if now - v.timestamp > self._tool_result_ttl
        ]
        for key in expired_keys:
            del self.tool_results[key]
        return len(expired_keys)

    def clear_tool_results(self) -> None:
        """Clear all cached tool results."""
        self.tool_results.clear()
        self.updated_at = datetime.now()


class ConversationStore:
    """In-memory conversation store with LRU eviction.

    Provides thread-safe conversation state management with automatic
    cleanup of old conversations.

    Example:
        ```python
        store = ConversationStore(max_conversations=100)

        # Add messages
        await store.add_message("conv-1", Message.user("Hello"))
        await store.add_message("conv-1", Message.assistant("Hi there!"))

        # Get history
        messages = await store.get_messages("conv-1")

        # Clear conversation
        await store.clear("conv-1")
        ```
    """

    def __init__(
        self,
        max_conversations: int = 1000,
        max_messages_per_conversation: int = 100,
        ttl_seconds: Optional[int] = 3600,
    ):
        """Initialize the conversation store.

        Args:
            max_conversations: Maximum number of conversations to keep
            max_messages_per_conversation: Max messages per conversation
            ttl_seconds: Time-to-live for conversations (None for no expiry)
        """
        self._conversations: OrderedDict[str, ConversationState] = OrderedDict()
        self._lock = RLock()
        self._max_conversations = max_conversations
        self._max_messages = max_messages_per_conversation
        self._ttl_seconds = ttl_seconds

    async def get_or_create(
        self,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationState:
        """Get or create a conversation.

        Args:
            conversation_id: Conversation identifier
            metadata: Optional initial metadata

        Returns:
            ConversationState instance
        """
        with self._lock:
            if conversation_id in self._conversations:
                # Move to end (LRU)
                self._conversations.move_to_end(conversation_id)
                return self._conversations[conversation_id]

            # Create new conversation
            state = ConversationState(
                conversation_id=conversation_id,
                metadata=metadata or {},
            )
            self._conversations[conversation_id] = state

            # Evict old conversations if needed
            self._evict_if_needed()

            return state

    async def get(self, conversation_id: str) -> Optional[ConversationState]:
        """Get a conversation by ID.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ConversationState or None if not found
        """
        with self._lock:
            state = self._conversations.get(conversation_id)
            if state:
                # Check TTL
                if self._is_expired(state):
                    del self._conversations[conversation_id]
                    return None
                # Move to end (LRU)
                self._conversations.move_to_end(conversation_id)
            return state

    async def add_message(
        self,
        conversation_id: str,
        message: Union[Message, Dict[str, Any]],
    ) -> None:
        """Add a message to a conversation.

        Creates the conversation if it doesn't exist.

        Args:
            conversation_id: Conversation identifier
            message: Message to add
        """
        state = await self.get_or_create(conversation_id)

        with self._lock:
            state.add_message(message)

            # Trim messages if needed
            if len(state.messages) > self._max_messages:
                # Keep system messages and trim oldest
                system_msgs = [m for m in state.messages if m.get("role") == "system"]
                other_msgs = [m for m in state.messages if m.get("role") != "system"]
                keep_count = self._max_messages - len(system_msgs)
                state.messages = system_msgs + other_msgs[-keep_count:]

    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get messages from a conversation.

        Args:
            conversation_id: Conversation identifier
            limit: Optional message limit

        Returns:
            List of Messages (empty if conversation not found)
        """
        state = await self.get(conversation_id)
        if state is None:
            return []
        return state.get_messages(limit)

    async def update_metadata(
        self,
        conversation_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Update conversation metadata.

        Args:
            conversation_id: Conversation identifier
            metadata: Metadata to merge
        """
        state = await self.get(conversation_id)
        if state:
            with self._lock:
                state.metadata.update(metadata)
                state.updated_at = datetime.now()

    async def clear(self, conversation_id: str) -> bool:
        """Clear a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if cleared, False if not found
        """
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                return True
            return False

    async def list_conversations(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List all conversations.

        Args:
            limit: Optional limit on results

        Returns:
            List of conversation summaries
        """
        with self._lock:
            # Clean expired conversations
            self._clean_expired()

            conversations = []
            items = list(self._conversations.items())
            if limit:
                items = items[-limit:]

            for conv_id, state in items:
                conversations.append(
                    {
                        "conversation_id": conv_id,
                        "message_count": state.message_count,
                        "created_at": state.created_at.isoformat(),
                        "updated_at": state.updated_at.isoformat(),
                        "metadata": state.metadata,
                    }
                )

            return conversations

    async def cleanup(self) -> int:
        """Clean up expired conversations.

        Returns:
            Number of conversations removed
        """
        with self._lock:
            return self._clean_expired()

    def _evict_if_needed(self) -> None:
        """Evict oldest conversations if over limit."""
        while len(self._conversations) > self._max_conversations:
            # Remove oldest (first item)
            oldest_id = next(iter(self._conversations))
            del self._conversations[oldest_id]
            logger.debug(f"Evicted conversation: {oldest_id}")

    def _is_expired(self, state: ConversationState) -> bool:
        """Check if a conversation is expired.

        Args:
            state: Conversation state

        Returns:
            True if expired
        """
        if self._ttl_seconds is None:
            return False

        age = (datetime.now() - state.updated_at).total_seconds()
        return age > self._ttl_seconds

    def _clean_expired(self) -> int:
        """Remove expired conversations.

        Returns:
            Number removed
        """
        if self._ttl_seconds is None:
            return 0

        expired = [
            conv_id
            for conv_id, state in self._conversations.items()
            if self._is_expired(state)
        ]

        for conv_id in expired:
            del self._conversations[conv_id]

        if expired:
            logger.debug(f"Cleaned {len(expired)} expired conversations")

        return len(expired)


class AgentMemoryBridge:
    """Bridge between agent memory and DAG execution context.

    Links conversations to DAG jobs for proper result propagation
    and state management across distributed execution.
    """

    def __init__(self, conversation_store: ConversationStore):
        """Initialize the bridge.

        Args:
            conversation_store: ConversationStore instance
        """
        self._store = conversation_store
        self._dag_conversation_map: Dict[str, str] = {}  # dag_id -> conversation_id
        self._lock = RLock()

    async def link_dag_to_conversation(
        self,
        dag_id: str,
        conversation_id: str,
    ) -> None:
        """Link a DAG execution to a conversation.

        Args:
            dag_id: DAG identifier
            conversation_id: Conversation identifier
        """
        with self._lock:
            self._dag_conversation_map[dag_id] = conversation_id
            logger.debug(f"Linked DAG {dag_id} to conversation {conversation_id}")

    async def get_conversation_for_dag(
        self,
        dag_id: str,
    ) -> Optional[str]:
        """Get the conversation ID for a DAG.

        Args:
            dag_id: DAG identifier

        Returns:
            Conversation ID or None
        """
        with self._lock:
            return self._dag_conversation_map.get(dag_id)

    async def propagate_result(
        self,
        dag_id: str,
        job_id: str,
        result: Any,
    ) -> bool:
        """Propagate a job result back to the conversation.

        Args:
            dag_id: DAG identifier
            job_id: Job identifier
            result: Job result

        Returns:
            True if propagated successfully
        """
        conversation_id = await self.get_conversation_for_dag(dag_id)
        if not conversation_id:
            logger.warning(f"No conversation found for DAG {dag_id}")
            return False

        # Add result as tool result message
        result_message = Message.tool_result(
            tool_call_id=job_id,
            content=str(result),
            name=f"dag_job_{job_id}",
        )

        await self._store.add_message(conversation_id, result_message)
        logger.debug(
            f"Propagated result from job {job_id} to conversation {conversation_id}"
        )

        return True

    async def unlink_dag(self, dag_id: str) -> bool:
        """Remove DAG-conversation link.

        Args:
            dag_id: DAG identifier

        Returns:
            True if unlinked, False if not found
        """
        with self._lock:
            if dag_id in self._dag_conversation_map:
                del self._dag_conversation_map[dag_id]
                return True
            return False
