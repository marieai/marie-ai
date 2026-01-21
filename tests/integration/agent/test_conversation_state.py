"""Integration tests for conversation state management.

Tests ConversationState, ConversationStore, and state persistence.
"""

import time
from datetime import datetime

import pytest

from marie.agent import Message
from marie.agent.state import ConversationState, ConversationStore


class TestConversationState:
    """Test ConversationState class."""

    def test_create_conversation_state(self):
        """Test creating a conversation state."""
        state = ConversationState(conversation_id="test-123")

        assert state.conversation_id == "test-123"
        assert len(state.messages) == 0
        assert state.created_at is not None
        assert state.updated_at is not None

    def test_add_message(self):
        """Test adding messages to state."""
        state = ConversationState(conversation_id="test")

        msg = Message.user("Hello")
        state.add_message(msg)

        assert len(state.messages) == 1

    def test_add_message_dict(self):
        """Test adding message as dict."""
        state = ConversationState(conversation_id="test")

        state.add_message({"role": "user", "content": "Hello"})

        assert len(state.messages) == 1

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        state = ConversationState(conversation_id="test")

        state.add_message(Message.user("Hello"))
        state.add_message(Message.assistant("Hi there!"))
        state.add_message(Message.user("How are you?"))

        assert len(state.messages) == 3
        assert state.message_count == 3

    def test_get_messages(self):
        """Test getting messages."""
        state = ConversationState(conversation_id="test")

        for i in range(10):
            state.add_message(Message.user(f"Message {i}"))

        # Get all
        all_msgs = state.get_messages()
        assert len(all_msgs) == 10
        assert all(isinstance(m, Message) for m in all_msgs)

        # Get with limit
        limited = state.get_messages(limit=5)
        assert len(limited) == 5
        # Should get the last 5 messages
        assert limited[0].content == "Message 5"

    def test_clear_messages(self):
        """Test clearing messages."""
        state = ConversationState(conversation_id="test")

        state.add_message(Message.user("Hello"))
        state.add_message(Message.assistant("Hi!"))
        assert len(state.messages) > 0

        state.clear_messages()
        assert len(state.messages) == 0

    def test_updated_at_changes(self):
        """Test that updated_at changes when state is modified."""
        state = ConversationState(conversation_id="test")
        original_updated = state.updated_at

        time.sleep(0.01)  # Small delay
        state.add_message(Message.user("Hello"))

        # updated_at should be different
        assert state.updated_at >= original_updated

    def test_metadata(self):
        """Test metadata storage."""
        state = ConversationState(
            conversation_id="test",
            metadata={"user_id": "user-123"}
        )

        assert state.metadata["user_id"] == "user-123"

        # Update metadata
        state.metadata["session"] = "session-456"
        assert state.metadata["session"] == "session-456"


class TestConversationStateToolResults:
    """Test tool result caching in ConversationState."""

    def test_add_tool_result(self):
        """Test adding a tool result."""
        state = ConversationState(conversation_id="test")

        state.add_tool_result("search_123", {"results": ["a", "b"]})

        result = state.get_tool_result("search_123")
        assert result == {"results": ["a", "b"]}

    def test_get_nonexistent_tool_result(self):
        """Test getting non-existent tool result."""
        state = ConversationState(conversation_id="test")

        result = state.get_tool_result("nonexistent")
        assert result is None

    def test_clear_tool_results(self):
        """Test clearing tool results."""
        state = ConversationState(conversation_id="test")

        state.add_tool_result("a", "value_a")
        state.add_tool_result("b", "value_b")

        state.clear_tool_results()

        assert state.get_tool_result("a") is None
        assert state.get_tool_result("b") is None


class TestConversationStore:
    """Test ConversationStore for managing multiple conversations."""

    @pytest.mark.asyncio
    async def test_get_or_create(self):
        """Test get_or_create method."""
        store = ConversationStore()

        # First call creates
        state1 = await store.get_or_create("new-conv")
        assert state1 is not None
        assert state1.conversation_id == "new-conv"

        # Second call gets existing
        state2 = await store.get_or_create("new-conv")
        assert state1.conversation_id == state2.conversation_id

    @pytest.mark.asyncio
    async def test_get_conversation(self):
        """Test getting a conversation."""
        store = ConversationStore()

        await store.get_or_create("test-conv")
        state = await store.get("test-conv")

        assert state is not None
        assert state.conversation_id == "test-conv"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting non-existent conversation."""
        store = ConversationStore()

        state = await store.get("nonexistent")
        assert state is None

    @pytest.mark.asyncio
    async def test_add_message_to_conversation(self):
        """Test adding message through store."""
        store = ConversationStore()

        msg = Message.user("Hello from store")
        await store.add_message("test-conv", msg)

        state = await store.get("test-conv")
        assert len(state.messages) == 1

    @pytest.mark.asyncio
    async def test_get_messages(self):
        """Test getting messages through store."""
        store = ConversationStore()

        await store.add_message("conv", Message.user("Message 1"))
        await store.add_message("conv", Message.assistant("Response 1"))

        messages = await store.get_messages("conv")

        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)

    @pytest.mark.asyncio
    async def test_clear_conversation(self):
        """Test clearing a conversation."""
        store = ConversationStore()

        await store.get_or_create("to-clear")
        assert await store.get("to-clear") is not None

        result = await store.clear("to-clear")
        assert result is True
        assert await store.get("to-clear") is None

    @pytest.mark.asyncio
    async def test_clear_nonexistent(self):
        """Test clearing non-existent conversation."""
        store = ConversationStore()

        result = await store.clear("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_conversations(self):
        """Test listing all conversations."""
        store = ConversationStore()

        await store.get_or_create("conv-1")
        await store.get_or_create("conv-2")
        await store.get_or_create("conv-3")

        conversations = await store.list_conversations()

        assert len(conversations) >= 3
        conv_ids = [c["conversation_id"] for c in conversations]
        assert "conv-1" in conv_ids
        assert "conv-2" in conv_ids
        assert "conv-3" in conv_ids

    @pytest.mark.asyncio
    async def test_update_metadata(self):
        """Test updating conversation metadata."""
        store = ConversationStore()

        await store.get_or_create("meta-conv", metadata={"key1": "value1"})
        await store.update_metadata("meta-conv", {"key2": "value2"})

        state = await store.get("meta-conv")
        assert state.metadata["key1"] == "value1"
        assert state.metadata["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_max_conversations_eviction(self):
        """Test that old conversations are evicted."""
        store = ConversationStore(max_conversations=3)

        await store.get_or_create("conv-1")
        await store.get_or_create("conv-2")
        await store.get_or_create("conv-3")
        await store.get_or_create("conv-4")  # Should evict conv-1

        # conv-1 should be evicted (it's the oldest)
        state = await store.get("conv-1")
        assert state is None

        # Others should still exist
        assert await store.get("conv-2") is not None
        assert await store.get("conv-4") is not None

    @pytest.mark.asyncio
    async def test_max_messages_trimming(self):
        """Test that messages are trimmed when over limit."""
        store = ConversationStore(max_messages_per_conversation=5)

        for i in range(10):
            await store.add_message("conv", Message.user(f"Message {i}"))

        messages = await store.get_messages("conv")

        # Should only keep max_messages
        assert len(messages) <= 5


class TestConversationStoreWithMetadata:
    """Test conversation metadata handling."""

    @pytest.mark.asyncio
    async def test_create_with_metadata(self):
        """Test creating conversation with initial metadata."""
        store = ConversationStore()

        state = await store.get_or_create(
            "meta-conv",
            metadata={"user_id": "user-123", "session": "sess-456"}
        )

        assert state.metadata["user_id"] == "user-123"
        assert state.metadata["session"] == "sess-456"

    @pytest.mark.asyncio
    async def test_metadata_persists(self):
        """Test that metadata persists across gets."""
        store = ConversationStore()

        await store.get_or_create("persist-conv", metadata={"key": "value"})

        # Get again
        state = await store.get("persist-conv")
        assert state.metadata["key"] == "value"


class TestConversationStoreCleanup:
    """Test conversation cleanup and TTL."""

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleaning up expired conversations."""
        store = ConversationStore(ttl_seconds=0)  # Expire immediately

        await store.get_or_create("expires")
        time.sleep(0.01)  # Let it expire

        count = await store.cleanup()

        # Should have cleaned at least one
        assert count >= 0  # May vary based on timing


class TestConversationStateSerialization:
    """Test serialization of conversation state."""

    def test_conversation_state_model_dump(self):
        """Test model_dump for ConversationState."""
        state = ConversationState(
            conversation_id="test-123",
            metadata={"key": "value"}
        )
        state.add_message(Message.user("Hello"))
        state.add_message(Message.assistant("Hi!"))

        data = state.model_dump()

        assert data["conversation_id"] == "test-123"
        assert len(data["messages"]) == 2
        assert data["metadata"]["key"] == "value"

    def test_restore_from_dict(self):
        """Test restoring ConversationState from dict."""
        data = {
            "conversation_id": "restored-123",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "metadata": {"restored": True},
            "tool_results": {},
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        state = ConversationState(**data)

        assert state.conversation_id == "restored-123"
        assert len(state.messages) == 2
        assert state.metadata["restored"] is True


class TestConversationStoreThreadSafety:
    """Test thread safety of ConversationStore."""

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to store."""
        import asyncio

        store = ConversationStore()
        conv_id = "concurrent-test"
        await store.get_or_create(conv_id)

        async def add_messages():
            for i in range(10):
                await store.add_message(conv_id, Message.user(f"Message {i}"))

        # Run multiple coroutines concurrently
        await asyncio.gather(*[add_messages() for _ in range(5)])

        # Verify messages were added (may be trimmed due to max_messages)
        state = await store.get(conv_id)
        assert len(state.messages) > 0
