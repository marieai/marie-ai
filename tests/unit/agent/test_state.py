"""Tests for Marie agent state management."""

from datetime import datetime, timedelta

import pytest

from marie.agent.message import Message
from marie.agent.state.conversation import (
    AgentMemoryBridge,
    ConversationState,
    ConversationStore,
)


class TestConversationState:
    """Tests for ConversationState class."""

    def test_create_state(self):
        """Test creating conversation state."""
        state = ConversationState(conversation_id="conv-123")
        assert state.conversation_id == "conv-123"
        assert state.message_count == 0
        assert len(state.messages) == 0

    def test_add_message_dict(self):
        """Test adding message as dict."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message({"role": "user", "content": "Hello"})
        assert state.message_count == 1
        assert state.messages[0]["content"] == "Hello"

    def test_add_message_object(self):
        """Test adding Message object."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message(Message.user("Hello"))
        assert state.message_count == 1

    def test_get_messages(self):
        """Test getting messages."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message(Message.user("First"))
        state.add_message(Message.assistant("Second"))
        state.add_message(Message.user("Third"))

        # Get all
        all_msgs = state.get_messages()
        assert len(all_msgs) == 3

        # Get with limit
        limited = state.get_messages(limit=2)
        assert len(limited) == 2
        assert limited[0].content == "Second"
        assert limited[1].content == "Third"

    def test_clear_messages(self):
        """Test clearing messages."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message(Message.user("Hello"))
        state.clear_messages()
        assert state.message_count == 0

    def test_tool_results(self):
        """Test tool result caching."""
        state = ConversationState(conversation_id="conv-123")
        state.add_tool_result("call_1", {"result": "data"})

        result = state.get_tool_result("call_1")
        assert result == {"result": "data"}

        missing = state.get_tool_result("nonexistent")
        assert missing is None

    def test_updated_at_changes(self):
        """Test that updated_at changes on modifications."""
        state = ConversationState(conversation_id="conv-123")
        original_time = state.updated_at

        state.add_message(Message.user("Hello"))
        assert state.updated_at >= original_time


class TestConversationStore:
    """Tests for ConversationStore class."""

    @pytest.mark.asyncio
    async def test_get_or_create(self):
        """Test getting or creating conversation."""
        store = ConversationStore()

        # Create new
        state1 = await store.get_or_create("conv-1")
        assert state1.conversation_id == "conv-1"

        # Get existing
        state2 = await store.get_or_create("conv-1")
        assert state1 is state2

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting non-existent conversation."""
        store = ConversationStore()
        state = await store.get("nonexistent")
        assert state is None

    @pytest.mark.asyncio
    async def test_add_message(self):
        """Test adding message to conversation."""
        store = ConversationStore()
        await store.add_message("conv-1", Message.user("Hello"))

        messages = await store.get_messages("conv-1")
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_get_messages_empty(self):
        """Test getting messages from non-existent conversation."""
        store = ConversationStore()
        messages = await store.get_messages("nonexistent")
        assert messages == []

    @pytest.mark.asyncio
    async def test_update_metadata(self):
        """Test updating conversation metadata."""
        store = ConversationStore()
        await store.get_or_create("conv-1", metadata={"key": "value"})
        await store.update_metadata("conv-1", {"new_key": "new_value"})

        state = await store.get("conv-1")
        assert state.metadata["key"] == "value"
        assert state.metadata["new_key"] == "new_value"

    @pytest.mark.asyncio
    async def test_clear_conversation(self):
        """Test clearing a conversation."""
        store = ConversationStore()
        await store.get_or_create("conv-1")

        result = await store.clear("conv-1")
        assert result is True

        state = await store.get("conv-1")
        assert state is None

    @pytest.mark.asyncio
    async def test_clear_nonexistent(self):
        """Test clearing non-existent conversation."""
        store = ConversationStore()
        result = await store.clear("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_conversations(self):
        """Test listing conversations."""
        store = ConversationStore()
        await store.get_or_create("conv-1")
        await store.get_or_create("conv-2")
        await store.get_or_create("conv-3")

        convos = await store.list_conversations()
        assert len(convos) == 3

        # Test with limit
        limited = await store.list_conversations(limit=2)
        assert len(limited) == 2

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when over limit."""
        store = ConversationStore(max_conversations=2)

        await store.get_or_create("conv-1")
        await store.get_or_create("conv-2")
        await store.get_or_create("conv-3")  # Should evict conv-1

        state1 = await store.get("conv-1")
        state3 = await store.get("conv-3")

        assert state1 is None  # Evicted
        assert state3 is not None

    @pytest.mark.asyncio
    async def test_message_trimming(self):
        """Test message trimming when over limit."""
        store = ConversationStore(max_messages_per_conversation=3)

        for i in range(5):
            await store.add_message("conv-1", Message.user(f"Message {i}"))

        messages = await store.get_messages("conv-1")
        assert len(messages) <= 3

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup of expired conversations."""
        store = ConversationStore(ttl_seconds=1)
        await store.get_or_create("conv-1")

        # Wait for expiry
        import asyncio
        await asyncio.sleep(1.1)

        removed = await store.cleanup()
        assert removed >= 1


class TestAgentMemoryBridge:
    """Tests for AgentMemoryBridge class."""

    @pytest.mark.asyncio
    async def test_link_dag_to_conversation(self):
        """Test linking DAG to conversation."""
        store = ConversationStore()
        bridge = AgentMemoryBridge(store)

        await bridge.link_dag_to_conversation("dag-1", "conv-1")

        conv_id = await bridge.get_conversation_for_dag("dag-1")
        assert conv_id == "conv-1"

    @pytest.mark.asyncio
    async def test_get_conversation_for_unknown_dag(self):
        """Test getting conversation for unknown DAG."""
        store = ConversationStore()
        bridge = AgentMemoryBridge(store)

        conv_id = await bridge.get_conversation_for_dag("unknown")
        assert conv_id is None

    @pytest.mark.asyncio
    async def test_unlink_dag(self):
        """Test unlinking DAG from conversation."""
        store = ConversationStore()
        bridge = AgentMemoryBridge(store)

        await bridge.link_dag_to_conversation("dag-1", "conv-1")
        result = await bridge.unlink_dag("dag-1")
        assert result is True

        conv_id = await bridge.get_conversation_for_dag("dag-1")
        assert conv_id is None

    @pytest.mark.asyncio
    async def test_propagate_result(self):
        """Test propagating result to conversation."""
        store = ConversationStore()
        bridge = AgentMemoryBridge(store)

        await store.get_or_create("conv-1")
        await bridge.link_dag_to_conversation("dag-1", "conv-1")

        result = await bridge.propagate_result("dag-1", "job-1", "Result data")
        assert result is True

        messages = await store.get_messages("conv-1")
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_propagate_result_unknown_dag(self):
        """Test propagating result for unknown DAG."""
        store = ConversationStore()
        bridge = AgentMemoryBridge(store)

        result = await bridge.propagate_result("unknown", "job-1", "Result")
        assert result is False
