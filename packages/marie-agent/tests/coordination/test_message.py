"""Unit tests for AgentMessage and routing signals."""

from __future__ import annotations

from marie.agent.coordination.message import (
    AgentMessage,
    AgentMessageType,
    ReservedReceiver,
    create_error_message,
    create_result_message,
    create_task_message,
)


class TestReservedReceiver:
    """Tests for ReservedReceiver enum."""

    def test_all_reserved_receivers_defined(self):
        """Ensure all expected reserved receivers exist."""
        assert ReservedReceiver.END == "__end__"
        assert ReservedReceiver.SELF == "__self__"
        assert ReservedReceiver.PREV == "__prev__"
        assert ReservedReceiver.START == "__start__"
        assert ReservedReceiver.NEXT == "__next__"
        assert ReservedReceiver.COORDINATOR == "__coord__"
        assert ReservedReceiver.BROADCAST == "*"

    def test_reserved_receivers_are_strings(self):
        """Reserved receivers should be string-compatible."""
        # Enum .value is the underlying string
        assert ReservedReceiver.END.value == "__end__"
        assert ReservedReceiver.SELF.value == "__self__"
        # str enum comparison works directly
        assert ReservedReceiver.END == "__end__"


class TestAgentMessage:
    """Tests for AgentMessage model."""

    def test_message_creation_minimal(self):
        """Test creating message with minimal fields."""
        msg = AgentMessage(sender="agent_a", receiver="agent_b")
        assert msg.sender == "agent_a"
        assert msg.receiver == "agent_b"
        assert msg.msg_type == AgentMessageType.RESULT
        assert msg.content == ""
        assert msg.msg_id is not None
        assert msg.timestamp is not None

    def test_message_creation_full(self):
        """Test creating message with all fields."""
        msg = AgentMessage(
            sender="planner",
            receiver="executor",
            msg_type=AgentMessageType.TASK,
            content="Execute extraction",
            metadata={"priority": "high"},
            trace={"run_id": "123"},
        )
        assert msg.sender == "planner"
        assert msg.msg_type == AgentMessageType.TASK
        assert msg.content == "Execute extraction"
        assert msg.metadata["priority"] == "high"
        assert msg.trace["run_id"] == "123"

    def test_is_terminal_with_enum(self):
        """Test is_terminal with ReservedReceiver enum."""
        msg = AgentMessage(
            sender="validator",
            receiver=ReservedReceiver.END,
        )
        assert msg.is_terminal() is True

    def test_is_terminal_with_string(self):
        """Test is_terminal with string value."""
        msg = AgentMessage(
            sender="validator",
            receiver="__end__",
        )
        assert msg.is_terminal() is True

    def test_is_not_terminal(self):
        """Test is_terminal returns False for non-terminal."""
        msg = AgentMessage(sender="a", receiver="agent_b")
        assert msg.is_terminal() is False

    def test_is_retry_with_enum(self):
        """Test is_retry with ReservedReceiver enum."""
        msg = AgentMessage(
            sender="validator",
            receiver=ReservedReceiver.SELF,
        )
        assert msg.is_retry() is True

    def test_is_retry_with_string(self):
        """Test is_retry with string value."""
        msg = AgentMessage(sender="validator", receiver="__self__")
        assert msg.is_retry() is True

    def test_is_not_retry(self):
        """Test is_retry returns False for non-retry."""
        msg = AgentMessage(sender="a", receiver="agent_b")
        assert msg.is_retry() is False

    def test_is_broadcast(self):
        """Test is_broadcast detection."""
        msg_enum = AgentMessage(sender="a", receiver=ReservedReceiver.BROADCAST)
        msg_str = AgentMessage(sender="a", receiver="*")

        assert msg_enum.is_broadcast() is True
        assert msg_str.is_broadcast() is True

    def test_is_coordinator_decision(self):
        """Test is_coordinator_decision detection."""
        msg_enum = AgentMessage(sender="a", receiver=ReservedReceiver.COORDINATOR)
        msg_str = AgentMessage(sender="a", receiver="__coord__")

        assert msg_enum.is_coordinator_decision() is True
        assert msg_str.is_coordinator_decision() is True

    def test_is_prev(self):
        """Test is_prev detection."""
        msg_enum = AgentMessage(sender="a", receiver=ReservedReceiver.PREV)
        msg_str = AgentMessage(sender="a", receiver="__prev__")

        assert msg_enum.is_prev() is True
        assert msg_str.is_prev() is True

    def test_is_start(self):
        """Test is_start detection."""
        msg_enum = AgentMessage(sender="a", receiver=ReservedReceiver.START)
        msg_str = AgentMessage(sender="a", receiver="__start__")

        assert msg_enum.is_start() is True
        assert msg_str.is_start() is True

    def test_is_next(self):
        """Test is_next detection."""
        msg_enum = AgentMessage(sender="a", receiver=ReservedReceiver.NEXT)
        msg_str = AgentMessage(sender="a", receiver="__next__")

        assert msg_enum.is_next() is True
        assert msg_str.is_next() is True

    def test_with_trace(self):
        """Test adding trace fields to message."""
        msg = AgentMessage(
            sender="a",
            receiver="b",
            trace={"run_id": "123"},
        )
        new_msg = msg.with_trace(span_id="456", latency_ms=50)

        # Original unchanged
        assert "span_id" not in msg.trace
        # New message has both
        assert new_msg.trace["run_id"] == "123"
        assert new_msg.trace["span_id"] == "456"
        assert new_msg.trace["latency_ms"] == 50


class TestMessageFactories:
    """Tests for message factory functions."""

    def test_create_task_message(self):
        """Test task message factory."""
        msg = create_task_message(
            sender="coordinator",
            receiver="planner",
            content="Plan document extraction",
            document_id="doc-123",
        )
        assert msg.sender == "coordinator"
        assert msg.receiver == "planner"
        assert msg.msg_type == AgentMessageType.TASK
        assert msg.content == "Plan document extraction"
        assert msg.metadata["document_id"] == "doc-123"

    def test_create_result_message(self):
        """Test result message factory."""
        msg = create_result_message(
            sender="executor",
            receiver="validator",
            content="Extracted 5 entities",
            entity_count=5,
        )
        assert msg.sender == "executor"
        assert msg.receiver == "validator"
        assert msg.msg_type == AgentMessageType.RESULT
        assert msg.content == "Extracted 5 entities"
        assert msg.metadata["entity_count"] == 5

    def test_create_error_message(self):
        """Test error message factory."""
        msg = create_error_message(
            sender="executor",
            error="Connection timeout",
            attempt=3,
        )
        assert msg.sender == "executor"
        assert msg.receiver == ReservedReceiver.COORDINATOR
        assert msg.msg_type == AgentMessageType.ERROR
        assert msg.content == "Connection timeout"
        assert msg.metadata["attempt"] == 3

    def test_create_error_message_custom_receiver(self):
        """Test error message with custom receiver."""
        msg = create_error_message(
            sender="executor",
            error="Validation failed",
            receiver="planner",
        )
        assert msg.receiver == "planner"


class TestAgentMessageType:
    """Tests for AgentMessageType enum."""

    def test_all_message_types_defined(self):
        """Ensure all expected message types exist."""
        assert AgentMessageType.TASK == "task"
        assert AgentMessageType.RESULT == "result"
        assert AgentMessageType.VALIDATION == "validation"
        assert AgentMessageType.ERROR == "error"
        assert AgentMessageType.CONTROL == "control"
