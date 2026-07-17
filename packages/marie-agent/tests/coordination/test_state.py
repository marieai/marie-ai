"""Unit tests for AgentWorkflowState."""

from __future__ import annotations

from marie.agent.coordination.message import (
    AgentMessage,
    AgentMessageType,
    ReservedReceiver,
    create_result_message,
    create_task_message,
)
from marie.agent.coordination.state import (
    AgentWorkflowState,
    AgentWorkflowStatus,
    create_workflow_state,
)


class TestAgentWorkflowStatus:
    """Tests for AgentWorkflowStatus enum."""

    def test_all_statuses_defined(self):
        """Ensure all expected statuses exist."""
        assert AgentWorkflowStatus.PENDING == "pending"
        assert AgentWorkflowStatus.RUNNING == "running"
        assert AgentWorkflowStatus.COMPLETED == "completed"
        assert AgentWorkflowStatus.FAILED == "failed"
        assert AgentWorkflowStatus.PAUSED == "paused"
        assert AgentWorkflowStatus.CANCELLED == "cancelled"


class TestAgentWorkflowStateCreation:
    """Tests for workflow state creation."""

    def test_state_creation_defaults(self):
        """Test creating state with defaults."""
        state = AgentWorkflowState()
        assert state.workflow_id is not None
        assert state.goal == ""
        assert state.status == AgentWorkflowStatus.PENDING
        assert state.mailbox == []
        assert state.communication_edges == []
        assert state.active_agent is None
        assert state.step == 0
        assert state.step_history == []
        assert state.errors == []
        assert state.shared_data == {}

    def test_state_creation_with_values(self):
        """Test creating state with specific values."""
        state = AgentWorkflowState(
            workflow_id="wf-123",
            goal="Process document",
        )
        assert state.workflow_id == "wf-123"
        assert state.goal == "Process document"

    def test_create_workflow_state_helper(self):
        """Test create_workflow_state factory function."""
        state = create_workflow_state(
            goal="Extract entities",
            workflow_id="wf-456",
        )
        assert state.workflow_id == "wf-456"
        assert state.goal == "Extract entities"

    def test_create_workflow_state_with_initial_agent(self):
        """Test create_workflow_state with initial agent."""
        state = create_workflow_state(
            goal="Process",
            initial_agent="planner",
        )
        assert len(state.mailbox) == 1
        assert state.mailbox[0].receiver == "planner"
        assert state.mailbox[0].sender == "coordinator"


class TestMailboxOperations:
    """Tests for mailbox message handling."""

    def test_post_message(self):
        """Test posting message to mailbox."""
        state = AgentWorkflowState()
        msg = create_task_message(
            sender="coordinator",
            receiver="planner",
            content="Start planning",
        )
        state.post_message(msg)

        assert len(state.mailbox) == 1
        assert state.mailbox[0] == msg
        assert state.step == 1

    def test_post_message_updates_edges(self):
        """Test that posting message updates communication edges."""
        state = AgentWorkflowState()
        msg = create_task_message(
            sender="planner",
            receiver="executor",
            content="Execute task",
        )
        state.post_message(msg)

        assert len(state.communication_edges) == 1
        edge = state.communication_edges[0]
        assert edge[0] == "planner"
        assert edge[1] == "executor"
        assert edge[2] == "task"

    def test_last_message_property(self):
        """Test last_message property."""
        state = AgentWorkflowState()
        assert state.last_message is None

        msg1 = create_task_message("a", "b", "first")
        msg2 = create_result_message("b", "c", "second")
        state.post_message(msg1)
        state.post_message(msg2)

        assert state.last_message == msg2

    def test_get_messages_for_agent(self):
        """Test filtering messages by agent."""
        state = AgentWorkflowState()
        state.post_message(create_task_message("a", "executor", "task 1"))
        state.post_message(create_result_message("executor", "validator", "result"))
        state.post_message(create_task_message("validator", "executor", "task 2"))

        executor_msgs = state.get_messages_for("executor")
        assert len(executor_msgs) == 2

    def test_get_messages_for_agent_with_type(self):
        """Test filtering messages by agent and type."""
        state = AgentWorkflowState()
        state.post_message(create_task_message("a", "executor", "task"))
        state.post_message(create_result_message("a", "executor", "result"))

        task_msgs = state.get_messages_for("executor", AgentMessageType.TASK)
        assert len(task_msgs) == 1
        assert task_msgs[0].msg_type == AgentMessageType.TASK


class TestRoutingLogic:
    """Tests for next_agent routing."""

    def test_next_agent_empty_mailbox(self):
        """Test next_agent with empty mailbox."""
        state = AgentWorkflowState()
        assert state.next_agent() is None

    def test_next_agent_direct_name(self):
        """Test next_agent with direct agent name."""
        state = AgentWorkflowState()
        state.post_message(create_task_message("a", "executor", "task"))
        assert state.next_agent() == "executor"

    def test_next_agent_end_signal(self):
        """Test next_agent with __end__ signal."""
        state = AgentWorkflowState()
        state.post_message(
            AgentMessage(
                sender="validator",
                receiver=ReservedReceiver.END,
                content="Done",
            )
        )
        assert state.next_agent() is None
        assert state.status == AgentWorkflowStatus.COMPLETED

    def test_next_agent_self_signal(self):
        """Test next_agent with __self__ signal."""
        state = AgentWorkflowState()
        state.post_message(
            AgentMessage(
                sender="executor",
                receiver=ReservedReceiver.SELF,
                content="Retry",
            )
        )
        assert state.next_agent() == "executor"

    def test_next_agent_prev_signal(self):
        """Test next_agent with __prev__ signal."""
        state = AgentWorkflowState()
        state.step_history = ["planner", "executor", "validator"]
        state.post_message(
            AgentMessage(
                sender="validator",
                receiver=ReservedReceiver.PREV,
                content="Go back",
            )
        )
        # previous_agent returns step_history[-2]
        assert state.next_agent() == "executor"

    def test_next_agent_prev_no_history(self):
        """Test next_agent with __prev__ but no history."""
        state = AgentWorkflowState()
        state.post_message(
            AgentMessage(
                sender="a",
                receiver=ReservedReceiver.PREV,
                content="Go back",
            )
        )
        assert state.next_agent() is None

    def test_next_agent_start_signal(self):
        """Test next_agent with __start__ signal."""
        state = AgentWorkflowState()
        state.step_history = ["planner", "executor", "validator"]
        state.post_message(
            AgentMessage(
                sender="validator",
                receiver=ReservedReceiver.START,
                content="Restart",
            )
        )
        assert state.next_agent() == "planner"

    def test_next_agent_start_no_history(self):
        """Test next_agent with __start__ but no history."""
        state = AgentWorkflowState()
        state.post_message(
            AgentMessage(
                sender="a",
                receiver=ReservedReceiver.START,
                content="Restart",
            )
        )
        assert state.next_agent() is None

    def test_next_agent_coordinator_signal(self):
        """Test next_agent with __coord__ signal."""
        state = AgentWorkflowState()
        state.post_message(
            AgentMessage(
                sender="a",
                receiver=ReservedReceiver.COORDINATOR,
                content="Need LLM decision",
            )
        )
        assert state.next_agent() is None

    def test_next_agent_broadcast_signal(self):
        """Test next_agent with * (broadcast) signal."""
        state = AgentWorkflowState()
        state.post_message(
            AgentMessage(
                sender="a",
                receiver=ReservedReceiver.BROADCAST,
                content="Fan out",
            )
        )
        assert state.next_agent() == "__broadcast__"


class TestAgentExecution:
    """Tests for agent execution tracking."""

    def test_record_agent_start(self):
        """Test recording agent start."""
        state = AgentWorkflowState()
        state.record_agent_start("executor")

        assert state.active_agent == "executor"
        assert state.step_history == ["executor"]
        assert state.status == AgentWorkflowStatus.RUNNING

    def test_record_agent_complete(self):
        """Test recording agent completion."""
        state = AgentWorkflowState()
        state.record_agent_start("executor")
        state.record_agent_complete("executor", "Extracted 5 entities")

        assert state.active_agent is None
        assert len(state.accumulated_messages) == 1
        assert state.accumulated_messages[0]["content"] == "Extracted 5 entities"

    def test_thread_agent_output(self):
        """Test threading agent output to accumulated messages."""
        state = AgentWorkflowState()
        state.thread_agent_output("planner", "Plan created")
        state.thread_agent_output("executor", "Executed plan")

        assert len(state.accumulated_messages) == 2
        assert state.accumulated_messages[0]["name"] == "planner"
        assert state.accumulated_messages[1]["name"] == "executor"

    def test_build_messages_for_agent(self):
        """Test building message list for agent execution."""
        state = AgentWorkflowState()
        state.thread_agent_output("planner", "Prior output")

        base_messages = [{"role": "user", "content": "Process document"}]
        messages = state.build_messages_for_agent("executor", base_messages)

        assert len(messages) == 2
        assert messages[0]["content"] == "Process document"
        assert messages[1]["content"] == "Prior output"


class TestStatusTransitions:
    """Tests for workflow status transitions."""

    def test_fail_sets_status(self):
        """Test fail() sets status to FAILED."""
        state = AgentWorkflowState()
        state.fail("Something went wrong")

        assert state.status == AgentWorkflowStatus.FAILED
        assert "Something went wrong" in state.errors

    def test_complete_sets_status(self):
        """Test complete() sets status to COMPLETED."""
        state = AgentWorkflowState()
        state.record_agent_start("executor")
        state.complete()

        assert state.status == AgentWorkflowStatus.COMPLETED
        assert state.active_agent is None

    def test_is_terminal_completed(self):
        """Test is_terminal for completed workflow."""
        state = AgentWorkflowState(status=AgentWorkflowStatus.COMPLETED)
        assert state.is_terminal() is True

    def test_is_terminal_failed(self):
        """Test is_terminal for failed workflow."""
        state = AgentWorkflowState(status=AgentWorkflowStatus.FAILED)
        assert state.is_terminal() is True

    def test_is_terminal_cancelled(self):
        """Test is_terminal for cancelled workflow."""
        state = AgentWorkflowState(status=AgentWorkflowStatus.CANCELLED)
        assert state.is_terminal() is True

    def test_is_terminal_running(self):
        """Test is_terminal for running workflow."""
        state = AgentWorkflowState(status=AgentWorkflowStatus.RUNNING)
        assert state.is_terminal() is False


class TestSerialization:
    """Tests for state serialization (checkpointing)."""

    def test_to_dict(self):
        """Test serializing state to dict."""
        state = AgentWorkflowState(
            workflow_id="wf-123",
            goal="Test serialization",
        )
        state.post_message(create_task_message("a", "b", "test"))

        data = state.to_dict()
        assert data["workflow_id"] == "wf-123"
        assert data["goal"] == "Test serialization"
        assert len(data["mailbox"]) == 1

    def test_from_dict(self):
        """Test deserializing state from dict."""
        original = AgentWorkflowState(
            workflow_id="wf-456",
            goal="Test deserialization",
        )
        original.post_message(create_task_message("a", "b", "test"))
        original.record_agent_start("b")

        data = original.to_dict()
        restored = AgentWorkflowState.from_dict(data)

        assert restored.workflow_id == "wf-456"
        assert restored.goal == "Test deserialization"
        assert len(restored.mailbox) == 1
        assert restored.step_history == ["b"]

    def test_round_trip_serialization(self):
        """Test full round-trip serialization preserves data."""
        state = AgentWorkflowState(
            workflow_id="wf-789",
            goal="Round trip test",
            status=AgentWorkflowStatus.RUNNING,
        )
        state.post_message(create_task_message("coord", "planner", "plan"))
        state.post_message(create_result_message("planner", "executor", "done"))
        state.record_agent_start("planner")
        state.record_agent_complete("planner", "Plan output")
        state.shared_data["key"] = "value"
        state.record_error("Warning: minor issue")

        data = state.to_dict()
        restored = AgentWorkflowState.from_dict(data)

        assert restored.workflow_id == state.workflow_id
        assert restored.goal == state.goal
        assert len(restored.mailbox) == len(state.mailbox)
        assert len(restored.communication_edges) == len(state.communication_edges)
        assert restored.step_history == state.step_history
        assert restored.accumulated_messages == state.accumulated_messages
        assert restored.shared_data == state.shared_data
        assert restored.errors == state.errors
