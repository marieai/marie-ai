"""Unit tests for CheckpointStore."""

from __future__ import annotations

import pytest

from marie.agent.coordination.checkpoint import InMemoryCheckpointStore
from marie.agent.coordination.message import create_task_message
from marie.agent.coordination.state import (
    AgentWorkflowState,
    AgentWorkflowStatus,
    create_workflow_state,
)


class TestInMemoryCheckpointStore:
    """Tests for in-memory checkpoint implementation."""

    @pytest.fixture
    def store(self):
        """In-memory store for testing."""
        return InMemoryCheckpointStore()

    @pytest.fixture
    def sample_state(self):
        """Sample workflow state for testing."""
        state = create_workflow_state(
            goal="Test checkpoint",
            workflow_id="wf-checkpoint-test",
        )
        state.post_message(create_task_message("coord", "planner", "Start"))
        state.record_agent_start("planner")
        state.status = AgentWorkflowStatus.RUNNING
        return state

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, store, sample_state):
        """Test saving a workflow checkpoint."""
        await store.save(sample_state.workflow_id, sample_state)
        # No error means success

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, store, sample_state):
        """Test loading a saved checkpoint."""
        await store.save(sample_state.workflow_id, sample_state)
        loaded = await store.load(sample_state.workflow_id)

        assert loaded is not None
        assert loaded.workflow_id == sample_state.workflow_id
        assert loaded.goal == sample_state.goal
        assert loaded.status == sample_state.status

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        """Test loading a nonexistent checkpoint."""
        loaded = await store.load("nonexistent-workflow")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, store, sample_state):
        """Test deleting a checkpoint."""
        await store.save(sample_state.workflow_id, sample_state)
        await store.delete(sample_state.workflow_id)

        loaded = await store.load(sample_state.workflow_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a nonexistent checkpoint (no error)."""
        await store.delete("nonexistent-workflow")
        # Should not raise

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self, store):
        """Test listing checkpoints when empty."""
        checkpoints = await store.list_checkpoints()
        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, store):
        """Test listing all checkpoints."""
        for i in range(3):
            state = create_workflow_state(
                goal=f"Test {i}",
                workflow_id=f"wf-{i}",
            )
            await store.save(state.workflow_id, state)

        checkpoints = await store.list_checkpoints()
        assert len(checkpoints) == 3
        assert "wf-0" in checkpoints
        assert "wf-1" in checkpoints
        assert "wf-2" in checkpoints

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_prefix(self, store):
        """Test listing checkpoints with prefix filter."""
        await store.save("project-a-wf-1", create_workflow_state("A1", "project-a-wf-1"))
        await store.save("project-a-wf-2", create_workflow_state("A2", "project-a-wf-2"))
        await store.save("project-b-wf-1", create_workflow_state("B1", "project-b-wf-1"))

        project_a = await store.list_checkpoints("project-a")
        assert len(project_a) == 2
        assert "project-a-wf-1" in project_a
        assert "project-a-wf-2" in project_a
        assert "project-b-wf-1" not in project_a

    @pytest.mark.asyncio
    async def test_upsert_existing_checkpoint(self, store, sample_state):
        """Test updating an existing checkpoint."""
        await store.save(sample_state.workflow_id, sample_state)

        # Modify state
        sample_state.status = AgentWorkflowStatus.COMPLETED
        sample_state.post_message(create_task_message("executor", "__end__", "Done"))

        # Save again
        await store.save(sample_state.workflow_id, sample_state)

        # Load and verify update
        loaded = await store.load(sample_state.workflow_id)
        assert loaded.status == AgentWorkflowStatus.COMPLETED
        assert len(loaded.mailbox) == 2

    @pytest.mark.asyncio
    async def test_preserves_full_state(self, store):
        """Test that checkpoint preserves all state fields."""
        state = create_workflow_state(
            goal="Full state test",
            workflow_id="wf-full-state",
        )
        state.post_message(create_task_message("coord", "planner", "task 1"))
        state.post_message(create_task_message("planner", "executor", "task 2"))
        state.record_agent_start("planner")
        state.record_agent_complete("planner", "Plan output")
        state.shared_data["key"] = "value"
        state.shared_data["nested"] = {"a": 1, "b": [1, 2, 3]}
        state.record_error("Minor warning")

        await store.save(state.workflow_id, state)
        loaded = await store.load(state.workflow_id)

        assert loaded.workflow_id == state.workflow_id
        assert loaded.goal == state.goal
        assert len(loaded.mailbox) == len(state.mailbox)
        assert len(loaded.communication_edges) == len(state.communication_edges)
        assert loaded.step_history == state.step_history
        assert loaded.accumulated_messages == state.accumulated_messages
        assert loaded.shared_data["key"] == "value"
        assert loaded.shared_data["nested"]["b"] == [1, 2, 3]
        assert loaded.errors == state.errors

    @pytest.mark.asyncio
    async def test_clear_store(self, store, sample_state):
        """Test clearing all checkpoints."""
        await store.save(sample_state.workflow_id, sample_state)
        await store.save("wf-2", create_workflow_state("Test 2", "wf-2"))

        store.clear()

        checkpoints = await store.list_checkpoints()
        assert len(checkpoints) == 0
