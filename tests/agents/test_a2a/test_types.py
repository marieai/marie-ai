"""Tests for A2A SDK type definitions.

These tests verify that the SDK types work as expected and that
Marie's re-exports are functioning correctly.
"""

import uuid

import pytest

from marie.agent.a2a import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)


class TestTextPart:
    """Tests for TextPart model."""

    def test_create_text_part(self):
        part = TextPart(text="Hello, world!")
        assert part.text == "Hello, world!"
        assert part.kind == "text"

    def test_text_part_serialization(self):
        part = TextPart(text="Test")
        data = part.model_dump(by_alias=True)
        assert data["text"] == "Test"
        assert data["kind"] == "text"


class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self):
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[TextPart(text="Hello")],
        )
        assert message.role == Role.user
        assert len(message.parts) == 1

    def test_create_agent_message(self):
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[TextPart(text="Response")],
            task_id="task-123",
            context_id="ctx-456",
        )
        assert message.role == Role.agent
        assert message.task_id == "task-123"
        assert message.context_id == "ctx-456"

    def test_message_serialization(self):
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[TextPart(text="Test")],
        )
        data = message.model_dump(by_alias=True, exclude_none=True)
        assert data["role"] == "user"
        assert "parts" in data


class TestTaskStatus:
    """Tests for TaskStatus model."""

    def test_create_task_status(self):
        status = TaskStatus(state=TaskState.working)
        assert status.state == TaskState.working

    def test_task_status_with_message(self):
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[TextPart(text="Processing...")],
        )
        status = TaskStatus(
            state=TaskState.working,
            message=message,
        )
        assert status.message is not None


class TestTask:
    """Tests for Task model."""

    def test_create_task(self):
        task = Task(
            id="task-123",
            context_id="ctx-456",
            status=TaskStatus(state=TaskState.submitted),
        )
        assert task.id == "task-123"
        assert task.context_id == "ctx-456"
        assert task.status.state == TaskState.submitted
        assert task.kind == "task"

    def test_task_with_history(self):
        user_msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[TextPart(text="Hello")],
        )
        agent_msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[TextPart(text="Hi")],
        )

        task = Task(
            id="task-123",
            context_id="ctx-456",
            status=TaskStatus(state=TaskState.completed),
            history=[user_msg, agent_msg],
        )
        assert len(task.history) == 2

    def test_task_with_artifacts(self):
        artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            parts=[TextPart(text="Result")],
            name="output",
        )
        task = Task(
            id="task-123",
            context_id="ctx-456",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[artifact],
        )
        assert len(task.artifacts) == 1
        assert task.artifacts[0].name == "output"

    def test_task_serialization(self):
        task = Task(
            id="task-123",
            context_id="ctx-456",
            status=TaskStatus(state=TaskState.completed),
        )
        data = task.model_dump(by_alias=True, exclude_none=True)
        assert data["id"] == "task-123"
        assert data["contextId"] == "ctx-456"
        assert data["status"]["state"] == "completed"
        assert data["kind"] == "task"


class TestAgentCard:
    """Tests for AgentCard model."""

    def test_create_agent_card(self):
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            url="http://localhost:9000",
            version="1.0.0",
            skills=[],
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(),
        )
        assert card.name == "Test Agent"
        assert card.url == "http://localhost:9000"

    def test_agent_card_with_skills(self):
        skill = AgentSkill(
            id="echo",
            name="Echo",
            description="Echoes input",
            tags=["echo", "utility"],
        )
        card = AgentCard(
            name="Echo Agent",
            description="An echo agent",
            url="http://localhost:9000",
            version="1.0.0",
            skills=[skill],
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(),
        )
        assert len(card.skills) == 1
        assert card.skills[0].id == "echo"

    def test_agent_card_with_capabilities(self):
        caps = AgentCapabilities(
            streaming=True,
            push_notifications=False,
        )
        card = AgentCard(
            name="Streaming Agent",
            description="An agent with streaming",
            url="http://localhost:9000",
            version="1.0.0",
            skills=[],
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=caps,
        )
        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is False

    def test_agent_card_serialization(self):
        card = AgentCard(
            name="Test",
            url="http://test.com",
            version="1.0.0",
            skills=[],
            description="A test agent",
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(),
        )
        data = card.model_dump(by_alias=True, exclude_none=True)
        assert data["name"] == "Test"
        assert data["url"] == "http://test.com"


class TestArtifact:
    """Tests for Artifact model."""

    def test_create_artifact(self):
        artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            parts=[TextPart(text="Result")],
        )
        assert len(artifact.parts) == 1

    def test_artifact_with_metadata(self):
        artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            parts=[TextPart(text="Data")],
            name="output",
            description="The output data",
        )
        assert artifact.name == "output"
        assert artifact.description == "The output data"


class TestTaskState:
    """Tests for TaskState enum."""

    def test_task_states(self):
        assert TaskState.submitted.value == "submitted"
        assert TaskState.working.value == "working"
        assert TaskState.input_required.value == "input-required"
        assert TaskState.completed.value == "completed"
        assert TaskState.canceled.value == "canceled"
        assert TaskState.failed.value == "failed"

    def test_task_state_from_string(self):
        assert TaskState("completed") == TaskState.completed
        assert TaskState("working") == TaskState.working
