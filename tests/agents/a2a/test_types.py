"""Tests for A2A type definitions."""

import pytest

from marie.agent.a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Part,
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
            role=Role.USER,
            parts=[TextPart(text="Hello")],
        )
        assert message.role == Role.USER
        assert len(message.parts) == 1
        assert message.message_id is not None

    def test_create_agent_message(self):
        message = Message(
            role=Role.AGENT,
            parts=[TextPart(text="Response")],
            task_id="task-123",
            context_id="ctx-456",
        )
        assert message.role == Role.AGENT
        assert message.task_id == "task-123"
        assert message.context_id == "ctx-456"

    def test_message_serialization(self):
        message = Message(
            role=Role.USER,
            parts=[TextPart(text="Test")],
        )
        data = message.model_dump(by_alias=True)
        assert data["role"] == "user"
        assert "messageId" in data
        assert "parts" in data


class TestTaskStatus:
    """Tests for TaskStatus model."""

    def test_create_task_status(self):
        status = TaskStatus(state=TaskState.WORKING)
        assert status.state == TaskState.WORKING

    def test_task_status_with_message(self):
        message = Message(
            role=Role.AGENT,
            parts=[TextPart(text="Processing...")],
        )
        status = TaskStatus(
            state=TaskState.WORKING,
            message=message,
        )
        assert status.message is not None


class TestTask:
    """Tests for Task model."""

    def test_create_task(self):
        task = Task(
            status=TaskStatus(state=TaskState.SUBMITTED),
        )
        assert task.id is not None
        assert task.context_id is not None
        assert task.status.state == TaskState.SUBMITTED
        assert task.kind == "task"

    def test_task_with_history(self):
        user_msg = Message(role=Role.USER, parts=[TextPart(text="Hello")])
        agent_msg = Message(role=Role.AGENT, parts=[TextPart(text="Hi")])

        task = Task(
            status=TaskStatus(state=TaskState.COMPLETED),
            history=[user_msg, agent_msg],
        )
        assert len(task.history) == 2

    def test_task_with_artifacts(self):
        artifact = Artifact(
            parts=[TextPart(text="Result")],
            name="output",
        )
        task = Task(
            status=TaskStatus(state=TaskState.COMPLETED),
            artifacts=[artifact],
        )
        assert len(task.artifacts) == 1
        assert task.artifacts[0].name == "output"

    def test_task_serialization(self):
        task = Task(
            id="task-123",
            context_id="ctx-456",
            status=TaskStatus(state=TaskState.COMPLETED),
        )
        data = task.model_dump(by_alias=True)
        assert data["id"] == "task-123"
        assert data["contextId"] == "ctx-456"
        assert data["status"]["state"] == "completed"
        assert data["kind"] == "task"


class TestAgentCard:
    """Tests for AgentCard model."""

    def test_create_agent_card(self):
        card = AgentCard(
            name="Test Agent",
            url="http://localhost:9000",
        )
        assert card.name == "Test Agent"
        assert card.url == "http://localhost:9000"
        assert card.version == "1.0.0"

    def test_agent_card_with_skills(self):
        skill = AgentSkill(
            id="echo",
            name="Echo",
            description="Echoes input",
        )
        card = AgentCard(
            name="Echo Agent",
            url="http://localhost:9000",
            skills=[skill],
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
            url="http://localhost:9000",
            capabilities=caps,
        )
        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is False

    def test_agent_card_serialization(self):
        card = AgentCard(
            name="Test",
            url="http://test.com",
            description="A test agent",
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
        )
        data = card.model_dump(by_alias=True)
        assert data["name"] == "Test"
        assert data["url"] == "http://test.com"
        assert "defaultInputModes" in data


class TestArtifact:
    """Tests for Artifact model."""

    def test_create_artifact(self):
        artifact = Artifact(
            parts=[TextPart(text="Result")],
        )
        assert artifact.artifact_id is not None
        assert len(artifact.parts) == 1

    def test_artifact_with_metadata(self):
        artifact = Artifact(
            parts=[TextPart(text="Data")],
            name="output",
            description="The output data",
        )
        assert artifact.name == "output"
        assert artifact.description == "The output data"


class TestTaskState:
    """Tests for TaskState enum."""

    def test_task_states(self):
        assert TaskState.SUBMITTED.value == "submitted"
        assert TaskState.WORKING.value == "working"
        assert TaskState.INPUT_REQUIRED.value == "input-required"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.CANCELED.value == "canceled"
        assert TaskState.FAILED.value == "failed"

    def test_task_state_from_string(self):
        assert TaskState("completed") == TaskState.COMPLETED
        assert TaskState("working") == TaskState.WORKING
