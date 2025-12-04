"""Tests for Marie agent backends."""

import pytest

from marie.agent.backends.base import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    CompositeBackend,
    ToolCallRecord,
)
from marie.agent.message import Message


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert AgentStatus.PENDING.value == "pending"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.CANCELLED.value == "cancelled"


class TestToolCallRecord:
    """Tests for ToolCallRecord class."""

    def test_create_record(self):
        """Test creating tool call record."""
        record = ToolCallRecord(
            tool_name="search",
            tool_args={"query": "test"},
            result="Found 10 results",
            duration_ms=150.5,
        )
        assert record.tool_name == "search"
        assert record.tool_args == {"query": "test"}
        assert record.result == "Found 10 results"
        assert record.error is None

    def test_create_error_record(self):
        """Test creating error tool call record."""
        record = ToolCallRecord(
            tool_name="failing_tool",
            tool_args={},
            error="Tool execution failed",
        )
        assert record.error == "Tool execution failed"
        assert record.result is None


class TestAgentResult:
    """Tests for AgentResult class."""

    def test_create_result(self):
        """Test creating agent result."""
        result = AgentResult(
            output="Hello, world!",
            status=AgentStatus.COMPLETED,
            iterations=3,
        )
        assert result.output == "Hello, world!"
        assert result.status == AgentStatus.COMPLETED
        assert result.iterations == 3
        assert result.is_complete is True

    def test_output_text_string(self):
        """Test output_text with string output."""
        result = AgentResult(output="Text output")
        assert result.output_text == "Text output"

    def test_output_text_message(self):
        """Test output_text with Message output."""
        msg = Message.assistant("Message output")
        result = AgentResult(output=msg)
        assert result.output_text == "Message output"

    def test_output_text_list(self):
        """Test output_text with list output."""
        msgs = [
            Message.assistant("First"),
            Message.assistant("Second"),
        ]
        result = AgentResult(output=msgs)
        assert result.output_text == "Second"

    def test_result_with_tool_calls(self):
        """Test result with tool call history."""
        tool_calls = [
            ToolCallRecord(tool_name="tool1", tool_args={}),
            ToolCallRecord(tool_name="tool2", tool_args={"x": 1}),
        ]
        result = AgentResult(
            output="Done",
            tool_calls=tool_calls,
        )
        assert len(result.tool_calls) == 2

    def test_failed_result(self):
        """Test failed result."""
        result = AgentResult(
            output=None,
            status=AgentStatus.FAILED,
            error="Something went wrong",
            is_complete=False,
        )
        assert result.status == AgentStatus.FAILED
        assert result.error == "Something went wrong"
        assert result.is_complete is False


class TestBackendConfig:
    """Tests for BackendConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = BackendConfig()
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300.0
        assert config.stream is True
        assert config.return_intermediate is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = BackendConfig(
            max_iterations=5,
            timeout_seconds=60.0,
            stream=False,
            extra={"custom_key": "custom_value"},
        )
        assert config.max_iterations == 5
        assert config.extra["custom_key"] == "custom_value"


class TestAgentBackend:
    """Tests for AgentBackend class."""

    def test_backend_initialization(self, mock_backend):
        """Test backend initialization."""
        assert mock_backend.config is not None

    @pytest.mark.asyncio
    async def test_backend_run(self, mock_backend, sample_user_message):
        """Test backend run method."""
        result = await mock_backend.run([sample_user_message])
        assert isinstance(result, AgentResult)
        assert result.status == AgentStatus.COMPLETED

    def test_backend_sync_run(self, mock_backend, sample_user_message):
        """Test synchronous run wrapper."""
        result = mock_backend.run_sync([sample_user_message])
        assert isinstance(result, AgentResult)

    def test_get_config(self, mock_backend):
        """Test getting config."""
        config = mock_backend.get_config()
        assert isinstance(config, BackendConfig)

    def test_update_config(self, mock_backend):
        """Test updating config."""
        mock_backend.update_config(max_iterations=20)
        assert mock_backend.config.max_iterations == 20


class TestCompositeBackend:
    """Tests for CompositeBackend class."""

    @pytest.mark.asyncio
    async def test_composite_backend_run(self, mock_backend, sample_user_message):
        """Test composite backend delegates to primary."""
        composite = CompositeBackend(primary=mock_backend)
        result = await composite.run([sample_user_message])
        assert isinstance(result, AgentResult)

    def test_add_delegate(self, mock_backend):
        """Test adding delegate backend."""
        composite = CompositeBackend(primary=mock_backend)
        delegate = mock_backend  # Use same mock for simplicity
        composite.add_delegate("rag", delegate)
        assert "rag" in composite.delegates

    def test_remove_delegate(self, mock_backend):
        """Test removing delegate backend."""
        composite = CompositeBackend(
            primary=mock_backend,
            delegates={"rag": mock_backend},
        )
        removed = composite.remove_delegate("rag")
        assert removed is not None
        assert "rag" not in composite.delegates

    def test_get_available_tools(self, mock_backend):
        """Test getting tools from all backends."""
        composite = CompositeBackend(primary=mock_backend)
        tools = composite.get_available_tools()
        assert isinstance(tools, list)
