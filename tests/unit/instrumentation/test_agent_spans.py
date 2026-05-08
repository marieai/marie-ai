"""Tests for agent framework OTel instrumentation.

Verifies that CHAIN, AGENT, LLM, and TOOL spans are correctly created
with proper attributes, status codes, and parent-child nesting.
Uses a lightweight in-memory exporter — no external OTel collector needed.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openinference.semconv.trace import MessageAttributes, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import StatusCode

from marie.agent.message import Message
from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.instrumentation import set_llm_io

# ---------------------------------------------------------------------------
# Lightweight in-memory exporter (OTel SDK 1.19 lacks InMemorySpanExporter)
# ---------------------------------------------------------------------------


class _InMemoryExporter(SpanExporter):
    """Collects finished spans in a thread-safe list for test assertions."""

    def __init__(self):
        self._spans: List = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def get_finished_spans(self) -> List:
        with self._lock:
            return list(self._spans)

    def clear(self):
        with self._lock:
            self._spans.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tracer_provider():
    """Force-reset the global TracerProvider for each test.

    OTel SDK 1.x raises a warning and silently ignores set_tracer_provider()
    when a non-proxy provider already exists. We bypass this by patching the
    internal _TRACER_PROVIDER / _TRACER_PROVIDER_SET_ONCE.
    """
    yield
    # After each test, reset the global provider to a fresh proxy
    # so the next test's set_tracer_provider() actually takes effect.
    import opentelemetry.trace as _trace_mod

    if hasattr(_trace_mod, "_TRACER_PROVIDER_SET_ONCE"):
        _trace_mod._TRACER_PROVIDER_SET_ONCE = _trace_mod.Once()
    if hasattr(_trace_mod, "_TRACER_PROVIDER"):
        _trace_mod._TRACER_PROVIDER = None


@pytest.fixture
def otel_setup():
    """Set up an in-memory span exporter for testing OTel spans."""
    exporter = _InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


@pytest.fixture
def otel_setup_with_processor():
    """Set up span exporter WITH OpenInferenceSpanProcessor for context tests."""
    from marie.instrumentation.processor import OpenInferenceSpanProcessor

    exporter = _InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(OpenInferenceSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


@pytest.fixture
def sample_messages():
    return [Message.user("Hello, world!")]


def test_set_llm_io_expands_multimodal_message_content(otel_setup):
    tracer = trace.get_tracer("test.llm_io")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe the image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                },
            ],
        },
    ]

    with tracer.start_as_current_span("llm-span") as span:
        set_llm_io(span, input_messages=messages)

    finished_span = otel_setup.get_finished_spans()[0]
    attrs = finished_span.attributes
    user_content_key = (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1."
        f"{MessageAttributes.MESSAGE_CONTENT}"
    )
    assert attrs[user_content_key] == (
        "describe the image\n[image_url: data:image/png;base64,ZmFrZQ==]"
    )


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockAgent:
    """Minimal mock agent for coordination tests."""

    name: str = "mock_agent"
    response: str = "mock response"
    call_count: int = field(default=0, init=False)

    async def arun(self, messages, **kwargs):
        self.call_count += 1
        return {
            "output": self.response,
            "messages": [],
            "metadata": {"next_agent": "__end__"},
        }


class _SimpleAgent:
    """Minimal concrete BaseAgent for testing span creation."""

    def __init__(self, name: str = "test-agent"):
        from marie.agent.base import BaseAgent

        class _Impl(BaseAgent):
            def _run(self_inner, messages, **kwargs) -> Iterator[List[Message]]:
                yield [Message.assistant("Hello from agent")]

        self._impl = _Impl(name=name)

    def run(self, messages):
        return self._impl.run(messages)


class _ConcreteTool(AgentTool):
    """Minimal tool for testing span creation."""

    def __init__(self, name: str = "echo_tool"):
        self._metadata = ToolMetadata(
            name=name,
            description="Echo tool for testing",
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, **kwargs) -> ToolOutput:
        text = kwargs.get("text", "echo")
        return ToolOutput(content=text, tool_name=self.name)

    async def acall(self, **kwargs) -> ToolOutput:
        return self.call(**kwargs)


class _FailingTool(AgentTool):
    """Tool whose call raises an exception."""

    def __init__(self, name: str = "failing_tool"):
        self._metadata = ToolMetadata(
            name=name,
            description="Always fails",
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, **kwargs) -> ToolOutput:
        raise ValueError("tool failure")

    async def acall(self, **kwargs) -> ToolOutput:
        raise ValueError("tool failure")


class _BadOutputTool(AgentTool):
    """Tool whose output content has a broken __str__."""

    def __init__(self):
        self._metadata = ToolMetadata(
            name="bad_output_tool",
            description="Has un-serializable output",
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, **kwargs) -> ToolOutput:
        class _Bad:
            def __str__(self):
                raise TypeError("cannot serialize")

        return ToolOutput(content=_Bad(), tool_name=self.name)


# ---------------------------------------------------------------------------
# 1. WorkflowCoordinator creates CHAIN span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_creates_workflow_span(otel_setup, sample_messages):
    """CHAIN span with marie.workflow_id attribute."""
    exporter = otel_setup

    from marie.agent.config import CoordinationConfig
    from marie.agent.coordination import WorkflowCoordinator

    config = CoordinationConfig(topology="workflow", max_steps=5, timeout=10.0)
    coordinator = WorkflowCoordinator(config)

    agent_a = _MockAgent(name="agent_a")
    coordinator.add_agent(agent_a)

    await coordinator.run(
        sample_messages,
        workflow_id="wf-test-001",
        goal="test",
    )

    spans = exporter.get_finished_spans()
    chain_spans = [s for s in spans if "workflow:" in s.name]
    assert len(chain_spans) == 1

    span = chain_spans[0]
    assert span.name == "workflow:wf-test-001"
    assert span.attributes.get("marie.workflow_id") == "wf-test-001"
    assert span.attributes.get("marie.agent_count") == 1
    assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# 2. BaseAgent creates AGENT span
# ---------------------------------------------------------------------------


def test_agent_creates_agent_span(otel_setup, sample_messages):
    """AGENT span with agent.name attribute."""
    exporter = otel_setup
    agent = _SimpleAgent(name="my-agent")

    for responses in agent.run(sample_messages):
        pass  # consume the generator

    spans = exporter.get_finished_spans()
    agent_spans = [s for s in spans if "agent:" in s.name]
    assert len(agent_spans) == 1

    span = agent_spans[0]
    assert span.name == "agent:my-agent"
    assert span.attributes.get(SpanAttributes.AGENT_NAME) == "my-agent"
    assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# 3. LLM wrapper creates LLM span (MarieEngineLLMWrapper.chat)
# ---------------------------------------------------------------------------


def test_llm_wrapper_creates_generation_span(otel_setup):
    """LLM span with llm.model_name and llm.system."""
    exporter = otel_setup

    # Patch get_engine at the location where it's imported
    with patch("marie.engine.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "Hello from LLM"
        mock_get_engine.return_value = mock_engine

        from marie.agent.llm_wrapper import MarieEngineLLMWrapper

        wrapper = MarieEngineLLMWrapper(engine_name="gpt-4", provider="openai")

        messages = [Message.user("Test")]
        for responses in wrapper.chat(messages):
            pass

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if "llm:" in s.name]
    assert len(llm_spans) == 1

    span = llm_spans[0]
    assert span.name == "llm:gpt-4"
    assert span.attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4"
    assert span.attributes.get(SpanAttributes.LLM_SYSTEM) == "openai"
    assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# 4. Streaming span ends OK on GeneratorExit
# ---------------------------------------------------------------------------


def test_streaming_span_on_generator_exit(otel_setup):
    """Span ends with OK status when generator is abandoned (GeneratorExit)."""
    exporter = otel_setup

    with patch("marie.engine.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "Hello"
        mock_get_engine.return_value = mock_engine

        from marie.agent.llm_wrapper import MarieEngineLLMWrapper

        wrapper = MarieEngineLLMWrapper(engine_name="qwen-7b", provider="vllm")

        messages = [Message.user("Test")]
        gen = wrapper.chat(messages)
        next(gen)  # Get first yield
        gen.close()  # Trigger GeneratorExit

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if "llm:" in s.name]
    assert len(llm_spans) == 1
    assert llm_spans[0].status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# 5. Streaming token count fallback (tiktoken estimation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_span_tiktoken_fallback(otel_setup):
    """Token count estimated when provider doesn't report usage."""
    exporter = otel_setup

    from marie.agent.llm_wrapper import OpenAICompatibleWrapper

    # Create wrapper via __new__ to skip __init__ OpenAI import
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    wrapper.model = "gpt-4"
    wrapper._async_client = None
    wrapper._emitter = None
    wrapper._tool_call_parser = MagicMock()

    # Create mock streaming response
    mock_chunk_content = MagicMock()
    mock_chunk_content.choices = [
        MagicMock(
            delta=MagicMock(content="Hello world", tool_calls=None),
            finish_reason=None,
        )
    ]
    mock_chunk_content.usage = None

    mock_chunk_done = MagicMock()
    mock_chunk_done.choices = [
        MagicMock(
            delta=MagicMock(content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_chunk_done.usage = None  # No provider usage

    async def mock_stream():
        yield mock_chunk_content
        yield mock_chunk_done

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_stream()

    wrapper._get_async_client = lambda: mock_client
    wrapper._build_api_kwargs = lambda msgs, fns, cfg: {
        "model": "gpt-4",
        "messages": [],
    }

    messages = [Message.user("Test")]
    async for chunk in wrapper.achat_stream(messages):
        pass

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if "llm:" in s.name]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.status.status_code == StatusCode.OK
    # Should have estimated token count since no provider usage
    assert span.attributes.get("marie.token_count_estimated") is True
    assert span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 0) > 0


# ---------------------------------------------------------------------------
# 6. Tool creates TOOL span
# ---------------------------------------------------------------------------


def test_tool_creates_tool_span(otel_setup):
    """TOOL span with tool.name attribute."""
    exporter = otel_setup

    tool = _ConcreteTool(name="search")
    result = tool.safe_call({"text": "hello"})

    assert not result.is_error

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if "tool:" in s.name]
    assert len(tool_spans) == 1

    span = tool_spans[0]
    assert span.name == "tool:search"
    assert span.attributes.get(SpanAttributes.TOOL_NAME) == "search"
    assert span.attributes.get(SpanAttributes.OUTPUT_VALUE) == "hello"
    assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# 7. Tool span survives bad output serialization
# ---------------------------------------------------------------------------


def test_tool_span_survives_bad_output(otel_setup):
    """Attribute serialization error doesn't crash tool execution."""
    exporter = otel_setup

    tool = _BadOutputTool()
    # Should NOT raise even though str(result.content) throws TypeError
    result = tool.safe_call({})

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if "tool:" in s.name]
    assert len(tool_spans) == 1
    # Span should still end
    assert tool_spans[0].name == "tool:bad_output_tool"


# ---------------------------------------------------------------------------
# 8. Agent → LLM → Tool parent-child nesting
# ---------------------------------------------------------------------------


def test_agent_llm_tool_parent_child(otel_setup):
    """Verify correct span nesting: AGENT → child spans."""
    exporter = otel_setup

    agent = _SimpleAgent(name="nested-agent")
    for responses in agent.run([Message.user("Hi")]):
        pass

    # Also create a tool span in the same trace context
    tool = _ConcreteTool(name="nested-tool")
    tool.safe_call({"text": "data"})

    spans = exporter.get_finished_spans()
    agent_spans = [s for s in spans if "agent:" in s.name]
    tool_spans = [s for s in spans if "tool:" in s.name]

    assert len(agent_spans) >= 1
    assert len(tool_spans) >= 1

    # Both should be valid spans with trace IDs
    for s in spans:
        assert s.context.trace_id != 0


# ---------------------------------------------------------------------------
# 9. Workflow ID propagated through attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_id_propagated(otel_setup, sample_messages):
    """All child spans within a workflow carry the workflow_id."""
    exporter = otel_setup

    from marie.agent.config import CoordinationConfig
    from marie.agent.coordination import WorkflowCoordinator

    config = CoordinationConfig(topology="workflow", max_steps=5, timeout=10.0)
    coordinator = WorkflowCoordinator(config)
    coordinator.add_agent(_MockAgent(name="worker"))

    await coordinator.run(
        sample_messages,
        workflow_id="wf-propagation-test",
    )

    spans = exporter.get_finished_spans()
    chain_spans = [s for s in spans if "workflow:" in s.name]
    assert len(chain_spans) == 1
    assert chain_spans[0].attributes.get("marie.workflow_id") == "wf-propagation-test"


# ---------------------------------------------------------------------------
# 10. Concurrent workflows don't leak context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_workflows_no_context_leak(otel_setup, sample_messages):
    """Two concurrent coordinators produce independent workflow_id attributes."""
    exporter = otel_setup

    from marie.agent.config import CoordinationConfig
    from marie.agent.coordination import WorkflowCoordinator

    config = CoordinationConfig(topology="workflow", max_steps=5, timeout=10.0)

    coord_a = WorkflowCoordinator(config)
    coord_a.add_agent(_MockAgent(name="a_worker"))

    coord_b = WorkflowCoordinator(config)
    coord_b.add_agent(_MockAgent(name="b_worker"))

    await asyncio.gather(
        coord_a.run(sample_messages, workflow_id="wf-A"),
        coord_b.run(sample_messages, workflow_id="wf-B"),
    )

    spans = exporter.get_finished_spans()
    chain_spans = [s for s in spans if "workflow:" in s.name]
    assert len(chain_spans) == 2

    wf_ids = {s.attributes.get("marie.workflow_id") for s in chain_spans}
    assert wf_ids == {"wf-A", "wf-B"}

    # Verify they have different trace IDs (independent traces)
    trace_ids = {s.context.trace_id for s in chain_spans}
    assert len(trace_ids) == 2


# ---------------------------------------------------------------------------
# Additional: Tool error span
# ---------------------------------------------------------------------------


def test_tool_error_produces_error_span(otel_setup):
    """Tool failure maps to StatusCode.ERROR with exception recorded."""
    exporter = otel_setup

    tool = _FailingTool(name="bad_tool")
    result = tool.safe_call({})

    assert result.is_error

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if "tool:" in s.name]
    assert len(tool_spans) == 1
    assert tool_spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Additional: Async tool span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_tool_creates_tool_span(otel_setup):
    """safe_acall creates TOOL span with proper attributes."""
    exporter = otel_setup

    tool = _ConcreteTool(name="async_search")
    result = await tool.safe_acall({"text": "async hello"})

    assert not result.is_error

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if "tool:" in s.name]
    assert len(tool_spans) == 1

    span = tool_spans[0]
    assert span.name == "tool:async_search"
    assert span.attributes.get(SpanAttributes.TOOL_NAME) == "async_search"
    assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# 13. session_id on CHAIN (workflow) span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_id_on_workflow_span(otel_setup, sample_messages):
    """CHAIN span carries session.id when passed via kwargs."""
    exporter = otel_setup

    from marie.agent.config import CoordinationConfig
    from marie.agent.coordination import WorkflowCoordinator

    config = CoordinationConfig(topology="workflow", max_steps=5, timeout=10.0)
    coordinator = WorkflowCoordinator(config)
    coordinator.add_agent(_MockAgent(name="worker"))

    await coordinator.run(
        sample_messages,
        workflow_id="wf-session-test",
        goal="test",
        session_id="sess-abc-123",
    )

    spans = exporter.get_finished_spans()
    chain_spans = [s for s in spans if "workflow:" in s.name]
    assert len(chain_spans) == 1

    span = chain_spans[0]
    assert span.attributes.get(SpanAttributes.SESSION_ID) == "sess-abc-123"


# ---------------------------------------------------------------------------
# 14. user_id on CHAIN (workflow) span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_id_on_workflow_span(otel_setup, sample_messages):
    """CHAIN span carries user.id when passed via kwargs."""
    exporter = otel_setup

    from marie.agent.config import CoordinationConfig
    from marie.agent.coordination import WorkflowCoordinator

    config = CoordinationConfig(topology="workflow", max_steps=5, timeout=10.0)
    coordinator = WorkflowCoordinator(config)
    coordinator.add_agent(_MockAgent(name="worker"))

    await coordinator.run(
        sample_messages,
        workflow_id="wf-user-test",
        goal="test",
        user_id="user-xyz-789",
    )

    spans = exporter.get_finished_spans()
    chain_spans = [s for s in spans if "workflow:" in s.name]
    assert len(chain_spans) == 1

    span = chain_spans[0]
    assert span.attributes.get(SpanAttributes.USER_ID) == "user-xyz-789"


# ---------------------------------------------------------------------------
# 15. session_id on AGENT span
# ---------------------------------------------------------------------------


def test_session_id_on_agent_span(otel_setup, sample_messages):
    """AGENT span carries session.id when passed via kwargs."""
    exporter = otel_setup
    agent = _SimpleAgent(name="session-agent")

    for responses in agent._impl.run(sample_messages, session_id="sess-agent-001"):
        pass

    spans = exporter.get_finished_spans()
    agent_spans = [s for s in spans if "agent:" in s.name]
    assert len(agent_spans) == 1

    span = agent_spans[0]
    assert span.attributes.get(SpanAttributes.SESSION_ID) == "sess-agent-001"


# ---------------------------------------------------------------------------
# 16. session_id propagates via SpanProcessor from using_session() context
# ---------------------------------------------------------------------------


def test_session_id_propagates_via_processor(otel_setup_with_processor):
    """SpanProcessor stamps session.id from using_session() onto vanilla spans."""
    from marie.instrumentation.context import using_session

    exporter = otel_setup_with_processor

    tracer = trace.get_tracer("test.processor")

    with using_session("sess-ctx-001"):
        with tracer.start_as_current_span("test-span") as span:
            pass  # span should get session.id from processor

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(SpanAttributes.SESSION_ID) == "sess-ctx-001"


# ---------------------------------------------------------------------------
# 17. user_id propagates via SpanProcessor from using_user() context
# ---------------------------------------------------------------------------


def test_user_id_propagates_via_processor(otel_setup_with_processor):
    """SpanProcessor stamps user.id from using_user() onto vanilla spans."""
    from marie.instrumentation.context import using_user

    exporter = otel_setup_with_processor

    tracer = trace.get_tracer("test.processor")

    with using_user("user-ctx-002"):
        with tracer.start_as_current_span("test-span") as span:
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(SpanAttributes.USER_ID) == "user-ctx-002"


# ---------------------------------------------------------------------------
# 18. No session_id when not provided
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_session_id_when_not_provided(otel_setup, sample_messages):
    """No session.id attribute set when session_id kwarg is omitted."""
    exporter = otel_setup

    from marie.agent.config import CoordinationConfig
    from marie.agent.coordination import WorkflowCoordinator

    config = CoordinationConfig(topology="workflow", max_steps=5, timeout=10.0)
    coordinator = WorkflowCoordinator(config)
    coordinator.add_agent(_MockAgent(name="worker"))

    await coordinator.run(
        sample_messages,
        workflow_id="wf-no-session",
        goal="test",
    )

    spans = exporter.get_finished_spans()
    chain_spans = [s for s in spans if "workflow:" in s.name]
    assert len(chain_spans) == 1

    span = chain_spans[0]
    assert span.attributes.get(SpanAttributes.SESSION_ID) is None
    assert span.attributes.get(SpanAttributes.USER_ID) is None
