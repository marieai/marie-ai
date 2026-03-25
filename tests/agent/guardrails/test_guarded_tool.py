"""Tests for GuardedTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.chain import GuardrailChain
from marie.agent.guardrails.guarded_tool import GuardedTool
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult
from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class MockTool(AgentTool):
    """Mock tool for testing."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="mock_tool",
            description="A mock tool for testing",
        )

    def call(self, **kwargs) -> ToolOutput:
        return ToolOutput(
            content="mock result",
            tool_name="mock_tool",
            raw_input=kwargs,
        )

    async def acall(self, **kwargs) -> ToolOutput:
        return ToolOutput(
            content="mock async result",
            tool_name="mock_tool",
            raw_input=kwargs,
        )


class AllowGuardrail(Guardrail):
    """Guardrail that allows all tools."""

    name = "allow"
    phase = "tool_call"

    async def evaluate(self, content, context):
        return GuardrailResult(action=GuardrailAction.ALLOW)


class BlockGuardrail(Guardrail):
    """Guardrail that blocks all tools."""

    name = "block"
    phase = "tool_call"

    async def evaluate(self, content, context):
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            message="Tool blocked by guardrail",
        )


class EscalateGuardrail(Guardrail):
    """Guardrail that escalates."""

    name = "escalate"
    phase = "tool_call"

    async def evaluate(self, content, context):
        return GuardrailResult(
            action=GuardrailAction.ESCALATE,
            message="Requires approval",
        )


class TestGuardedTool:
    """Tests for GuardedTool wrapper."""

    def test_metadata_delegation(self):
        """GuardedTool should delegate metadata to inner tool."""
        inner = MockTool()
        chain = GuardrailChain([AllowGuardrail()])
        guarded = GuardedTool(inner, chain)

        assert guarded.name == "mock_tool"
        assert guarded.description == "A mock tool for testing"
        assert guarded.metadata == inner.metadata

    def test_function_definition_delegation(self):
        """GuardedTool should delegate function definition to inner tool."""
        inner = MockTool()
        chain = GuardrailChain([AllowGuardrail()])
        guarded = GuardedTool(inner, chain)

        assert guarded.get_function_definition() == inner.get_function_definition()
        assert guarded.to_openai_tool() == inner.to_openai_tool()

    @pytest.mark.asyncio
    async def test_acall_allowed(self):
        """Async call should proceed when guardrails allow."""
        inner = MockTool()
        chain = GuardrailChain([AllowGuardrail()])
        guarded = GuardedTool(inner, chain)

        result = await guarded.acall(query="test")

        assert result.content == "mock async result"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_acall_blocked(self):
        """Async call should return error when guardrails block."""
        inner = MockTool()
        chain = GuardrailChain([BlockGuardrail()])
        guarded = GuardedTool(inner, chain)

        result = await guarded.acall(query="test")

        assert "blocked" in result.content.lower()
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_acall_escalated(self):
        """Async call should return error when guardrails escalate."""
        inner = MockTool()
        chain = GuardrailChain([EscalateGuardrail()])
        guarded = GuardedTool(inner, chain)

        result = await guarded.acall(query="test")

        assert "human approval" in result.content.lower()
        assert result.is_error is True

    def test_call_allowed(self):
        """Sync call should proceed when guardrails allow."""
        inner = MockTool()
        chain = GuardrailChain([AllowGuardrail()])
        guarded = GuardedTool(inner, chain)

        result = guarded.call(query="test")

        assert result.content == "mock result"
        assert result.is_error is False

    def test_call_blocked(self):
        """Sync call should return error when guardrails block."""
        inner = MockTool()
        chain = GuardrailChain([BlockGuardrail()])
        guarded = GuardedTool(inner, chain)

        result = guarded.call(query="test")

        assert "blocked" in result.content.lower()
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_empty_chain_allows(self):
        """Empty guardrail chain should allow all calls."""
        inner = MockTool()
        chain = GuardrailChain([])
        guarded = GuardedTool(inner, chain)

        result = await guarded.acall(query="test")

        assert result.content == "mock async result"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_guardrail_context_has_tool_name(self):
        """Guardrail should receive tool name in context."""
        inner = MockTool()

        class ContextCheckGuardrail(Guardrail):
            name = "context_check"
            phase = "tool_call"
            context_received = None

            async def evaluate(self, content, context):
                self.context_received = context
                return GuardrailResult(action=GuardrailAction.ALLOW)

        guard = ContextCheckGuardrail()
        chain = GuardrailChain([guard])
        guarded = GuardedTool(inner, chain)

        await guarded.acall(query="test")

        assert guard.context_received is not None
        assert guard.context_received["tool_name"] == "mock_tool"
        assert guard.context_received["phase"] == "tool_call"

    def test_repr(self):
        """GuardedTool should have informative repr."""
        inner = MockTool()
        chain = GuardrailChain([AllowGuardrail()])
        guarded = GuardedTool(inner, chain)

        repr_str = repr(guarded)
        assert "GuardedTool" in repr_str
        assert "MockTool" in repr_str
