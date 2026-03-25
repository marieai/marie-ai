"""Tests for GuardrailChain."""

from __future__ import annotations

import pytest

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.chain import GuardrailChain
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult


class AllowGuardrail(Guardrail):
    """Test guardrail that always allows."""

    name = "allow"
    phase = "before"

    async def evaluate(self, content, context):
        return GuardrailResult(action=GuardrailAction.ALLOW, score=0.0)


class BlockGuardrail(Guardrail):
    """Test guardrail that always blocks."""

    name = "block"
    phase = "before"

    async def evaluate(self, content, context):
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            score=1.0,
            message="Blocked by test",
        )


class ModifyGuardrail(Guardrail):
    """Test guardrail that modifies content."""

    name = "modify"
    phase = "before"

    async def evaluate(self, content, context):
        return GuardrailResult(
            action=GuardrailAction.MODIFY,
            score=0.5,
            modified_content=f"[modified] {content}",
        )


class EscalateGuardrail(Guardrail):
    """Test guardrail that escalates."""

    name = "escalate"
    phase = "before"

    async def evaluate(self, content, context):
        return GuardrailResult(
            action=GuardrailAction.ESCALATE,
            score=0.8,
            message="Requires review",
        )


class ErrorGuardrail(Guardrail):
    """Test guardrail that raises an error."""

    name = "error"
    phase = "before"

    async def evaluate(self, content, context):
        raise ValueError("Test error")


class PriorityGuardrail(Guardrail):
    """Test guardrail that records its name in metadata."""

    name = "priority"
    phase = "before"

    async def evaluate(self, content, context):
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            metadata={"order": context.get("order", []) + [self.name]},
        )


class TestGuardrailChain:
    """Tests for GuardrailChain."""

    @pytest.mark.asyncio
    async def test_empty_chain_allows(self):
        """Empty chain should allow all content."""
        chain = GuardrailChain([])
        assert chain.is_empty

        result = await chain.run("test content", {"phase": "before"})

        assert result.action == GuardrailAction.ALLOW
        assert result.results == []
        assert result.final_content == "test content"

    @pytest.mark.asyncio
    async def test_single_allow_guardrail(self):
        """Chain with single allow guardrail."""
        chain = GuardrailChain([AllowGuardrail()])

        result = await chain.run("test content", {"phase": "before"})

        assert result.action == GuardrailAction.ALLOW
        assert len(result.results) == 1
        assert result.results[0].action == GuardrailAction.ALLOW

    @pytest.mark.asyncio
    async def test_single_block_guardrail(self):
        """Chain with single block guardrail should block."""
        chain = GuardrailChain([BlockGuardrail()])

        result = await chain.run("test content", {"phase": "before"})

        assert result.action == GuardrailAction.BLOCK
        assert len(result.results) == 1
        assert result.results[0].message == "Blocked by test"

    @pytest.mark.asyncio
    async def test_block_short_circuits(self):
        """Block should short-circuit chain."""
        # Block comes before Allow (higher priority)
        block = BlockGuardrail(GuardrailConfig(priority=200))
        allow = AllowGuardrail(GuardrailConfig(priority=100))
        chain = GuardrailChain([allow, block])  # Order doesn't matter, sorted by priority

        result = await chain.run("test content", {"phase": "before"})

        assert result.action == GuardrailAction.BLOCK
        assert len(result.results) == 1  # Only block ran

    @pytest.mark.asyncio
    async def test_escalate_short_circuits(self):
        """Escalate should short-circuit chain."""
        escalate = EscalateGuardrail(GuardrailConfig(priority=200))
        allow = AllowGuardrail(GuardrailConfig(priority=100))
        chain = GuardrailChain([allow, escalate])

        result = await chain.run("test content", {"phase": "before"})

        assert result.action == GuardrailAction.ESCALATE
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_modify_propagates_content(self):
        """Modified content should propagate to next guardrail."""
        modify = ModifyGuardrail(GuardrailConfig(priority=200))
        allow = AllowGuardrail(GuardrailConfig(priority=100))
        chain = GuardrailChain([allow, modify])

        result = await chain.run("test content", {"phase": "before"})

        assert result.action == GuardrailAction.MODIFY
        assert result.final_content == "[modified] test content"
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Errors in guardrails should not crash the chain."""
        error = ErrorGuardrail(GuardrailConfig(priority=200))
        allow = AllowGuardrail(GuardrailConfig(priority=100))
        chain = GuardrailChain([allow, error])

        result = await chain.run("test content", {"phase": "before"})

        # Chain should continue despite error
        assert result.action == GuardrailAction.ALLOW
        assert len(result.results) == 2
        # Error guardrail adds a result with error metadata
        assert "error" in result.results[0].metadata

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Guardrails should run in priority order (highest first)."""
        low = PriorityGuardrail(GuardrailConfig(priority=50))
        low.name = "low"
        medium = PriorityGuardrail(GuardrailConfig(priority=100))
        medium.name = "medium"
        high = PriorityGuardrail(GuardrailConfig(priority=150))
        high.name = "high"

        chain = GuardrailChain([low, high, medium])  # Random order

        result = await chain.run("test", {"phase": "before"})

        # Check execution order via result names
        names = [r.guardrail_name for r in result.results]
        assert names == ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_disabled_guardrails_skipped(self):
        """Disabled guardrails should be skipped."""
        enabled = AllowGuardrail(GuardrailConfig(enabled=True))
        disabled = BlockGuardrail(GuardrailConfig(enabled=False))
        chain = GuardrailChain([enabled, disabled])

        result = await chain.run("test content", {"phase": "before"})

        assert result.action == GuardrailAction.ALLOW
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_multiple_modify_chains(self):
        """Multiple modifiers should chain their modifications."""
        class PrefixModifier(Guardrail):
            name = "prefix"
            phase = "before"

            async def evaluate(self, content, context):
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    modified_content=f"[prefix]{content}",
                )

        class SuffixModifier(Guardrail):
            name = "suffix"
            phase = "before"

            async def evaluate(self, content, context):
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    modified_content=f"{content}[suffix]",
                )

        prefix = PrefixModifier(GuardrailConfig(priority=200))
        suffix = SuffixModifier(GuardrailConfig(priority=100))
        chain = GuardrailChain([suffix, prefix])

        result = await chain.run("test", {"phase": "before"})

        assert result.action == GuardrailAction.MODIFY
        assert result.final_content == "[prefix]test[suffix]"
