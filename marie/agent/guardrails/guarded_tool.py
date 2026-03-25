"""Guarded tool wrapper for tool-call guardrails.

This module provides GuardedTool, which wraps an AgentTool with
guardrail checks before execution.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, Optional

from marie.agent.guardrails.chain import GuardrailChain
from marie.agent.guardrails.result import GuardrailAction
from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.guardrails.guarded_tool")


class GuardedTool(AgentTool):
    """Wraps an AgentTool with guardrail checks before execution.

    The LLM sees the inner tool's schema unchanged. Guardrails run
    on the tool name only (context["tool_name"]) - raw args are NOT
    passed into the guardrail chain to avoid leaking sensitive data.

    Why not use middleware for tool-call guardrails?
    ------------------------------------------------
    AgentTool.safe_call() emits `tool.start` via emit_sync(), but
    emit_sync() is fire-and-forget when an event loop is running -
    it calls loop.create_task() and returns None. Since safe_call()
    is called from within chat_endpoint (async), blocking listeners
    cannot prevent tool execution.

    GuardedTool wraps the inner tool directly, running guardrails
    synchronously before delegating to _inner.call().

    Example:
        ```python
        chain = GuardrailChain([ToolScopeGuardrail(config)])
        guarded = GuardedTool(original_tool, chain)

        # Call methods work as expected, but check guardrails first
        result = guarded.call(query="test")  # May return error if blocked
        ```
    """

    def __init__(self, inner: AgentTool, chain: GuardrailChain):
        """Initialize the guarded tool wrapper.

        Args:
            inner: The inner tool to wrap
            chain: Guardrail chain to run before tool execution

        Note:
            Does NOT call super().__init__() - AgentTool is abstract
            and all behavior delegates to _inner.
        """
        self._inner = inner
        self._chain = chain

    @property
    def metadata(self) -> ToolMetadata:
        """Return the inner tool's metadata unchanged."""
        return self._inner.metadata

    @property
    def name(self) -> str:
        """Get the tool name from inner tool."""
        return self._inner.name

    @property
    def description(self) -> str:
        """Get the tool description from inner tool."""
        return self._inner.description

    def _make_blocked_output(self, message: str) -> ToolOutput:
        """Create an error ToolOutput for blocked calls.

        Args:
            message: Error message explaining why the tool was blocked

        Returns:
            ToolOutput with is_error=True
        """
        return ToolOutput(
            content=f"Tool call blocked: {message}",
            tool_name=self._inner.metadata.name,
            is_error=True,
        )

    async def _run_guards(self) -> Optional[ToolOutput]:
        """Run guardrail chain against the tool name.

        Returns:
            ToolOutput on block/escalate, None on allow
        """
        if self._chain.is_empty:
            return None

        ctx = {
            "phase": "tool_call",
            "tool_name": self._inner.metadata.name,
        }

        try:
            result = await self._chain.run(self._inner.metadata.name, ctx)

            if result.action == GuardrailAction.BLOCK:
                msg = (
                    result.results[-1].message
                    if result.results
                    else "blocked by policy"
                )
                logger.warning(f"Tool {self._inner.name} blocked by guardrail: {msg}")
                return self._make_blocked_output(msg)

            if result.action == GuardrailAction.ESCALATE:
                logger.info(f"Tool {self._inner.name} requires human approval")
                return self._make_blocked_output(
                    "This tool requires human approval (not available in direct chat)."
                )

        except Exception as e:
            logger.error(f"Guardrail check failed for tool {self._inner.name}: {e}")
            # On guardrail error, allow the tool to run (fail-open)
            # This prevents guardrail bugs from breaking all tool calls

        return None

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the tool synchronously with guardrail checks.

        ReactAgent._call_tool() calls tool.safe_call() which calls this.
        The executor's event loop IS running (we're inside chat_endpoint),
        so we cannot use run_until_complete(). Instead we spawn a thread
        with its own event loop.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            ToolOutput from the inner tool, or error if blocked
        """

        def _run_guards_sync():
            return asyncio.run(self._run_guards())

        # Run guardrails in a separate thread to avoid event loop conflicts
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                blocked = pool.submit(_run_guards_sync).result(timeout=10.0)
        except concurrent.futures.TimeoutError:
            logger.error(f"Guardrail check timed out for tool {self._inner.name}")
            # On timeout, allow the tool to run (fail-open)
            blocked = None
        except Exception as e:
            logger.error(f"Guardrail check error for tool {self._inner.name}: {e}")
            blocked = None

        if blocked is not None:
            return blocked

        return self._inner.call(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the tool asynchronously with guardrail checks.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            ToolOutput from the inner tool, or error if blocked
        """
        blocked = await self._run_guards()
        if blocked is not None:
            return blocked

        return await self._inner.acall(*args, **kwargs)

    def get_function_definition(self) -> Dict[str, Any]:
        """Get function definition from inner tool."""
        return self._inner.get_function_definition()

    def to_openai_tool(self, skip_length_check: bool = False) -> Dict[str, Any]:
        """Convert to OpenAI tool format using inner tool."""
        return self._inner.to_openai_tool(skip_length_check=skip_length_check)

    def __repr__(self) -> str:
        return f"GuardedTool({self._inner!r})"
