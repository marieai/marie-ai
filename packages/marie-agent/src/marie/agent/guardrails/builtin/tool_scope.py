"""Tool scope guardrail.

Controls which tools can be executed based on allowlist/blocklist.
"""

from __future__ import annotations

from typing import Any, List, Set

from pydantic import Field

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.registry import register_guardrail
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult


class ToolScopeConfig(GuardrailConfig):
    """Configuration for tool scope control.

    Attributes:
        allowed: List of allowed tool names (empty = all allowed)
        blocked: List of blocked tool names
        block_message: Message when tool is blocked
    """

    allowed: List[str] = Field(
        default_factory=list,
        description="Allowed tool names (empty = all allowed)",
    )
    blocked: List[str] = Field(
        default_factory=list,
        description="Blocked tool names",
    )
    block_message: str = Field(
        default="Tool not permitted in current context",
        description="Message when tool is blocked",
    )


@register_guardrail("tool_scope", "tool_call")
class ToolScopeGuardrail(Guardrail):
    """Control which tools can be executed.

    Implements tool-level access control using allowlist and blocklist:
    - If `allowed` is non-empty, only listed tools can run
    - Tools in `blocked` are always blocked
    - Blocklist takes precedence over allowlist

    Example:
        ```yaml
        guardrails:
          tool_call:
            - type: tool_scope
              config:
                allowed:
                  - search
                  - calculator
                  - weather
                blocked:
                  - dangerous_tool
        ```
    """

    name = "tool_scope"
    phase = "tool_call"
    config_class = ToolScopeConfig

    def __init__(self, config: ToolScopeConfig = None):
        super().__init__(config or ToolScopeConfig())
        self._allowed_set: Set[str] = set()
        self._blocked_set: Set[str] = set()
        self._initialize_sets()

    def _initialize_sets(self) -> None:
        """Initialize allow/block sets."""
        config = self.config
        if isinstance(config, ToolScopeConfig):
            self._allowed_set = set(config.allowed)
            self._blocked_set = set(config.blocked)

    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Evaluate if tool is allowed.

        Args:
            content: Tool name to check
            context: Execution context with tool_name

        Returns:
            GuardrailResult with BLOCK if tool not permitted
        """
        # Get tool name from content or context
        tool_name = (
            content if isinstance(content, str) else context.get("tool_name", "")
        )

        if not tool_name:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                score=0.0,
                metadata={"reason": "no_tool_name"},
            )

        config = self.config
        if not isinstance(config, ToolScopeConfig):
            config = ToolScopeConfig()

        # Check blocklist first (takes precedence)
        if tool_name in self._blocked_set:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                score=1.0,
                message=f"{config.block_message}: '{tool_name}' is blocked",
                metadata={
                    "tool_name": tool_name,
                    "reason": "blocked",
                },
            )

        # Check allowlist (if non-empty)
        if self._allowed_set and tool_name not in self._allowed_set:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                score=1.0,
                message=f"{config.block_message}: '{tool_name}' not in allowed list",
                metadata={
                    "tool_name": tool_name,
                    "reason": "not_allowed",
                    "allowed_tools": list(self._allowed_set),
                },
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            score=0.0,
            metadata={
                "tool_name": tool_name,
            },
        )
