"""Base guardrail classes.

This module provides the abstract base class for all guardrails.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from marie.agent.guardrails.result import GuardrailResult


class GuardrailConfig(BaseModel):
    """Base configuration for guardrails.

    Subclasses add guardrail-specific fields. The base config provides
    common settings for enabling/disabling and priority ordering.

    Attributes:
        enabled: Whether the guardrail is active
        mode: Operation mode ('strict' blocks on violation, 'permissive' logs only)
        priority: Execution priority (higher runs first)
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Whether this guardrail is enabled",
    )
    mode: str = Field(
        default="strict",
        description="Operation mode: 'strict' or 'permissive'",
    )
    priority: int = Field(
        default=100,
        description="Execution priority (higher runs first)",
    )


class Guardrail(ABC):
    """Base class for all guardrails.

    Guardrails evaluate content at specific phases of agent execution:
    - before: Before the agent processes input
    - after: After the agent generates output
    - tool_call: Before a tool is executed

    Subclasses MUST set `name` and `phase` as class attributes,
    and MAY override `config_class` for custom configuration.

    Example:
        ```python
        @register_guardrail("my_guardrail", "before")
        class MyGuardrail(Guardrail):
            name = "my_guardrail"
            phase = "before"

            async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
                if contains_bad_content(content):
                    return GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        message="Content contains prohibited material",
                    )
                return GuardrailResult(action=GuardrailAction.ALLOW)
        ```
    """

    name: ClassVar[str]
    phase: ClassVar[str]  # "before" | "after" | "tool_call"
    config_class: ClassVar[Type[GuardrailConfig]] = GuardrailConfig

    def __init__(self, config: Optional[GuardrailConfig] = None):
        """Initialize the guardrail.

        Args:
            config: Guardrail configuration. If None, uses default config.
        """
        self.config = config or self.config_class()

    @abstractmethod
    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Evaluate content against this guardrail.

        Args:
            content: The content to evaluate (text for before/after,
                    tool name for tool_call)
            context: Execution context containing:
                - phase: Current execution phase
                - agent_name: Name of the agent
                - conversation_id: Conversation identifier
                - user_id: User identifier (if available)
                - Additional phase-specific context

        Returns:
            GuardrailResult indicating the action to take
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, phase={self.phase!r})"
