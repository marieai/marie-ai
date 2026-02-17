"""AgentCard generation from Marie agents.

This module provides utilities for generating A2A AgentCards from
Marie agent configurations and tool registrations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from marie.agent.a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
)

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent
    from marie.agent.config import AgentConfig
    from marie.agent.tools.base import AgentTool

logger = logging.getLogger(__name__)


class AgentCardBuilder:
    """Builder for creating A2A AgentCards from Marie agents.

    Provides a fluent interface for constructing AgentCards with
    automatic skill generation from registered tools.

    Example:
        card = (
            AgentCardBuilder()
            .with_name("My Agent")
            .with_url("http://localhost:8000")
            .with_description("A helpful assistant")
            .from_agent(my_agent)
            .build()
        )
    """

    def __init__(self) -> None:
        self._name: Optional[str] = None
        self._description: Optional[str] = None
        self._url: Optional[str] = None
        self._version: str = "1.0.0"
        self._skills: list[AgentSkill] = []
        self._capabilities: Optional[AgentCapabilities] = None
        self._input_modes: list[str] = ["text", "text/plain"]
        self._output_modes: list[str] = ["text", "text/plain"]
        self._provider: Optional[AgentProvider] = None
        self._documentation_url: Optional[str] = None
        self._icon_url: Optional[str] = None

    def with_name(self, name: str) -> "AgentCardBuilder":
        """Set the agent name."""
        self._name = name
        return self

    def with_description(self, description: str) -> "AgentCardBuilder":
        """Set the agent description."""
        self._description = description
        return self

    def with_url(self, url: str) -> "AgentCardBuilder":
        """Set the agent URL."""
        self._url = url
        return self

    def with_version(self, version: str) -> "AgentCardBuilder":
        """Set the agent version."""
        self._version = version
        return self

    def with_capabilities(
        self,
        streaming: bool = False,
        push_notifications: bool = False,
        state_transition_history: bool = False,
    ) -> "AgentCardBuilder":
        """Set agent capabilities."""
        self._capabilities = AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
            state_transition_history=state_transition_history,
        )
        return self

    def with_provider(
        self,
        organization: str,
        url: Optional[str] = None,
    ) -> "AgentCardBuilder":
        """Set provider information."""
        self._provider = AgentProvider(organization=organization, url=url)
        return self

    def with_input_modes(self, modes: list[str]) -> "AgentCardBuilder":
        """Set supported input modes (MIME types)."""
        self._input_modes = modes
        return self

    def with_output_modes(self, modes: list[str]) -> "AgentCardBuilder":
        """Set supported output modes (MIME types)."""
        self._output_modes = modes
        return self

    def with_skill(
        self,
        id: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        examples: Optional[list[str]] = None,
    ) -> "AgentCardBuilder":
        """Add a skill manually."""
        self._skills.append(
            AgentSkill(
                id=id,
                name=name,
                description=description,
                tags=tags,
                examples=examples,
            )
        )
        return self

    def with_documentation_url(self, url: str) -> "AgentCardBuilder":
        """Set documentation URL."""
        self._documentation_url = url
        return self

    def with_icon_url(self, url: str) -> "AgentCardBuilder":
        """Set icon URL."""
        self._icon_url = url
        return self

    def from_agent(self, agent: "BaseAgent") -> "AgentCardBuilder":
        """Populate card fields from a Marie agent.

        Extracts name, description, and tools from the agent to
        auto-generate the AgentCard.
        """
        if agent.name and not self._name:
            self._name = agent.name
        if agent.description and not self._description:
            self._description = agent.description

        # Generate skills from tools
        for tool in agent.function_map.values():
            skill = self._tool_to_skill(tool)
            self._skills.append(skill)

        return self

    def from_config(self, config: "AgentConfig") -> "AgentCardBuilder":
        """Populate card fields from agent configuration."""
        if config.name and not self._name:
            self._name = config.name
        if config.description and not self._description:
            self._description = config.description
        return self

    def _tool_to_skill(self, tool: "AgentTool") -> AgentSkill:
        """Convert a Marie tool to an A2A skill."""
        func_def = tool.get_function_definition()
        params = func_def.get("parameters", {})

        # Extract parameter examples from schema
        examples = []
        properties = params.get("properties", {})
        if properties:
            example_parts = []
            for name, schema in properties.items():
                if "example" in schema:
                    example_parts.append(f"{name}={schema['example']}")
            if example_parts:
                examples.append(f"{tool.name}({', '.join(example_parts)})")

        return AgentSkill(
            id=tool.name,
            name=tool.name.replace("_", " ").title(),
            description=tool.description,
            tags=self._extract_tags(tool),
            examples=examples if examples else None,
        )

    def _extract_tags(self, tool: "AgentTool") -> list[str]:
        """Extract tags from tool metadata or name."""
        tags = []

        # Extract from tool name
        name_parts = tool.name.lower().split("_")
        if len(name_parts) > 1:
            tags.append(name_parts[0])  # First part as category

        return tags if tags else None

    def build(self) -> AgentCard:
        """Build the AgentCard.

        Returns:
            The constructed AgentCard.

        Raises:
            ValueError: If required fields are missing.
        """
        if not self._name:
            raise ValueError("Agent name is required")
        if not self._url:
            raise ValueError("Agent URL is required")

        return AgentCard(
            name=self._name,
            description=self._description,
            url=self._url,
            version=self._version,
            skills=self._skills if self._skills else None,
            capabilities=self._capabilities
            or AgentCapabilities(streaming=False, push_notifications=False),
            default_input_modes=self._input_modes,
            default_output_modes=self._output_modes,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )


def agent_card_from_agent(
    agent: "BaseAgent",
    url: str,
    version: str = "1.0.0",
    streaming: bool = False,
    push_notifications: bool = False,
) -> AgentCard:
    """Create an AgentCard from a Marie agent.

    Convenience function for quick card generation.

    Args:
        agent: The Marie agent.
        url: The agent's A2A endpoint URL.
        version: Agent version string.
        streaming: Whether streaming is supported.
        push_notifications: Whether push notifications are supported.

    Returns:
        The generated AgentCard.
    """
    return (
        AgentCardBuilder()
        .with_url(url)
        .with_version(version)
        .with_capabilities(
            streaming=streaming,
            push_notifications=push_notifications,
        )
        .from_agent(agent)
        .build()
    )


def agent_card_from_config(
    config: "AgentConfig",
    url: str,
    version: str = "1.0.0",
) -> AgentCard:
    """Create an AgentCard from agent configuration.

    Args:
        config: The agent configuration.
        url: The agent's A2A endpoint URL.
        version: Agent version string.

    Returns:
        The generated AgentCard.
    """
    return (
        AgentCardBuilder()
        .with_url(url)
        .with_version(version)
        .from_config(config)
        .build()
    )
