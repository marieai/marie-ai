"""A2A Remote Agent Tool for Marie agent toolbox.

This module provides an AgentTool implementation that allows Marie agents
to delegate tasks to external A2A-compatible agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from marie.agent.a2a.client import A2AClient
from marie.agent.a2a.discovery import A2AAgentDiscovery, AgentRegistry
from marie.agent.a2a.errors import A2AClientError, AgentDiscoveryError
from marie.agent.a2a.types import AgentCard, Message, Role, Task, TextPart
from marie.agent.tools.base import AgentTool, ToolOutput

logger = logging.getLogger(__name__)


class A2ARemoteAgentTool(AgentTool):
    """Tool for delegating tasks to external A2A agents.

    Allows Marie agents to discover and communicate with external
    A2A-compatible agents as part of their tool repertoire.

    Example:
        # Create tool for a specific agent
        tool = A2ARemoteAgentTool(
            agent_url="http://calculator-agent:9000",
            name="calculator",
            description="Performs mathematical calculations",
        )

        # Use in an agent
        agent = MyAgent(function_list=[tool])

        # Or create dynamically from discovery
        tool = await A2ARemoteAgentTool.from_discovery(
            url="http://remote-agent:9000"
        )
    """

    def __init__(
        self,
        agent_url: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        agent_card: Optional[AgentCard] = None,
        timeout: float = 60.0,
    ):
        """Initialize the A2A remote agent tool.

        Args:
            agent_url: URL of the remote A2A agent.
            name: Tool name (defaults to agent name from card).
            description: Tool description (defaults to agent description).
            agent_card: Pre-fetched agent card (optional).
            timeout: Request timeout in seconds.
        """
        self._agent_url = agent_url
        self._agent_card = agent_card
        self._timeout = timeout
        self._client: Optional[A2AClient] = None

        # Set name and description from card if available
        if agent_card:
            tool_name = name or self._sanitize_name(agent_card.name)
            tool_description = description or agent_card.description or ""
        else:
            tool_name = name or "remote_agent"
            tool_description = description or f"Remote A2A agent at {agent_url}"

        super().__init__(
            name=tool_name,
            description=tool_description,
        )

    @classmethod
    async def from_discovery(
        cls,
        url: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        timeout: float = 60.0,
    ) -> "A2ARemoteAgentTool":
        """Create a tool by discovering an A2A agent.

        Args:
            url: URL of the remote A2A agent.
            name: Optional tool name override.
            description: Optional description override.
            timeout: Request timeout.

        Returns:
            Configured A2ARemoteAgentTool.

        Raises:
            AgentDiscoveryError: If agent discovery fails.
        """
        discovery = A2AAgentDiscovery(timeout=timeout)
        try:
            card = await discovery.discover(url)
            return cls(
                agent_url=url,
                name=name,
                description=description,
                agent_card=card,
                timeout=timeout,
            )
        finally:
            await discovery.close()

    async def _get_client(self) -> A2AClient:
        """Get or create the A2A client."""
        if self._client is None:
            if self._agent_card:
                self._client = A2AClient(
                    agent_card=self._agent_card,
                    timeout=self._timeout,
                )
            else:
                self._client = await A2AClient.from_url(
                    self._agent_url,
                    timeout=self._timeout,
                )
                self._agent_card = self._client.agent_card
        return self._client

    async def close(self) -> None:
        """Close the A2A client."""
        if self._client:
            await self._client.close()
            self._client = None

    def _sanitize_name(self, name: str) -> str:
        """Sanitize agent name for use as tool name."""
        # Convert to snake_case and remove invalid characters
        sanitized = name.lower().replace(" ", "_").replace("-", "_")
        # Remove consecutive underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_")

    def get_function_definition(self) -> Dict[str, Any]:
        """Get the function definition for LLM function calling."""
        properties: Dict[str, Any] = {
            "message": {
                "type": "string",
                "description": "The message to send to the remote agent",
            },
        }
        required = ["message"]

        # Add skill-specific parameters if available
        if self._agent_card and self._agent_card.skills:
            skill_names = [s.id for s in self._agent_card.skills]
            if skill_names:
                properties["skill"] = {
                    "type": "string",
                    "description": f"Specific skill to invoke: {', '.join(skill_names)}",
                    "enum": skill_names,
                }

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _call(
        self,
        args: Union[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolOutput:
        """Synchronous call - runs the async version."""
        return asyncio.get_event_loop().run_until_complete(self._acall(args, **kwargs))

    async def _acall(
        self,
        args: Union[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolOutput:
        """Asynchronously call the remote A2A agent."""
        try:
            # Parse arguments
            if isinstance(args, str):
                try:
                    params = json.loads(args)
                except json.JSONDecodeError:
                    # Treat as plain message
                    params = {"message": args}
            else:
                params = args

            message_text = params.get("message", "")
            skill = params.get("skill")

            # Add skill hint to message if specified
            if skill:
                message_text = f"[Skill: {skill}] {message_text}"

            # Get client and send message
            client = await self._get_client()
            result = await client.send_message(message_text)

            # Extract response
            response_text = self._extract_response(result)

            return ToolOutput(
                content=response_text,
                is_error=False,
            )

        except A2AClientError as e:
            logger.error(f"A2A client error: {e}")
            return ToolOutput(
                content=f"Remote agent error: {e.message}",
                is_error=True,
            )
        except Exception as e:
            logger.exception(f"Error calling remote agent: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                is_error=True,
            )

    def _extract_response(self, result: Union[Message, Task]) -> str:
        """Extract text response from result."""
        if isinstance(result, Task):
            # Get from artifacts
            if result.artifacts:
                texts = []
                for artifact in result.artifacts:
                    for part in artifact.parts:
                        if isinstance(part, TextPart):
                            texts.append(part.text)
                        elif hasattr(part, "text"):
                            texts.append(part.text)
                if texts:
                    return "\n".join(texts)

            # Get from last message in history
            if result.history:
                for msg in reversed(result.history):
                    if msg.role == Role.AGENT:
                        return self._extract_message_text(msg)

            return f"Task completed with status: {result.status.state}"

        elif isinstance(result, Message):
            return self._extract_message_text(result)

        return str(result)

    def _extract_message_text(self, message: Message) -> str:
        """Extract text from message parts."""
        texts = []
        for part in message.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
            elif hasattr(part, "text"):
                texts.append(part.text)
        return "\n".join(texts)

    @property
    def agent_card(self) -> Optional[AgentCard]:
        """Get the remote agent's card."""
        return self._agent_card

    @property
    def skills(self) -> List[str]:
        """Get available skills from the remote agent."""
        if self._agent_card and self._agent_card.skills:
            return [s.id for s in self._agent_card.skills]
        return []


class A2AAgentToolkit:
    """Toolkit for managing multiple A2A remote agent tools.

    Provides utilities for creating and managing a collection of
    A2A remote agent tools for use in multi-agent scenarios.

    Example:
        toolkit = A2AAgentToolkit()
        await toolkit.register_agent("calc", "http://calc-agent:9000")
        await toolkit.register_agent("search", "http://search-agent:9000")

        # Get tools for an agent
        tools = toolkit.get_tools()

        # Create agent with all tools
        agent = MyAgent(function_list=tools)
    """

    def __init__(self, timeout: float = 60.0):
        """Initialize the toolkit.

        Args:
            timeout: Default timeout for agent calls.
        """
        self._tools: Dict[str, A2ARemoteAgentTool] = {}
        self._discovery = A2AAgentDiscovery(timeout=timeout)
        self._timeout = timeout

    async def close(self) -> None:
        """Close all tools and discovery service."""
        for tool in self._tools.values():
            await tool.close()
        await self._discovery.close()

    async def register_agent(
        self,
        name: str,
        url: str,
        description: Optional[str] = None,
    ) -> A2ARemoteAgentTool:
        """Register an A2A agent as a tool.

        Args:
            name: Tool name.
            url: Agent URL.
            description: Optional description override.

        Returns:
            The created tool.
        """
        card = await self._discovery.discover(url)
        tool = A2ARemoteAgentTool(
            agent_url=url,
            name=name,
            description=description,
            agent_card=card,
            timeout=self._timeout,
        )
        self._tools[name] = tool
        return tool

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent tool.

        Args:
            name: Tool name.

        Returns:
            True if removed, False if not found.
        """
        if name in self._tools:
            # Note: async close should be called separately
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[A2ARemoteAgentTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools(self) -> List[A2ARemoteAgentTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def list_agents(self) -> Dict[str, str]:
        """List all registered agents.

        Returns:
            Dictionary mapping names to URLs.
        """
        return {name: tool._agent_url for name, tool in self._tools.items()}
