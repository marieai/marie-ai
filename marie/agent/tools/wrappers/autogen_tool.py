"""AutoGenTeamTool - Wrap AutoGen multi-agent teams as agent tools.

This module provides tools that wrap AutoGen group chats and teams,
enabling the Qwen meta-planner to delegate to specialized agent teams.
"""

from __future__ import annotations

import asyncio
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    try:
        from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
    except ImportError:
        GroupChat = Any
        GroupChatManager = Any

logger = MarieLogger("marie.agent.tools.wrappers.autogen")


class AutoGenToolInput(BaseModel):
    """Default input schema for AutoGen tools."""

    message: str = Field(..., description="Message to send to the agent team")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for the conversation",
    )
    max_turns: Optional[int] = Field(
        default=None,
        description="Maximum conversation turns",
    )


class AutoGenTeamTool(AgentTool):
    """Tool that wraps an AutoGen multi-agent team.

    Enables the meta-planner to delegate complex tasks to specialized
    AutoGen agent teams for collaborative problem-solving.

    Example:
        ```python
        from autogen import AssistantAgent, GroupChat, GroupChatManager

        # Create AutoGen agents
        researcher = AssistantAgent("researcher", llm_config=config)
        analyst = AssistantAgent("analyst", llm_config=config)
        writer = AssistantAgent("writer", llm_config=config)

        # Create group chat
        group_chat = GroupChat(
            agents=[researcher, analyst, writer],
            messages=[],
            max_round=10,
        )
        manager = GroupChatManager(groupchat=group_chat)

        # Wrap as tool
        tool = AutoGenTeamTool.from_group_chat(
            group_chat=group_chat,
            manager=manager,
            name="research_team",
            description="A team of agents for research and analysis tasks",
        )

        # Use in Qwen agent
        result = tool.call(message="Research the latest AI trends")
        ```
    """

    def __init__(
        self,
        group_chat: "GroupChat",
        manager: "GroupChatManager",
        initiator: Any,  # UserProxyAgent or similar
        metadata: ToolMetadata,
        result_extractor: Optional[Callable[[List[Dict]], str]] = None,
    ):
        """Initialize AutoGenTeamTool.

        Args:
            group_chat: AutoGen GroupChat instance
            manager: GroupChatManager instance
            initiator: Agent that initiates the conversation
            metadata: Tool metadata
            result_extractor: Custom function to extract result from messages
        """
        self._group_chat = group_chat
        self._manager = manager
        self._initiator = initiator
        self._metadata = metadata
        self._result_extractor = result_extractor or self._default_result_extractor

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def _default_result_extractor(self, messages: List[Dict[str, Any]]) -> str:
        """Extract result from conversation messages.

        Args:
            messages: List of conversation messages

        Returns:
            Extracted result string
        """
        if not messages:
            return "No response generated."

        # Get the last substantive message (not from user proxy)
        result_parts = []
        for msg in reversed(messages):
            content = msg.get("content", "")
            name = msg.get("name", "")
            role = msg.get("role", "")

            # Skip empty or system messages
            if not content or role == "system":
                continue

            # Skip user proxy messages that just relay
            if "user_proxy" in name.lower():
                continue

            # Found a substantive message
            result_parts.append(f"[{name}]: {content}")

            # Get last 3 substantive messages for context
            if len(result_parts) >= 3:
                break

        if result_parts:
            # Return in chronological order
            return "\n\n".join(reversed(result_parts))

        # Fallback: return last message
        last_msg = messages[-1]
        return last_msg.get("content", str(last_msg))

    def _reset_group_chat(self) -> None:
        """Reset group chat state for a new conversation."""
        self._group_chat.messages.clear()

    def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the AutoGen team synchronously.

        Args:
            **kwargs: Input arguments (message, context, max_turns)

        Returns:
            ToolOutput with team result
        """
        message = kwargs.get("message", kwargs.get("input", ""))
        context = kwargs.get("context", {})
        max_turns = kwargs.get("max_turns")

        try:
            # Reset for new conversation
            self._reset_group_chat()

            # Update max_round if specified
            original_max_round = self._group_chat.max_round
            if max_turns is not None:
                self._group_chat.max_round = max_turns

            # Build initial message with context
            if context:
                full_message = f"{message}\n\nContext:\n{json.dumps(context, indent=2)}"
            else:
                full_message = message

            # Initiate chat
            self._initiator.initiate_chat(
                self._manager,
                message=full_message,
            )

            # Restore max_round
            if max_turns is not None:
                self._group_chat.max_round = original_max_round

            # Extract result from messages
            result = self._result_extractor(self._group_chat.messages)

            return ToolOutput(
                content=result,
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"messages": self._group_chat.messages},
            )

        except Exception as e:
            logger.error(f"AutoGen team '{self.name}' failed: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=None,
                is_error=True,
            )

    async def acall(self, **kwargs: Any) -> ToolOutput:
        """Execute the AutoGen team asynchronously.

        Args:
            **kwargs: Input arguments

        Returns:
            ToolOutput with team result
        """
        message = kwargs.get("message", kwargs.get("input", ""))
        context = kwargs.get("context", {})
        max_turns = kwargs.get("max_turns")

        try:
            self._reset_group_chat()

            original_max_round = self._group_chat.max_round
            if max_turns is not None:
                self._group_chat.max_round = max_turns

            if context:
                full_message = f"{message}\n\nContext:\n{json.dumps(context, indent=2)}"
            else:
                full_message = message

            # Check for async initiate_chat
            if hasattr(self._initiator, "a_initiate_chat"):
                await self._initiator.a_initiate_chat(
                    self._manager,
                    message=full_message,
                )
            else:
                # Run sync in thread pool
                await asyncio.to_thread(
                    self._initiator.initiate_chat,
                    self._manager,
                    message=full_message,
                )

            if max_turns is not None:
                self._group_chat.max_round = original_max_round

            result = self._result_extractor(self._group_chat.messages)

            return ToolOutput(
                content=result,
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"messages": self._group_chat.messages},
            )

        except Exception as e:
            logger.error(f"AutoGen team '{self.name}' failed: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=None,
                is_error=True,
            )

    @classmethod
    def from_group_chat(
        cls,
        group_chat: "GroupChat",
        manager: "GroupChatManager",
        name: str,
        description: str,
        initiator: Optional[Any] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
        result_extractor: Optional[Callable] = None,
    ) -> "AutoGenTeamTool":
        """Create an AutoGenTeamTool from a group chat.

        Args:
            group_chat: AutoGen GroupChat instance
            manager: GroupChatManager instance
            name: Tool name
            description: Tool description
            initiator: Agent to initiate conversations (creates UserProxy if None)
            fn_schema: Custom input schema
            result_extractor: Custom result extractor

        Returns:
            Configured AutoGenTeamTool
        """
        # Create initiator if not provided
        if initiator is None:
            try:
                from autogen import UserProxyAgent

                initiator = UserProxyAgent(
                    name="user_proxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=0,
                    code_execution_config=False,
                )
            except ImportError:
                raise ImportError(
                    "AutoGen is required for AutoGenTeamTool. "
                    "Install with: pip install pyautogen"
                )

        metadata = ToolMetadata(
            name=name,
            description=description,
            fn_schema=fn_schema or AutoGenToolInput,
        )

        return cls(
            group_chat=group_chat,
            manager=manager,
            initiator=initiator,
            metadata=metadata,
            result_extractor=result_extractor,
        )


class AutoGenAgentTool(AgentTool):
    """Tool that wraps a single AutoGen agent for direct interaction.

    Useful when you want to delegate to a specific specialized agent
    rather than a full team.
    """

    class InputSchema(BaseModel):
        """Input schema for single agent interaction."""

        message: str = Field(..., description="Message to send to the agent")

    def __init__(
        self,
        agent: Any,  # AutoGen AssistantAgent
        user_proxy: Any,  # AutoGen UserProxyAgent
        metadata: ToolMetadata,
    ):
        """Initialize AutoGenAgentTool.

        Args:
            agent: AutoGen agent to interact with
            user_proxy: User proxy for initiating chat
            metadata: Tool metadata
        """
        self._agent = agent
        self._user_proxy = user_proxy
        self._metadata = metadata

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, message: str, **kwargs: Any) -> ToolOutput:
        """Interact with the AutoGen agent.

        Args:
            message: Message to send
            **kwargs: Additional arguments

        Returns:
            ToolOutput with agent response
        """
        try:
            # Initiate chat
            self._user_proxy.initiate_chat(
                self._agent,
                message=message,
                max_turns=kwargs.get("max_turns", 2),
            )

            # Get last agent response
            if self._agent.chat_messages:
                last_msg = list(self._agent.chat_messages.values())[-1]
                if last_msg:
                    response = last_msg[-1].get("content", "")
                else:
                    response = "No response"
            else:
                response = "No response"

            return ToolOutput(
                content=response,
                tool_name=self.name,
                raw_input={"message": message},
                raw_output={"messages": self._agent.chat_messages},
            )

        except Exception as e:
            logger.error(f"AutoGen agent '{self.name}' failed: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input={"message": message},
                raw_output=None,
                is_error=True,
            )

    async def acall(self, message: str, **kwargs: Any) -> ToolOutput:
        """Interact with the agent asynchronously."""
        return await asyncio.to_thread(self.call, message, **kwargs)

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_proxy: Optional[Any] = None,
    ) -> "AutoGenAgentTool":
        """Create an AutoGenAgentTool from an agent.

        Args:
            agent: AutoGen AssistantAgent instance
            name: Tool name (defaults to agent name)
            description: Tool description (defaults to agent system message)
            user_proxy: User proxy for chat (creates one if None)

        Returns:
            Configured AutoGenAgentTool
        """
        try:
            from autogen import UserProxyAgent
        except ImportError:
            raise ImportError(
                "AutoGen is required. Install with: pip install pyautogen"
            )

        # Get defaults from agent
        if name is None:
            name = getattr(agent, "name", "autogen_agent")

        if description is None:
            system_msg = getattr(agent, "system_message", "")
            description = (
                system_msg[:200] + "..."
                if len(system_msg) > 200
                else system_msg or f"Interact with {name}"
            )

        # Create user proxy if needed
        if user_proxy is None:
            user_proxy = UserProxyAgent(
                name=f"{name}_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )

        metadata = ToolMetadata(
            name=name,
            description=description,
            fn_schema=cls.InputSchema,
        )

        return cls(
            agent=agent,
            user_proxy=user_proxy,
            metadata=metadata,
        )


def create_research_team_tool(
    llm_config: Dict[str, Any],
    name: str = "research_team",
    description: str = "A team of AI agents for research, analysis, and writing tasks",
) -> AutoGenTeamTool:
    """Create a pre-configured research team tool.

    Args:
        llm_config: LLM configuration for AutoGen agents
        name: Tool name
        description: Tool description

    Returns:
        Configured AutoGenTeamTool with researcher, analyst, and writer agents
    """
    try:
        from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
    except ImportError:
        raise ImportError("AutoGen is required. Install with: pip install pyautogen")

    # Create specialized agents
    researcher = AssistantAgent(
        name="researcher",
        system_message=(
            "You are a research specialist. Your job is to gather information, "
            "find relevant sources, and provide factual data. Focus on accuracy "
            "and comprehensiveness."
        ),
        llm_config=llm_config,
    )

    analyst = AssistantAgent(
        name="analyst",
        system_message=(
            "You are a data analyst. Your job is to analyze information provided "
            "by the researcher, identify patterns, draw conclusions, and provide "
            "insights. Focus on logical reasoning and clear analysis."
        ),
        llm_config=llm_config,
    )

    writer = AssistantAgent(
        name="writer",
        system_message=(
            "You are a technical writer. Your job is to synthesize the research "
            "and analysis into clear, well-structured content. Focus on clarity, "
            "organization, and readability. Provide the final summary."
        ),
        llm_config=llm_config,
    )

    # Create user proxy
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Create group chat
    group_chat = GroupChat(
        agents=[user_proxy, researcher, analyst, writer],
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    return AutoGenTeamTool.from_group_chat(
        group_chat=group_chat,
        manager=manager,
        name=name,
        description=description,
        initiator=user_proxy,
    )
