"""AutoGen agent backend for Marie agent framework.

This module provides a backend that wraps AutoGen multi-agent teams,
enabling collaborative multi-agent workflows within Marie.
"""

from __future__ import annotations

import asyncio
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from pydantic import Field

from marie.agent.backends.base import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    ToolCallRecord,
)
from marie.agent.message import Message
from marie.agent.tools.base import AgentTool
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    try:
        from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
    except ImportError:
        GroupChat = Any
        GroupChatManager = Any

logger = MarieLogger("marie.agent.backends.autogen")


class AutoGenBackendConfig(BackendConfig):
    """Configuration for AutoGen backend."""

    max_rounds: int = Field(default=10, description="Maximum conversation rounds")
    speaker_selection: str = Field(
        default="auto",
        description="Speaker selection method (auto, round_robin, random, manual)",
    )
    allow_repeat_speaker: bool = Field(
        default=True,
        description="Whether to allow the same speaker twice in a row",
    )
    human_input_mode: str = Field(
        default="NEVER",
        description="Human input mode (NEVER, ALWAYS, TERMINATE)",
    )
    code_execution: bool = Field(
        default=False,
        description="Whether to enable code execution",
    )
    llm_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM configuration for AutoGen agents",
    )


class AutoGenAgentBackend(AgentBackend):
    """Backend that wraps AutoGen multi-agent teams.

    Enables using AutoGen group chats and agent teams as backends
    for collaborative problem-solving tasks.

    Example:
        ```python
        from autogen import AssistantAgent, GroupChat, GroupChatManager

        # Create agents
        researcher = AssistantAgent("researcher", llm_config=config)
        analyst = AssistantAgent("analyst", llm_config=config)

        # Create group chat
        group_chat = GroupChat(agents=[researcher, analyst], max_round=10)
        manager = GroupChatManager(groupchat=group_chat)

        # Create backend
        backend = AutoGenAgentBackend(
            group_chat=group_chat,
            manager=manager,
            config=AutoGenBackendConfig(max_rounds=10),
        )

        # Run
        messages = [Message.user("Analyze the latest AI trends")]
        result = await backend.run(messages)
        print(result.output_text)
        ```
    """

    def __init__(
        self,
        group_chat: "GroupChat",
        manager: "GroupChatManager",
        initiator: Optional[Any] = None,
        config: Optional[AutoGenBackendConfig] = None,
        result_extractor: Optional[Callable[[List[Dict]], str]] = None,
        **kwargs: Any,
    ):
        """Initialize AutoGen backend.

        Args:
            group_chat: AutoGen GroupChat instance
            manager: GroupChatManager instance
            initiator: Agent to initiate conversations
            config: Backend configuration
            result_extractor: Custom function to extract result from messages
            **kwargs: Additional arguments
        """
        if config is None:
            config = AutoGenBackendConfig(**kwargs)

        super().__init__(config=config)

        self._group_chat = group_chat
        self._manager = manager
        self._initiator = initiator or self._create_initiator()
        self._result_extractor = result_extractor or self._default_result_extractor

    @property
    def autogen_config(self) -> AutoGenBackendConfig:
        """Get typed configuration."""
        return self.config  # type: ignore

    def _create_initiator(self) -> Any:
        """Create a user proxy agent for initiating chats."""
        try:
            from autogen import UserProxyAgent

            return UserProxyAgent(
                name="user_proxy",
                human_input_mode=self.autogen_config.human_input_mode,
                max_consecutive_auto_reply=0,
                code_execution_config=(
                    {"work_dir": "workspace"}
                    if self.autogen_config.code_execution
                    else False
                ),
            )
        except ImportError:
            raise ImportError(
                "AutoGen is required. Install with: pip install pyautogen"
            )

    def _default_result_extractor(self, messages: List[Dict[str, Any]]) -> str:
        """Extract result from conversation messages.

        Args:
            messages: AutoGen conversation messages

        Returns:
            Extracted result string
        """
        if not messages:
            return "No response generated."

        # Collect substantive messages (skip user proxy relays)
        substantive = []
        for msg in messages:
            content = msg.get("content", "")
            name = msg.get("name", "")
            role = msg.get("role", "")

            if not content or role == "system":
                continue

            if "user_proxy" in name.lower():
                continue

            substantive.append({"name": name, "content": content})

        if not substantive:
            # Fallback to last message
            return messages[-1].get("content", str(messages[-1]))

        # Format last few substantive messages
        result_parts = []
        for msg in substantive[-3:]:
            result_parts.append(f"[{msg['name']}]: {msg['content']}")

        return "\n\n".join(result_parts)

    def _reset(self) -> None:
        """Reset group chat state for new conversation."""
        self._group_chat.messages.clear()

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the AutoGen team conversation.

        Args:
            messages: Input messages (last user message used to initiate)
            tools: Tools to make available (registered with agents)
            config: Optional override configuration
            **kwargs: Additional arguments

        Returns:
            AgentResult with team output
        """
        start_time = time.time()

        try:
            # Reset for new conversation
            self._reset()

            # Extract query from messages
            query = ""
            context_parts = []
            for msg in messages:
                if msg.role == "user":
                    query = msg.text_content
                elif msg.role == "system":
                    context_parts.append(msg.text_content)

            # Build initial message with context
            if context_parts:
                full_message = (
                    f"Context:\n{chr(10).join(context_parts)}\n\nTask: {query}"
                )
            else:
                full_message = query

            # Update max_round if specified
            original_max_round = self._group_chat.max_round
            if kwargs.get("max_rounds"):
                self._group_chat.max_round = kwargs["max_rounds"]

            # Initiate chat
            if hasattr(self._initiator, "a_initiate_chat"):
                await self._initiator.a_initiate_chat(
                    self._manager,
                    message=full_message,
                )
            else:
                await asyncio.to_thread(
                    self._initiator.initiate_chat,
                    self._manager,
                    message=full_message,
                )

            # Restore max_round
            self._group_chat.max_round = original_max_round

            # Extract result
            output_text = self._result_extractor(self._group_chat.messages)

            # Build response message
            response_msg = Message.assistant(content=output_text)

            duration_ms = (time.time() - start_time) * 1000

            return AgentResult(
                output=response_msg,
                messages=[
                    {"role": "assistant", "content": output_text},
                ],
                status=AgentStatus.COMPLETED,
                is_complete=True,
                iterations=len(self._group_chat.messages),
                metadata={
                    "duration_ms": duration_ms,
                    "total_messages": len(self._group_chat.messages),
                    "agents": [a.name for a in self._group_chat.agents],
                    "raw_messages": self._group_chat.messages,
                },
            )

        except Exception as e:
            logger.error(f"AutoGen backend failed: {e}")
            return AgentResult(
                output="",
                status=AgentStatus.FAILED,
                error=str(e),
                is_complete=False,
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tools registered with AutoGen agents."""
        tools = []

        for agent in self._group_chat.agents:
            if hasattr(agent, "function_map"):
                for name, func in agent.function_map.items():
                    tools.append(
                        {
                            "name": f"{agent.name}_{name}",
                            "description": getattr(func, "__doc__", f"Tool {name}"),
                        }
                    )

        return tools

    def add_agent(self, agent: Any) -> None:
        """Add an agent to the group chat.

        Args:
            agent: AutoGen agent to add
        """
        if agent not in self._group_chat.agents:
            self._group_chat.agents.append(agent)

    def remove_agent(self, agent_name: str) -> Optional[Any]:
        """Remove an agent from the group chat.

        Args:
            agent_name: Name of agent to remove

        Returns:
            Removed agent or None
        """
        for i, agent in enumerate(self._group_chat.agents):
            if agent.name == agent_name:
                return self._group_chat.agents.pop(i)
        return None

    @classmethod
    def from_agents(
        cls,
        agents: List[Any],
        llm_config: Dict[str, Any],
        max_rounds: int = 10,
        speaker_selection: str = "auto",
        **kwargs: Any,
    ) -> "AutoGenAgentBackend":
        """Create backend from a list of agents.

        Args:
            agents: List of AutoGen agents
            llm_config: LLM configuration for manager
            max_rounds: Maximum conversation rounds
            speaker_selection: Speaker selection method
            **kwargs: Additional arguments

        Returns:
            Configured AutoGenAgentBackend
        """
        try:
            from autogen import GroupChat, GroupChatManager, UserProxyAgent
        except ImportError:
            raise ImportError(
                "AutoGen is required. Install with: pip install pyautogen"
            )

        # Create user proxy for initiation
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )

        # Create group chat with all agents
        all_agents = [user_proxy] + list(agents)
        group_chat = GroupChat(
            agents=all_agents,
            messages=[],
            max_round=max_rounds,
            speaker_selection_method=speaker_selection,
        )

        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
        )

        config = AutoGenBackendConfig(
            max_rounds=max_rounds,
            speaker_selection=speaker_selection,
            llm_config=llm_config,
            **kwargs,
        )

        return cls(
            group_chat=group_chat,
            manager=manager,
            initiator=user_proxy,
            config=config,
        )


class SingleAgentBackend(AgentBackend):
    """Backend for single AutoGen agent interaction.

    Simpler than full group chat for cases where only one
    specialized agent is needed.
    """

    def __init__(
        self,
        agent: Any,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ):
        """Initialize single agent backend.

        Args:
            agent: AutoGen AssistantAgent
            config: Backend configuration
            **kwargs: Additional arguments
        """
        super().__init__(config=config or BackendConfig(**kwargs))

        self._agent = agent
        self._user_proxy = self._create_user_proxy()

    def _create_user_proxy(self) -> Any:
        """Create user proxy for chat initiation."""
        try:
            from autogen import UserProxyAgent

            return UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )
        except ImportError:
            raise ImportError(
                "AutoGen is required. Install with: pip install pyautogen"
            )

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute single agent conversation.

        Args:
            messages: Input messages
            tools: Not used
            config: Not used
            **kwargs: Additional arguments

        Returns:
            AgentResult with agent response
        """
        try:
            # Get query
            query = ""
            for msg in reversed(messages):
                if msg.role == "user":
                    query = msg.text_content
                    break

            # Initiate chat
            await asyncio.to_thread(
                self._user_proxy.initiate_chat,
                self._agent,
                message=query,
                max_turns=kwargs.get("max_turns", 2),
            )

            # Get response
            if self._agent.chat_messages:
                agent_messages = list(self._agent.chat_messages.values())
                if agent_messages and agent_messages[-1]:
                    response = agent_messages[-1][-1].get("content", "")
                else:
                    response = "No response"
            else:
                response = "No response"

            return AgentResult(
                output=Message.assistant(content=response),
                messages=[{"role": "assistant", "content": response}],
                status=AgentStatus.COMPLETED,
                is_complete=True,
            )

        except Exception as e:
            logger.error(f"Single agent backend failed: {e}")
            return AgentResult(
                output="",
                status=AgentStatus.FAILED,
                error=str(e),
                is_complete=False,
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tools from agent."""
        if hasattr(self._agent, "function_map"):
            return [
                {"name": name, "description": getattr(func, "__doc__", "")}
                for name, func in self._agent.function_map.items()
            ]
        return []


def create_research_backend(
    llm_config: Dict[str, Any],
    max_rounds: int = 10,
) -> AutoGenAgentBackend:
    """Create a pre-configured research team backend.

    Creates a team with researcher, analyst, and writer agents
    for comprehensive research tasks.

    Args:
        llm_config: LLM configuration for agents
        max_rounds: Maximum conversation rounds

    Returns:
        Configured AutoGenAgentBackend
    """
    try:
        from autogen import AssistantAgent
    except ImportError:
        raise ImportError("AutoGen is required. Install with: pip install pyautogen")

    researcher = AssistantAgent(
        name="researcher",
        system_message=(
            "You are a research specialist. Gather information, find sources, "
            "and provide factual data. Focus on accuracy and comprehensiveness."
        ),
        llm_config=llm_config,
    )

    analyst = AssistantAgent(
        name="analyst",
        system_message=(
            "You are a data analyst. Analyze information from the researcher, "
            "identify patterns, draw conclusions, and provide insights."
        ),
        llm_config=llm_config,
    )

    writer = AssistantAgent(
        name="writer",
        system_message=(
            "You are a technical writer. Synthesize research and analysis into "
            "clear, well-structured content. Provide the final summary."
        ),
        llm_config=llm_config,
    )

    return AutoGenAgentBackend.from_agents(
        agents=[researcher, analyst, writer],
        llm_config=llm_config,
        max_rounds=max_rounds,
        speaker_selection="round_robin",
    )


def create_coding_backend(
    llm_config: Dict[str, Any],
    max_rounds: int = 15,
    code_execution: bool = True,
) -> AutoGenAgentBackend:
    """Create a pre-configured coding team backend.

    Creates a team with coder and reviewer agents for
    software development tasks.

    Args:
        llm_config: LLM configuration for agents
        max_rounds: Maximum conversation rounds
        code_execution: Whether to enable code execution

    Returns:
        Configured AutoGenAgentBackend
    """
    try:
        from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
    except ImportError:
        raise ImportError("AutoGen is required. Install with: pip install pyautogen")

    coder = AssistantAgent(
        name="coder",
        system_message=(
            "You are a senior software engineer. Write clean, efficient, "
            "well-documented code. Follow best practices and design patterns."
        ),
        llm_config=llm_config,
    )

    reviewer = AssistantAgent(
        name="reviewer",
        system_message=(
            "You are a code reviewer. Review code for bugs, security issues, "
            "and improvements. Provide constructive feedback."
        ),
        llm_config=llm_config,
    )

    executor = UserProxyAgent(
        name="executor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=(
            {"work_dir": "workspace", "use_docker": False} if code_execution else False
        ),
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    agents = [user_proxy, coder, reviewer]
    if code_execution:
        agents.append(executor)

    group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=max_rounds,
        speaker_selection_method="auto",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    config = AutoGenBackendConfig(
        max_rounds=max_rounds,
        code_execution=code_execution,
        llm_config=llm_config,
    )

    return AutoGenAgentBackend(
        group_chat=group_chat,
        manager=manager,
        initiator=user_proxy,
        config=config,
    )
