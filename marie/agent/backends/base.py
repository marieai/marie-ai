"""Base backend interface for Marie agent framework.

This module provides the abstract base class for agent backends,
enabling pluggable execution strategies (Qwen-style, Haystack, AutoGen, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from marie.agent.message import Message
from marie.agent.tools.base import AgentTool


class AgentStatus(str, Enum):
    """Agent execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolCallRecord(BaseModel):
    """Record of a tool call during agent execution."""

    tool_name: str = Field(..., description="Name of the tool called")
    tool_args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )
    result: Optional[str] = Field(default=None, description="Result from the tool")
    error: Optional[str] = Field(
        default=None, description="Error message if tool failed"
    )
    duration_ms: Optional[float] = Field(
        default=None, description="Execution time in milliseconds"
    )


class AgentResult(BaseModel):
    """Result from agent execution.

    Contains the output, tool call history, and conversation state for continuation.
    """

    output: Any = Field(..., description="Primary output from the agent")
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation messages generated during execution",
    )
    tool_calls: List[ToolCallRecord] = Field(
        default_factory=list,
        description="Record of tool calls made during execution",
    )
    conversation_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="State to persist for conversation continuation",
    )
    status: AgentStatus = Field(
        default=AgentStatus.COMPLETED,
        description="Execution status",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed",
    )
    iterations: int = Field(
        default=0,
        description="Number of reasoning iterations performed",
    )
    is_complete: bool = Field(
        default=True,
        description="Whether the agent completed its task",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the execution",
    )

    @property
    def output_text(self) -> str:
        """Get output as text."""
        if isinstance(self.output, str):
            return self.output
        if isinstance(self.output, Message):
            return self.output.text_content
        if isinstance(self.output, list) and self.output:
            last = self.output[-1]
            if isinstance(last, Message):
                return last.text_content
            if isinstance(last, dict):
                return last.get("content", str(last))
        return str(self.output)


class BackendConfig(BaseModel):
    """Configuration for agent backends."""

    max_iterations: int = Field(default=10, description="Maximum reasoning iterations")
    timeout_seconds: float = Field(default=300.0, description="Execution timeout")
    stream: bool = Field(default=True, description="Whether to stream responses")
    return_intermediate: bool = Field(
        default=False,
        description="Whether to return intermediate steps",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific configuration",
    )


class AgentBackend(ABC):
    """Abstract base class for agent backends.

    Backends provide different execution strategies for agents:
    - QwenAgentBackend: Native Qwen-style ReAct execution
    - HaystackAgentBackend: Wraps Haystack pipelines
    - AutoGenAgentBackend: Wraps AutoGen multi-agent teams

    This abstraction allows the AgentExecutor to work with any backend
    while maintaining a consistent interface.

    Example:
        ```python
        class MyBackend(AgentBackend):
            async def run(self, messages, tools, config, **kwargs):
                # Custom execution logic
                result = await self._execute(messages, tools)
                return AgentResult(output=result)

            def get_available_tools(self):
                return [{"name": "tool1", "description": "..."}]


        backend = MyBackend(config=BackendConfig(max_iterations=20))
        result = await backend.run(messages, tools, config)
        ```
    """

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ):
        """Initialize the backend.

        Args:
            config: Backend configuration
            **kwargs: Additional backend-specific arguments
        """
        self.config = config or BackendConfig()
        self._extra_config = kwargs

    @abstractmethod
    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the agent with the given messages and tools.

        Args:
            messages: Input conversation messages
            tools: Available tools (name -> AgentTool mapping)
            config: Optional override configuration
            **kwargs: Additional execution arguments

        Returns:
            AgentResult with output and execution metadata
        """
        pass

    def run_sync(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous wrapper for run().

        Properly handles running from both sync and async contexts.

        Args:
            messages: Input messages
            tools: Available tools
            config: Optional configuration
            **kwargs: Additional arguments

        Returns:
            AgentResult
        """
        from marie.helper import run_async

        return run_async(self.run(messages, tools, config, **kwargs))

    @abstractmethod
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of tools available to this backend.

        Returns:
            List of tool definitions in OpenAI function format
        """
        pass

    def get_config(self) -> BackendConfig:
        """Get the current backend configuration."""
        return self.config

    def update_config(self, **kwargs: Any) -> None:
        """Update backend configuration.

        Args:
            **kwargs: Configuration fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


class CompositeBackend(AgentBackend):
    """Backend that can delegate to multiple sub-backends.

    Useful for implementing the meta-planner pattern where
    a primary backend (e.g., Qwen) can delegate to specialized
    backends (e.g., Haystack for RAG, AutoGen for teams) via tool calls.

    The primary backend calls delegate backends through tools - all execution
    happens within a single agent task, not as separate DAG jobs.
    """

    def __init__(
        self,
        primary: AgentBackend,
        delegates: Optional[Dict[str, AgentBackend]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ):
        """Initialize composite backend.

        Args:
            primary: Primary backend for main execution
            delegates: Named backends for delegation (called via tools)
            config: Configuration
            **kwargs: Additional arguments
        """
        super().__init__(config=config, **kwargs)
        self.primary = primary
        self.delegates = delegates or {}

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run using primary backend.

        The primary backend executes to completion, calling delegate
        backends through tools as needed during its execution loop.
        """
        # Run primary backend - it handles all tool calls internally
        return await self.primary.run(messages, tools, config, **kwargs)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tools from primary and all delegates."""
        tools = self.primary.get_available_tools()

        for name, delegate in self.delegates.items():
            delegate_tools = delegate.get_available_tools()
            # Prefix delegate tools to avoid conflicts
            for tool in delegate_tools:
                tool["name"] = f"{name}_{tool['name']}"
            tools.extend(delegate_tools)

        return tools

    def add_delegate(self, name: str, backend: AgentBackend) -> None:
        """Add a delegate backend.

        Args:
            name: Name for the delegate
            backend: Backend instance
        """
        self.delegates[name] = backend

    def remove_delegate(self, name: str) -> Optional[AgentBackend]:
        """Remove a delegate backend.

        Args:
            name: Name of the delegate to remove

        Returns:
            The removed backend, or None if not found
        """
        return self.delegates.pop(name, None)
