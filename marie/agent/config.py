"""Configuration models for Marie agent framework.

This module provides configuration classes that support both
YAML file loading and Python-based configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.config")


class LLMConfig(BaseModel):
    """Configuration for LLM backends.

    Supports marie.engine and OpenAI-compatible backends.
    """

    model_config = ConfigDict(extra="allow")

    backend: Literal["marie", "openai"] = Field(
        default="marie",
        description="LLM backend to use",
    )
    engine_name: str = Field(
        default="qwen2_5_vl_7b",
        description="Engine name for marie backend",
    )
    provider: str = Field(
        default="vllm",
        description="Provider for marie backend (vllm, openai, etc.)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name for OpenAI backend",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI backend",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate",
    )

    def to_wrapper_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for LLM wrapper initialization."""
        if self.backend == "marie":
            return {
                "engine_name": self.engine_name,
                "provider": self.provider,
            }
        elif self.backend == "openai":
            kwargs: Dict[str, Any] = {}
            if self.model:
                kwargs["model"] = self.model
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return kwargs
        return {}


class ToolConfig(BaseModel):
    """Configuration for a single tool."""

    name: str = Field(..., description="Tool name")
    enabled: bool = Field(default=True, description="Whether tool is enabled")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific configuration",
    )


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str = Field(..., description="Server name")
    url: str = Field(..., description="Server URL")
    enabled: bool = Field(default=True, description="Whether server is enabled")
    tools: List[str] = Field(
        default_factory=list,
        description="List of tools to expose from this server",
    )


class MCPConfig(BaseModel):
    """Configuration for MCP (Model Context Protocol) integration."""

    enabled: bool = Field(default=False, description="Enable MCP support")
    servers: List[MCPServerConfig] = Field(
        default_factory=list,
        description="MCP servers to connect to",
    )


class MemoryConfig(BaseModel):
    """Configuration for agent memory."""

    type: Literal["chat_buffer", "summary", "vector", "none"] = Field(
        default="chat_buffer",
        description="Memory type",
    )
    max_messages: int = Field(
        default=100,
        description="Maximum messages to retain",
    )
    summary_interval: int = Field(
        default=10,
        description="Summarize every N messages (for summary type)",
    )


class AgentConfig(BaseModel):
    """Main configuration for an agent.

    Supports both programmatic configuration and YAML file loading.

    Example YAML:
        ```yaml
        agent:
          name: my_agent
          backend: qwen_agent
          system_message: "You are a helpful assistant."
          max_iterations: 10

          llm:
            backend: marie
            engine_name: qwen2_5_vl_7b
            provider: vllm

          tools:
            - search
            - calculator
            - name: custom_tool
              config:
                timeout: 30

          memory:
            type: chat_buffer
            max_messages: 50
        ```

    Example Python:
        ```python
        config = AgentConfig(
            name="my_agent",
            backend="qwen_agent",
            llm=LLMConfig(engine_name="qwen2_5_vl_7b"),
            tools=["search", "calculator"],
        )
        ```
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="agent", description="Agent name")
    description: str = Field(default="", description="Agent description")
    backend: Literal["qwen_agent", "haystack", "autogen"] = Field(
        default="qwen_agent",
        description="Agent backend type",
    )
    system_message: str = Field(
        default="You are a helpful assistant.",
        description="System message for the agent",
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum iterations for agent loops",
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration",
    )
    tools: List[Union[str, ToolConfig]] = Field(
        default_factory=list,
        description="Tools available to the agent",
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory configuration",
    )
    mcp: MCPConfig = Field(
        default_factory=MCPConfig,
        description="MCP configuration",
    )

    @field_validator("tools", mode="before")
    @classmethod
    def normalize_tools(cls, v: Any) -> List[Union[str, ToolConfig]]:
        """Normalize tool specifications."""
        if not v:
            return []

        result = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                if "name" in item:
                    result.append(ToolConfig(**item))
                else:
                    # Assume simple dict with name as key
                    for name, config in item.items():
                        result.append(ToolConfig(name=name, config=config or {}))
            elif isinstance(item, ToolConfig):
                result.append(item)
            else:
                raise ValueError(f"Invalid tool specification: {item}")

        return result

    def get_tool_list(self) -> List[Union[str, Dict[str, Any]]]:
        """Get tools as a list for agent initialization.

        Returns:
            List of tool names or config dicts
        """
        result: List[Union[str, Dict[str, Any]]] = []
        for tool in self.tools:
            if isinstance(tool, str):
                result.append(tool)
            elif isinstance(tool, ToolConfig):
                if tool.enabled:
                    if tool.config:
                        result.append({"name": tool.name, **tool.config})
                    else:
                        result.append(tool.name)
        return result

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AgentConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            AgentConfig instance

        Example:
            ```python
            config = AgentConfig.from_yaml("config/agent.yaml")
            ```
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle nested 'agent' key
        if "agent" in data:
            data = data["agent"]

        return cls(**data)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "AgentConfig":
        """Load configuration from a YAML string.

        Args:
            yaml_string: YAML configuration string

        Returns:
            AgentConfig instance
        """
        data = yaml.safe_load(yaml_string)

        if "agent" in data:
            data = data["agent"]

        return cls(**data)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export configuration to YAML.

        Args:
            path: Optional path to write file

        Returns:
            YAML string
        """
        # Convert to dict with serializable types
        data = {"agent": self.model_dump(exclude_none=True)}

        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(yaml_str)

        return yaml_str


class ExecutorConfig(BaseModel):
    """Configuration for AgentExecutor.

    Extends AgentConfig with executor-specific settings.
    """

    model_config = ConfigDict(extra="allow")

    agent: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent configuration",
    )
    workspace: str = Field(
        default="/tmp/marie_agent",
        description="Workspace directory for the executor",
    )
    timeout: int = Field(
        default=300,
        description="Request timeout in seconds",
    )
    max_concurrent: int = Field(
        default=4,
        description="Maximum concurrent requests",
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExecutorConfig":
        """Load executor configuration from YAML."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)


def load_config(
    path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentConfig:
    """Load agent configuration from various sources.

    Priority: kwargs > config_dict > path

    Args:
        path: Path to YAML config file
        config_dict: Configuration dictionary
        **kwargs: Override configuration values

    Returns:
        AgentConfig instance

    Example:
        ```python
        # From file
        config = load_config(path="agent.yaml")

        # From dict
        config = load_config(config_dict={"name": "my_agent"})

        # With overrides
        config = load_config(
            path="agent.yaml",
            max_iterations=20,
        )
        ```
    """
    base_config: Dict[str, Any] = {}

    # Load from file
    if path:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                file_data = yaml.safe_load(f)
                if "agent" in file_data:
                    base_config.update(file_data["agent"])
                else:
                    base_config.update(file_data)

    # Merge config dict
    if config_dict:
        base_config.update(config_dict)

    # Apply overrides
    base_config.update(kwargs)

    return AgentConfig(**base_config)
