"""Configuration models for Marie agent framework.

This module provides configuration classes that support both
YAML file loading and Python-based configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from marie.mem0 import Mem0Config

logger = logging.getLogger("marie.agent.config")


class LLMConfig(BaseModel):
    """Configuration for an OpenAI-compatible LLM endpoint.

    The model may be served by OpenAI, Marie, vLLM, or a proxy as long as the
    endpoint implements the OpenAI API.

    Example YAML configurations:

        # OpenAI direct
        llm:
          model: gpt-4o

        # Claude via LiteLLM proxy
        llm:
          model: claude/claude-sonnet-4-20250514
          base_url: http://localhost:4000

        # Any provider via LiteLLM
        llm:
          model: anthropic/claude-3-opus
          base_url: ${LITELLM_BASE_URL}
    """

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(
        default=None,
        description="Model name. For LiteLLM use format: provider/model (e.g., claude/claude-sonnet-4)",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (uses OPENAI_API_KEY or LITELLM_API_KEY env var if not provided)",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL. Set to LiteLLM server URL for Claude/other providers.",
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
        kwargs: Dict[str, Any] = {}
        if self.model:
            kwargs["model"] = self.model
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


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
    url: Optional[str] = Field(default=None, description="Server URL")
    server_id: Optional[str] = Field(
        default=None,
        description="Registered MCP server ID. Prefer this when using a persisted server registration.",
    )
    auth_type: Literal["none", "static_headers"] = Field(
        default="none",
        description="Authentication mode for direct MCP server access",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Static headers for direct MCP server access",
    )
    enabled: bool = Field(default=True, description="Whether server is enabled")
    tools: List[Union[str, Dict[str, Any]]] = Field(
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


class SkillsConfig(BaseModel):
    """Configuration for agent skills system."""

    enabled: bool = Field(
        default=True,
        description="Enable skill routing (slash commands, auto-matching)",
    )
    auto_match: bool = Field(
        default=True,
        description="Auto-match skills from message content",
    )
    default_skills: List[str] = Field(
        default_factory=list,
        description="Skills to always load (by name)",
    )
    skill_paths: List[str] = Field(
        default_factory=list,
        description="Additional directories to search for skills",
    )


class GuardrailEntry(BaseModel):
    """Single guardrail configuration entry.

    Attributes:
        type: Guardrail type name (e.g., 'pii', 'prompt_injection')
        config: Guardrail-specific configuration
    """

    type: str = Field(..., description="Guardrail type name")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Guardrail-specific configuration",
    )


class GuardrailsConfig(BaseModel):
    """Phase-separated guardrail configuration.

    Guardrails run at three phases:
    - before: Before agent processes input
    - after: After agent generates output
    - tool_call: Before a tool is executed

    Example YAML:
        ```yaml
        guardrails:
          before:
            - type: prompt_injection
            - type: pii
              config:
                check_email: true
          after:
            - type: pii
            - type: secrets
          tool_call:
            - type: tool_scope
              config:
                allowed: [search, calculator]
        ```
    """

    before: List[GuardrailEntry] = Field(
        default_factory=list,
        description="Guardrails to run before agent processes input",
    )
    after: List[GuardrailEntry] = Field(
        default_factory=list,
        description="Guardrails to run after agent generates output",
    )
    tool_call: List[GuardrailEntry] = Field(
        default_factory=list,
        description="Guardrails to run before tool execution",
    )


class CoordinationConfig(BaseModel):
    """Configuration for agent coordination.

    Enables multi-agent coordination capabilities. Agents with the same
    group_id share memory and can coordinate their execution.

    Example YAML:
        ```yaml
        agent:
          name: document_analyzer
          coordination:
            topology: parallel
            merge_strategy: aggregate
            max_concurrent: 5
            timeout: 30.0
            group_id: document-processing
            shared_memory_enabled: true
            routing_policy: sequential
            routing_sequence:
              - agent1
              - agent2
            max_steps: 20
            max_retries_per_agent: 3
            checkpoint_enabled: true
            checkpoint_store: sqlite
            audit_enabled: true
        ```
    """

    model_config = ConfigDict(extra="allow")

    topology: str = Field(
        default="sequential",
        description="Execution topology: parallel (fan-out), sequential (chain), or custom registered topology",
    )
    merge_strategy: Literal["aggregate", "vote", "first_wins", "best_score"] = Field(
        default="aggregate",
        description="Strategy for combining results from multiple agents",
    )
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum concurrent agent executions for parallel topology",
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Timeout in seconds for coordination operations",
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Group identifier for shared memory scoping",
    )
    shared_memory_enabled: bool = Field(
        default=False,
        description="Enable shared memory across coordinated agents",
    )
    routing_policy: Optional[str] = Field(
        default=None,
        description="Routing policy: 'sequential', 'llm', or custom policy name. If None, uses message-driven routing.",
    )
    routing_sequence: Optional[List[str]] = Field(
        default=None,
        description="Agent sequence for sequential routing policy",
    )
    max_steps: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum workflow steps before termination",
    )
    max_retries_per_agent: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts per agent on failure",
    )
    checkpoint_enabled: bool = Field(
        default=False,
        description="Enable workflow state checkpointing for recovery",
    )
    checkpoint_store: Literal["sqlite", "postgresql"] = Field(
        default="sqlite",
        description="Checkpoint storage backend",
    )
    audit_enabled: bool = Field(
        default=False,
        description="Enable structured audit logging of agent communications",
    )


class AgentConfig(BaseModel):
    """Main configuration for an agent.

    Supports both programmatic configuration and YAML file loading.

    Example YAML:
        ```yaml
        agent:
          name: my_agent
          backend: openai
          system_message: "You are a helpful assistant."
          max_iterations: 10

          llm:
            model: qwen2_5_vl_7b
            base_url: http://localhost:8000/v1

          tools:
            - search
            - calculator
            - name: custom_tool
              config:
                timeout: 30

          skills:
            enabled: true
            auto_match: true
            default_skills:
              - document-extraction

          memory:
            type: chat_buffer
            max_messages: 50
        ```

    Example Python:
        ```python
        config = AgentConfig(
            name="my_agent",
            backend="openai",
            llm=LLMConfig(
                model="qwen2_5_vl_7b",
                base_url="http://localhost:8000/v1",
            ),
            tools=["search", "calculator"],
            skills=SkillsConfig(enabled=True, default_skills=["document-extraction"]),
        )
        ```
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="agent", description="Agent name")
    description: str = Field(default="", description="Agent description")
    backend: Literal["openai", "haystack", "autogen"] = Field(
        default="openai",
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
    skills: SkillsConfig = Field(
        default_factory=SkillsConfig,
        description="Skills system configuration",
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory configuration",
    )
    mcp: MCPConfig = Field(
        default_factory=MCPConfig,
        description="MCP configuration",
    )
    mem0: Mem0Config = Field(
        default_factory=Mem0Config,
        description="Mem0 memory configuration",
    )
    coordination: Optional[CoordinationConfig] = Field(
        default=None,
        description="Multi-agent coordination configuration",
    )
    sub_agents: Optional[List[str]] = Field(
        default=None,
        description="List of sub-agent names for coordination (resolved at runtime)",
    )
    guardrails: Optional[GuardrailsConfig] = Field(
        default=None,
        description="Guardrails configuration for input/output validation",
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
        if self.mcp.enabled:
            for server in self.mcp.servers:
                if not server.enabled:
                    continue

                for tool in server.tools:
                    if isinstance(tool, str):
                        result.append(
                            {
                                "type": "mcp",
                                "tool_name": tool,
                                "server_id": server.server_id,
                                "server_name": server.name,
                                "server_url": server.url,
                                "auth_type": server.auth_type,
                                "headers": server.headers,
                            }
                        )
                        continue

                    tool_name = tool.get("name")
                    if not tool_name:
                        raise ValueError("MCP tool configs must include a name")

                    result.append(
                        {
                            "type": "mcp",
                            "tool_name": tool_name,
                            "server_id": server.server_id,
                            "server_name": server.name,
                            "server_url": server.url,
                            "auth_type": server.auth_type,
                            "headers": server.headers,
                            "description": tool.get("description"),
                            "input_schema": tool.get("input_schema")
                            or tool.get("inputSchema"),
                        }
                    )
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
        default="/tmp/marie.agent",
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
