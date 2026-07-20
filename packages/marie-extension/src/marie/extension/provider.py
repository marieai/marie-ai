from typing import Literal

from pydantic import Field

from marie.extension.agent import AgentDefinition
from marie.extension.model import ModelDefinition
from marie.extension.settings import (
    CredentialDefinition,
    ExtensionModel,
    ImplementationRef,
)
from marie.extension.tool import ToolDefinition

ProviderType = Literal[
    "tool_provider",
    "model_provider",
    "datasource_provider",
    "trigger_provider",
    "endpoint_provider",
    "agent_strategy_provider",
    "mcp_provider",
    "http_ui_provider",
]


class ProviderDefinition(ExtensionModel):
    ref: str
    type: ProviderType
    provider_id: str | None = Field(default=None, alias="providerId")
    display_name: str | dict[str, str] | None = Field(default=None, alias="displayName")
    description: str | dict[str, str] | None = None
    icon: str | None = None
    implementation: ImplementationRef | None = None
    credentials: list[CredentialDefinition] = Field(default_factory=list)
    configuration_methods: list[str] = Field(
        default_factory=list, alias="configurationMethods"
    )
    supported_model_types: list[str] = Field(
        default_factory=list, alias="supportedModelTypes"
    )
    tools: list[ToolDefinition] = Field(default_factory=list)
    models: list[ModelDefinition] = Field(default_factory=list)
    agents: list[AgentDefinition] = Field(default_factory=list)
