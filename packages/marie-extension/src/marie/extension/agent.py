from typing import Any

from pydantic import Field

from marie.extension.runtime import RuntimePolicy
from marie.extension.settings import (
    CredentialDefinition,
    ExtensionModel,
    ImplementationRef,
    SchemaTrack,
)


class AgentOutput(ExtensionModel):
    schema_ref: str | None = Field(default=None, alias="schemaRef")
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


class AgentDefinition(ExtensionModel):
    ref: str
    name: str
    display_name: str | dict[str, str] | None = Field(default=None, alias="displayName")
    description: str | dict[str, str] | None = None
    implementation: ImplementationRef | None = None
    invocation_schema: SchemaTrack = Field(
        default_factory=SchemaTrack, alias="invocationSchema"
    )
    output: AgentOutput = Field(default_factory=AgentOutput)
    credentials: list[CredentialDefinition] = Field(default_factory=list)
    model_capabilities: list[str] = Field(
        default_factory=list, alias="modelCapabilities"
    )
    runtime_policy: RuntimePolicy | None = Field(default=None, alias="runtimePolicy")
