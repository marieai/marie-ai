from pydantic import Field

from marie.extension.settings import (
    ExtensionModel,
    ImplementationRef,
    SchemaTrack,
)


class ToolOutput(ExtensionModel):
    message_types: list[str] = Field(default_factory=list, alias="messageTypes")
    schema_ref: str | None = Field(default=None, alias="schemaRef")
    schema_: dict[str, object] | None = Field(default=None, alias="schema")


class ToolDefinition(ExtensionModel):
    ref: str
    name: str
    display_name: str | dict[str, str] | None = Field(default=None, alias="displayName")
    description: str | dict[str, str] | None = None
    implementation: ImplementationRef | None = None
    invocation_schema: SchemaTrack = Field(
        default_factory=SchemaTrack, alias="invocationSchema"
    )
    configuration_schema: SchemaTrack = Field(
        default_factory=SchemaTrack, alias="configurationSchema"
    )
    output: ToolOutput = Field(default_factory=ToolOutput)
