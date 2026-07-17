from pydantic import Field

from marie.extension.settings import ExtensionModel, ImplementationRef, SchemaTrack


class TriggerDefinition(ExtensionModel):
    ref: str
    name: str
    display_name: str | dict[str, str] | None = Field(default=None, alias="displayName")
    implementation: ImplementationRef | None = None
    configuration_schema: SchemaTrack = Field(
        default_factory=SchemaTrack, alias="configurationSchema"
    )
    event_schema: dict[str, object] | None = Field(default=None, alias="eventSchema")
