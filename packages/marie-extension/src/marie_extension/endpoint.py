from pydantic import Field

from marie_extension.settings import ExtensionModel, ImplementationRef, SchemaTrack


class EndpointDefinition(ExtensionModel):
    ref: str
    name: str
    path: str
    methods: list[str] = Field(default_factory=lambda: ["POST"])
    implementation: ImplementationRef | None = None
    configuration_schema: SchemaTrack = Field(
        default_factory=SchemaTrack, alias="configurationSchema"
    )
