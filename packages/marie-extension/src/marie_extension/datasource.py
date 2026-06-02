from pydantic import Field

from marie_extension.settings import ExtensionModel, ImplementationRef, SchemaTrack


class DatasourceDefinition(ExtensionModel):
    ref: str
    name: str
    display_name: str | dict[str, str] | None = Field(default=None, alias="displayName")
    description: str | dict[str, str] | None = None
    implementation: ImplementationRef | None = None
    configuration_schema: SchemaTrack = Field(
        default_factory=SchemaTrack, alias="configurationSchema"
    )
    output_schema: dict[str, object] | None = Field(default=None, alias="outputSchema")
