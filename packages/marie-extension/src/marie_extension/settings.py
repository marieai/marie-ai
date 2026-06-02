from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtensionModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ImplementationRef(ExtensionModel):
    language: str | None = None
    source: str | None = None
    class_name: str | None = Field(default=None, alias="class")
    provider_source: str | None = Field(default=None, alias="providerSource")
    model_sources: list[str] = Field(default_factory=list, alias="modelSources")


class ParameterOption(ExtensionModel):
    value: str
    label: str | dict[str, str] | None = None


class ParameterDefinition(ExtensionModel):
    key: str
    type: str
    title: str | dict[str, str] | None = None
    description: str | dict[str, str] | None = None
    required: bool = False
    default: Any = None
    options: list[str | ParameterOption] = Field(default_factory=list)
    minimum: float | int | None = Field(default=None, alias="min")
    maximum: float | int | None = Field(default=None, alias="max")
    form: Literal["llm", "form", "schema"] | None = Field(
        default=None,
        alias="x-marie-form",
    )


class SchemaTrack(ExtensionModel):
    required: list[str] = Field(default_factory=list)
    parameters: list[ParameterDefinition] = Field(default_factory=list)


class CredentialSecretPolicy(ExtensionModel):
    owner: str | None = None
    shareable: bool = False


class CredentialDefinition(ExtensionModel):
    key: str
    type: str
    required: bool = False
    label: str | dict[str, str] | None = None
    help: str | dict[str, str] | None = None
    placeholder: str | dict[str, str] | None = None
    secret: CredentialSecretPolicy | None = None
