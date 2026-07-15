"""Validated Marie-side models for the document extraction wire contract."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class ArtifactDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    role: Literal["document"] = "document"


class ProviderProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    provider_version: str
    canonical_format: str
    backend: str | None = None


class ExtractionSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    outcome: Literal["success"]
    result_kind: Literal["semantic_document", "structured_document"]
    artifact: ArtifactDescriptor
    provenance: ProviderProvenance
    metadata: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class NotExtractable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    outcome: Literal["not_extractable"]
    canonical_format: str
    reason: str
    attempted_providers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FormatCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_format: str
    aliases: list[str]
    extensions: list[str]
    mime_types: list[str]
    intents: list[str]
    result_kinds: list[str]
    providers: list[str]


class CapabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    plugin_version: str
    ready: bool
    formats: list[FormatCapability]


ExtractionResult = Annotated[
    Union[ExtractionSuccess, NotExtractable], Field(discriminator="outcome")
]
_RESULT_ADAPTER = TypeAdapter(ExtractionResult)


def parse_extraction_result(payload: object) -> ExtractionResult:
    """Validate and discriminate one plugin extraction result."""
    return _RESULT_ADAPTER.validate_python(payload)
