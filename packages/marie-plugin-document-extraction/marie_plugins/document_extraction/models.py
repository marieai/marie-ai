"""Typed document extraction plugin contract."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = '1.0'


class ResultKind(str, Enum):
    SEMANTIC_DOCUMENT = 'semantic_document'
    STRUCTURED_DOCUMENT = 'structured_document'


class ArtifactDescriptor(BaseModel):
    model_config = ConfigDict(extra='forbid')

    path: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    role: Literal['document'] = 'document'


class ProviderProvenance(BaseModel):
    model_config = ConfigDict(extra='forbid')

    provider: str
    provider_version: str
    canonical_format: str
    backend: str | None = None


class ExtractionSuccess(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal['1.0'] = SCHEMA_VERSION
    outcome: Literal['success'] = 'success'
    result_kind: ResultKind
    artifact: ArtifactDescriptor
    provenance: ProviderProvenance
    metadata: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class NotExtractable(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal['1.0'] = SCHEMA_VERSION
    outcome: Literal['not_extractable'] = 'not_extractable'
    canonical_format: str
    reason: str
    attempted_providers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FormatCapability(BaseModel):
    model_config = ConfigDict(extra='forbid')

    canonical_format: str
    aliases: list[str]
    extensions: list[str]
    mime_types: list[str]
    intents: list[str] = Field(default_factory=lambda: ['semantic'])
    result_kinds: list[ResultKind]
    providers: list[str]


class CapabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal['1.0'] = SCHEMA_VERSION
    plugin_version: str
    ready: bool
    formats: list[FormatCapability]


class ProviderDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    media_type: str = 'text/markdown'
    result_kind: ResultKind = ResultKind.SEMANTIC_DOCUMENT
    provider: str
    provider_version: str
    backend: str | None = None
    metadata: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
