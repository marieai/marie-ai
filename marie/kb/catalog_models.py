"""Validated inputs for the authenticated Studio-to-KB catalog boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Nonempty = Annotated[str, Field(min_length=1, max_length=255)]
Revision = Annotated[int, Field(ge=0, strict=True)]
Backend = Literal['pgvector', 'postgresql_text']


class Scope(BaseModel):
    model_config = ConfigDict(extra='forbid')
    organization_id: UUID
    actor_id: UUID
    tenant_id: UUID


class ListIndexes(BaseModel):
    model_config = ConfigDict(extra='forbid')
    organization_id: UUID
    actor_id: UUID
    tenant_id: UUID | None = None
    tenantIds: list[UUID] | None = Field(default=None, max_length=10000)
    indexIds: list[UUID] | None = Field(default=None, max_length=10000)


class IndexScope(Scope):
    indexId: UUID


class GetIndex(Scope):
    indexId: UUID | None = None
    name: Nonempty | None = None

    @model_validator(mode='after')
    def selector(self) -> GetIndex:
        if self.indexId is None and self.name is None:
            raise ValueError('indexId or name is required')
        return self


class ProcessingParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    parse_mode: (
        Literal[
            'basic',
            'llm',
            'agent',
            'auto',
            'document_llm',
            'document_agent',
            'page',
            'paragraph',
            'semantic',
        ]
        | None
    ) = None
    multimodal: bool | None = None
    retrieval_backend: Backend | None = None
    cache_options: dict[str, bool] | None = None
    layout_options: dict[str, bool] | None = None

    @model_validator(mode='after')
    def supported_options(self) -> ProcessingParams:
        if self.cache_options and set(self.cache_options) - {'invalidate', 'disabled'}:
            raise ValueError('Unsupported cache option')
        if self.layout_options and set(self.layout_options) - {
            'precise_bounding_boxes',
            'line_level_bounding_boxes',
        }:
            raise ValueError('Unsupported layout option')
        return self


class IndexSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chunkSize: Annotated[int, Field(ge=100, le=8000, strict=True)] | None = None
    chunkOverlap: Annotated[int, Field(ge=0, le=1000, strict=True)] | None = None
    segmentationMode: Literal['page', 'paragraph', 'semantic'] | None = None
    hybridEnabled: bool | None = None
    hybridAlpha: Annotated[float, Field(ge=0, le=1)] | None = None
    multimodalEnabled: bool | None = None
    retrievalBackend: Backend | None = None
    metadata: dict[str, Any] | None = None
    runParams: ProcessingParams | None = None

    @field_validator('metadata')
    @classmethod
    def metadata_controls(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value and set(value) & {
            'configurationRevision',
            'removalPending',
            'retrievalBackend',
        }:
            raise ValueError('Use dedicated fields for server-controlled settings')
        if value and any(
            isinstance(value.get(key), dict) and value[key].get('enabled') is True
            for key in ('graph', 'reranking', 'rerank', 'inspector')
        ):
            raise ValueError('This capability is not available')
        return value


class CreateIndex(Scope, IndexSettings):
    name: Annotated[str, Field(min_length=1, max_length=100, pattern=r'^[a-z0-9-]+$')]
    indexId: UUID | None = None
    workspaceId: UUID | None = None
    embeddingModel: Annotated[str, Field(min_length=1, max_length=255)] = (
        'Qwen/Qwen3-VL-Embedding-2B'
    )
    embeddingDim: Annotated[int, Field(ge=64, le=2048, strict=True)] = 1024


class UpdateIndex(IndexScope):
    expectedConfigurationRevision: Revision
    data: IndexSettings


class SourceScope(IndexScope):
    sourceId: UUID


class ListSources(IndexScope):
    sourceType: Nonempty | None = None


def valid_key(value: str) -> str:
    if (
        not value
        or value.startswith('/')
        or '\\' in value
        or '://' in value
        or any(ord(char) < 32 for char in value)
        or any(part in ('.', '..') for part in value.split('/'))
    ):
        raise ValueError('Invalid storage object key')
    return value


def valid_filename(value: str) -> str:
    valid_key(value)
    if '/' in value:
        raise ValueError('File name must be a single path segment')
    return value


class Upload(IndexScope):
    fileName: Annotated[str, Field(min_length=1, max_length=500)]
    contentType: Annotated[str, Field(min_length=1, max_length=100)] = (
        'application/octet-stream'
    )
    _filename = field_validator('fileName')(valid_filename)


class ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    cron: Annotated[str, Field(min_length=1, max_length=200)]
    timezone: Annotated[str, Field(min_length=1, max_length=200)] = 'UTC'

    @model_validator(mode='after')
    def valid_schedule(self) -> ScheduleConfig:
        from croniter import croniter

        if len(self.cron.split()) != 5 or not croniter.is_valid(self.cron):
            raise ValueError('Enter a five-field cron expression')
        try:
            ZoneInfo(self.timezone)
        except (KeyError, ValueError) as error:
            raise ValueError('Enter an IANA timezone') from error
        return self


class SaveSource(IndexScope):
    expectedConfigurationRevision: Revision
    sourceId: UUID | None = None
    name: Annotated[str, Field(min_length=1, max_length=255)]
    sourceType: Literal[
        'file',
        'url',
        'text',
        'jsonl',
        'submission',
        'datasource',
        'repository',
        's3_import',
    ]
    extensionInstallId: UUID | None = None
    datasourceRef: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    datasetInputs: dict[str, Any] | None = None
    syncMode: Literal['manual', 'scheduled'] | None = None
    triggerConfigId: UUID | None = None
    status: (
        Literal['pending', 'processing', 'ready', 'error', 'registered', 'active']
        | None
    ) = None
    metadata: dict[str, Any] | None = None
    schedule: ScheduleConfig | None = None

    @field_validator('metadata')
    @classmethod
    def metadata_controls(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value and set(value) & {'removalPending', 'codeIndex', 'syncDagId'}:
            raise ValueError('Source runtime state is server controlled')
        return value


class Origin(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['kb', 'submission'] = 'kb'
    submissionId: UUID | None = None
    submissionDocumentId: UUID | None = None

    @model_validator(mode='after')
    def owner(self) -> Origin:
        if self.kind == 'submission' and (
            self.submissionId is None or self.submissionDocumentId is None
        ):
            raise ValueError(
                'Submission ownership requires submission and document IDs'
            )
        if self.kind == 'kb' and (self.submissionId or self.submissionDocumentId):
            raise ValueError('KB-owned storage cannot claim a submission owner')
        return self


class FileAdmission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sourceId: UUID
    name: Annotated[str, Field(min_length=1, max_length=500)]
    s3Key: Annotated[str, Field(min_length=1, max_length=1024)]
    size: Annotated[int, Field(ge=0, le=9223372036854775807)]
    contentType: Annotated[str, Field(min_length=1, max_length=100)] = (
        'application/octet-stream'
    )
    origin: Origin = Field(default_factory=Origin)
    upstreamPlanner: Nonempty | None = None
    upstreamFabrics: list[Nonempty] = Field(default_factory=list, max_length=100)
    _filename = field_validator('name')(valid_filename)
    _key = field_validator('s3Key')(valid_key)


class AdmitDocuments(IndexScope):
    bucket: Annotated[
        str, Field(min_length=1, max_length=255, pattern=r'^[A-Za-z0-9][A-Za-z0-9.-]*$')
    ]
    files: list[FileAdmission] = Field(min_length=1, max_length=1000)


class DocumentScope(Scope):
    indexId: UUID | None = None
    documentId: UUID


class DocumentMutation(DocumentScope):
    expectedIndexingDagId: UUID | None = None
    bucket: (
        Annotated[
            str,
            Field(min_length=1, max_length=255, pattern=r'^[A-Za-z0-9][A-Za-z0-9.-]*$'),
        ]
        | None
    ) = None


class ReindexFailed(IndexScope):
    bucket: (
        Annotated[
            str,
            Field(min_length=1, max_length=255, pattern=r'^[A-Za-z0-9][A-Za-z0-9.-]*$'),
        ]
        | None
    ) = None


class ListDocuments(IndexScope):
    sourceId: UUID | None = None
    search: Annotated[str, Field(max_length=500)] | None = None


class WithdrawSource(SourceScope):
    expectedConfigurationRevision: Revision | None = None


class SetSourceSchedule(SourceScope):
    expectedConfigurationRevision: Revision
    action: Literal['save', 'activate', 'pause']
    schedule: ScheduleConfig | None = None


class OwnsSchedule(BaseModel):
    model_config = ConfigDict(extra='forbid')
    organization_id: UUID
    actor_id: UUID
    tenant_id: UUID | None = None
    externalId: UUID


class SourceSchedules(BaseModel):
    model_config = ConfigDict(extra='forbid')
    organization_id: UUID
    actor_id: UUID
    tenant_id: UUID | None = None
    workspaceId: UUID


class StorageReferences(Scope):
    keys: list[Annotated[str, Field(min_length=1, max_length=2048)]] = Field(
        max_length=5000
    )
    excludeDocumentId: UUID | None = None
    excludeIndexId: UUID | None = None


class Citation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sourceId: UUID
    itemId: Annotated[str, Field(min_length=1, max_length=1024)]
    sectionId: Annotated[str, Field(min_length=1, max_length=1024)]
    indexedDagId: UUID
    sourceDigest: Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]


class CheckCitation(IndexScope):
    citation: Citation


class Audit(Scope):
    entityType: Literal['kb_index', 'kb_source', 'kb_document']
    entityId: UUID
    action: Annotated[str, Field(min_length=1, max_length=30)]
    previousValue: Any = None
    newValue: Any = None
    ipAddress: Annotated[str, Field(max_length=45)] | None = None
    userAgent: Annotated[str, Field(max_length=4096)] | None = None


class ListAudit(IndexScope):
    page: Annotated[int, Field(ge=1, strict=True)] = 1
    pageSize: Annotated[int, Field(ge=1, le=200, strict=True)] = 50
    action: Annotated[str, Field(min_length=1, max_length=30)] | None = None


class SubmissionScope(Scope):
    submissionId: UUID


class RegisterSubmission(SubmissionScope):
    bucket: Annotated[
        str, Field(min_length=1, max_length=255, pattern=r'^[A-Za-z0-9][A-Za-z0-9.-]*$')
    ]
    indexIds: list[UUID] | None = Field(default=None, max_length=1000)
    documentIds: list[UUID] | None = Field(default=None, max_length=1000)
    upstreamPlanner: Nonempty | None = None
    upstreamFabrics: list[Nonempty] = Field(default_factory=list, max_length=100)
    expectedQueryPlanTemplateId: UUID | None = None


class SubmissionData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Annotated[str, Field(min_length=1, max_length=255)] | None = None
    description: Annotated[str, Field(max_length=10000)] | None = None
    queryPlanTemplateId: UUID | None = None


class SubmissionDestinations(SubmissionScope):
    kbIndexIds: list[UUID] = Field(max_length=1000)
    enableSemanticSearch: bool
    expectedUpdatedAt: datetime | None = None
    bucket: Annotated[
        str, Field(min_length=1, max_length=255, pattern=r'^[A-Za-z0-9][A-Za-z0-9.-]*$')
    ]
    upstreamPlanner: Nonempty | None = None
    upstreamFabrics: list[Nonempty] = Field(default_factory=list, max_length=100)
    expectedQueryPlanTemplateId: UUID | None = None
    data: SubmissionData | None = None
    registerExistingDocuments: bool = True


class SubmissionFile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: UUID | None = None
    fileName: Annotated[str, Field(min_length=1, max_length=500)]
    fileSize: Annotated[int, Field(ge=0, le=9223372036854775807)]
    contentType: Annotated[str, Field(min_length=1, max_length=100)] = (
        'application/octet-stream'
    )
    storageKey: Annotated[str, Field(min_length=1, max_length=1024)]
    storageImportAuthorized: bool = False
    _filename = field_validator('fileName')(valid_filename)
    _key = field_validator('storageKey')(valid_key)


class AdmitSubmission(SubmissionScope):
    bucket: Annotated[
        str, Field(min_length=1, max_length=255, pattern=r'^[A-Za-z0-9][A-Za-z0-9.-]*$')
    ]
    files: list[SubmissionFile] = Field(min_length=1, max_length=1000)
    upstreamPlanner: Nonempty | None = None
    upstreamFabrics: list[Nonempty] = Field(default_factory=list, max_length=100)
    expectedQueryPlanTemplateId: UUID | None = None


class WithdrawSubmission(SubmissionScope):
    documentId: UUID | None = None
    retainedIndexIds: list[UUID] = Field(default_factory=list, max_length=1000)


COMMANDS: dict[str, type[BaseModel]] = {
    'list_indexes': ListIndexes,
    'get_index': GetIndex,
    'create_index': CreateIndex,
    'update_index_configuration': UpdateIndex,
    'list_sources': ListSources,
    'get_source': SourceScope,
    'prepare_upload': Upload,
    'save_source': SaveSource,
    'admit_documents': AdmitDocuments,
    'get_document': DocumentScope,
    'list_documents': ListDocuments,
    'reindex_document': DocumentMutation,
    'reindex_failed': ReindexFailed,
    'withdraw_document': DocumentMutation,
    'finalize_document_removal': DocumentScope,
    'withdraw_source': WithdrawSource,
    'finalize_source_removal': SourceScope,
    'withdraw_index': IndexScope,
    'finalize_index_removal': IndexScope,
    'storage_references': StorageReferences,
    'check_text_citation': CheckCitation,
    'document_timeline': DocumentScope,
    'record_audit': Audit,
    'list_audit': ListAudit,
    'set_submission_destinations': SubmissionDestinations,
    'register_submission_documents': RegisterSubmission,
    'admit_submission_documents': AdmitSubmission,
    'withdraw_submission': WithdrawSubmission,
    'set_source_schedule': SetSourceSchedule,
    'owns_schedule': OwnsSchedule,
    'document_evidence': DocumentScope,
    'source_schedules': SourceSchedules,
}
