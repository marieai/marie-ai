"""
Submission domain models.

These dataclasses mirror the Prisma schema (owned by marie-studio) and provide
typed access to submission data from marie-ai.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SubmissionStatus(str, Enum):
    """Submission operational status."""

    OPEN = "open"
    CLOSED = "closed"


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexingStatus(str, Enum):
    """RAG indexing status."""

    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Submission:
    """Submission record from marie-studio."""

    id: str
    name: str
    tenant_id: str
    status: SubmissionStatus = SubmissionStatus.OPEN
    description: Optional[str] = None
    source: str = "manual"
    trigger_id: Optional[str] = None
    external_ref: Optional[str] = None
    query_plan_template_id: Optional[str] = None
    kb_index_id: Optional[str] = None
    enable_semantic_search: bool = False
    total_documents: int = 0
    processed_documents: int = 0
    closed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Submission":
        """Create from database row."""
        return cls(
            id=str(row["id"]),
            name=row["name"],
            tenant_id=str(row["tenant_id"]),
            status=SubmissionStatus(row.get("status", "open")),
            description=row.get("description"),
            source=row.get("source", "manual"),
            trigger_id=str(row["trigger_id"]) if row.get("trigger_id") else None,
            external_ref=row.get("external_ref"),
            query_plan_template_id=(
                str(row["query_plan_template_id"])
                if row.get("query_plan_template_id")
                else None
            ),
            kb_index_id=(
                str(row["kb_index_id"]) if row.get("kb_index_id") else None
            ),
            enable_semantic_search=row.get("enable_semantic_search", False),
            total_documents=row.get("total_documents", 0),
            processed_documents=row.get("processed_documents", 0),
            closed_at=row.get("closed_at"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            created_by=row.get("created_by"),
        )


@dataclass
class SubmissionDocument:
    """Document within a submission."""

    id: str
    submission_id: str
    file_name: str
    file_size: int
    content_type: str
    storage_key: str  # S3 URI
    status: DocumentStatus = DocumentStatus.PENDING
    page_count: Optional[int] = None
    document_type: Optional[str] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    dag_id: Optional[str] = None  # Deprecated: use workflows relation
    job_id: Optional[str] = None  # Deprecated: use workflows relation
    hitl_request_id: Optional[str] = None
    indexing_status: Optional[str] = None
    indexing_error: Optional[str] = None
    indexed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "SubmissionDocument":
        """Create from database row."""
        extracted = row.get("extracted_fields")
        if isinstance(extracted, str):
            import json

            extracted = json.loads(extracted) if extracted else {}

        return cls(
            id=str(row["id"]),
            submission_id=str(row["submission_id"]),
            file_name=row["file_name"],
            file_size=int(row["file_size"]),
            content_type=row["content_type"],
            storage_key=row["storage_key"],
            status=DocumentStatus(row.get("status", "pending")),
            page_count=row.get("page_count"),
            document_type=row.get("document_type"),
            error_message=row.get("error_message"),
            confidence_score=row.get("confidence_score"),
            extracted_fields=extracted or {},
            dag_id=str(row["dag_id"]) if row.get("dag_id") else None,
            job_id=str(row["job_id"]) if row.get("job_id") else None,
            hitl_request_id=(
                str(row["hitl_request_id"]) if row.get("hitl_request_id") else None
            ),
            indexing_status=row.get("indexing_status"),
            indexing_error=row.get("indexing_error"),
            indexed_at=row.get("indexed_at"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


@dataclass
class SubmissionDocumentWorkflow:
    """Workflow execution record for a document."""

    id: str
    document_id: str
    workflow_type: str
    dag_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "SubmissionDocumentWorkflow":
        """Create from database row."""
        return cls(
            id=str(row["id"]),
            document_id=str(row["document_id"]),
            workflow_type=row["workflow_type"],
            dag_id=str(row["dag_id"]) if row.get("dag_id") else None,
            status=WorkflowStatus(row.get("status", "pending")),
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            error_message=row.get("error_message"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


@dataclass
class SubmissionRagIndex:
    """Link between submission and RAG index."""

    id: str
    submission_id: str
    kb_index_id: str
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "SubmissionRagIndex":
        """Create from database row."""
        return cls(
            id=str(row["id"]),
            submission_id=str(row["submission_id"]),
            kb_index_id=str(row["kb_index_id"]),
            created_at=row.get("created_at"),
        )
