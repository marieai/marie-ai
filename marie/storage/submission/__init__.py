"""
Submission storage module.

Provides typed access to submission data from marie-ai.
Schema is owned by marie-studio (Prisma).
"""

from marie.storage.submission.storage import SubmissionStorage
from marie.storage.submission.types import (
    DocumentStatus,
    IndexingStatus,
    Submission,
    SubmissionDocument,
    SubmissionDocumentWorkflow,
    SubmissionRagIndex,
    SubmissionStatus,
    WorkflowStatus,
)

__all__ = [
    "SubmissionStorage",
    "Submission",
    "SubmissionDocument",
    "SubmissionDocumentWorkflow",
    "SubmissionRagIndex",
    "SubmissionStatus",
    "DocumentStatus",
    "IndexingStatus",
    "WorkflowStatus",
]
