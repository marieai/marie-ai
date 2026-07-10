"""
Submission Document sensor for RAG indexing.

Monitors S3 for new submission document uploads and triggers
RAG indexing workflows when semantic search is enabled.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.data_sink.base import DataSinkProvider, FileObject
from marie.sensors.definitions.data_sink.s3_sensor import S3DataSinkSensor
from marie.sensors.registry import register_sensor
from marie.sensors.types import RunRequest, SensorResult, SensorType
from marie.storage.submission import SubmissionStorage


# Separate sensor type for submission documents
# This allows independent control vs generic DATA_SINK sensors
class SubmissionSensorType:
    """Submission-specific sensor type (uses DATA_SINK infrastructure)."""

    SUBMISSION_DOCUMENT = "submission_document"


@register_sensor(SensorType.DATA_SINK, subtype="submission_document")
class SubmissionDocumentSensor(S3DataSinkSensor):
    """
    Sensor for monitoring submission document uploads.

    Watches S3 for files matching the pattern:
        tenants/{tenant_id}/submissions/{submission_id}/{filename}

    When a new file is detected:
    1. Looks up the submission to check if semantic search is enabled
    2. Finds the document record by storage_key
    3. Gets linked RAG indexes
    4. Creates workflow records and emits RunRequests

    Configuration:
        provider: "s3"
        bucket: str - S3 bucket name
        prefix: str - Prefix to watch (default: "tenants/")
        region: str - AWS region
        endpoint_url: str - Custom endpoint for S3-compatible storage

    Run Config:
        {
            "uri": "s3://bucket/tenants/t1/submissions/s1/doc.pdf",
            "ref_id": "<document_uuid>",
            "ref_type": "submission_document",
            "workflow_type": "rag_indexing",
            "source_id": "submission:<submission_id>",
            "index_name": "<rag_index_id>",
            "node_type": "document",
            "tenant_id": "<tenant_id>",
            "submission_id": "<submission_id>"
        }
    """

    # Pattern: tenants/{tenant_id}/submissions/{submission_id}/{filename}
    PATH_PATTERN = re.compile(r"tenants/([^/]+)/submissions/([^/]+)/(.+)")

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)
        self._submission_storage: Optional[SubmissionStorage] = None

    @property
    def submission_storage(self) -> SubmissionStorage:
        """Lazy-init submission storage."""
        if self._submission_storage is None:
            postgres_url = os.getenv("DATABASE_URL")
            if not postgres_url:
                raise ValueError(
                    "DATABASE_URL environment variable required for SubmissionDocumentSensor"
                )
            self._submission_storage = SubmissionStorage(postgres_url)
        return self._submission_storage

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate sensor by checking for new submission documents.

        Overrides parent to add submission-aware filtering and
        workflow record creation.
        """
        # Parse cursor as ISO timestamp
        cursor_timestamp: Optional[datetime] = None
        if context.cursor:
            try:
                cursor_timestamp = datetime.fromisoformat(context.cursor)
            except ValueError:
                context.log_warning(f"Invalid cursor format: {context.cursor}")

        # List objects after the cursor timestamp
        try:
            objects = await self.list_objects(after_timestamp=cursor_timestamp)
        except Exception as e:
            context.log_error(f"Failed to list objects: {e}")
            from marie.sensors.exceptions import SensorEvaluationError

            raise SensorEvaluationError(
                f"Failed to list objects in {self.bucket}: {e}",
                sensor_id=self.sensor_id,
                cause=e,
            )

        # Filter to submission documents only
        submission_objects = []
        for obj in objects:
            if self.PATH_PATTERN.match(obj.key):
                submission_objects.append(obj)

        if not submission_objects:
            return SensorResult.skip(
                "No new submission documents detected",
                cursor=context.cursor,
            )

        # Process each submission document
        run_requests = []
        latest_timestamp: Optional[datetime] = None

        for obj in submission_objects:
            requests = await self._process_submission_document(obj, context)
            run_requests.extend(requests)

            # Track latest timestamp for cursor
            if latest_timestamp is None or obj.last_modified > latest_timestamp:
                latest_timestamp = obj.last_modified

        # Update cursor
        new_cursor = (
            latest_timestamp.isoformat() if latest_timestamp else context.cursor
        )

        if not run_requests:
            return SensorResult.skip(
                "No documents require RAG indexing",
                cursor=new_cursor,
            )

        context.log_info(
            f"Created {len(run_requests)} RAG indexing workflow(s) for submission documents"
        )

        return SensorResult.fire_multiple(run_requests, cursor=new_cursor)

    async def _process_submission_document(
        self,
        obj: FileObject,
        context: SensorEvaluationContext,
    ) -> List[RunRequest]:
        """
        Process a single submission document.

        Returns RunRequests for each RAG index the document should be indexed to.
        """
        match = self.PATH_PATTERN.match(obj.key)
        if not match:
            return []

        tenant_id, submission_id, filename = match.groups()

        # Check if submission has semantic search enabled
        submission = self.submission_storage.get_submission(submission_id)
        if not submission:
            context.log_warning(f"Submission not found: {submission_id}")
            return []

        if not submission.enable_semantic_search:
            # Semantic search not enabled, skip RAG indexing
            return []

        # Get document record by storage key
        s3_uri = f"s3://{self.bucket}/{obj.key}"
        document = self.submission_storage.get_document_by_storage_key(s3_uri)
        if not document:
            context.log_warning(f"Document not found for storage_key: {s3_uri}")
            return []

        # Get linked RAG indexes
        rag_index_ids = self.submission_storage.get_rag_indexes_for_submission(
            submission_id
        )
        if not rag_index_ids and submission.rag_index_id:
            # Fallback to single rag_index_id on submission
            rag_index_ids = [submission.rag_index_id]

        if not rag_index_ids:
            context.log_info(
                f"No RAG indexes linked to submission {submission_id}, skipping"
            )
            return []

        # Create RunRequest for each RAG index
        run_requests = []
        for rag_index_id in rag_index_ids:
            # Create workflow record (upsert)
            workflow = self.submission_storage.create_document_workflow(
                document_id=document.id,
                workflow_type="rag_indexing",
            )

            run_key = self.build_run_key(
                "rag_indexing",
                document.id,
                rag_index_id,
            )

            run_requests.append(
                RunRequest(
                    run_key=run_key,
                    job_name="rag_indexing",
                    run_config={
                        # Standard metadata fields
                        "uri": s3_uri,
                        "ref_id": document.id,
                        "ref_type": "submission_document",
                        "workflow_type": "rag_indexing",
                        # RAG-specific parameters
                        "source_id": f"submission:{submission_id}",
                        "index_name": rag_index_id,
                        "node_type": "document",
                        # Context for status updates
                        "tenant_id": tenant_id,
                        "submission_id": submission_id,
                        "workflow_id": workflow.id,
                    },
                    tags={
                        "trigger": "submission_document",
                        "provider": self.provider.value,
                        "sensor_id": self.sensor_id,
                        "tenant_id": tenant_id,
                    },
                )
            )

        return run_requests

    def validate_config(self) -> None:
        """Validate submission document sensor configuration."""
        super().validate_config()

        # Ensure prefix matches submission path structure
        prefix = self.get_config_value("prefix", "")
        if prefix and not prefix.startswith("tenants"):
            from marie.sensors.exceptions import SensorConfigError

            raise SensorConfigError(
                "SubmissionDocumentSensor prefix must start with 'tenants' "
                f"to match submission path pattern, got: {prefix}",
                field="prefix",
            )
