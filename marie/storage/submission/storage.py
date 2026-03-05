"""
Submission storage for marie-ai access to marie-studio data.

Uses PostgresqlMixin for connection pooling and provides read/write
access to submission tables owned by marie-studio (Prisma).
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from marie.logging_core.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin
from marie.storage.submission.types import (
    IndexingStatus,
    Submission,
    SubmissionDocument,
    SubmissionDocumentWorkflow,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


class SubmissionStorage(PostgresqlMixin):
    """
    Storage for Submission and SubmissionDocument access.

    Provides read/write operations for submission data. The schema is
    owned by marie-studio (Prisma), this class provides typed access
    from marie-ai.
    """

    def __init__(
        self,
        postgres_url: str,
        schema: str = "marie_studio",
        pool_size: int = 5,
        connect_timeout: int = 10,
    ):
        """
        Initialize submission storage.

        Args:
            postgres_url: PostgreSQL connection URL
            schema: Schema name (default: marie_studio)
            pool_size: Connection pool size
            connect_timeout: Connection timeout in seconds
        """
        self._postgres_url = postgres_url
        self._schema = schema
        self._pool_size = pool_size
        self._connect_timeout = connect_timeout
        self._started = False
        self.logger = MarieLogger("submission_storage")

    def _parse_url(self) -> Dict[str, Any]:
        """Parse PostgreSQL URL into connection parameters."""
        parsed = urlparse(self._postgres_url)
        return {
            "hostname": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "username": parsed.username or "postgres",
            "password": parsed.password or "",
            "database": parsed.path.lstrip("/") or "marie",
            "schema": self._schema,
            "max_connections": self._pool_size,
            "min_connections": 1,
            "default_table": "submissions",
            "application_name": "marie_submission_storage",
            "connect_timeout": self._connect_timeout,
        }

    def _qualified_table(self, table: str) -> str:
        """Return fully qualified table name."""
        return f"{self._schema}.{table}"

    def start(self) -> None:
        """Initialize connection pool."""
        if self._started:
            return
        config = self._parse_url()
        self._setup_storage(config, connection_only=True)
        self._started = True
        self.logger.info("Submission storage initialized")

    def stop(self) -> None:
        """Close connection pool."""
        if hasattr(self, "postgreSQL_pool") and self.postgreSQL_pool:
            self.postgreSQL_pool.closeall()
        self._started = False

    def _ensure_started(self) -> None:
        """Ensure storage is started."""
        if not self._started:
            self.start()

    def get_submission(self, submission_id: str) -> Optional[Submission]:
        """Get submission by ID."""
        self._ensure_started()
        conn = self._get_connection()
        try:
            cursor = self._execute_sql_gracefully(
                f"SELECT * FROM {self._qualified_table('submissions')} WHERE id = %s",
                (submission_id,),
                return_cursor=True,
                connection=conn,
            )
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return Submission.from_row(dict(zip(columns, row)))
            return None
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def get_document_by_id(self, document_id: str) -> Optional[SubmissionDocument]:
        """Get document by ID."""
        self._ensure_started()
        conn = self._get_connection()
        try:
            cursor = self._execute_sql_gracefully(
                f"SELECT * FROM {self._qualified_table('submission_documents')} WHERE id = %s",
                (document_id,),
                return_cursor=True,
                connection=conn,
            )
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return SubmissionDocument.from_row(dict(zip(columns, row)))
            return None
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def get_document_by_storage_key(
        self, storage_key: str
    ) -> Optional[SubmissionDocument]:
        """Get document by S3 storage key (exact match)."""
        self._ensure_started()
        conn = self._get_connection()
        try:
            cursor = self._execute_sql_gracefully(
                f"SELECT * FROM {self._qualified_table('submission_documents')} WHERE storage_key = %s",
                (storage_key,),
                return_cursor=True,
                connection=conn,
            )
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return SubmissionDocument.from_row(dict(zip(columns, row)))
            return None
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def update_document_indexing_status(
        self,
        document_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """
        Update document indexing status.

        Args:
            document_id: Document UUID
            status: Indexing status (pending, indexed, failed)
            error: Error message if status is 'failed'
        """
        self._ensure_started()
        conn = self._get_connection()
        try:
            self._execute_sql_gracefully(
                f"""
                UPDATE {self._qualified_table('submission_documents')}
                SET indexing_status = %s,
                    indexing_error = %s,
                    indexed_at = CASE WHEN %s = 'indexed' THEN NOW() ELSE indexed_at END,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (status, error, status, document_id),
                connection=conn,
            )
        finally:
            self._close_connection(conn)

    def get_rag_indexes_for_submission(self, submission_id: str) -> List[str]:
        """Get RAG index IDs linked to a submission."""
        self._ensure_started()
        conn = self._get_connection()
        try:
            cursor = self._execute_sql_gracefully(
                f"SELECT rag_index_id FROM {self._qualified_table('submission_rag_indexes')} WHERE submission_id = %s",
                (submission_id,),
                return_cursor=True,
                connection=conn,
            )
            return [str(row[0]) for row in cursor.fetchall()]
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def create_document_workflow(
        self,
        document_id: str,
        workflow_type: str,
        dag_id: Optional[str] = None,
    ) -> SubmissionDocumentWorkflow:
        """
        Create a workflow record for a document.

        Uses UPSERT to handle duplicate workflow_type per document.
        """
        self._ensure_started()
        conn = self._get_connection()
        try:
            cursor = self._execute_sql_gracefully(
                f"""
                INSERT INTO {self._qualified_table('submission_document_workflows')}
                    (document_id, workflow_type, dag_id, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (document_id, workflow_type)
                DO UPDATE SET
                    dag_id = EXCLUDED.dag_id,
                    status = EXCLUDED.status,
                    updated_at = NOW()
                RETURNING *
                """,
                (document_id, workflow_type, dag_id, WorkflowStatus.PENDING.value),
                return_cursor=True,
                connection=conn,
            )
            row = cursor.fetchone()
            columns = [desc[0] for desc in cursor.description]
            return SubmissionDocumentWorkflow.from_row(dict(zip(columns, row)))
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def update_workflow_status(
        self,
        workflow_id: str,
        status: str,
        error: Optional[str] = None,
        dag_id: Optional[str] = None,
    ) -> None:
        """Update workflow execution status."""
        self._ensure_started()
        conn = self._get_connection()
        try:
            started_clause = (
                "started_at = CASE WHEN %s = 'running' AND started_at IS NULL THEN NOW() ELSE started_at END"
            )
            completed_clause = (
                "completed_at = CASE WHEN %s IN ('completed', 'failed') THEN NOW() ELSE completed_at END"
            )
            dag_clause = "dag_id = COALESCE(%s, dag_id)"

            self._execute_sql_gracefully(
                f"""
                UPDATE {self._qualified_table('submission_document_workflows')}
                SET status = %s,
                    error_message = %s,
                    {dag_clause},
                    {started_clause},
                    {completed_clause},
                    updated_at = NOW()
                WHERE id = %s
                """,
                (status, error, dag_id, status, status, workflow_id),
                connection=conn,
            )
        finally:
            self._close_connection(conn)

    def get_workflow_by_dag_id(
        self, dag_id: str
    ) -> Optional[SubmissionDocumentWorkflow]:
        """Get workflow by DAG ID."""
        self._ensure_started()
        conn = self._get_connection()
        try:
            cursor = self._execute_sql_gracefully(
                f"SELECT * FROM {self._qualified_table('submission_document_workflows')} WHERE dag_id = %s",
                (dag_id,),
                return_cursor=True,
                connection=conn,
            )
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return SubmissionDocumentWorkflow.from_row(dict(zip(columns, row)))
            return None
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def get_document_workflows(
        self, document_id: str
    ) -> List[SubmissionDocumentWorkflow]:
        """Get all workflows for a document."""
        self._ensure_started()
        conn = self._get_connection()
        try:
            cursor = self._execute_sql_gracefully(
                f"SELECT * FROM {self._qualified_table('submission_document_workflows')} WHERE document_id = %s ORDER BY created_at",
                (document_id,),
                return_cursor=True,
                connection=conn,
            )
            columns = [desc[0] for desc in cursor.description]
            return [
                SubmissionDocumentWorkflow.from_row(dict(zip(columns, row)))
                for row in cursor.fetchall()
            ]
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)
