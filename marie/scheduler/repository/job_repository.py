import asyncio
import glob
import os
import random
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg
from psycopg.types.json import Jsonb

from marie.constants import __default_psql_dir__, __default_schema_dir__
from marie.excepts import RuntimeFailToStart
from marie.logging_core.logger import MarieLogger
from marie.query_planner.base import QueryPlan
from marie.scheduler.fixtures import create_sql_from_file
from marie.scheduler.models import RecoveredRunLease, WorkInfo
from marie.scheduler.repository.plans import (
    cancel_jobs,
    cancel_pending_jobs_for_dag,
    complete_jobs,
    complete_jobs_by_attempt,
    complete_jobs_by_id,
    count_dag_states,
    count_job_states,
    create_queue,
    fail_jobs_by_attempt,
    fail_jobs_by_id,
    insert_dag,
    insert_job,
    insert_job_search_document,
    insert_version,
    load_dag,
    mark_as_active_dags,
    mark_as_active_jobs,
    resume_jobs,
    version_table_exists,
)
from marie.scheduler.search_documents import build_job_search_documents
from marie.scheduler.state import WorkState
from marie.storage.database.postgres import PostgresqlMixin

DEFAULT_SCHEMA = "marie_scheduler"
DEFAULT_JOB_TABLE = "job"


class JobRepository(PostgresqlMixin):
    """
    Repository for all database operations related to jobs and DAGs.
    Provides a clean data access layer with no business logic.
    """

    def __init__(self, config: Dict[str, Any], max_workers: int = 5):
        """
        Initialize the job repository.

        :param config: Database configuration
        :param max_workers: Number of thread pool workers for DB operations
        """
        super().__init__()
        self.logger = MarieLogger(JobRepository.__name__)
        self._setup_storage(config, connection_only=True)

        # Thread pool for blocking database operations
        self._db_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="db-executor"
        )
        self._loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

    # ==================== Job CRUD Operations ====================

    async def get_job_by_id(self, job_id: str) -> Optional[WorkInfo]:
        """
        Retrieve a job by its ID.

        :param job_id: The job ID to retrieve
        :return: WorkInfo object if found, None otherwise
        """

        def db_call():
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT
                        id, name, priority, state, retry_limit, start_after,
                        expire_in, data, retry_delay, retry_backoff, keep_until,
                        dag_id, job_level, soft_sla, hard_sla,
                        run_owner, run_attempt_id
                    FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                    WHERE id = %s
                    """,
                    (job_id,),
                )
                record = cursor.fetchone()
                conn.commit()
                if record:
                    return self._record_to_work_info(record)
                return None
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error getting job '{job_id}': {error}")
                if conn:
                    conn.rollback()
                return None
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def get_job_by_policy(self, ref_type: str, ref_id: str) -> Optional[WorkInfo]:
        """
        Find a job by its reference type and reference ID.

        :param ref_type: Reference type from job metadata
        :param ref_id: Reference ID from job metadata
        :return: WorkInfo object if found, None otherwise
        """

        def db_call():
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT
                        id, name, priority, state, retry_limit, start_after,
                        expire_in, data, retry_delay, retry_backoff, keep_until,
                        dag_id, job_level, soft_sla, hard_sla,
                        run_owner, run_attempt_id
                    FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                    WHERE data->'metadata'->>'ref_type' = %s
                    AND data->'metadata'->>'ref_id' = %s
                    """,
                    (ref_type, ref_id),
                )
                record = cursor.fetchone()
                conn.commit()
                if record:
                    return self._record_to_work_info(record)
                return None
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error getting job by policy: {error}")
                if conn:
                    conn.rollback()
                return None
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def list_jobs(
        self,
        queue: Optional[str] = None,
        state: Optional[WorkState] = None,
        limit: int = 1000,
    ) -> List[WorkInfo]:
        """
        List jobs with optional filters.

        :param queue: Filter by queue name
        :param state: Filter by job state
        :param limit: Maximum number of jobs to return
        :return: List of WorkInfo objects
        """

        def db_call():
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Build WHERE clause
                where_clauses = []
                params = []

                if queue:
                    where_clauses.append("name = %s")
                    params.append(queue)

                if state:
                    where_clauses.append("state = %s")
                    params.append(state.value)

                where_sql = (
                    "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                )

                cursor.execute(
                    f"""
                    SELECT
                        id, name, priority, state, retry_limit, start_after,
                        expire_in, data, retry_delay, retry_backoff, keep_until,
                        dag_id, job_level, soft_sla, hard_sla,
                        run_owner, run_attempt_id
                    FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                    {where_sql}
                    ORDER BY created_on DESC
                    LIMIT %s
                    """,
                    (*params, limit),
                )
                records = cursor.fetchall()
                conn.commit()

                return [self._record_to_work_info(r) for r in records]
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error listing jobs: {error}")
                if conn:
                    conn.rollback()
                return []
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def create_job(self, work_info: WorkInfo) -> bool:
        """
        Insert a new job into the database.

        :param work_info: Job information to insert
        :return: True if successful, False otherwise
        """

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                query = insert_job(DEFAULT_SCHEMA, work_info)
                self._execute_sql_gracefully(query, connection=conn)
                return True
            except Exception as error:
                self.logger.error(f"Error creating job: {error}")
                traceback.print_exc()
                return False
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from the database.

        :param job_id: Job ID to delete
        :return: True if deleted, False otherwise
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"DELETE FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE} WHERE id = %s",
                    (job_id,),
                )
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error deleting job '{job_id}': {error}")
                if conn:
                    conn.rollback()
                return False
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    # ==================== Job State Management ====================

    async def update_job_state(
        self,
        job_id: str,
        state: WorkState,
        output: Optional[Dict] = None,
        started_on: Optional[datetime] = None,
        completed_on: Optional[datetime] = None,
    ) -> bool:
        """
        Update the state of a job with optional timestamps.

        :param job_id: Job ID
        :param state: New state
        :param output: Optional output data
        :param started_on: Optional start timestamp
        :param completed_on: Optional completion timestamp
        :return: True if updated, False otherwise
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Build dynamic UPDATE fields
                update_fields = ["state = %s"]
                params = [state.value]

                if output:
                    update_fields.append("output = %s")
                    params.append(Jsonb(output))

                if started_on:
                    update_fields.append("started_on = %s")
                    params.append(started_on)

                if completed_on:
                    update_fields.append("completed_on = %s")
                    params.append(completed_on)

                params.append(job_id)  # WHERE clause parameter

                cursor.execute(
                    f"""
                    UPDATE {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                    """,
                    tuple(params),
                )

                updated = cursor.rowcount > 0
                conn.commit()
                return updated
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error updating job state: {error}")
                if conn:
                    conn.rollback()
                return False
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def update_job_metadata(
        self,
        job_id: str,
        queue_name: str,
        metadata_updates: Dict[str, Any],
    ) -> bool:
        """
        Update specific metadata fields for a job (e.g., branch_metadata).

        :param job_id: Job ID
        :param queue_name: Queue/partition name
        :param metadata_updates: Dict of fields to update (e.g., {"branch_metadata": {...}})
        :return: True if updated, False otherwise
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Build dynamic UPDATE fields for JSONB columns
                update_fields = []
                params = []

                for field_name, field_value in metadata_updates.items():
                    update_fields.append(f"{field_name} = %s")
                    params.append(Jsonb(field_value))

                params.extend([queue_name, job_id])  # WHERE clause parameters

                cursor.execute(
                    f"""
                    UPDATE {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                    SET {', '.join(update_fields)}
                    WHERE name = %s AND id = %s
                    """,
                    tuple(params),
                )

                updated = cursor.rowcount > 0
                conn.commit()
                return updated
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error updating job metadata: {error}")
                if conn:
                    conn.rollback()
                return False
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def mark_jobs_as_active(self, job_ids: List[str], job_name: str) -> int:
        """
        Mark multiple jobs as active.

        :param job_ids: List of job IDs to mark as active
        :param job_name: Name of the job queue
        :return: Number of jobs marked as active
        """
        if not job_ids:
            return 0

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                query = mark_as_active_jobs(DEFAULT_SCHEMA, job_name, job_ids)
                result = self._execute_sql_gracefully(query, connection=conn)
                return len(result) if result else 0
            except Exception as error:
                self.logger.error(f"Error marking jobs as active: {error}")
                return 0
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def complete_jobs(
        self, job_ids: List[str], output: Optional[Dict] = None
    ) -> int:
        """
        Mark jobs as completed.

        :param job_ids: List of job IDs to complete
        :param output: Optional output data
        :return: Number of jobs completed
        """
        if not job_ids:
            return 0

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                if output:
                    query = complete_jobs_by_id(DEFAULT_SCHEMA, job_ids, Jsonb(output))
                else:
                    query = complete_jobs_by_id(DEFAULT_SCHEMA, job_ids, None)

                result = self._execute_sql_gracefully(query, connection=conn)
                # Extract count from SELECT COUNT(*) result: [(count,)]
                return result[0][0] if result and len(result) > 0 else 0
            except Exception as error:
                self.logger.error(f"Error completing jobs: {error}")
                return 0
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def fail_jobs(self, job_ids: List[str], error_message: str) -> int:
        """
        Mark jobs as failed.

        :param job_ids: List of job IDs to fail
        :param error_message: Error message
        :return: Number of jobs failed
        """
        if not job_ids:
            return 0

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                output = {"error": error_message}
                query = fail_jobs_by_id(DEFAULT_SCHEMA, job_ids, output)
                result = self._execute_sql_gracefully(query, connection=conn)
                # Extract count from SELECT COUNT(*) result: [(count,)]
                return result[0][0] if result and len(result) > 0 else 0
            except Exception as error:
                self.logger.error(f"Error failing jobs: {error}")
                return 0
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def cancel_jobs(self, job_ids: List[str]) -> int:
        """
        Cancel jobs.

        :param job_ids: List of job IDs to cancel
        :return: Number of jobs cancelled
        """
        if not job_ids:
            return 0

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                query = cancel_jobs(DEFAULT_SCHEMA, job_ids)
                result = self._execute_sql_gracefully(query, connection=conn)
                # Extract count from SELECT COUNT(*) result: [(count,)]
                return result[0][0] if result and len(result) > 0 else 0
            except Exception as error:
                self.logger.error(f"Error cancelling jobs: {error}")
                return 0
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def resume_jobs(self, job_ids: List[str]) -> int:
        """
        Resume suspended jobs.

        :param job_ids: List of job IDs to resume
        :return: Number of jobs resumed
        """
        if not job_ids:
            return 0

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                query = resume_jobs(DEFAULT_SCHEMA, job_ids)
                result = self._execute_sql_gracefully(query, connection=conn)
                # Extract count from SELECT COUNT(*) result: [(count,)]
                return result[0][0] if result and len(result) > 0 else 0
            except Exception as error:
                self.logger.error(f"Error resuming jobs: {error}")
                return 0
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    # ==================== Lease Operations ====================

    async def lease_jobs(
        self, job_ids: List[str], owner: str, ttl_seconds: int, job_name: str
    ) -> set[str]:
        """
        Try to lease the given job IDs.

        :param job_ids: List of job IDs to lease
        :param owner: Lease owner identifier
        :param ttl_seconds: Lease TTL in seconds
        :param job_name: Job name
        :return: Set of successfully leased job IDs
        """
        if not job_ids:
            return set()

        norm_ids = list(dict.fromkeys(str(i) for i in job_ids))

        def _lease_sync() -> set[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                sql = f"""
                    SELECT unnest({DEFAULT_SCHEMA}.lease_jobs_by_id(
                        %s::uuid[],          -- _ids
                        %s::interval,        -- _ttl
                        %s,                  -- _owner
                        %s                   -- _name
                    ))
                """
                ttl_interval = f"{int(ttl_seconds)} seconds"
                params = (norm_ids, ttl_interval, owner, job_name)
                cursor = self._execute_sql_gracefully(
                    sql, data=params, return_cursor=True, connection=conn
                )
                if not cursor:
                    return set()

                leased: set[str] = set()
                for row in cursor.fetchall():
                    if len(row) >= 1 and row[0] is not None:
                        leased.add(str(row[0]))
                return leased
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _lease_sync)

    async def extend_run_lease(
        self,
        job_ids: List[str],
        owner: str,
        run_attempt_id: str,
        extend_seconds: int,
    ) -> set[str]:
        """
        Extend the active run lease for the current durable attempt.

        :param job_ids: List of job IDs to extend
        :param owner: Run owner identifier
        :param run_attempt_id: Durable run attempt identifier
        :param extend_seconds: Extension duration in seconds
        :return: Set of successfully extended job IDs
        """
        if not job_ids:
            return set()

        norm_ids = list(dict.fromkeys(str(i) for i in job_ids))

        def _extend_sync() -> set[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                sql = f"""
                    SELECT unnest({DEFAULT_SCHEMA}.extend_run_lease(
                        %s::uuid[],          -- _ids
                        %s,                  -- _run_owner
                        %s::uuid,            -- _run_attempt_id
                        %s::interval         -- _extend_by
                    ))
                """
                params = (
                    norm_ids,
                    owner,
                    run_attempt_id,
                    f"{int(extend_seconds)} seconds",
                )
                cursor = self._execute_sql_gracefully(
                    sql, data=params, return_cursor=True, connection=conn
                )
                if not cursor:
                    return set()

                extended: set[str] = set()
                for row in cursor.fetchall():
                    if len(row) >= 1 and row[0] is not None:
                        extended.add(str(row[0]))
                return extended
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _extend_sync)

    async def activate_from_lease(
        self, job_ids: List[str], owner: str, run_ttl_seconds: int
    ) -> dict[str, str]:
        """
        Promote leased jobs to active once dispatch is acknowledged.

        :param job_ids: List of job IDs to activate
        :param owner: Lease owner identifier
        :param run_ttl_seconds: Run TTL in seconds
        :return: Mapping of successfully activated job IDs to run attempt IDs
        """
        if not job_ids:
            return {}

        def _activate_sync() -> dict[str, str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                params = (
                    job_ids,
                    owner,
                    f"{run_ttl_seconds} seconds",
                )
                cursor.execute(
                    f"""
                    SELECT unnest({DEFAULT_SCHEMA}.activate_from_lease(
                        %s::uuid[], %s, %s::interval
                    )) AS id
                    """,
                    params,
                )
                activated_ids = [
                    str(row[0]) for row in cursor.fetchall() if row[0] is not None
                ]
                if not activated_ids:
                    conn.commit()
                    return {}

                cursor.execute(
                    f"""
                    SELECT id, run_attempt_id
                    FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                    WHERE id = ANY(%s::uuid[])
                    """,
                    (activated_ids,),
                )
                rows = cursor.fetchall()
                missing_attempt = [
                    str(row[0]) for row in rows if row[0] is not None and row[1] is None
                ]
                if missing_attempt:
                    raise RuntimeError(
                        "activate_from_lease returned active jobs without "
                        f"run_attempt_id: {missing_attempt}. Apply "
                        "config/psql/schema/lease/000_durable_attempts.sql."
                    )
                attempts = {
                    str(row[0]): str(row[1])
                    for row in rows
                    if row[0] is not None and row[1] is not None
                }
                missing_rows = set(activated_ids) - set(attempts)
                if missing_rows:
                    raise RuntimeError(
                        "activate_from_lease returned ids that could not be "
                        f"read back from job table: {sorted(missing_rows)}"
                    )
                conn.commit()
                return attempts
            except Exception:
                if conn:
                    conn.rollback()
                raise
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _activate_sync)

    async def validate_durable_scheduler_schema(
        self, schema: str = DEFAULT_SCHEMA
    ) -> None:
        def _validate_sync() -> None:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = %s
                          AND table_name = 'job'
                          AND column_name = 'run_attempt_id'
                    )
                    """,
                    (schema,),
                )
                has_run_attempt_id = bool(cursor.fetchone()[0])
                if not has_run_attempt_id:
                    raise RuntimeFailToStart(
                        f"{schema}.job.run_attempt_id is missing. Apply "
                        "config/psql/schema/lease/000_durable_attempts.sql before starting."
                    )

                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_enum e
                        JOIN pg_type t ON t.oid = e.enumtypid
                        JOIN pg_namespace n ON n.oid = t.typnamespace
                        WHERE n.nspname = %s
                          AND t.typname = 'job_state'
                          AND e.enumlabel = 'skipped'
                    )
                    """,
                    (schema,),
                )
                has_skipped_state = bool(cursor.fetchone()[0])
                if not has_skipped_state:
                    raise RuntimeFailToStart(
                        f"{schema}.job_state is missing value 'skipped'. Apply "
                        "config/psql/schema/lease/000_durable_attempts.sql before starting."
                    )

                cursor.execute(
                    """
                    SELECT pg_get_functiondef(p.oid)
                    FROM pg_proc p
                    JOIN pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = %s
                      AND p.proname = 'activate_from_lease'
                    ORDER BY p.oid DESC
                    LIMIT 1
                    """,
                    (schema,),
                )
                row = cursor.fetchone()
                function_def = row[0] if row else ""
                if (
                    "run_attempt_id" not in function_def
                    or "lease_owner = _run_owner" not in function_def
                ):
                    raise RuntimeFailToStart(
                        f"{schema}.activate_from_lease is not the durable "
                        "attempt-aware version. Apply "
                        "config/psql/schema/lease/002_activate_from_lease.sql before starting."
                    )

                cursor.execute(
                    """
                    SELECT pg_get_functiondef(p.oid)
                    FROM pg_proc p
                    JOIN pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = %s
                      AND p.proname = 'extend_run_lease'
                    ORDER BY p.oid DESC
                    LIMIT 1
                    """,
                    (schema,),
                )
                row = cursor.fetchone()
                function_def = row[0] if row else ""
                if "_run_attempt_id" not in function_def:
                    raise RuntimeFailToStart(
                        f"{schema}.extend_run_lease is not attempt-aware. Apply "
                        "config/psql/schema/lease/003_extend_run_lease.sql before starting."
                    )

                cursor.execute(
                    f"""
                    SELECT id
                    FROM {schema}.{DEFAULT_JOB_TABLE}
                    WHERE state::text = 'active'
                      AND run_attempt_id IS NULL
                    LIMIT 10
                    """
                )
                bad_rows = [str(row[0]) for row in cursor.fetchall()]
                if bad_rows:
                    raise RuntimeFailToStart(
                        "Active jobs without run_attempt_id found: "
                        f"{bad_rows}. Reset/retry these rows before starting; "
                        "do not repair them in the scheduler hot path."
                    )
            finally:
                self._close_cursor(cursor)
                if conn and not conn.closed:
                    conn.rollback()
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, _validate_sync)

    async def release_lease(self, job_ids: List[str]) -> set[str]:
        """
        Release leases for the given job IDs.

        :param job_ids: List of job IDs to release
        :return: Set of successfully released job IDs
        """
        if not job_ids:
            return set()

        def _release_sync() -> set[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                sql = f"SELECT unnest({DEFAULT_SCHEMA}.release_lease(%s::uuid[])) AS id"
                params = (job_ids,)
                cursor = self._execute_sql_gracefully(
                    sql, data=params, return_cursor=True, connection=conn
                )
                rows = cursor.fetchall() if cursor else []
                return {row[0] for row in rows}
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _release_sync)

    @staticmethod
    def _recovery_start_after(
        *, retry_delay: int, retry_backoff: bool, retry_count: int
    ) -> datetime:
        now = datetime.now(timezone.utc)
        if not retry_backoff:
            return now + timedelta(seconds=int(retry_delay or 0))

        backoff_factor = 2 ** min(16, int(retry_count) + 1) / 2
        base_delay = int(retry_delay or 0) * backoff_factor
        return now + timedelta(seconds=base_delay + base_delay * random.random())

    async def recover_expired_run_leases(
        self, limit: int = 1000
    ) -> list[RecoveredRunLease]:
        """
        Recover active run leases whose heartbeats have expired.

        Claiming and applying recovery decisions happen in one transaction so
        row locks acquired by claim_expired_run_leases remain held until apply.
        """

        def _recover_sync() -> list[RecoveredRunLease]:
            conn = None
            cursor = None
            recovered: list[RecoveredRunLease] = []
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT
                        id,
                        name,
                        dag_id,
                        previous_state,
                        retry_count,
                        retry_limit,
                        retry_delay,
                        retry_backoff,
                        start_after,
                        run_owner,
                        run_attempt_id,
                        run_lease_expires_at
                    FROM {DEFAULT_SCHEMA}.claim_expired_run_leases(%s)
                    """,
                    (int(limit),),
                )
                rows = cursor.fetchall()
                for row in rows:
                    job_id = str(row[0])
                    name = row[1]
                    dag_id = str(row[2]) if row[2] is not None else None
                    previous_state = row[3]
                    retry_count = int(row[4])
                    retry_limit = int(row[5])
                    retry_delay = int(row[6] or 0)
                    retry_backoff = bool(row[7])
                    previous_run_owner = row[9]
                    previous_run_attempt_id = str(row[10])
                    reason = {
                        "recovery": {
                            "reason_code": "RUN_LEASE_EXPIRED",
                            "previous_run_owner": previous_run_owner,
                            "previous_run_attempt_id": previous_run_attempt_id,
                        }
                    }

                    if retry_count < retry_limit:
                        start_after = self._recovery_start_after(
                            retry_delay=retry_delay,
                            retry_backoff=retry_backoff,
                            retry_count=retry_count,
                        )
                        cursor.execute(
                            f"""
                            UPDATE {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                            SET state                = 'retry',
                                start_after          = %s,
                                completed_on         = NULL,
                                output               = %s,
                                lease_owner          = NULL,
                                lease_expires_at     = NULL,
                                run_owner            = NULL,
                                run_attempt_id       = NULL,
                                run_lease_expires_at = NULL
                            WHERE id = %s
                              AND state = 'active'
                              AND run_owner = %s
                              AND run_attempt_id = %s::uuid
                              AND run_lease_expires_at <= now()
                            RETURNING id
                            """,
                            (
                                start_after,
                                Jsonb(reason),
                                job_id,
                                previous_run_owner,
                                previous_run_attempt_id,
                            ),
                        )
                        if cursor.fetchone() is None:
                            continue
                        recovered.append(
                            RecoveredRunLease(
                                id=job_id,
                                name=name,
                                previous_state=previous_state,
                                recovered_state="retry",
                                dag_id=dag_id,
                                retry_count=retry_count,
                                retry_limit=retry_limit,
                                start_after=start_after,
                                previous_run_owner=previous_run_owner,
                                previous_run_attempt_id=previous_run_attempt_id,
                            )
                        )
                    else:
                        cursor.execute(
                            f"""
                            UPDATE {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                            SET state                = 'failed',
                                completed_on         = now(),
                                output               = %s,
                                lease_owner          = NULL,
                                lease_expires_at     = NULL,
                                run_lease_expires_at = NULL
                            WHERE id = %s
                              AND state = 'active'
                              AND run_owner = %s
                              AND run_attempt_id = %s::uuid
                              AND run_lease_expires_at <= now()
                            RETURNING id
                            """,
                            (
                                Jsonb(reason),
                                job_id,
                                previous_run_owner,
                                previous_run_attempt_id,
                            ),
                        )
                        if cursor.fetchone() is None:
                            continue
                        recovered.append(
                            RecoveredRunLease(
                                id=job_id,
                                name=name,
                                previous_state=previous_state,
                                recovered_state="failed",
                                dag_id=dag_id,
                                retry_count=retry_count,
                                retry_limit=retry_limit,
                                start_after=None,
                                previous_run_owner=previous_run_owner,
                                previous_run_attempt_id=previous_run_attempt_id,
                            )
                        )

                conn.commit()
                return recovered
            except Exception:
                if conn:
                    conn.rollback()
                raise
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _recover_sync)

    # ==================== DAG Operations ====================

    async def get_dag_by_id(self, dag_id: str) -> Optional[QueryPlan]:
        """
        Retrieve a DAG by its ID.

        :param dag_id: DAG ID
        :return: QueryPlan object if found, None otherwise
        """

        def db_call() -> Optional[QueryPlan]:
            conn = None
            try:
                conn = self._get_connection()
                query = load_dag(DEFAULT_SCHEMA, dag_id)
                result = self._execute_sql_gracefully(query, connection=conn)

                if result and len(result) > 0:
                    row = result[0]
                    serialized_dag = row[0]  # JSON data (single column from query)
                    return QueryPlan.model_validate(serialized_dag)
                return None
            except Exception as error:
                self.logger.error(f"Error getting DAG '{dag_id}': {error}")
                traceback.print_exc()
                return None
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def create_dag_with_jobs(
        self,
        dag_id: str,
        plan: QueryPlan,
        dag_nodes: List[WorkInfo],
        work_info: WorkInfo,
    ) -> Tuple[bool, str]:
        """
        Create a DAG and its associated jobs in a single transaction.
        This is the primary method for DAG creation during job submission.

        :param dag_id: Unique identifier for the DAG
        :param plan: The query plan representing the DAG structure
        :param dag_nodes: List of WorkInfo objects representing individual jobs in the DAG
        :param work_info: Original WorkInfo object containing DAG-level metadata (SLA, planner)
        :return: Tuple of (new_key_added: bool, new_dag_key: str)
        """

        def db_call() -> Tuple[bool, str]:
            dag_name = f"{dag_id}_dag"
            connection = None
            cursor = None
            new_key_added = False
            try:
                json_serialized_dag = plan.model_dump()
                planner_name = work_info.data.get("metadata", {}).get("planner", None)
                search_documents = build_job_search_documents(
                    plan=plan,
                    dag_nodes=dag_nodes,
                    planner=planner_name,
                )

                connection = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    insert_dag(
                        DEFAULT_SCHEMA,
                        dag_id,
                        dag_name,
                        json_serialized_dag,
                        work_info.soft_sla,
                        work_info.hard_sla,
                        planner_name,
                    ),
                    connection=connection,
                    return_cursor=True,
                    commit=False,
                )
                # Check if DAG was inserted (ON CONFLICT DO NOTHING returns no rows if exists)
                result = cursor.fetchone() if cursor else None
                new_dag_key = result[0] if result else None
                # new_key_added indicates if this is a new submission (DAG was created)
                new_key_added = result is not None
                self._close_cursor(cursor)

                # Insert all jobs in the DAG
                for dag_work_info in dag_nodes:
                    cursor = self._execute_sql_gracefully(
                        insert_job(DEFAULT_SCHEMA, dag_work_info),
                        connection=connection,
                        return_cursor=True,
                        commit=False,
                    )
                    self._close_cursor(cursor)

                for search_document in search_documents:
                    cursor = self._execute_sql_gracefully(
                        insert_job_search_document(DEFAULT_SCHEMA, search_document),
                        connection=connection,
                        return_cursor=True,
                        commit=False,
                    )
                    self._close_cursor(cursor)

                connection.commit()
                return new_key_added, new_dag_key
            except (Exception, psycopg.Error) as error:
                if connection:
                    connection.rollback()
                raise ValueError(f"Job creation failed for {dag_id}: {error}")
            finally:
                self._close_cursor(cursor)
                self._close_connection(connection)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def create_dag(self, dag: QueryPlan, jobs: List[WorkInfo]) -> bool:
        """
        Create a DAG and its associated jobs.
        DEPRECATED: Use create_dag_with_jobs for new code.

        :param dag: QueryPlan DAG definition
        :param jobs: List of jobs belonging to this DAG
        :return: True if successful, False otherwise
        """

        def db_call():
            conn = None
            try:
                conn = self._get_connection()

                # Insert DAG - extract DAG metadata from first job
                if not jobs:
                    raise ValueError("Cannot create DAG without jobs")

                first_job = jobs[0]
                dag_id = first_job.dag_id
                if not dag_id:
                    raise ValueError("Jobs must have dag_id set")

                dag_name = f"{dag_id}_dag"
                serialized_dag = dag.model_dump()

                # Extract planner name from first job metadata
                planner_name = first_job.data.get("metadata", {}).get("planner", None)

                # Use SLA from the first job (representing DAG-level SLA)
                dag_query = insert_dag(
                    DEFAULT_SCHEMA,
                    dag_id,
                    dag_name,
                    serialized_dag,
                    first_job.soft_sla,
                    first_job.hard_sla,
                    planner_name,
                )
                self._execute_sql_gracefully(dag_query, connection=conn)

                # Insert jobs
                for job in jobs:
                    job_query = insert_job(DEFAULT_SCHEMA, job)
                    self._execute_sql_gracefully(job_query, connection=conn)

                return True
            except Exception as error:
                self.logger.error(f"Error creating DAG: {error}")
                traceback.print_exc()
                return False
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def mark_dag_as_active(self, dag_id: str) -> bool:
        """
        Mark a DAG as active.

        :param dag_id: DAG ID
        :return: True if successful, False otherwise
        """

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                query = mark_as_active_dags(DEFAULT_SCHEMA, [dag_id])
                result = self._execute_sql_gracefully(query, connection=conn)
                return len(result) > 0 if result else False
            except Exception as error:
                self.logger.error(f"Error marking DAG as active: {error}")
                raise error
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def resolve_dag_state(self, dag_id: str) -> Optional[str]:
        """
        Resolve the final state of a DAG by calling the database function.
        This function determines if all jobs in the DAG are complete and what the final state is.

        :param dag_id: DAG ID to resolve
        :return: Final DAG state ('completed', 'failed', or None if still in progress)
        """

        def db_call() -> Optional[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {DEFAULT_SCHEMA}.resolve_dag_state(%s::uuid);",
                    (dag_id,),
                )
                result = cursor.fetchone()
                conn.commit()
                return result[0] if result else None
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error resolving DAG state for {dag_id}: {error}")
                if conn:
                    conn.rollback()
                raise
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def get_active_dag_ids(self, dag_ids: List[str]) -> Set[str]:
        """
        Validates which DAGs from the provided list are still active in the database.
        This is used for syncing in-memory DAG cache with the database state.

        :param dag_ids: List of DAG IDs to validate
        :return: Set of DAG IDs that are still active in database
        """
        if not dag_ids:
            return set()

        def db_call() -> Set[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Build parameterized query
                placeholders = ','.join(['%s'] * len(dag_ids))
                query = f"""
                    SELECT id FROM {DEFAULT_SCHEMA}.dag
                    WHERE id IN ({placeholders}) AND state = 'active'
                """
                cursor.execute(query, dag_ids)
                valid_dag_records = cursor.fetchall()
                conn.commit()

                # Return set of valid DAG IDs
                return {str(record[0]) for record in valid_dag_records}
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error validating active DAGs: {error}")
                if conn:
                    conn.rollback()
                raise
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def load_dag_and_jobs(
        self, dag_id: str
    ) -> Tuple[Optional[Dict], List[Tuple]]:
        """
        Load a DAG and its jobs from the database.

        :param dag_id: DAG ID
        :return: Tuple of (serialized_dag dict, list of job rows)
        """

        def _load():
            conn = self._get_connection()
            cur = None
            try:
                # Get the DAG
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT dag_id, serialized_dag
                    FROM marie_scheduler.hydrate_frontier_dags()
                    WHERE dag_id = %s
                    """,
                    (dag_id,),
                )
                dag_row = cur.fetchone()
                if not dag_row:
                    return None, []

                _, serialized_dag = dag_row

                # Get the jobs for this DAG
                cur.execute(
                    """
                    SELECT dag_id, job
                    FROM marie_scheduler.hydrate_frontier_jobs(ARRAY[%s]::uuid[])
                    """,
                    (dag_id,),
                )
                job_rows = cur.fetchall()

                conn.commit()
                return serialized_dag, job_rows
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                if cur and not cur.closed:
                    self._close_cursor(cur)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _load)

    async def get_job_priorities(self, job_ids: List[str]) -> Dict[str, int]:
        """
        Load current persisted priorities for the given jobs.
        """

        if not job_ids:
            return {}

        def _load() -> Dict[str, int]:
            conn = None
            cur = None
            try:
                conn = self._get_connection()
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT id, priority
                    FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                    WHERE id = ANY(%s::uuid[])
                    """,
                    (job_ids,),
                )
                rows = cur.fetchall()
                conn.commit()
                return {str(job_id): int(priority) for job_id, priority in rows}
            except Exception:
                if conn:
                    conn.rollback()
                raise
            finally:
                self._close_cursor(cur)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _load)

    # ==================== State Counting ====================

    async def count_job_states(self) -> Dict[str, Dict[str, int]]:
        """
        Count jobs by state and queue.

        :return: Dictionary of queue names to state counts
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                query = count_job_states(DEFAULT_SCHEMA)
                cursor = self._execute_sql_gracefully(
                    query, return_cursor=True, connection=conn
                )
                counts = cursor.fetchall() if cursor else []

                state_count_default = {
                    key.lower(): 0 for key in WorkState.__members__.keys()
                }
                states = {"queues": {}}

                for item in counts:
                    name, state, size = item
                    if name:
                        if name not in states["queues"]:
                            states["queues"][name] = state_count_default.copy()
                        queue = states["queues"][name]
                        queue[state or "all"] = int(size)

                # Calculate the 'all' column as sum of all state columns
                for queue in states["queues"].values():
                    queue["all"] = sum(v for k, v in queue.items() if k != "all")

                return states
            except Exception as error:
                self.logger.error(f"Error counting job states: {error}")
                return {"queues": {}}
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def count_dag_states(self) -> Dict[str, Dict[str, int]]:
        """
        Count DAGs by state.

        :return: Dictionary of DAG state counts
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                query = count_dag_states(DEFAULT_SCHEMA)
                cursor = self._execute_sql_gracefully(
                    query, return_cursor=True, connection=conn
                )
                counts = cursor.fetchall() if cursor else []

                state_count_default = {
                    key.lower(): 0 for key in WorkState.__members__.keys()
                }
                states = {"queues": {}}

                for item in counts:
                    name, state, size = item
                    if name:
                        if name not in states["queues"]:
                            states["queues"][name] = state_count_default.copy()
                        queue = states["queues"][name]
                        queue[state or "all"] = int(size)

                # Calculate the 'all' column
                for queue in states["queues"].values():
                    queue["all"] = sum(v for k, v in queue.items() if k != "all")

                return states
            except Exception as error:
                self.logger.error(f"Error counting DAG states: {error}")
                return {"queues": {}}
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    # ==================== Monitoring Operations ====================

    async def update_monitor_time(
        self, monitor_state_interval_seconds: int
    ) -> Optional[bool]:
        """
        Updates the monitoring timestamp in the version table.
        This is used to track when the scheduler last checked for expired/stale jobs.

        :param monitor_state_interval_seconds: Minimum seconds between monitor updates
        :return: True if timestamp was updated, None if skipped (interval not elapsed)
        """

        def db_call() -> Optional[bool]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Update monitored_on timestamp if interval has elapsed
                query = f"""
                    UPDATE {DEFAULT_SCHEMA}.version
                    SET monitored_on = now()
                    WHERE EXTRACT(EPOCH FROM (now() - COALESCE(monitored_on, now() - interval '1 week'))) > %s
                    RETURNING true
                """
                cursor.execute(query, (monitor_state_interval_seconds,))
                result = cursor.fetchone()
                conn.commit()
                return result[0] if result else None
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error updating monitor time: {error}")
                if conn:
                    conn.rollback()
                return None
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    # ==================== Queue Operations ====================

    async def create_queue(self, queue_name: str) -> bool:
        """
        Create a new queue.

        :param queue_name: Name of the queue to create
        :return: True if successful
        :raises RuntimeFailToStart: If queue creation fails, indicating bootstrap failure
        """

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                query = create_queue(DEFAULT_SCHEMA, queue_name, {})
                self._execute_sql_gracefully(query, connection=conn)
                return True
            except Exception as error:
                self.logger.error(f"Error creating queue '{queue_name}': {error}")
                # Re-raise as RuntimeFailToStart since this indicates a bootstrap failure
                from marie.excepts import RuntimeFailToStart

                raise RuntimeFailToStart(
                    f"Failed to create queue '{queue_name}' during bootstrap. "
                    f"This indicates the database schema may be incompatible or corrupt. "
                    f"Error: {error}"
                )
            finally:
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    # ==================== Helper Methods ====================

    def _record_to_work_info(self, record: Any) -> WorkInfo:
        """
        Convert a database record to a WorkInfo object.

        :param record: Database record tuple
        :return: WorkInfo object
        """
        (
            id_,
            name,
            priority,
            state,
            retry_limit,
            start_after,
            expire_in,
            data,
            retry_delay,
            retry_backoff,
            keep_until,
            dag_id,
            job_level,
            soft_sla,
            hard_sla,
            run_owner,
            run_attempt_id,
        ) = record

        wi = WorkInfo(
            id=str(id_),
            name=name,
            priority=priority,
            state=WorkState(state) if state else None,
            retry_limit=retry_limit,
            start_after=start_after,
            expire_in_seconds=int(expire_in.total_seconds()) if expire_in else 0,
            data=data,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            keep_until=keep_until,
            dag_id=str(dag_id) if dag_id is not None else None,
            job_level=job_level,
            soft_sla=soft_sla,
            hard_sla=hard_sla,
            run_owner=run_owner,
            run_attempt_id=str(run_attempt_id) if run_attempt_id else None,
        )
        return wi

    def create_tables(self, schema: str = DEFAULT_SCHEMA):
        """
        Create all database tables, functions, and triggers for the scheduler.

        SQL files are auto-discovered and loaded based on their numeric prefix.
        Files with 3-digit prefix (e.g., 001_schema.sql) are loaded first in sorted order.

        :param schema: The name of the schema where the tables will be created
        :return: None
        :raises RuntimeFailToStart: If table creation fails
        """
        version = 1

        # Auto-discover SQL files in schema directory
        schema_dir = __default_schema_dir__
        all_sql_paths = glob.glob(os.path.join(schema_dir, "*.sql"))

        # Separate numbered files (e.g., 001_schema.sql) from non-numbered files
        numbered_pattern = re.compile(r"^\d{3}_")
        numbered_files = []

        for path in all_sql_paths:
            filename = os.path.basename(path)
            if numbered_pattern.match(filename):
                numbered_files.append(filename)

        # Sort numbered files by their prefix
        numbered_files.sort()

        self.logger.info(f"Loading {len(numbered_files)} SQL files from {schema_dir}")

        # Load all numbered SQL files
        commands = [
            create_sql_from_file(schema, os.path.join(schema_dir, fname))
            for fname in numbered_files
        ]

        # Add version insert
        commands.append(insert_version(schema, version))

        # Load lease-related SQL files from subdirectory
        lease_dir = os.path.join(schema_dir, "lease")
        if os.path.isdir(lease_dir):
            lease_files = sorted(glob.glob(os.path.join(lease_dir, "*.sql")))
            self.logger.info(f"Loading {len(lease_files)} lease SQL files")
            commands.extend([create_sql_from_file(schema, f) for f in lease_files])

        # Add cron job initialization
        commands.append(
            create_sql_from_file(
                schema, os.path.join(__default_psql_dir__, "cron_job_init.sql")
            )
        )

        query = ";\n".join(commands)

        locked_query = f"""
           BEGIN;
           SET LOCAL statement_timeout = '30s';
           SELECT pg_try_advisory_lock(1);
           {query};
           SELECT pg_advisory_unlock(1);
           COMMIT;
           """

        # Write query to temp file for review
        tmp_path = "/tmp/marie/psql"
        os.makedirs(tmp_path, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        query_file = os.path.join(tmp_path, f"locked_query_{timestamp}.sql")

        try:
            with open(query_file, 'w') as f:
                f.write(locked_query)
            self.logger.info(f"Wrote locked query to: {query_file}")
        except Exception as e:
            self.logger.error(f"Failed to write query to file: {e}")

        conn = None
        try:
            conn = self._get_connection()
            self._execute_sql_gracefully(locked_query, connection=conn)
            self.logger.info(f"Successfully created tables in schema '{schema}'")
        except (Exception, psycopg.Error) as error:
            if isinstance(error, psycopg.errors.DuplicateTable):
                self.logger.warning("Tables already exist, skipping creation.")
            else:
                self.logger.error(f"Error creating tables: {error}")
                raise RuntimeFailToStart(
                    f"Failed to create tables in schema '{schema}': {error}"
                )
        finally:
            self._close_connection(conn)

    async def wipe(self, schema: str = DEFAULT_SCHEMA) -> None:
        """
        Clear all data from job and archive tables.

        :param schema: The schema containing the tables
        :return: None
        :raises RuntimeFailToStart: If wipe operation fails
        """
        query = f"TRUNCATE {schema}.job, {schema}.archive"
        self.logger.info(f"Wiping all data in schema '{schema}'")

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                self._execute_sql_gracefully(query, connection=conn)
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error wiping tables: {error}")
                raise RuntimeFailToStart(
                    f"Failed to wipe tables in schema '{schema}': {error}"
                )
            finally:
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, db_call)

    async def is_installed(self, schema: str = DEFAULT_SCHEMA) -> bool:
        """
        Check if the scheduler tables are installed.

        :param schema: The schema to check
        :return: True if tables are installed, False otherwise
        :raises RuntimeFailToStart: If check fails
        """

        def db_call():
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    version_table_exists(schema), return_cursor=True, connection=conn
                )

                if cursor and cursor.rowcount > 0:
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        return True
                return False
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error checking tables: {error}")
                raise RuntimeFailToStart(
                    f"Unable to check installation in schema '{schema}': {error}"
                )
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def get_defined_queues(self, schema: str = DEFAULT_SCHEMA) -> set[str]:
        """
        Get the set of defined queues from the database.

        :param schema: The schema containing the queue table
        :return: Set of queue names
        """

        def db_call():
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(f"SELECT name FROM {schema}.queue")
                rows = cursor.fetchall()
                conn.commit()
                return {row[0] for row in rows}
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error getting defined queues: {error}")
                return set()
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    # ==================== Job State Transitions ====================

    async def cancel_job(
        self, job_id: str, queue_name: str, schema: str = DEFAULT_SCHEMA
    ) -> None:
        """
        Cancel a job by its ID.

        :param job_id: The ID of the job to cancel
        :param queue_name: The name of the queue
        :param schema: The database schema (default: marie_scheduler)
        """
        self.logger.info(f"Cancelling job: {job_id}")

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                self._execute_sql_gracefully(
                    cancel_jobs(schema, queue_name, [job_id]),
                    connection=conn,
                )
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error cancelling job: {error}")
            finally:
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, db_call)

    async def cancel_pending_jobs_for_dag(
        self,
        dag_id: str,
        output_metadata: dict = None,
        schema: str = DEFAULT_SCHEMA,
    ) -> int:
        """
        Cancel all non-started jobs for a DAG.

        :param dag_id: The DAG ID to cancel pending jobs for
        :param output_metadata: Optional metadata to merge into job output
        :param schema: The database schema (default: marie_scheduler)
        :return: Number of jobs cancelled
        """
        self.logger.info(f"Cancelling pending jobs for DAG: {dag_id}")

        def db_call() -> int:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    cancel_pending_jobs_for_dag(
                        schema,
                        dag_id,
                        output_metadata or {},
                    ),
                    return_cursor=True,
                    connection=conn,
                )
                return cursor.fetchone()[0]
            except (Exception, psycopg.Error) as error:
                self.logger.error(
                    f"Error cancelling pending jobs for DAG {dag_id}: {error}"
                )
                return 0
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def resume_job(
        self, job_id: str, queue_name: str, schema: str = DEFAULT_SCHEMA
    ) -> None:
        """
        Resume a job by its ID.

        :param job_id: The ID of the job to resume
        :param queue_name: The name of the queue
        :param schema: The database schema (default: marie_scheduler)
        """
        self.logger.info(f"Resuming job: {job_id}")

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                self._execute_sql_gracefully(
                    resume_jobs(schema, queue_name, [job_id]), connection=conn
                )
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error resuming job: {error}")
            finally:
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, db_call)

    async def complete_job(
        self,
        job_id: str,
        queue_name: str,
        output_metadata: dict = None,
        force: bool = False,
        schema: str = DEFAULT_SCHEMA,
        run_owner: str | None = None,
        run_attempt_id: str | None = None,
    ) -> int:
        """
        Mark a job as completed.

        :param job_id: The ID of the job to complete
        :param queue_name: The name of the queue
        :param output_metadata: Optional metadata to store with completion
        :param force: If True, complete job regardless of current state
        :param schema: The database schema (default: marie_scheduler)
        :return: Number of jobs completed (0 or 1)
        """
        self.logger.info(f"Completing job: {job_id}")

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()

                if force:
                    query = complete_jobs_by_id(
                        schema,
                        queue_name,
                        [job_id],
                        {"on_complete": "done", **(output_metadata or {})},
                    )
                elif run_owner and run_attempt_id:
                    query = complete_jobs_by_attempt(
                        schema,
                        queue_name,
                        [job_id],
                        {"on_complete": "done", **(output_metadata or {})},
                        run_owner,
                        run_attempt_id,
                    )
                else:
                    query = complete_jobs(
                        schema,
                        queue_name,
                        [job_id],
                        {"on_complete": "done", **(output_metadata or {})},
                    )

                cursor = self._execute_sql_gracefully(
                    query,
                    return_cursor=True,
                    connection=conn,
                )
                counts = cursor.fetchone()[0]

                if counts > 0:
                    self.logger.debug(f"Completed job: {job_id} : {counts}")
                else:
                    # Job completion failed - likely state was changed (e.g., reset_all())
                    self.logger.warning(
                        f"Job {job_id} completion ignored - state was changed (likely reset). "
                        "Job will be re-executed from fresh state."
                    )
                return counts
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error completing job: {error}")
                return 0
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def fail_job(
        self,
        job_id: str,
        queue_name: str,
        output_metadata: dict = None,
        schema: str = DEFAULT_SCHEMA,
        run_owner: str | None = None,
        run_attempt_id: str | None = None,
    ) -> Tuple[int, Optional[str]]:
        """
        Mark a job as failed or for retry.

        :param job_id: The ID of the job to mark as failed
        :param queue_name: The name of the queue
        :param output_metadata: Optional metadata to store with failure
        :param schema: The database schema (default: marie_scheduler)
        :return: Tuple of (count, final_state) where final_state is 'retry' or 'failed'
        """
        self.logger.info(f"Processing job failure: {job_id}")

        def db_call() -> Tuple[int, Optional[str]]:
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                if run_owner and run_attempt_id:
                    query = fail_jobs_by_attempt(
                        schema,
                        queue_name,
                        [job_id],
                        {"on_complete": "failed", **(output_metadata or {})},
                        run_owner,
                        run_attempt_id,
                    )
                else:
                    query = fail_jobs_by_id(
                        schema,
                        queue_name,
                        [job_id],
                        {"on_complete": "failed", **(output_metadata or {})},
                    )

                cursor = self._execute_sql_gracefully(
                    query,
                    return_cursor=True,
                    connection=conn,
                )
                result = cursor.fetchone()
                count = result[0]
                final_state = result[1] if len(result) > 1 else None

                if count > 0:
                    if final_state == 'retry':
                        self.logger.info(
                            f"Job {job_id} marked for retry (will be retried)"
                        )
                    else:
                        self.logger.info(
                            f"Job {job_id} failed permanently (no retries remaining)"
                        )
                else:
                    self.logger.error(f"Failed to update job state: {job_id}")

                return (count, final_state)
            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error processing job failure: {error}")
                return (0, None)
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def mark_jobs_as_skipped(
        self,
        job_ids: list[str],
        queue_name: str,
        output_metadata: dict = None,
        schema: str = DEFAULT_SCHEMA,
    ) -> int:
        """
        Mark jobs as SKIPPED. This is used for branch paths that were not selected.

        :param job_ids: List of job IDs to mark as skipped
        :param queue_name: The name of the queue
        :param output_metadata: Optional metadata to store (e.g., skip_reason)
        :param schema: The database schema (default: marie_scheduler)
        :return: Number of jobs marked as skipped
        """
        if not job_ids:
            return 0

        self.logger.info(f"Marking {len(job_ids)} jobs as SKIPPED: {job_ids}")

        def db_call():
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Update jobs to SKIPPED state
                cursor.execute(
                    f"""
                    UPDATE {schema}.{queue_name}
                    SET state = 'skipped',
                        completed_on = NOW(),
                        output = %s
                    WHERE id = ANY(%s)
                      AND state NOT IN ('completed', 'failed', 'cancelled', 'skipped')
                    """,
                    (
                        Jsonb({"on_skip": "skipped", **(output_metadata or {})}),
                        job_ids,
                    ),
                )

                count = cursor.rowcount
                conn.commit()

                if count > 0:
                    self.logger.info(f"Marked {count} jobs as SKIPPED")
                else:
                    self.logger.warning(
                        "No jobs were marked as SKIPPED (may already be in terminal state)"
                    )

                return count

            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error marking jobs as skipped: {error}")
                if conn:
                    conn.rollback()
                return 0
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def close(self):
        """
        Close the repository and cleanup resources.
        """
        if hasattr(self, '_db_executor'):
            self._db_executor.shutdown(wait=True)
        if hasattr(self, 'postgreSQL_pool'):
            self.postgreSQL_pool.close()
