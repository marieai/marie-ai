import asyncio
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2
import psycopg2.extras
from psycopg2.extras import Json

from marie.constants import __default_psql_dir__, __default_schema_dir__
from marie.excepts import RuntimeFailToStart
from marie.logging_core.logger import MarieLogger
from marie.query_planner.base import QueryPlan
from marie.scheduler.fixtures import *  # noqa: F403
from marie.scheduler.models import WorkInfo
from marie.scheduler.repository.plans import (
    cancel_jobs,
    complete_jobs,
    complete_jobs_by_id,
    count_dag_states,
    count_job_states,
    create_queue,
    fail_jobs_by_id,
    insert_dag,
    insert_job,
    insert_version,
    load_dag,
    mark_as_active_dags,
    mark_as_active_jobs,
    resume_jobs,
    version_table_exists,
)
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
                        dag_id, job_level
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
            except (Exception, psycopg2.Error) as error:
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
                        dag_id, job_level
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
            except (Exception, psycopg2.Error) as error:
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
                        dag_id, job_level
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
            except (Exception, psycopg2.Error) as error:
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
            except (Exception, psycopg2.Error) as error:
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
                    params.append(Json(output))

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
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error updating job state: {error}")
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
                    query = complete_jobs_by_id(DEFAULT_SCHEMA, job_ids, Json(output))
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
                output = Json({"error": error_message})
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

        norm_ids = list({str(i) for i in job_ids})

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

    async def activate_from_lease(
        self, job_ids: List[str], owner: str, run_ttl_seconds: int
    ) -> set[str]:
        """
        Promote leased jobs to active once dispatch is acknowledged.

        :param job_ids: List of job IDs to activate
        :param owner: Lease owner identifier
        :param run_ttl_seconds: Run TTL in seconds
        :return: Set of successfully activated job IDs
        """
        if not job_ids:
            return set()

        def _activate_sync() -> set[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                sql = (
                    f"SELECT unnest({DEFAULT_SCHEMA}.activate_from_lease("
                    f"%s::uuid[], %s, %s::interval)) AS id"
                )
                params = (
                    job_ids,
                    owner,
                    f"{run_ttl_seconds} seconds",
                )
                cursor = self._execute_sql_gracefully(
                    sql, data=params, return_cursor=True, connection=conn
                )
                rows = cursor.fetchall() if cursor else []
                return {row[0] for row in rows}
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _activate_sync)

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
                # Extract planner name from work_info metadata
                planner_name = work_info.data.get("metadata", {}).get("planner", None)

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
                )
                # Check if DAG was inserted (ON CONFLICT DO NOTHING returns no rows if exists)
                result = cursor.fetchone() if cursor else None
                new_dag_key = result[0] if result else None
                # new_key_added indicates if this is a new submission (DAG was created)
                new_key_added = result is not None
                self._close_cursor(cursor)

                # Insert all jobs in the DAG
                for i, dag_work_info in enumerate(dag_nodes):
                    cursor = self._execute_sql_gracefully(
                        insert_job(DEFAULT_SCHEMA, dag_work_info),
                        connection=connection,
                        return_cursor=True,
                    )
                    self._close_cursor(cursor)

                connection.commit()
                return new_key_added, new_dag_key
            except (Exception, psycopg2.Error) as error:
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
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error resolving DAG state for {dag_id}: {error}")
                if conn:
                    conn.rollback()
                return None
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
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error validating active DAGs: {error}")
                if conn:
                    conn.rollback()
                return set()
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
            except (Exception, psycopg2.Error) as error:
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
        :return: True if successful, False otherwise
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
                return False
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
            dag_id=dag_id,
            job_level=job_level,
        )
        return wi

    # ==================== Schema Management ====================

    def create_tables(self, schema: str = DEFAULT_SCHEMA):
        """
        Create all database tables, functions, and triggers for the scheduler.

        :param schema: The name of the schema where the tables will be created
        :return: None
        :raises RuntimeFailToStart: If table creation fails
        """
        version = 1
        commands = [
            create_schema(schema),
            create_version_table(schema),
            create_table_queue(schema),
            create_job_state_enum(schema),
            create_job_table(schema),
            create_primary_key_job(schema),
            create_job_history_table(schema),
            create_job_update_trigger_function(schema),
            create_job_update_trigger(schema),
            clone_job_table_for_archive(schema),
            create_schedule_table(schema),
            create_subscription_table(schema),
            add_archived_on_to_archive(schema),
            add_archived_on_index_to_archive(schema),
            add_id_index_to_archive(schema),
            create_index_job_name(schema),
            create_index_job_fetch(schema),
            create_queue_function(schema),
            delete_queue_function(schema),
            insert_version(schema, version),
            create_exponential_backoff_function(schema),
            create_dag_table(schema),
            create_dag_table_history(schema),
            create_dag_history_trigger_function(schema),
        ]

        # SQL files to be loaded
        sql_files = [
            "job_dependencies.sql",
            "fetch_next_job.sql",
            "create_indexes.sql",
            "create_constraints.sql",
            "resolve_dag_state.sql",
            "count_job_states.sql",
            "count_dag_states.sql",
            "refresh_job_priority.sql",
            "delete_dags_and_jobs.sql",
            "delete_failed_dags_and_jobs.sql",
            "delete_orphaned_jobs.sql",
            "jobs_with_unmet_dependencies.sql",
            "notify_dag_state_change.sql",
            "purge_non_started_work.sql",
            "ready_jobs_view.sql",
            "refresh_dag_durations.sql",
            "refresh_job_durations.sql",
            "reset_active_dags_and_jobs.sql",
            "reset_all.sql",
            "reset_dag.sql",
            "reset_job.sql",
            "reset_completed_dags_and_jobs.sql",
            "suspend_non_started_work.sql",
            "unsuspend_work.sql",
            "sync_job_dependencies.sql",
            "lease/release_expired_leases.sql",
            "lease/release_lease.sql",
            "lease/activate_from_lease.sql",
            "lease/clear_all_leases.sql",
            "lease/hydrate_frontier.sql",
            "lease/hydrate_frontier_jobs.sql",
            "lease/lease_jobs_by_id.sql",
            "lease/reap_expired_leases.sql",
        ]

        commands.extend(
            [
                create_sql_from_file(
                    schema, os.path.join(__default_schema_dir__, fname)
                )
                for fname in sql_files
            ]
        )

        commands.extend(
            [
                create_sql_from_file(
                    schema, os.path.join(__default_psql_dir__, "cron_job_init.sql")
                )
            ]
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
        except (Exception, psycopg2.Error) as error:
            if isinstance(error, psycopg2.errors.DuplicateTable):
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
            except (Exception, psycopg2.Error) as error:
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
            except (Exception, psycopg2.Error) as error:
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
            except (Exception, psycopg2.Error) as error:
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
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error cancelling job: {error}")
            finally:
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, db_call)

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
            except (Exception, psycopg2.Error) as error:
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

                # Choose appropriate completion function based on force flag
                if force:
                    query = complete_jobs_by_id(
                        schema,
                        queue_name,
                        [job_id],
                        {"on_complete": "done", **(output_metadata or {})},
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
            except (Exception, psycopg2.Error) as error:
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
    ) -> int:
        """
        Mark a job as failed.

        :param job_id: The ID of the job to mark as failed
        :param queue_name: The name of the queue
        :param output_metadata: Optional metadata to store with failure
        :param schema: The database schema (default: marie_scheduler)
        :return: Number of jobs marked as failed (0 or 1)
        """
        self.logger.info(f"Marking job as failed: {job_id}")

        def db_call():
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    fail_jobs_by_id(
                        schema,
                        queue_name,
                        [job_id],
                        {"on_complete": "failed", **(output_metadata or {})},
                    ),
                    return_cursor=True,
                    connection=conn,
                )
                counts = cursor.fetchone()[0]
                if counts > 0:
                    self.logger.info(f"Completed failed job: {job_id}")
                else:
                    self.logger.error(f"Error completing failed job: {job_id}")
                return counts
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error completing failed job: {error}")
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
            self.postgreSQL_pool.closeall()
