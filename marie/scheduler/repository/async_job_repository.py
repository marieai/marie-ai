from __future__ import annotations

import glob
import os
import random
import re
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import psycopg
from psycopg.rows import tuple_row
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
    insert_job_search_documents,
    insert_jobs,
    insert_version,
    load_dag,
    mark_as_active_dags,
    mark_as_active_jobs,
    resume_jobs,
)
from marie.scheduler.search_documents import build_job_search_documents
from marie.scheduler.state import WorkState
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

DEFAULT_SCHEMA = "marie_scheduler"
DEFAULT_JOB_TABLE = "job"
SCHEDULER_SCHEMA_VERSION = 72


class _GuardrailRouteConflict(RuntimeError):
    pass


def _scheduler_sql_paths() -> tuple[str, str]:
    psql_dir = os.environ.get("MARIE_PSQL_DIR", __default_psql_dir__)
    schema_dir = os.environ.get("MARIE_SCHEMA_DIR")
    if schema_dir is None:
        schema_dir = (
            os.path.join(psql_dir, "schema")
            if psql_dir != __default_psql_dir__
            else __default_schema_dir__
        )
    return psql_dir, schema_dir


class AsyncJobRepository:
    def __init__(
        self,
        config: Dict[str, Any],
        pool: AsyncPostgresConnectionPool | None = None,
    ) -> None:
        self.logger = MarieLogger(AsyncJobRepository.__name__)
        self._config = config
        self._pool = pool or AsyncPostgresConnectionPool()
        self._owns_pool = pool is None
        self._closed = False

    async def initialize(self) -> None:
        await self._pool.initialize(
            self._config,
            row_factory=tuple_row,
            autocommit=True,
        )

    async def get_job_by_id(self, job_id: str) -> Optional[WorkInfo]:
        query = f"""
            SELECT id, name, priority, state, retry_limit, start_after,
                   expire_in, data, retry_delay, retry_backoff, keep_until,
                   dag_id, job_level, soft_sla, hard_sla,
                   run_owner, run_attempt_id, branch_metadata
            FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
            WHERE id = %s
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, job_id)
        return self._record_to_work_info(row) if row else None

    async def get_job_by_policy(self, ref_type: str, ref_id: str) -> Optional[WorkInfo]:
        query = f"""
            SELECT id, name, priority, state, retry_limit, start_after,
                   expire_in, data, retry_delay, retry_backoff, keep_until,
                   dag_id, job_level, soft_sla, hard_sla,
                   run_owner, run_attempt_id, branch_metadata
            FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
            WHERE data->'metadata'->>'ref_type' = %s
              AND data->'metadata'->>'ref_id' = %s
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, ref_type, ref_id)
        return self._record_to_work_info(row) if row else None

    async def list_jobs(
        self,
        queue: Optional[str] = None,
        state: Optional[WorkState | str | Sequence[WorkState | str]] = None,
        limit: int = 1000,
        fetch_size: int = 1000,
    ) -> List[WorkInfo]:
        if fetch_size <= 0:
            raise ValueError("fetch_size must be greater than zero")
        states = (
            []
            if state is None
            else ([state] if isinstance(state, (WorkState, str)) else state)
        )
        state_values = [
            item.value if isinstance(item, WorkState) else WorkState(item.lower()).value
            for item in states
        ]
        where: list[str] = []
        params: list[Any] = []
        if queue:
            where.append("name = %s")
            params.append(queue)
        if state_values:
            where.append(f"state = ANY(%s::{DEFAULT_SCHEMA}.job_state[])")
            params.append(state_values)
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        limit_sql = "LIMIT %s" if limit > 0 else ""
        if limit > 0:
            params.append(limit)
        query = f"""
            SELECT id, name, priority, state, retry_limit, start_after,
                   expire_in, data, retry_delay, retry_backoff, keep_until,
                   dag_id, job_level, soft_sla, hard_sla,
                   run_owner, run_attempt_id, branch_metadata
            FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
            {where_sql}
            ORDER BY created_on DESC
            {limit_sql}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [self._record_to_work_info(row) for row in rows]

    async def create_job(self, work_info: WorkInfo) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    insert_jobs(DEFAULT_SCHEMA),
                    Jsonb([work_info.model_dump(mode="json")]),
                )
        except Exception as error:
            self.logger.error(f"Error creating job: {error}")
            return False
        return True

    async def delete_job(self, job_id: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                DELETE FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                WHERE id = %s::uuid
                RETURNING id
                """,
                job_id,
            )
        return row is not None

    async def update_job_state(
        self,
        job_id: str,
        state: WorkState,
        output: Optional[Dict] = None,
        started_on: Optional[datetime] = None,
        completed_on: Optional[datetime] = None,
    ) -> bool:
        fields = ["state = %s"]
        params: list[Any] = [state.value]
        if output is not None:
            fields.append("output = %s")
            params.append(Jsonb(output))
        if started_on is not None:
            fields.append("started_on = %s")
            params.append(started_on)
        if completed_on is not None:
            fields.append("completed_on = %s")
            params.append(completed_on)
        params.append(job_id)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                SET {', '.join(fields)}
                WHERE id = %s::uuid
                RETURNING id
                """,
                *params,
            )
        return row is not None

    async def update_job_metadata(
        self,
        job_id: str,
        queue_name: str,
        metadata_updates: Dict[str, Any],
    ) -> bool:
        if not metadata_updates:
            return False
        for field_name in metadata_updates:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field_name):
                raise ValueError(f"Invalid job metadata field: {field_name!r}")
        fields = [f"{field_name} = %s" for field_name in metadata_updates]
        params = [Jsonb(value) for value in metadata_updates.values()]
        params.extend((queue_name, job_id))
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                SET {', '.join(fields)}
                WHERE name = %s AND id = %s::uuid
                RETURNING id
                """,
                *params,
            )
        return row is not None

    async def mark_jobs_as_active(self, job_ids: List[str], job_name: str) -> int:
        if not job_ids:
            return 0
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                mark_as_active_jobs(DEFAULT_SCHEMA, job_name, job_ids)
            )
        return len(rows)

    async def _job_ids_by_queue(
        self, conn: Any, job_ids: Sequence[str]
    ) -> dict[str, list[str]]:
        rows = await conn.fetch(
            f"""
            SELECT name, array_agg(id)
            FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
            WHERE id = ANY(%s::uuid[])
            GROUP BY name
            """,
            list(job_ids),
        )
        return {
            str(name): [str(job_id) for job_id in queue_job_ids]
            for name, queue_job_ids in rows
        }

    async def complete_jobs(
        self, job_ids: List[str], output: Optional[Dict] = None
    ) -> int:
        if not job_ids:
            return 0
        count = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                grouped = await self._job_ids_by_queue(conn, job_ids)
                for queue_name, queue_job_ids in grouped.items():
                    value = await conn.fetchval(
                        complete_jobs_by_id(
                            DEFAULT_SCHEMA,
                            queue_name,
                            queue_job_ids,
                            output,
                        )
                    )
                    count += int(value or 0)
        return count

    async def fail_jobs(self, job_ids: List[str], error_message: str) -> int:
        if not job_ids:
            return 0
        count = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                grouped = await self._job_ids_by_queue(conn, job_ids)
                for queue_name, queue_job_ids in grouped.items():
                    row = await conn.fetchrow(
                        fail_jobs_by_id(
                            DEFAULT_SCHEMA,
                            queue_name,
                            queue_job_ids,
                            {"error": error_message},
                        )
                    )
                    count += int(row[0] if row else 0)
        return count

    async def cancel_jobs(self, job_ids: List[str]) -> int:
        if not job_ids:
            return 0
        count = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                grouped = await self._job_ids_by_queue(conn, job_ids)
                for queue_name, queue_job_ids in grouped.items():
                    value = await conn.fetchval(
                        cancel_jobs(
                            DEFAULT_SCHEMA,
                            queue_name,
                            queue_job_ids,
                        )
                    )
                    count += int(value or 0)
        return count

    async def resume_jobs(self, job_ids: List[str]) -> int:
        if not job_ids:
            return 0
        count = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                grouped = await self._job_ids_by_queue(conn, job_ids)
                for queue_name, queue_job_ids in grouped.items():
                    value = await conn.fetchval(
                        resume_jobs(
                            DEFAULT_SCHEMA,
                            queue_name,
                            queue_job_ids,
                        )
                    )
                    count += int(value or 0)
        return count

    async def get_dag_by_id(self, dag_id: str) -> Optional[QueryPlan]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(load_dag(DEFAULT_SCHEMA, dag_id))
        return QueryPlan.model_validate(row[0]) if row else None

    async def get_active_dag_ids(self, dag_ids: List[str]) -> Set[str]:
        if not dag_ids:
            return set()
        query = f"""
            SELECT id FROM {DEFAULT_SCHEMA}.dag
            WHERE id = ANY(%s::uuid[]) AND state = 'active'
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, dag_ids)
        return {str(row[0]) for row in rows}

    async def get_job_priorities(self, job_ids: List[str]) -> Dict[str, int]:
        if not job_ids:
            return {}
        query = f"""
            SELECT id, priority FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
            WHERE id = ANY(%s::uuid[])
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, job_ids)
        return {str(row[0]): int(row[1]) for row in rows}

    async def discover_hydratable_dags(self, limit: int = 0) -> List[Tuple[str, Dict]]:
        limit_sql = " LIMIT %s" if limit > 0 else ""
        params = (limit,) if limit > 0 else ()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT dag_id, serialized_dag
                FROM {DEFAULT_SCHEMA}.hydrate_frontier_dags(){limit_sql}
                """,
                *params,
            )
        return [(str(row[0]), row[1]) for row in rows]

    async def load_hydratable_jobs(self, dag_ids: List[str]) -> List[Tuple[str, Dict]]:
        if not dag_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT dag_id, job
                FROM {DEFAULT_SCHEMA}.hydrate_frontier_jobs(%s::uuid[])
                """,
                dag_ids,
            )
        return [(str(row[0]), row[1]) for row in rows]

    async def release_expired_leases(self) -> int:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(
                f"SELECT {DEFAULT_SCHEMA}.release_expired_leases()"
            )
        return int(value or 0)

    @staticmethod
    def _recovery_start_after(
        *, retry_delay: int, retry_backoff: bool, retry_count: int
    ) -> datetime:
        now = datetime.now(timezone.utc)
        if not retry_backoff:
            return now + timedelta(seconds=retry_delay)
        factor = 2 ** min(16, retry_count + 1) / 2
        delay = retry_delay * factor
        return now + timedelta(seconds=delay + delay * random.random())

    async def recover_expired_run_leases(
        self, limit: int = 1000
    ) -> list[RecoveredRunLease]:
        recovered: list[RecoveredRunLease] = []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    f"""
                    SELECT id, name, dag_id, previous_state, retry_count,
                           retry_limit, retry_delay, retry_backoff, start_after,
                           run_owner, run_attempt_id, run_lease_expires_at
                    FROM {DEFAULT_SCHEMA}.claim_expired_run_leases(%s)
                    """,
                    int(limit),
                )
                for row in rows:
                    job_id = str(row[0])
                    dag_id = str(row[2]) if row[2] is not None else None
                    retry_count = int(row[4])
                    retry_limit = int(row[5])
                    previous_owner = row[9]
                    attempt_id = str(row[10])
                    reason = Jsonb(
                        {
                            "recovery": {
                                "reason_code": "RUN_LEASE_EXPIRED",
                                "previous_run_owner": previous_owner,
                                "previous_run_attempt_id": attempt_id,
                            }
                        }
                    )
                    if retry_count < retry_limit:
                        start_after = self._recovery_start_after(
                            retry_delay=int(row[6] or 0),
                            retry_backoff=bool(row[7]),
                            retry_count=retry_count,
                        )
                        updated = await conn.fetchrow(
                            f"""
                            UPDATE {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                            SET state = 'retry', start_after = %s,
                                completed_on = NULL, output = %s,
                                lease_owner = NULL, lease_expires_at = NULL,
                                run_owner = NULL, run_attempt_id = NULL,
                                run_lease_expires_at = NULL
                            WHERE id = %s::uuid AND state = 'active'
                              AND run_owner = %s
                              AND run_attempt_id = %s::uuid
                              AND run_lease_expires_at <= now()
                            RETURNING id
                            """,
                            start_after,
                            reason,
                            job_id,
                            previous_owner,
                            attempt_id,
                        )
                        recovered_state = "retry"
                    else:
                        start_after = None
                        updated = await conn.fetchrow(
                            f"""
                            UPDATE {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                            SET state = 'failed', completed_on = now(), output = %s,
                                lease_owner = NULL, lease_expires_at = NULL,
                                run_lease_expires_at = NULL
                            WHERE id = %s::uuid AND state = 'active'
                              AND run_owner = %s
                              AND run_attempt_id = %s::uuid
                              AND run_lease_expires_at <= now()
                            RETURNING id
                            """,
                            reason,
                            job_id,
                            previous_owner,
                            attempt_id,
                        )
                        recovered_state = "failed"
                    if updated is None:
                        continue
                    await conn.execute(
                        f"""
                        UPDATE {DEFAULT_SCHEMA}.job_attempt
                        SET attempt_state = %s, recovery_at = NOW(),
                            recovery_state = %s,
                            recovery_reason = 'RUN_LEASE_EXPIRED', updated_on = NOW()
                        WHERE run_attempt_id = %s::uuid
                        """,
                        f"recovered_{recovered_state}",
                        recovered_state,
                        attempt_id,
                    )
                    recovered.append(
                        RecoveredRunLease(
                            id=job_id,
                            name=row[1],
                            previous_state=row[3],
                            recovered_state=recovered_state,
                            dag_id=dag_id,
                            retry_count=retry_count,
                            retry_limit=retry_limit,
                            start_after=start_after,
                            previous_run_owner=previous_owner,
                            previous_run_attempt_id=attempt_id,
                        )
                    )
        return recovered

    async def load_dag_and_jobs(
        self, dag_id: str
    ) -> Tuple[Optional[Dict], List[Tuple[str, Dict]]]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                dag_row = await conn.fetchrow(
                    f"""
                    SELECT dag_id, serialized_dag
                    FROM {DEFAULT_SCHEMA}.hydrate_frontier_dags()
                    WHERE dag_id = %s::uuid
                    """,
                    dag_id,
                )
                if dag_row is None:
                    return None, []
                rows = await conn.fetch(
                    f"""
                    SELECT dag_id, job
                    FROM {DEFAULT_SCHEMA}.hydrate_frontier_jobs(
                        ARRAY[%s]::uuid[]
                    )
                    """,
                    dag_id,
                )
        return dag_row[1], [(str(row[0]), row[1]) for row in rows]

    async def mark_dag_as_active(self, dag_id: str) -> bool:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(mark_as_active_dags(DEFAULT_SCHEMA, [dag_id]))
        return bool(rows)

    async def lease_jobs(
        self, job_ids: List[str], owner: str, ttl_seconds: int, job_name: str
    ) -> set[str]:
        if not job_ids:
            return set()
        query = f"""
            SELECT unnest({DEFAULT_SCHEMA}.lease_jobs_by_id(
                %s::uuid[], %s::interval, %s, %s
            ))
        """
        params = (
            list(dict.fromkeys(str(item) for item in job_ids)),
            f"{int(ttl_seconds)} seconds",
            owner,
            job_name,
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return {str(row[0]) for row in rows if row[0] is not None}

    async def extend_run_lease(
        self,
        job_ids: List[str],
        owner: str,
        run_attempt_id: str,
        extend_seconds: int,
    ) -> set[str]:
        if not job_ids:
            return set()
        query = f"""
            SELECT unnest({DEFAULT_SCHEMA}.extend_run_lease(
                %s::uuid[], %s, %s::uuid, %s::interval
            ))
        """
        params = (
            list(dict.fromkeys(str(item) for item in job_ids)),
            owner,
            run_attempt_id,
            f"{int(extend_seconds)} seconds",
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return {str(row[0]) for row in rows if row[0] is not None}

    async def release_lease(self, job_ids: List[str]) -> set[str]:
        if not job_ids:
            return set()
        query = f"SELECT unnest({DEFAULT_SCHEMA}.release_lease(%s::uuid[]))"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, job_ids)
        return {str(row[0]) for row in rows}

    async def activate_from_lease(
        self,
        job_ids: List[str],
        owner: str,
        run_ttl_seconds: int,
        gateway_instance_id: str | None = None,
    ) -> dict[str, str]:
        if not job_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT job_id, run_attempt_id
                FROM {DEFAULT_SCHEMA}.activate_from_lease(
                    %s::uuid[], %s, %s::interval, %s
                )
                """,
                job_ids,
                owner,
                f"{run_ttl_seconds} seconds",
                gateway_instance_id,
            )
        return {str(row[0]): str(row[1]) for row in rows}

    async def record_job_attempt_dispatch_started(
        self,
        *,
        job_id: str,
        job_name: str,
        dag_id: str,
        run_owner: str,
        run_attempt_id: str,
        scheduler_lease_owner: str,
        gateway_instance_id: str | None,
        executor: str | None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {DEFAULT_SCHEMA}.job_attempt (
                    run_attempt_id, job_id, job_name, dag_id, run_owner,
                    scheduler_lease_owner, gateway_instance_id, executor,
                    attempt_state, dispatch_started_at, updated_on
                )
                VALUES (%s::uuid, %s::uuid, %s, %s::uuid, %s, %s, %s, %s,
                        'dispatching', NOW(), NOW())
                ON CONFLICT (run_attempt_id) DO UPDATE
                SET executor = COALESCE(EXCLUDED.executor, job_attempt.executor),
                    gateway_instance_id = COALESCE(
                        EXCLUDED.gateway_instance_id,
                        job_attempt.gateway_instance_id
                    ),
                    scheduler_lease_owner = EXCLUDED.scheduler_lease_owner,
                    attempt_state = 'dispatching',
                    dispatch_started_at = COALESCE(
                        job_attempt.dispatch_started_at, NOW()
                    ),
                    updated_on = NOW()
                """,
                run_attempt_id,
                job_id,
                job_name,
                dag_id,
                run_owner,
                scheduler_lease_owner,
                gateway_instance_id,
                executor,
            )

    async def record_job_attempt_dispatch_result(
        self,
        *,
        run_attempt_id: str,
        confirmed: bool,
        error: str | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {DEFAULT_SCHEMA}.job_attempt
                SET attempt_state = %s,
                    dispatch_confirmed_at = CASE
                        WHEN %s THEN COALESCE(dispatch_confirmed_at, NOW())
                        ELSE dispatch_confirmed_at
                    END,
                    dispatch_error = %s,
                    updated_on = NOW()
                WHERE run_attempt_id = %s::uuid
                """,
                "dispatched" if confirmed else "dispatch_failed",
                confirmed,
                error,
                run_attempt_id,
            )

    async def record_job_attempt_terminal(
        self,
        *,
        job_id: str,
        job_name: str,
        dag_id: str,
        run_owner: str,
        run_attempt_id: str,
        scheduler_lease_owner: str,
        gateway_instance_id: str | None,
        terminal_status: str,
        terminal_work_state: str | None,
        source: str,
        accepted: bool,
        reject_reason: str | None = None,
    ) -> None:
        attempt_state = terminal_work_state or (
            "terminal_rejected" if not accepted else terminal_status.lower()
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""
                    UPDATE {DEFAULT_SCHEMA}.job_attempt
                    SET attempt_state = CASE
                            WHEN (terminal_accepted IS TRUE OR recovery_at IS NOT NULL)
                                 AND %s IS FALSE
                            THEN attempt_state ELSE %s
                        END,
                        terminal_at = COALESCE(terminal_at, NOW()),
                        terminal_status = CASE
                            WHEN terminal_accepted IS TRUE AND %s IS FALSE
                            THEN terminal_status ELSE %s
                        END,
                        terminal_work_state = CASE
                            WHEN terminal_accepted IS TRUE AND %s IS FALSE
                            THEN terminal_work_state ELSE %s
                        END,
                        terminal_source = CASE
                            WHEN terminal_accepted IS TRUE AND %s IS FALSE
                            THEN terminal_source ELSE %s
                        END,
                        terminal_gateway_instance_id = CASE
                            WHEN terminal_accepted IS TRUE AND %s IS FALSE
                            THEN terminal_gateway_instance_id ELSE %s
                        END,
                        terminal_scheduler_lease_owner = CASE
                            WHEN terminal_accepted IS TRUE AND %s IS FALSE
                            THEN terminal_scheduler_lease_owner ELSE %s
                        END,
                        terminal_accepted = CASE
                            WHEN terminal_accepted IS TRUE AND %s IS FALSE
                            THEN TRUE ELSE %s
                        END,
                        terminal_reject_reason = CASE
                            WHEN terminal_accepted IS TRUE AND %s IS FALSE
                            THEN terminal_reject_reason ELSE %s
                        END,
                        updated_on = NOW()
                    WHERE run_attempt_id = %s::uuid
                    RETURNING run_attempt_id
                    """,
                    accepted,
                    attempt_state,
                    accepted,
                    terminal_status,
                    accepted,
                    terminal_work_state,
                    accepted,
                    source,
                    accepted,
                    gateway_instance_id,
                    accepted,
                    scheduler_lease_owner,
                    accepted,
                    accepted,
                    accepted,
                    reject_reason,
                    run_attempt_id,
                )
                if row is not None:
                    return
                await conn.execute(
                    f"""
                    INSERT INTO {DEFAULT_SCHEMA}.job_attempt (
                        run_attempt_id, job_id, job_name, dag_id, run_owner,
                        scheduler_lease_owner, gateway_instance_id, attempt_state,
                        terminal_at, terminal_status, terminal_work_state,
                        terminal_source, terminal_gateway_instance_id,
                        terminal_scheduler_lease_owner, terminal_accepted,
                        terminal_reject_reason, metadata, updated_on
                    )
                    VALUES (
                        %s::uuid, %s::uuid, %s, %s::uuid, %s, %s, %s, %s,
                        NOW(), %s, %s, %s, %s, %s, %s, %s,
                        jsonb_build_object('missing_activation_audit', TRUE), NOW()
                    )
                    ON CONFLICT (run_attempt_id) DO NOTHING
                    """,
                    run_attempt_id,
                    job_id,
                    job_name,
                    dag_id,
                    run_owner,
                    scheduler_lease_owner,
                    gateway_instance_id,
                    attempt_state,
                    terminal_status,
                    terminal_work_state,
                    source,
                    gateway_instance_id,
                    scheduler_lease_owner,
                    accepted,
                    reject_reason,
                )

    async def cancel_job_attempt(
        self,
        job_id: str,
        queue_name: str,
        run_owner: str,
        run_attempt_id: str,
        schema: str = DEFAULT_SCHEMA,
    ) -> set[str]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {schema}.{DEFAULT_JOB_TABLE}
                SET completed_on = NOW(),
                    state = %s::{schema}.job_state,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    run_owner = NULL,
                    run_attempt_id = NULL,
                    run_lease_expires_at = NULL
                WHERE id = %s::uuid
                  AND name = %s
                  AND state = %s::{schema}.job_state
                  AND run_owner = %s
                  AND run_attempt_id = %s::uuid
                RETURNING id
                """,
                WorkState.CANCELLED.value,
                job_id,
                queue_name,
                WorkState.ACTIVE.value,
                run_owner,
                run_attempt_id,
            )
        return {str(row[0])} if row else set()

    async def cancel_job(
        self, job_id: str, queue_name: str, schema: str = DEFAULT_SCHEMA
    ) -> int:
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(cancel_jobs(schema, queue_name, [job_id]))
        return int(count or 0)

    async def cancel_pending_jobs_for_dag(
        self,
        dag_id: str,
        output_metadata: Optional[dict] = None,
        schema: str = DEFAULT_SCHEMA,
    ) -> int:
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                cancel_pending_jobs_for_dag(schema, dag_id, output_metadata or {})
            )
        return int(count or 0)

    async def resume_job(
        self, job_id: str, queue_name: str, schema: str = DEFAULT_SCHEMA
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.fetchval(resume_jobs(schema, queue_name, [job_id]))

    async def complete_job(
        self,
        job_id: str,
        queue_name: str,
        output_metadata: Optional[dict] = None,
        force: bool = False,
        schema: str = DEFAULT_SCHEMA,
        run_owner: str | None = None,
        run_attempt_id: str | None = None,
    ) -> int:
        output = {"on_complete": "done", **(output_metadata or {})}
        if force:
            query = complete_jobs_by_id(schema, queue_name, [job_id], output)
        elif run_owner and run_attempt_id:
            query = complete_jobs_by_attempt(
                schema,
                queue_name,
                [job_id],
                output,
                run_owner,
                run_attempt_id,
            )
        else:
            query = complete_jobs(schema, queue_name, [job_id], output)
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(query)
        return int(count or 0)

    async def fail_job(
        self,
        job_id: str,
        queue_name: str,
        output_metadata: Optional[dict] = None,
        schema: str = DEFAULT_SCHEMA,
        run_owner: str | None = None,
        run_attempt_id: str | None = None,
    ) -> Tuple[int, Optional[str]]:
        output = {"on_complete": "failed", **(output_metadata or {})}
        if run_owner and run_attempt_id:
            query = fail_jobs_by_attempt(
                schema,
                queue_name,
                [job_id],
                output,
                run_owner,
                run_attempt_id,
            )
        else:
            query = fail_jobs_by_id(schema, queue_name, [job_id], output)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query)
        if row is None:
            return 0, None
        return int(row[0]), row[1]

    async def mark_jobs_as_skipped(
        self,
        job_ids: list[str],
        queue_name: str,
        output_metadata: dict[str, Any] | None = None,
        schema: str = DEFAULT_SCHEMA,
    ) -> set[str]:
        if not job_ids:
            return set()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                UPDATE {schema}.{DEFAULT_JOB_TABLE}
                SET state = 'skipped',
                    completed_on = NOW(),
                    output = %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE name = %s
                  AND id = ANY(%s::uuid[])
                  AND state IN ('created', 'retry')
                RETURNING id
                """,
                Jsonb({"on_skip": "skipped", **(output_metadata or {})}),
                queue_name,
                job_ids,
            )
        skipped_ids = {str(row[0]) for row in rows}
        missing_ids = set(job_ids) - skipped_ids
        if missing_ids:
            self.logger.warning(
                f"Some jobs were not committed as skipped: {sorted(missing_ids)}"
            )
        return skipped_ids

    async def resolve_dag_state(self, dag_id: str) -> Optional[str]:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(
                f"SELECT {DEFAULT_SCHEMA}.resolve_dag_state(%s::uuid)", dag_id
            )
        return str(value) if value is not None else None

    async def update_monitor_time(
        self, monitor_state_interval_seconds: int
    ) -> Optional[bool]:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(
                f"""
                UPDATE {DEFAULT_SCHEMA}.version
                SET monitored_on = NOW()
                WHERE EXTRACT(
                    EPOCH FROM (
                        NOW() - COALESCE(monitored_on, NOW() - interval '1 week')
                    )
                ) > %s
                RETURNING TRUE
                """,
                monitor_state_interval_seconds,
            )
        return bool(value) if value is not None else None

    async def count_job_states(self) -> Dict[str, Dict[str, int]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(count_job_states(DEFAULT_SCHEMA))
        return self._format_state_counts(rows)

    async def count_dag_states(self) -> Dict[str, Dict[str, int]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(count_dag_states(DEFAULT_SCHEMA))
        return self._format_state_counts(rows)

    @staticmethod
    def _format_state_counts(rows: Sequence[Any]) -> Dict[str, Dict[str, int]]:
        state_defaults = {state.value: 0 for state in WorkState}
        queues: dict[str, dict[str, int]] = {}
        for name, state, size in rows:
            if not name:
                continue
            queue = queues.setdefault(str(name), state_defaults.copy())
            queue[state or "all"] = int(size)
        for queue in queues.values():
            queue["all"] = sum(
                count for state, count in queue.items() if state != "all"
            )
        return {"queues": queues}

    async def create_dag_with_jobs(
        self,
        dag_id: str,
        plan: QueryPlan,
        dag_nodes: List[WorkInfo],
        work_info: WorkInfo,
    ) -> Tuple[bool, Optional[str]]:
        dag_name = f"{dag_id}_dag"
        planner = work_info.data.get("metadata", {}).get("planner")
        search_documents = build_job_search_documents(
            plan=plan, dag_nodes=dag_nodes, planner=planner
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    insert_dag(
                        DEFAULT_SCHEMA,
                        dag_id,
                        dag_name,
                        plan.model_dump(),
                        work_info.soft_sla,
                        work_info.hard_sla,
                        planner,
                    )
                )
                if row is None:
                    return False, None
                await conn.execute(
                    insert_jobs(DEFAULT_SCHEMA),
                    Jsonb([node.model_dump(mode="json") for node in dag_nodes]),
                )
                await conn.execute(
                    insert_job_search_documents(DEFAULT_SCHEMA),
                    Jsonb([asdict(document) for document in search_documents]),
                )
        return True, str(row[0])

    async def create_dag(self, dag: QueryPlan, jobs: List[WorkInfo]) -> bool:
        """Create a DAG and jobs through the transactional submission path."""
        if not jobs:
            raise ValueError("Cannot create DAG without jobs")
        first_job = jobs[0]
        if not first_job.dag_id:
            raise ValueError("Jobs must have dag_id set")
        created, _ = await self.create_dag_with_jobs(
            str(first_job.dag_id), dag, jobs, first_job
        )
        return created

    async def diagnose_dag_activation_failure(self, dag_id: str) -> Dict[str, Any]:
        query = f"""
            SELECT d.state, d.started_on, d.completed_on,
                   COUNT(j.id),
                   COUNT(j.id) FILTER (WHERE j.state IN ('created', 'retry')),
                   COUNT(j.id) FILTER (WHERE j.state = 'active')
            FROM {DEFAULT_SCHEMA}.dag d
            LEFT JOIN {DEFAULT_SCHEMA}.job j ON j.dag_id = d.id
            WHERE d.id = %s::uuid
            GROUP BY d.id, d.state, d.started_on, d.completed_on
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, dag_id)
            if row is None:
                return {"reason": "dag_missing", "dag_id": dag_id}
            state_rows = await conn.fetch(
                f"""
                SELECT state::text, COUNT(*)
                FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE}
                WHERE dag_id = %s::uuid
                GROUP BY state
                ORDER BY state
                """,
                dag_id,
            )
            blocker_rows = await conn.fetch(
                f"""
                SELECT j.id, j.name, j.state::text, j.retry_count, j.retry_limit,
                       j.completed_on, j.output, j.run_attempt_id,
                       attempt.attempt_state, attempt.dispatch_error,
                       attempt.terminal_source, attempt.terminal_work_state,
                       attempt.recovery_reason, attempt.gateway_instance_id,
                       attempt.executor, attempt.terminal_gateway_instance_id,
                       attempt.terminal_reject_reason
                FROM {DEFAULT_SCHEMA}.{DEFAULT_JOB_TABLE} j
                LEFT JOIN LATERAL (
                    SELECT a.attempt_state, a.dispatch_error,
                           a.terminal_source, a.terminal_work_state,
                           a.recovery_reason, a.gateway_instance_id,
                           a.executor, a.terminal_gateway_instance_id,
                           a.terminal_reject_reason
                    FROM {DEFAULT_SCHEMA}.job_attempt a
                    WHERE a.job_id = j.id
                    ORDER BY a.updated_on DESC
                    LIMIT 1
                ) attempt ON TRUE
                WHERE j.dag_id = %s::uuid
                  AND j.state::text IN ('failed', 'expired', 'cancelled')
                  AND (
                      j.state::text <> 'cancelled'
                      OR j.output->>'cancel_reason' IS DISTINCT FROM 'dag_failed'
                  )
                ORDER BY j.completed_on NULLS LAST, j.id
                """,
                dag_id,
            )
            historical_blocker_rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (h.id)
                       h.id, h.name, h.state::text, h.retry_count, h.retry_limit,
                       h.completed_on, h.output, h.run_attempt_id,
                       h.history_created_on
                FROM {DEFAULT_SCHEMA}.job_history h
                WHERE h.dag_id = %s::uuid
                  AND h.state::text IN ('failed', 'expired', 'cancelled')
                  AND (
                      h.state::text <> 'cancelled'
                      OR h.output->>'cancel_reason' IS DISTINCT FROM 'dag_failed'
                  )
                ORDER BY h.id, h.history_created_on DESC
                """,
                dag_id,
            )
            history_rows = await conn.fetch(
                f"""
                SELECT state, history_created_on
                FROM {DEFAULT_SCHEMA}.dag_history
                WHERE id = %s::uuid
                ORDER BY history_created_on DESC
                LIMIT 5
                """,
                dag_id,
            )
        state, started_on, completed_on, total, hydratable, active = row
        if state not in {WorkState.CREATED.value, WorkState.ACTIVE.value}:
            reason = "dag_state_not_activatable"
        elif int(hydratable or 0) == 0:
            reason = "no_hydratable_jobs"
        else:
            reason = "activation_update_returned_zero_rows"
        blocking_jobs = [
            {
                "job_id": str(blocker[0]),
                "queue": blocker[1],
                "state": blocker[2],
                "retry_count": int(blocker[3]),
                "retry_limit": int(blocker[4]),
                "completed_on": blocker[5],
                "output": blocker[6],
                "run_attempt_id": str(blocker[7]) if blocker[7] else None,
                "attempt_state": blocker[8],
                "dispatch_error": blocker[9],
                "terminal_source": blocker[10],
                "terminal_work_state": blocker[11],
                "recovery_reason": blocker[12],
                "gateway_instance_id": blocker[13],
                "executor": blocker[14],
                "terminal_gateway_instance_id": blocker[15],
                "terminal_reject_reason": blocker[16],
            }
            for blocker in blocker_rows
        ]
        historical_blocking_jobs = [
            {
                "job_id": str(blocker[0]),
                "queue": blocker[1],
                "state": blocker[2],
                "retry_count": int(blocker[3]),
                "retry_limit": int(blocker[4]),
                "completed_on": blocker[5],
                "output": blocker[6],
                "run_attempt_id": str(blocker[7]) if blocker[7] else None,
                "changed_on": blocker[8],
            }
            for blocker in historical_blocker_rows
        ]
        return {
            "reason": reason,
            "dag_id": dag_id,
            "dag_state": state,
            "started_on": started_on,
            "completed_on": completed_on,
            "total_jobs": int(total or 0),
            "hydratable_jobs": int(hydratable or 0),
            "active_jobs": int(active or 0),
            "job_states": {str(item[0]): int(item[1]) for item in state_rows},
            "blocking_jobs": blocking_jobs,
            "historical_blocking_jobs": historical_blocking_jobs,
            "dag_state_history": [
                {"state": item[0], "changed_on": item[1]} for item in history_rows
            ],
        }

    async def get_guardrail_report_decision(
        self,
        *,
        job_id: str,
        run_attempt_id: str,
        schema: str = DEFAULT_SCHEMA,
    ) -> Optional[Dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT
                    asset_key,
                    asset_version,
                    partition_key,
                    metadata->>'outcome',
                    metadata->>'evaluated_at'
                FROM {schema}.asset_materialization
                WHERE asset_key = %s
                  AND job_id = %s::uuid
                  AND node_task_id = %s
                  AND metadata->>'schema' = 'marie.guardrail-report/v1'
                  AND metadata->>'run_attempt_id' = %s
                """,
                f"guardrail/report/{job_id}",
                job_id,
                job_id,
                run_attempt_id,
            )
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError(
                f"Guardrail attempt {run_attempt_id} produced conflicting reports"
            )
        asset_key, version, partition_key, outcome, evaluated_at = rows[0]
        if outcome not in {"VALID", "INVALID"}:
            raise ValueError(f"Guardrail report has invalid outcome: {outcome!r}")
        return {
            "outcome": outcome,
            "evaluated_at": evaluated_at,
            "report_asset": {
                "asset_key": asset_key,
                "asset_version": version,
                "partition_key": partition_key,
            },
        }

    async def commit_guardrail_route(
        self,
        *,
        job_id: str,
        queue_name: str,
        run_owner: str,
        run_attempt_id: str,
        branch_metadata: Dict[str, Any],
        skipped_job_ids: list[str],
        schema: str = DEFAULT_SCHEMA,
    ) -> Tuple[bool, Set[str], Optional[str]]:
        try:
            return await self._commit_guardrail_route_transaction(
                job_id=job_id,
                queue_name=queue_name,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                branch_metadata=branch_metadata,
                skipped_job_ids=skipped_job_ids,
                schema=schema,
            )
        except _GuardrailRouteConflict as error:
            return False, set(), str(error)

    async def _commit_guardrail_route_transaction(
        self,
        *,
        job_id: str,
        queue_name: str,
        run_owner: str,
        run_attempt_id: str,
        branch_metadata: Dict[str, Any],
        skipped_job_ids: list[str],
        schema: str = DEFAULT_SCHEMA,
    ) -> Tuple[bool, Set[str], Optional[str]]:
        report_asset = branch_metadata["report_asset"]
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                report = await conn.fetchrow(
                    f"""
                    SELECT 1
                    FROM {schema}.asset_materialization
                    WHERE asset_key = %s
                      AND asset_version = %s
                      AND partition_key IS NOT DISTINCT FROM %s
                      AND job_id = %s::uuid
                      AND node_task_id = %s
                      AND metadata->>'schema' = 'marie.guardrail-report/v1'
                      AND metadata->>'outcome' = %s
                      AND metadata->>'evaluated_at' = %s
                      AND metadata->>'run_attempt_id' = %s
                    LIMIT 1
                    """,
                    report_asset["asset_key"],
                    report_asset["asset_version"],
                    report_asset.get("partition_key"),
                    job_id,
                    job_id,
                    branch_metadata["outcome"],
                    branch_metadata["evaluated_at"],
                    run_attempt_id,
                )
                if report is None:
                    return False, set(), "report_asset_not_materialized"

                completed = await conn.fetchrow(
                    f"""
                    UPDATE {schema}.{DEFAULT_JOB_TABLE}
                    SET completed_on = NOW(),
                        state = 'completed',
                        output = %s,
                        branch_metadata = %s,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        run_owner = NULL,
                        run_attempt_id = NULL,
                        run_lease_expires_at = NULL
                    WHERE name = %s
                      AND id = %s::uuid
                      AND state = 'active'
                      AND run_owner = %s
                      AND run_attempt_id = %s::uuid
                    RETURNING id
                    """,
                    Jsonb({"on_complete": "done"}),
                    Jsonb(branch_metadata),
                    queue_name,
                    job_id,
                    run_owner,
                    run_attempt_id,
                )
                if completed is None:
                    return False, set(), "stale_attempt"

                skipped_ids: set[str] = set()
                if skipped_job_ids:
                    rows = await conn.fetch(
                        f"""
                        UPDATE {schema}.{DEFAULT_JOB_TABLE}
                        SET completed_on = NOW(),
                            state = 'skipped',
                            output = %s,
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            run_owner = NULL,
                            run_attempt_id = NULL,
                            run_lease_expires_at = NULL
                        WHERE name = %s
                          AND id = ANY(%s::uuid[])
                          AND state IN ('created', 'retry')
                        RETURNING id
                        """,
                        Jsonb(
                            {
                                "on_skip": "skipped",
                                "skip_reason": {
                                    "guardrail_node_id": job_id,
                                    "selected_path_ids": branch_metadata[
                                        "selected_path_ids"
                                    ],
                                },
                            }
                        ),
                        queue_name,
                        skipped_job_ids,
                    )
                    skipped_ids = {str(row[0]) for row in rows}
                    if skipped_ids != set(skipped_job_ids):
                        raise _GuardrailRouteConflict("skip_state_conflict")
        return True, skipped_ids, None

    async def is_installed(self, schema: str = DEFAULT_SCHEMA) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT
                    to_regclass('{schema}.version'),
                    to_regclass('{schema}.sensor_tick'),
                    to_regclass('{schema}.job_attempt'),
                    to_regclass('{schema}.llm_queue_fabric_config')
                """
            )
            if not row or not all(row):
                return False
            return bool(
                await conn.fetchval(
                    f"SELECT EXISTS (SELECT 1 FROM {schema}.version WHERE version = %s)",
                    SCHEDULER_SCHEMA_VERSION,
                )
            )

    async def get_defined_queues(self, schema: str = DEFAULT_SCHEMA) -> set[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT name FROM {schema}.queue")
        return {str(row[0]) for row in rows}

    async def create_queue(self, queue_name: str) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(create_queue(DEFAULT_SCHEMA, queue_name, {}))
        except Exception as error:
            raise RuntimeFailToStart(
                f"Failed to create queue '{queue_name}' during bootstrap: {error}"
            ) from error
        return True

    async def create_tables(self, schema: str = DEFAULT_SCHEMA) -> None:
        psql_dir, schema_dir = _scheduler_sql_paths()
        numbered = sorted(
            os.path.basename(path)
            for path in glob.glob(os.path.join(schema_dir, "*.sql"))
            if re.match(r"^\d{3}_", os.path.basename(path))
        )
        migrations = [
            (name, create_sql_from_file(schema, os.path.join(schema_dir, name)))
            for name in numbered
        ]
        lease_dir = os.path.join(schema_dir, "lease")
        migrations.extend(
            (os.path.relpath(path, schema_dir), create_sql_from_file(schema, path))
            for path in sorted(glob.glob(os.path.join(lease_dir, "*.sql")))
        )
        migrations.append(
            (
                "cron_job_init.sql",
                create_sql_from_file(
                    schema, os.path.join(psql_dir, "cron_job_init.sql")
                ),
            )
        )
        migrations.append(
            (
                "schema version",
                insert_version(schema, SCHEDULER_SCHEMA_VERSION),
            )
        )
        self.logger.info(
            f"Applying {len(numbered)} scheduler schema files from {schema_dir}"
        )
        source = "advisory lock"
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    locked = await conn.fetchval("SELECT pg_try_advisory_lock(1)")
                if not locked:
                    raise RuntimeFailToStart(
                        "Scheduler schema installation lock is already held"
                    )
                try:
                    for source, command in migrations:
                        async with conn.transaction():
                            await conn.execute("SET LOCAL statement_timeout = '30s'")
                            await conn.execute(command)
                finally:
                    async with conn.transaction():
                        await conn.fetchval("SELECT pg_advisory_unlock(1)")
        except psycopg.Error as error:
            raise RuntimeFailToStart(
                f"Failed to apply scheduler schema file '{source}' "
                f"from '{schema_dir}': {error}"
            ) from error

    async def validate_durable_scheduler_schema(
        self, schema: str = DEFAULT_SCHEMA
    ) -> None:
        checks = (
            (
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = 'job'
                      AND column_name = 'run_attempt_id'
                )
                """,
                (schema,),
                f"{schema}.job.run_attempt_id is missing",
            ),
            (
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_enum e
                    JOIN pg_type t ON t.oid = e.enumtypid
                    JOIN pg_namespace n ON n.oid = t.typnamespace
                    WHERE n.nspname = %s AND t.typname = 'job_state'
                      AND e.enumlabel = 'skipped'
                )
                """,
                (schema,),
                f"{schema}.job_state is missing value 'skipped'",
            ),
            (
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = 'job_attempt'
                )
                """,
                (schema,),
                f"{schema}.job_attempt is missing",
            ),
        )
        async with self._pool.acquire() as conn:
            for query, params, message in checks:
                if not await conn.fetchval(query, *params):
                    raise RuntimeFailToStart(message)
            activation = await conn.fetchval(
                """
                SELECT pg_get_functiondef(p.oid)
                FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = %s AND p.proname = 'activate_from_lease'
                  AND p.pronargs = 4
                ORDER BY p.oid DESC LIMIT 1
                """,
                schema,
            )
            if (
                not activation
                or "run_attempt_id" not in activation
                or "lease_owner = _run_owner" not in activation
                or "INSERT INTO" not in activation
                or ".job_attempt" not in activation
                or "_gateway_instance_id" not in activation
            ):
                raise RuntimeFailToStart(
                    f"{schema}.activate_from_lease is not attempt-audited"
                )
            extension = await conn.fetchval(
                """
                SELECT pg_get_functiondef(p.oid)
                FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = %s AND p.proname = 'extend_run_lease'
                ORDER BY p.oid DESC LIMIT 1
                """,
                schema,
            )
            if not extension or "_run_attempt_id" not in extension:
                raise RuntimeFailToStart(
                    f"{schema}.extend_run_lease is not attempt-aware"
                )
            rows = await conn.fetch(
                f"""
                SELECT id FROM {schema}.{DEFAULT_JOB_TABLE}
                WHERE state::text = 'active' AND run_attempt_id IS NULL
                LIMIT 10
                """
            )
        if rows:
            raise RuntimeFailToStart(
                "Active jobs without run_attempt_id found: "
                f"{[str(row[0]) for row in rows]}"
            )

    async def wipe(self, schema: str = DEFAULT_SCHEMA) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"TRUNCATE {schema}.job, {schema}.archive")

    def _record_to_work_info(self, record: Any) -> WorkInfo:
        if len(record) == 17:
            record = (*record, None)
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
            branch_metadata,
        ) = record
        return WorkInfo(
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
            branch_metadata=branch_metadata,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_pool:
            await self._pool.close()
