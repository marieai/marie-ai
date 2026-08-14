from __future__ import annotations

import glob
import os
import random
import re
import time
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
from marie.utils.scheduler_trace import scheduler_trace

DEFAULT_SCHEMA = "marie_scheduler"
SCHEDULER_SCHEMA_VERSION = 87

OPERATIONAL_JOB_ATTENTION = {
    "any",
    "queued_too_long",
    "running_too_long",
    "stale_update",
    "retrying",
    "failed",
    "terminal_mismatch",
}
OPERATIONAL_SORTS = {"attention", "newest", "oldest", "updated"}
OPERATIONAL_JOB_SORTS = OPERATIONAL_SORTS | {"timeline"}
OPERATIONAL_QUEUED_TOO_LONG_SECONDS = 300
OPERATIONAL_RUNNING_TOO_LONG_SECONDS = 900
OPERATIONAL_STALE_UPDATE_SECONDS = 600
OPERATIONAL_ATTEMPT_ATTENTION = {
    "any",
    "active_too_long",
    "stale_update",
    "recovered",
    "terminal_rejected",
    "terminal_mismatch",
    "owner_mismatch",
}
OPERATIONAL_EVENT_SEVERITIES = {"info", "warning", "bad"}


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
            FROM {DEFAULT_SCHEMA}.job
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
            FROM {DEFAULT_SCHEMA}.job
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
            FROM {DEFAULT_SCHEMA}.job
            {where_sql}
            ORDER BY created_on DESC
            {limit_sql}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [self._record_to_work_info(row) for row in rows]

    async def list_operational_jobs(
        self,
        *,
        limit: int = 25,
        offset: int = 0,
        states: Sequence[str] | None = None,
        attention: str = "any",
        queue: str | None = None,
        search: str | None = None,
        sort: str = "attention",
        dag_id: str | None = None,
    ) -> Dict[str, Any]:
        """Return a bounded, payload-free page for operational consoles."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if attention not in OPERATIONAL_JOB_ATTENTION:
            raise ValueError(f"unsupported attention preset: {attention}")
        if sort not in OPERATIONAL_JOB_SORTS:
            raise ValueError(f"unsupported sort: {sort}")

        state_values = [WorkState(state.lower()).value for state in states or []]
        query = f"""
            SELECT *
            FROM {DEFAULT_SCHEMA}.list_operational_jobs(
                %s, %s, %s::text[], %s, %s, %s, %s, %s::uuid, %s, %s, %s
            )
        """
        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            result_rows = await conn.fetch(
                query,
                limit,
                offset,
                state_values or None,
                attention,
                queue,
                search,
                sort,
                dag_id,
                OPERATIONAL_QUEUED_TOO_LONG_SECONDS,
                OPERATIONAL_RUNNING_TOO_LONG_SECONDS,
                OPERATIONAL_STALE_UPDATE_SECONDS,
            )
        total = int(result_rows[0][0]) if result_rows else 0
        queues = [str(name) for name in result_rows[0][1]] if result_rows else []
        rows = [row[2:] for row in result_rows if row[2] is not None]
        query_ms = (time.perf_counter() - started) * 1_000.0
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_seconds": OPERATIONAL_RUNNING_TOO_LONG_SECONDS,
            "query_ms": query_ms,
            "page": {
                "limit": limit,
                "offset": offset,
                "total": total,
                "has_next": offset + len(rows) < total,
            },
            "filters": {
                "states": state_values,
                "attention": attention,
                "queue": queue,
                "search": search,
                "sort": sort,
            },
            "thresholds": self._operational_thresholds(),
            "facets": {"queues": queues},
            "items": [self._operational_job_from_row(row) for row in rows],
        }

    async def get_operational_job(self, job_id: str) -> Dict[str, Any] | None:
        """Return safe lifecycle and attempt metadata for one job."""
        last_update = (
            "COALESCE(ja.updated_on, j.completed_on, j.started_on, j.created_on)"
        )
        query = f"""
            SELECT
                j.id::text,
                j.name,
                j.state::text,
                j.dag_id::text,
                d.name,
                d.planner,
                j.priority,
                j.job_level,
                j.retry_count,
                j.retry_limit,
                j.created_on,
                j.started_on,
                j.completed_on,
                {last_update},
                EXTRACT(EPOCH FROM (NOW() - j.created_on)),
                EXTRACT(EPOCH FROM (NOW() - {last_update})),
                j.run_owner,
                j.run_attempt_id::text,
                ja.executor,
                ja.activated_at,
                ja.terminal_at,
                ja.terminal_status,
                ja.terminal_work_state,
                ja.terminal_source,
                ja.terminal_accepted
            FROM {DEFAULT_SCHEMA}.job AS j
            LEFT JOIN {DEFAULT_SCHEMA}.dag AS d ON d.id = j.dag_id
            LEFT JOIN {DEFAULT_SCHEMA}.job_attempt AS ja
              ON ja.run_attempt_id = j.run_attempt_id
            WHERE j.id = %s::uuid
            LIMIT 1
        """
        history_query = f"""
            SELECT state::text, history_created_on
            FROM (
                SELECT state, history_created_on
                FROM {DEFAULT_SCHEMA}.job_history
                WHERE id = %s::uuid
                ORDER BY history_created_on DESC
                LIMIT 32
            ) AS recent
            ORDER BY history_created_on ASC
        """
        attempts_query = f"""
            SELECT
                run_attempt_id::text,
                run_owner,
                scheduler_lease_owner,
                gateway_instance_id,
                executor,
                attempt_state,
                activated_at,
                terminal_at,
                terminal_status,
                terminal_work_state,
                terminal_source,
                terminal_accepted,
                recovery_at,
                recovery_state,
                created_on,
                updated_on
            FROM {DEFAULT_SCHEMA}.job_attempt
            WHERE job_id = %s::uuid
            ORDER BY activated_at DESC
            LIMIT 10
        """
        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, job_id)
            if row is None:
                return None
            history_rows = await conn.fetch(history_query, job_id)
            attempt_rows = await conn.fetch(attempts_query, job_id)
        item = self._operational_job_from_row(row)
        lifecycle = [
            {"state": str(history[0]), "at": self._iso_timestamp(history[1])}
            for history in history_rows
        ]
        if not lifecycle or lifecycle[-1]["state"] != item["state"]:
            lifecycle.append(
                {
                    "state": item["state"],
                    "at": item["last_updated_at"],
                }
            )
        item.update(
            {
                "query_ms": (time.perf_counter() - started) * 1_000.0,
                "thresholds": self._operational_thresholds(),
                "lifecycle": lifecycle,
                "attempts": [
                    {
                        "run_attempt_id": attempt[0],
                        "run_owner": attempt[1],
                        "scheduler_lease_owner": attempt[2],
                        "gateway_instance_id": attempt[3],
                        "executor": attempt[4],
                        "state": attempt[5],
                        "activated_at": self._iso_timestamp(attempt[6]),
                        "terminal_at": self._iso_timestamp(attempt[7]),
                        "terminal_status": attempt[8],
                        "terminal_work_state": attempt[9],
                        "terminal_source": attempt[10],
                        "terminal_accepted": attempt[11],
                        "recovery_at": self._iso_timestamp(attempt[12]),
                        "recovery_state": attempt[13],
                        "created_at": self._iso_timestamp(attempt[14]),
                        "updated_at": self._iso_timestamp(attempt[15]),
                    }
                    for attempt in attempt_rows
                ],
                "output_suppressed": True,
            }
        )
        return item

    async def list_operational_execution_history(
        self,
        *,
        job_id: str | None = None,
        dag_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any] | None:
        """Return bounded worker lifecycle and structured error details."""
        if (job_id is None) == (dag_id is None):
            raise ValueError("provide exactly one of job_id or dag_id")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        query = f"""
            SELECT *
            FROM {DEFAULT_SCHEMA}.list_operational_execution_history(
                %s::uuid, %s::uuid, %s, %s
            )
        """
        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            result_rows = await conn.fetch(query, job_id, dag_id, limit, offset)
        if not result_rows:
            return None
        total = int(result_rows[0][0])
        scope_dag_id = str(result_rows[0][1]) if result_rows[0][1] else None
        rows = [row[2:] for row in result_rows if row[2] is not None]
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "query_ms": (time.perf_counter() - started) * 1_000.0,
            "scope": {
                "job_id": job_id,
                "dag_id": scope_dag_id,
            },
            "page": {
                "limit": limit,
                "offset": offset,
                "total": total,
                "has_next": offset + len(rows) < total,
            },
            "items": [
                self._operational_execution_history_from_row(row) for row in rows
            ],
            "raw_runtime_environment_suppressed": True,
            "traceback_suppressed": True,
        }

    async def list_operational_attempts(
        self,
        *,
        limit: int = 25,
        offset: int = 0,
        states: Sequence[str] | None = None,
        attention: str = "any",
        gateway: str | None = None,
        executor: str | None = None,
        search: str | None = None,
        sort: str = "attention",
    ) -> Dict[str, Any]:
        """Return a bounded, payload-free execution-attempt page."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if attention not in OPERATIONAL_ATTEMPT_ATTENTION:
            raise ValueError(f"unsupported attempt attention preset: {attention}")
        if sort not in OPERATIONAL_SORTS:
            raise ValueError(f"unsupported sort: {sort}")

        state_values = [
            state.strip().lower() for state in states or [] if state.strip()
        ]
        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            result_rows = await conn.fetch(
                f"""
                SELECT *
                FROM {DEFAULT_SCHEMA}.list_operational_attempts(
                    %s, %s, %s::text[], %s, %s, %s, %s, %s, %s, %s
                )
                """,
                limit,
                offset,
                state_values or None,
                attention,
                gateway,
                executor,
                search,
                sort,
                OPERATIONAL_RUNNING_TOO_LONG_SECONDS,
                OPERATIONAL_STALE_UPDATE_SECONDS,
            )
        total = int(result_rows[0][0]) if result_rows else 0
        gateways = [str(value) for value in result_rows[0][1]] if result_rows else []
        executors = [str(value) for value in result_rows[0][2]] if result_rows else []
        rows = [row[3:] for row in result_rows if row[3] is not None]
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "query_ms": (time.perf_counter() - started) * 1_000.0,
            "page": {
                "limit": limit,
                "offset": offset,
                "total": total,
                "has_next": offset + len(rows) < total,
            },
            "filters": {
                "states": state_values,
                "attention": attention,
                "gateway": gateway,
                "executor": executor,
                "search": search,
                "sort": sort,
            },
            "thresholds": {
                "active_too_long_seconds": OPERATIONAL_RUNNING_TOO_LONG_SECONDS,
                "stale_update_seconds": OPERATIONAL_STALE_UPDATE_SECONDS,
            },
            "facets": {"gateways": gateways, "executors": executors},
            "items": [self._operational_attempt_from_row(row) for row in rows],
        }

    async def list_operational_events(
        self,
        *,
        limit: int = 25,
        before_at: datetime | None = None,
        before_id: str | None = None,
        window_seconds: int = 900,
        severity: str | None = None,
        component: str | None = None,
        search: str | None = None,
    ) -> Dict[str, Any]:
        """Return cursor-paged scheduler lifecycle events."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not 60 <= window_seconds <= 86_400:
            raise ValueError("window_seconds must be between 60 and 86400")
        if severity is not None and severity not in OPERATIONAL_EVENT_SEVERITIES:
            raise ValueError(f"unsupported event severity: {severity}")
        if (before_at is None) != (before_id is None):
            raise ValueError("before_at and before_id must be provided together")

        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT *
                FROM {DEFAULT_SCHEMA}.list_operational_events(
                    %s, %s::timestamptz, %s, %s, %s, %s, %s
                )
                """,
                limit,
                before_at,
                before_id,
                window_seconds,
                severity,
                component,
                search,
            )
        has_next = len(rows) > limit
        page_rows = rows[:limit]
        next_before_at = self._iso_timestamp(page_rows[-1][1]) if has_next else None
        next_before_id = str(page_rows[-1][0]) if has_next else None
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_seconds": window_seconds,
            "query_ms": (time.perf_counter() - started) * 1_000.0,
            "page": {
                "limit": limit,
                "has_next": has_next,
                "next_before_at": next_before_at,
                "next_before_id": next_before_id,
            },
            "filters": {
                "severity": severity,
                "component": component,
                "search": search,
            },
            "facets": {
                "severities": sorted(OPERATIONAL_EVENT_SEVERITIES),
                "components": [
                    "scheduler.attempt",
                    "scheduler.dag",
                    "scheduler.job",
                ],
            },
            "items": [self._operational_event_from_row(row) for row in page_rows],
        }

    async def get_operational_flow(
        self,
        *,
        window_seconds: int = 900,
        queue: str | None = None,
        queue_limit: int = 25,
    ) -> Dict[str, Any]:
        """Return a bounded scheduler flow-pressure snapshot."""
        if not 60 <= window_seconds <= 86_400:
            raise ValueError("window_seconds must be between 60 and 86400")
        if not 1 <= queue_limit <= 100:
            raise ValueError("queue_limit must be between 1 and 100")

        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {DEFAULT_SCHEMA}.get_operational_flow(%s, %s, %s)",
                window_seconds,
                queue,
                queue_limit,
            )
        if row is None:
            raise RuntimeError("operational flow query returned no row")

        arrivals = int(row[1])
        terminals = int(row[5])
        ready = int(row[7])
        seconds = float(window_seconds)
        arrival_rate = arrivals / seconds
        terminal_rate = terminals / seconds
        delta_rate = arrival_rate - terminal_rate
        queues = [
            self._operational_flow_queue(item, seconds) for item in (row[16] or [])
        ]
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "observed_at": self._iso_timestamp(row[0]),
            "window_seconds": window_seconds,
            "query_ms": (time.perf_counter() - started) * 1_000.0,
            "scope": {"queue": queue},
            "rates": {
                "arrival_per_second": arrival_rate,
                "ready_per_second": int(row[2]) / seconds,
                "attempt_activation_per_second": int(row[3]) / seconds,
                "start_per_second": int(row[4]) / seconds,
                "terminal_per_second": terminal_rate,
                "failure_per_second": int(row[6]) / seconds,
                "lease_per_second": None,
                "dispatch_per_second": None,
            },
            "pressure": {
                "state": self._flow_state(delta_rate),
                "backlog_delta_per_second": delta_rate,
                "ready": ready,
                "active": int(row[8]),
                "oldest_ready_seconds": self._optional_float(row[9]),
                "drain_seconds": ready / -delta_rate
                if delta_rate < 0 and ready
                else None,
            },
            "stages": [
                self._flow_stage("ready_to_running", row[10], row[11], row[12]),
                self._flow_stage("running_to_terminal", row[13], row[14], row[15]),
            ],
            "queues": queues,
        }

    async def get_operational_database_health(self) -> Dict[str, Any]:
        """Return safe PostgreSQL activity and connection-pool pressure."""
        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {DEFAULT_SCHEMA}.get_operational_database_health()"
            )
        stats_method = getattr(self._pool, "stats", None)
        stats = stats_method() if callable(stats_method) else {}
        pool_size = int(stats.get("pool_size", 0))
        pool_available = int(stats.get("pool_available", 0))
        return {
            "reachable": True,
            "latency_ms": (time.perf_counter() - started) * 1_000.0,
            "schema_version": SCHEDULER_SCHEMA_VERSION,
            "pool": {
                "minimum": stats.get("pool_min"),
                "maximum": stats.get("pool_max"),
                "size": stats.get("pool_size"),
                "available": stats.get("pool_available"),
                "used": max(0, pool_size - pool_available),
                "waiters": stats.get("requests_waiting"),
            },
            "active_sessions": int(row[0]) if row else None,
            "blocked_sessions": int(row[1]) if row else None,
            "oldest_transaction_seconds": self._optional_float(row[2]) if row else None,
        }

    async def list_operational_dags(
        self,
        *,
        limit: int = 25,
        offset: int = 0,
        states: Sequence[str] | None = None,
        attention: str = "any",
        queue: str | None = None,
        search: str | None = None,
        sort: str = "attention",
    ) -> Dict[str, Any]:
        """Return a bounded DAG page with database-side job rollups."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if attention not in OPERATIONAL_JOB_ATTENTION:
            raise ValueError(f"unsupported attention preset: {attention}")
        if sort not in OPERATIONAL_SORTS:
            raise ValueError(f"unsupported sort: {sort}")

        state_values = [WorkState(state.lower()).value for state in states or []]
        where = []
        params: list[Any] = []
        if state_values:
            where.append("LOWER(COALESCE(d.state, 'created')) = ANY(%s::text[])")
            params.append(state_values)
        if queue:
            where.append(
                f"EXISTS (SELECT 1 FROM {DEFAULT_SCHEMA}.job AS qj "
                "WHERE qj.dag_id = d.id AND qj.name = %s)"
            )
            params.append(queue)
        if search:
            pattern = f"%{search}%"
            where.append(
                "(d.id::text ILIKE %s OR d.name ILIKE %s OR "
                "COALESCE(d.planner, '') ILIKE %s)"
            )
            params.extend([pattern] * 3)
        dag_attention = self._operational_dag_attention_sql()
        if attention != "any":
            where.append(dag_attention[attention])
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        cte = self._operational_dag_stats_cte()
        from_sql = f"""
            FROM {DEFAULT_SCHEMA}.dag AS d
            LEFT JOIN job_stats AS js ON js.dag_id = d.id
            {where_sql}
        """
        order_sql = self._operational_dag_order_sql(sort, dag_attention)
        item_query = f"""
            {cte}
            SELECT
                d.id::text,
                d.name,
                LOWER(COALESCE(d.state, 'created')),
                d.planner,
                d.priority,
                d.task_count,
                d.created_on,
                d.started_on,
                d.completed_on,
                d.updated_on,
                EXTRACT(EPOCH FROM (NOW() - d.created_on)),
                EXTRACT(EPOCH FROM (
                    NOW() - GREATEST(d.updated_on, COALESCE(js.last_updated_on, d.updated_on))
                )),
                COALESCE(js.total, 0),
                COALESCE(js.created, 0),
                COALESCE(js.retry, 0),
                COALESCE(js.active, 0),
                COALESCE(js.completed, 0),
                COALESCE(js.skipped, 0),
                COALESCE(js.expired, 0),
                COALESCE(js.cancelled, 0),
                COALESCE(js.failed, 0),
                COALESCE(js.queues, ARRAY[]::text[]),
                COALESCE(js.queued_too_long, 0),
                COALESCE(js.running_too_long, 0),
                COALESCE(js.stale_update, 0),
                COALESCE(js.retrying, 0),
                COALESCE(js.failed_attention, 0),
                COALESCE(js.terminal_mismatch, 0)
            {from_sql}
            {order_sql}
            LIMIT %s OFFSET %s
        """
        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            total = int(
                await conn.fetchval(f"{cte} SELECT COUNT(*) {from_sql}", *params) or 0
            )
            rows = await conn.fetch(item_query, *params, limit, offset)
            queue_rows = await conn.fetch(
                f"SELECT name FROM {DEFAULT_SCHEMA}.queue ORDER BY name"
            )
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_seconds": OPERATIONAL_RUNNING_TOO_LONG_SECONDS,
            "query_ms": (time.perf_counter() - started) * 1_000.0,
            "page": {
                "limit": limit,
                "offset": offset,
                "total": total,
                "has_next": offset + len(rows) < total,
            },
            "filters": {
                "states": state_values,
                "attention": attention,
                "queue": queue,
                "search": search,
                "sort": sort,
            },
            "thresholds": self._operational_thresholds(),
            "facets": {"queues": [str(row[0]) for row in queue_rows]},
            "items": [self._operational_dag_from_row(row) for row in rows],
        }

    async def get_operational_dag(
        self,
        dag_id: str,
        *,
        job_limit: int = 25,
        job_offset: int = 0,
    ) -> Dict[str, Any] | None:
        """Return safe DAG metadata plus one bounded child-job page."""
        query = f"""
            SELECT *
            FROM {DEFAULT_SCHEMA}.get_operational_dag(%s::uuid, %s, %s, %s)
        """
        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                dag_id,
                OPERATIONAL_QUEUED_TOO_LONG_SECONDS,
                OPERATIONAL_RUNNING_TOO_LONG_SECONDS,
                OPERATIONAL_STALE_UPDATE_SECONDS,
            )
            if row is None:
                return None
        jobs = await self.list_operational_jobs(
            limit=job_limit,
            offset=job_offset,
            sort="timeline",
            dag_id=dag_id,
        )
        item = self._operational_dag_from_row(row)
        item.update(
            {
                "query_ms": (time.perf_counter() - started) * 1_000.0,
                "thresholds": self._operational_thresholds(),
                "lifecycle": row[28],
                "job_page": jobs,
                "data_suppressed": True,
            }
        )
        return item

    async def get_operational_throughput(
        self,
        *,
        lookback_hours: int = 24,
        planner: str | None = None,
        planner_limit: int = 25,
        task_limit: int = 25,
    ) -> Dict[str, Any]:
        """Return bounded scheduler completion-throughput reports."""
        if not 1 <= lookback_hours <= 720:
            raise ValueError("lookback_hours must be between 1 and 720")
        if not 1 <= planner_limit <= 100:
            raise ValueError("planner_limit must be between 1 and 100")
        if not 1 <= task_limit <= 100:
            raise ValueError("task_limit must be between 1 and 100")

        planner_name = planner.strip() if planner and planner.strip() else None
        system_query = f"""
            SELECT *
            FROM {DEFAULT_SCHEMA}.monitor_system_throughput(%s, %s)
        """
        planner_query = f"""
            SELECT *
            FROM {DEFAULT_SCHEMA}.monitor_planner_throughput(%s, %s)
            WHERE period = 'window_total'
            ORDER BY executor_tasks_completed DESC, plans_completed DESC, planner
            LIMIT %s
        """
        task_query = f"""
            SELECT *
            FROM {DEFAULT_SCHEMA}.monitor_task_throughput(%s, %s)
            WHERE period = 'window_total'
            ORDER BY tasks_completed DESC, tasks_failed DESC, planner, queue_name,
                     task_name, endpoint
            LIMIT %s
        """
        started = time.perf_counter()
        async with self._pool.acquire() as conn:
            system_rows = await conn.fetch(system_query, lookback_hours, planner_name)
            planner_rows = await conn.fetch(
                planner_query, lookback_hours, planner_name, planner_limit
            )
            task_rows = await conn.fetch(
                task_query, lookback_hours, planner_name, task_limit
            )

        system = [self._system_throughput_from_row(row) for row in system_rows]
        summary = next((row for row in system if row["period"] == "window_total"), None)
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "query_ms": (time.perf_counter() - started) * 1_000.0,
            "lookback_hours": lookback_hours,
            "planner": planner_name,
            "system": {
                "summary": summary,
                "hourly": [row for row in system if row["period"] == "hour"],
            },
            "planners": [
                self._planner_throughput_from_row(row) for row in planner_rows
            ],
            "tasks": [self._task_throughput_from_row(row) for row in task_rows],
            "limits": {
                "planners": planner_limit,
                "tasks": task_limit,
            },
        }

    @classmethod
    def _system_throughput_from_row(cls, row: Sequence[Any]) -> Dict[str, Any]:
        return {
            "period": str(row[0]),
            "period_start_utc": cls._iso_timestamp(row[1]),
            "period_end_utc": cls._iso_timestamp(row[2]),
            "partial": bool(row[3]),
            "plans_submitted": int(row[4]),
            "plans_completed": int(row[5]),
            "plans_failed": int(row[6]),
            "plans_expired": int(row[7]),
            "plans_cancelled": int(row[8]),
            "plan_success_rate_pct": cls._optional_float(row[9]),
            "tasks_completed": int(row[10]),
            "executor_tasks_completed": int(row[11]),
            "tasks_failed": int(row[12]),
            "tasks_expired": int(row[13]),
            "tasks_cancelled": int(row[14]),
            "tasks_skipped": int(row[15]),
            "task_success_rate_pct": cls._optional_float(row[16]),
            "avg_completed_plans_per_hour": cls._optional_float(row[17]),
            "avg_completed_executor_tasks_per_hour": cls._optional_float(row[18]),
        }

    @classmethod
    def _planner_throughput_from_row(cls, row: Sequence[Any]) -> Dict[str, Any]:
        return {
            "period": str(row[0]),
            "bucket_start_utc": cls._iso_timestamp(row[1]),
            "planner": str(row[2]),
            "plans_submitted": int(row[3]),
            "plans_completed": int(row[4]),
            "plans_failed": int(row[5]),
            "plans_expired": int(row[6]),
            "plans_cancelled": int(row[7]),
            "executor_tasks_completed": int(row[8]),
            "tasks_failed": int(row[9]),
            "tasks_expired": int(row[10]),
            "tasks_cancelled": int(row[11]),
        }

    @classmethod
    def _task_throughput_from_row(cls, row: Sequence[Any]) -> Dict[str, Any]:
        return {
            "period": str(row[0]),
            "bucket_start_utc": cls._iso_timestamp(row[1]),
            "planner": str(row[2]),
            "queue_name": str(row[3]),
            "task_name": row[4],
            "endpoint": row[5],
            "executor_backed": bool(row[6]),
            "tasks_completed": int(row[7]),
            "tasks_failed": int(row[8]),
            "tasks_expired": int(row[9]),
            "tasks_cancelled": int(row[10]),
            "tasks_skipped": int(row[11]),
            "avg_execution_seconds": cls._optional_float(row[12]),
            "p95_execution_seconds": cls._optional_float(row[13]),
        }

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _operational_thresholds() -> Dict[str, int]:
        return {
            "queued_too_long_seconds": OPERATIONAL_QUEUED_TOO_LONG_SECONDS,
            "running_too_long_seconds": OPERATIONAL_RUNNING_TOO_LONG_SECONDS,
            "stale_update_seconds": OPERATIONAL_STALE_UPDATE_SECONDS,
        }

    @classmethod
    def _operational_job_from_row(cls, row: Sequence[Any]) -> Dict[str, Any]:
        state = str(row[2])
        age_seconds = float(row[14] or 0.0)
        last_update_age_seconds = float(row[15] or 0.0)
        started_at = cls._iso_timestamp(row[11])
        attention = []

        def add_attention(
            code: str, severity: str, message: str, age: float | None = None
        ) -> None:
            attention.append(
                {
                    "code": code,
                    "severity": severity,
                    "message": message,
                    "age_seconds": age,
                }
            )

        terminal_states = {"completed", "skipped", "failed", "expired", "cancelled"}
        if (
            row[17]
            and state in terminal_states
            and (row[24] is False or (row[22] is not None and str(row[22]) != state))
        ):
            add_attention(
                "TERMINAL_MISMATCH",
                "bad",
                "job and current attempt disagree on terminal state",
                last_update_age_seconds,
            )
        if state in {"failed", "expired", "cancelled"}:
            add_attention("FAILED", "bad", f"job is {state}", last_update_age_seconds)
        if (
            state in {"active", "retry"}
            and last_update_age_seconds > OPERATIONAL_STALE_UPDATE_SECONDS
        ):
            add_attention(
                "STALE_UPDATE",
                "bad",
                "job has not recorded a recent operational update",
                last_update_age_seconds,
            )
        if state == "active" and row[11] is not None:
            running_seconds = cls._seconds_since(row[11])
            if running_seconds > OPERATIONAL_RUNNING_TOO_LONG_SECONDS:
                add_attention(
                    "RUNNING_TOO_LONG",
                    "warning",
                    "job has exceeded the running threshold",
                    running_seconds,
                )
        if (
            state in {"created", "retry"}
            and last_update_age_seconds > OPERATIONAL_QUEUED_TOO_LONG_SECONDS
        ):
            add_attention(
                "QUEUED_TOO_LONG",
                "warning",
                "job has exceeded the queue threshold",
                last_update_age_seconds,
            )
        if state == "retry":
            add_attention("RETRYING", "warning", "job is waiting for another attempt")
        return {
            "id": str(row[0]),
            "queue": str(row[1]),
            "state": state,
            "dag_id": str(row[3]),
            "dag_name": row[4],
            "planner": row[5],
            "priority": int(row[6]),
            "job_level": int(row[7]),
            "retry_count": int(row[8]),
            "retry_limit": int(row[9]),
            "created_at": cls._iso_timestamp(row[10]),
            "started_at": started_at,
            "completed_at": cls._iso_timestamp(row[12]),
            "last_updated_at": cls._iso_timestamp(row[13]),
            "age_seconds": age_seconds,
            "last_update_age_seconds": last_update_age_seconds,
            "run_owner": row[16],
            "run_attempt_id": str(row[17]) if row[17] else None,
            "executor": row[18],
            "attempt_activated_at": cls._iso_timestamp(row[19]),
            "attempt_terminal_at": cls._iso_timestamp(row[20]),
            "terminal_status": row[21],
            "terminal_work_state": row[22],
            "terminal_source": row[23],
            "terminal_accepted": row[24],
            "attention": attention,
        }

    @classmethod
    def _operational_attempt_from_row(cls, row: Sequence[Any]) -> Dict[str, Any]:
        messages = {
            "TERMINAL_REJECTED": ("bad", "terminal update was rejected"),
            "TERMINAL_MISMATCH": ("bad", "terminal status and work state disagree"),
            "OWNER_MISMATCH": ("warning", "activation and terminal owners differ"),
            "RECOVERED": ("warning", "attempt entered recovery"),
            "ACTIVE_TOO_LONG": ("warning", "attempt exceeded the active threshold"),
            "STALE_UPDATE": ("bad", "attempt has not recorded a recent update"),
        }
        attention = [
            {
                "code": str(code),
                "severity": messages[str(code)][0],
                "message": messages[str(code)][1],
                "age_seconds": float(row[22] or 0.0),
            }
            for code in row[23] or []
        ]
        return {
            "run_attempt_id": str(row[0]),
            "job_id": str(row[1]),
            "queue": str(row[2]),
            "dag_id": str(row[3]),
            "run_owner": str(row[4]),
            "scheduler_lease_owner": str(row[5]),
            "gateway_instance_id": row[6],
            "executor": row[7],
            "state": str(row[8]),
            "activated_at": cls._iso_timestamp(row[9]),
            "terminal_at": cls._iso_timestamp(row[10]),
            "terminal_status": row[11],
            "terminal_work_state": row[12],
            "terminal_source": row[13],
            "terminal_gateway_instance_id": row[14],
            "terminal_scheduler_lease_owner": row[15],
            "terminal_accepted": row[16],
            "recovery_at": cls._iso_timestamp(row[17]),
            "recovery_state": row[18],
            "created_at": cls._iso_timestamp(row[19]),
            "updated_at": cls._iso_timestamp(row[20]),
            "age_seconds": float(row[21] or 0.0),
            "last_update_age_seconds": float(row[22] or 0.0),
            "attention": attention,
        }

    @classmethod
    def _operational_event_from_row(cls, row: Sequence[Any]) -> Dict[str, Any]:
        return {
            "event_id": str(row[0]),
            "occurred_at": cls._iso_timestamp(row[1]),
            "severity": str(row[2]),
            "component": str(row[3]),
            "code": str(row[4]),
            "affected_type": str(row[5]),
            "affected_id": str(row[6]),
            "job_id": str(row[7]) if row[7] else None,
            "dag_id": str(row[8]) if row[8] else None,
            "run_attempt_id": str(row[9]) if row[9] else None,
            "executor": row[10],
            "gateway_instance_id": row[11],
            "summary": str(row[12]),
        }

    @classmethod
    def _operational_execution_history_from_row(
        cls, row: Sequence[Any]
    ) -> Dict[str, Any]:
        return {
            "history_id": int(row[0]),
            "job_id": str(row[1]),
            "queue": str(row[2]),
            "changed_at": cls._iso_timestamp(row[3]),
            "operation": row[4],
            "status": row[5],
            "worker_message": row[6],
            "run_attempt_id": row[7],
            "executor": row[8],
            "runtime_name": row[9],
            "executor_host": row[10],
            "endpoint": row[11],
            "error": (
                {
                    "type": row[12],
                    "message": row[13],
                    "file": row[14],
                    "function": row[15],
                    "line": row[16],
                }
                if any(value is not None for value in row[12:17])
                else None
            ),
        }

    @staticmethod
    def _flow_state(delta_per_second: float) -> str:
        if delta_per_second > 0.01:
            return "growing"
        if delta_per_second < -0.01:
            return "draining"
        return "stable"

    @classmethod
    def _operational_flow_queue(
        cls, item: Dict[str, Any], window_seconds: float
    ) -> Dict[str, Any]:
        arrivals = int(item["arrivals"])
        terminals = int(item["terminals"])
        ready = int(item["ready"])
        delta = (arrivals - terminals) / window_seconds
        return {
            "name": str(item["name"]),
            "arrival_per_second": arrivals / window_seconds,
            "terminal_per_second": terminals / window_seconds,
            "failure_per_second": int(item["failures"]) / window_seconds,
            "backlog_delta_per_second": delta,
            "state": cls._flow_state(delta),
            "ready": ready,
            "active": int(item["active"]),
            "oldest_ready_seconds": cls._optional_float(
                item.get("oldest_ready_seconds")
            ),
            "drain_seconds": ready / -delta if delta < 0 and ready else None,
        }

    @classmethod
    def _flow_stage(cls, name: str, p50: Any, p95: Any, maximum: Any) -> Dict[str, Any]:
        return {
            "name": name,
            "p50_seconds": cls._optional_float(p50),
            "p95_seconds": cls._optional_float(p95),
            "max_seconds": cls._optional_float(maximum),
        }

    @staticmethod
    def _operational_dag_stats_cte() -> str:
        last_update = (
            "COALESCE(ja.updated_on, j.completed_on, j.started_on, j.created_on)"
        )
        return f"""
            WITH job_stats AS (
                SELECT
                    j.dag_id,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE j.state::text = 'created') AS created,
                    COUNT(*) FILTER (WHERE j.state::text = 'retry') AS retry,
                    COUNT(*) FILTER (WHERE j.state::text = 'active') AS active,
                    COUNT(*) FILTER (WHERE j.state::text = 'completed') AS completed,
                    COUNT(*) FILTER (WHERE j.state::text = 'skipped') AS skipped,
                    COUNT(*) FILTER (WHERE j.state::text = 'expired') AS expired,
                    COUNT(*) FILTER (WHERE j.state::text = 'cancelled') AS cancelled,
                    COUNT(*) FILTER (WHERE j.state::text = 'failed') AS failed,
                    ARRAY_AGG(DISTINCT j.name ORDER BY j.name) AS queues,
                    MAX({last_update}) AS last_updated_on,
                    COUNT(*) FILTER (WHERE
                        j.state::text IN ('created', 'retry') AND
                        EXTRACT(EPOCH FROM (NOW() - {last_update})) >
                            {OPERATIONAL_QUEUED_TOO_LONG_SECONDS}
                    ) AS queued_too_long,
                    COUNT(*) FILTER (WHERE
                        j.state::text = 'active' AND j.started_on IS NOT NULL AND
                        EXTRACT(EPOCH FROM (NOW() - j.started_on)) >
                            {OPERATIONAL_RUNNING_TOO_LONG_SECONDS}
                    ) AS running_too_long,
                    COUNT(*) FILTER (WHERE
                        j.state::text IN ('active', 'retry') AND
                        EXTRACT(EPOCH FROM (NOW() - {last_update})) >
                            {OPERATIONAL_STALE_UPDATE_SECONDS}
                    ) AS stale_update,
                    COUNT(*) FILTER (WHERE j.state::text = 'retry') AS retrying,
                    COUNT(*) FILTER (WHERE
                        j.state::text IN ('failed', 'expired', 'cancelled')
                    ) AS failed_attention,
                    COUNT(*) FILTER (WHERE
                        j.run_attempt_id IS NOT NULL AND
                        j.state::text IN (
                            'completed', 'skipped', 'failed', 'expired', 'cancelled'
                        ) AND (
                            ja.terminal_accepted IS FALSE OR (
                                ja.terminal_work_state IS NOT NULL AND
                                ja.terminal_work_state <> j.state::text
                            )
                        )
                    ) AS terminal_mismatch
                FROM {DEFAULT_SCHEMA}.job AS j
                LEFT JOIN {DEFAULT_SCHEMA}.job_attempt AS ja
                  ON ja.run_attempt_id = j.run_attempt_id
                GROUP BY j.dag_id
            )
        """

    @staticmethod
    def _operational_dag_attention_sql() -> Dict[str, str]:
        return {
            "queued_too_long": "(COALESCE(js.queued_too_long, 0) > 0)",
            "running_too_long": (
                "(COALESCE(js.running_too_long, 0) > 0 OR ("
                "LOWER(COALESCE(d.state, 'created')) = 'active' AND "
                "d.started_on IS NOT NULL AND EXTRACT(EPOCH FROM "
                f"(NOW() - d.started_on)) > {OPERATIONAL_RUNNING_TOO_LONG_SECONDS}))"
            ),
            "stale_update": (
                "(COALESCE(js.stale_update, 0) > 0 OR ("
                "LOWER(COALESCE(d.state, 'created')) IN ('active', 'retry') AND "
                "EXTRACT(EPOCH FROM (NOW() - GREATEST(d.updated_on, "
                "COALESCE(js.last_updated_on, d.updated_on)))) > "
                f"{OPERATIONAL_STALE_UPDATE_SECONDS}))"
            ),
            "retrying": (
                "(LOWER(COALESCE(d.state, 'created')) = 'retry' OR "
                "COALESCE(js.retrying, 0) > 0)"
            ),
            "failed": (
                "(LOWER(COALESCE(d.state, 'created')) IN "
                "('failed', 'expired', 'cancelled') OR "
                "COALESCE(js.failed_attention, 0) > 0)"
            ),
            "terminal_mismatch": "(COALESCE(js.terminal_mismatch, 0) > 0)",
        }

    @staticmethod
    def _operational_dag_order_sql(sort: str, attention_sql: Dict[str, str]) -> str:
        if sort == "newest":
            return "ORDER BY d.created_on DESC, d.id"
        if sort == "oldest":
            return "ORDER BY d.created_on ASC, d.id"
        if sort == "updated":
            return (
                "ORDER BY GREATEST(d.updated_on, "
                "COALESCE(js.last_updated_on, d.updated_on)) DESC, d.id"
            )
        return f"""
            ORDER BY CASE
                WHEN {attention_sql['terminal_mismatch']} THEN 0
                WHEN {attention_sql['failed']} THEN 1
                WHEN {attention_sql['stale_update']} THEN 2
                WHEN {attention_sql['running_too_long']} THEN 3
                WHEN {attention_sql['queued_too_long']} THEN 4
                WHEN {attention_sql['retrying']} THEN 5
                ELSE 6
            END,
            d.created_on DESC,
            d.id
        """

    @classmethod
    def _operational_dag_from_row(cls, row: Sequence[Any]) -> Dict[str, Any]:
        state = str(row[2])
        attention = []

        def add_attention(code: str, severity: str, count: int) -> None:
            attention.append(
                {
                    "code": code,
                    "severity": severity,
                    "count": count,
                }
            )

        if int(row[27]) > 0:
            add_attention("TERMINAL_MISMATCH", "bad", int(row[27]))
        failed_count = int(row[26])
        if state in {"failed", "expired", "cancelled"}:
            failed_count = max(1, failed_count)
        if failed_count > 0:
            add_attention("FAILED", "bad", failed_count)
        if int(row[24]) > 0:
            add_attention("STALE_UPDATE", "bad", int(row[24]))
        if int(row[23]) > 0:
            add_attention("RUNNING_TOO_LONG", "warning", int(row[23]))
        if int(row[22]) > 0:
            add_attention("QUEUED_TOO_LONG", "warning", int(row[22]))
        retry_count = int(row[25])
        if state == "retry":
            retry_count = max(1, retry_count)
        if retry_count > 0:
            add_attention("RETRYING", "warning", retry_count)
        return {
            "id": str(row[0]),
            "name": str(row[1]),
            "state": state,
            "planner": row[3],
            "priority": int(row[4]),
            "task_count": int(row[5]),
            "created_at": cls._iso_timestamp(row[6]),
            "started_at": cls._iso_timestamp(row[7]),
            "completed_at": cls._iso_timestamp(row[8]),
            "updated_at": cls._iso_timestamp(row[9]),
            "age_seconds": float(row[10] or 0.0),
            "last_update_age_seconds": float(row[11] or 0.0),
            "jobs": {
                "total": int(row[12]),
                "created": int(row[13]),
                "retry": int(row[14]),
                "active": int(row[15]),
                "completed": int(row[16]),
                "skipped": int(row[17]),
                "expired": int(row[18]),
                "cancelled": int(row[19]),
                "failed": int(row[20]),
            },
            "queues": [str(queue) for queue in row[21]],
            "attention": attention,
        }

    @staticmethod
    def _iso_timestamp(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _seconds_since(value: Any) -> float:
        if not isinstance(value, datetime):
            return 0.0
        current = datetime.now(value.tzinfo or timezone.utc)
        return max(0.0, (current - value).total_seconds())

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
                DELETE FROM {DEFAULT_SCHEMA}.job
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
                UPDATE {DEFAULT_SCHEMA}.job
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
                UPDATE {DEFAULT_SCHEMA}.job
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
            FROM {DEFAULT_SCHEMA}.job
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
            SELECT id, priority FROM {DEFAULT_SCHEMA}.job
            WHERE id = ANY(%s::uuid[])
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, job_ids)
        return {str(row[0]): int(row[1]) for row in rows}

    async def discover_admission_candidates(
        self,
        *,
        limit: int,
        sla_interval_seconds: int,
        excluded_dag_ids: Sequence[str] = (),
    ) -> List[Tuple[str, Dict]]:
        """Return eligible DAGs in durable admission order."""
        if limit <= 0:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT candidate.dag_id, candidate.serialized_dag
                FROM {DEFAULT_SCHEMA}.admission_candidate_dags(
                    %s,
                    %s,
                    %s::uuid[]
                ) WITH ORDINALITY AS candidate(
                    dag_id,
                    serialized_dag,
                    admission_rank
                )
                ORDER BY candidate.admission_rank
                """,
                limit,
                max(1, int(sla_interval_seconds)),
                list(excluded_dag_ids),
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

    async def release_expired_leases(self, limit: int = 1000) -> int:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(
                f"SELECT {DEFAULT_SCHEMA}.release_expired_leases(%s)", limit
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
                            UPDATE {DEFAULT_SCHEMA}.job
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
                            UPDATE {DEFAULT_SCHEMA}.job
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

    async def defer_leased_job(
        self,
        *,
        job_id: str,
        owner: str,
        delay_seconds: float,
    ) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {DEFAULT_SCHEMA}.job
                SET start_after = NOW() + %s::interval,
                    lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE id = %s::uuid
                  AND state IN ('created', 'retry')
                  AND lease_owner = %s
                RETURNING id
                """,
                f"{delay_seconds} seconds",
                job_id,
                owner,
            )
        return row is not None

    async def activate_from_lease(
        self,
        job_ids: List[str],
        owner: str,
        run_ttl_seconds: int,
        gateway_instance_id: str | None = None,
        run_attempt_ids: Dict[str, str] | None = None,
    ) -> dict[str, str]:
        if not job_ids:
            return {}
        if run_attempt_ids is None:
            placeholders = "%s::uuid[], %s, %s::interval, %s"
            params = (
                job_ids,
                owner,
                f"{run_ttl_seconds} seconds",
                gateway_instance_id,
            )
        else:
            placeholders = "%s::uuid[], %s::uuid[], %s, %s::interval, %s"
            params = (
                job_ids,
                [run_attempt_ids[job_id] for job_id in job_ids],
                owner,
                f"{run_ttl_seconds} seconds",
                gateway_instance_id,
            )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT job_id, run_attempt_id
                FROM {DEFAULT_SCHEMA}.activate_from_lease(
                    {placeholders}
                )
                """,
                *params,
            )
        return {str(row[0]): str(row[1]) for row in rows}

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
        operation_started = time.perf_counter()
        pool_wait_started = time.perf_counter()
        async with self._pool.acquire() as conn:
            acquired_at = time.perf_counter()
            transaction_started = time.perf_counter()
            async with conn.transaction():
                sql_started = time.perf_counter()
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
                updated_existing = row is not None
                if row is None:
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
                sql_ms = (time.perf_counter() - sql_started) * 1000.0
            transaction_ms = (time.perf_counter() - transaction_started) * 1000.0
        scheduler_trace(
            "terminal_db_operation_completed",
            operation="attempt_terminal_audit",
            job_id=job_id,
            dag_id=dag_id,
            run_attempt_id=run_attempt_id,
            accepted=accepted,
            updated_existing=updated_existing,
            pool_wait_ms=(acquired_at - pool_wait_started) * 1000.0,
            sql_ms=sql_ms,
            commit_ms=max(0.0, transaction_ms - sql_ms),
            autocommit=False,
            total_ms=(time.perf_counter() - operation_started) * 1000.0,
        )

    async def transition_job_attempt_terminal(
        self,
        *,
        job_id: str,
        queue_name: str,
        dag_id: str,
        run_owner: str,
        run_attempt_id: str,
        scheduler_lease_owner: str,
        gateway_instance_id: str | None,
        terminal_status: str,
        source: str,
        output_metadata: Optional[dict] = None,
        schema: str = DEFAULT_SCHEMA,
    ) -> tuple[bool, Optional[str]]:
        if terminal_status == "SUCCEEDED":
            transition_ctes = f"""
                transitioned AS (
                    UPDATE {schema}.job AS job
                    SET completed_on = NOW(),
                        state = %s::{schema}.job_state,
                        output = %s::jsonb,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        run_owner = NULL,
                        run_lease_expires_at = NULL
                    WHERE job.name = %s
                      AND job.id = %s::uuid
                      AND job.state = %s::{schema}.job_state
                      AND job.run_owner = %s
                      AND job.run_attempt_id = %s::uuid
                    RETURNING job.state::text AS final_state
                )
            """
            transition_params: tuple[Any, ...] = (
                WorkState.COMPLETED.value,
                Jsonb({"on_complete": "done", **(output_metadata or {})}),
                queue_name,
                job_id,
                WorkState.ACTIVE.value,
                run_owner,
                run_attempt_id,
            )
        elif terminal_status == "FAILED":
            transition_ctes = f"""
                transitioned AS (
                    UPDATE {schema}.job AS job
                    SET state = CASE
                            WHEN job.retry_count < job.retry_limit
                            THEN %s::{schema}.job_state
                            ELSE %s::{schema}.job_state
                        END,
                        completed_on = CASE
                            WHEN job.retry_count < job.retry_limit THEN NULL
                            ELSE NOW()
                        END,
                        start_after = CASE
                            WHEN job.retry_count = job.retry_limit
                            THEN job.start_after
                            WHEN NOT job.retry_backoff
                            THEN NOW() + job.retry_delay * INTERVAL '1 second'
                            ELSE {schema}.exponential_backoff(
                                job.retry_delay,
                                job.retry_count
                            )
                        END,
                        output = %s::jsonb,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        run_owner = NULL,
                        run_attempt_id = CASE
                            WHEN job.retry_count < job.retry_limit THEN NULL
                            ELSE job.run_attempt_id
                        END,
                        run_lease_expires_at = NULL
                    WHERE job.name = %s
                      AND job.id = %s::uuid
                      AND job.state = %s::{schema}.job_state
                      AND job.run_owner = %s
                      AND job.run_attempt_id = %s::uuid
                    RETURNING job.name, job.data, job.output, job.retry_limit,
                              job.keep_until, job.start_after, job.dead_letter,
                              job.state::text AS final_state
                ),
                dead_lettered AS (
                    INSERT INTO {schema}.job (
                        name,
                        data,
                        output,
                        retry_limit,
                        keep_until
                    )
                    SELECT
                        transitioned.dead_letter,
                        transitioned.data,
                        transitioned.output,
                        transitioned.retry_limit,
                        transitioned.keep_until + (
                            transitioned.keep_until - transitioned.start_after
                        )
                    FROM transitioned
                    WHERE transitioned.final_state = %s
                      AND transitioned.dead_letter IS NOT NULL
                      AND transitioned.name <> transitioned.dead_letter
                    RETURNING id
                )
            """
            transition_params = (
                WorkState.RETRY.value,
                WorkState.FAILED.value,
                Jsonb({"on_complete": "failed", **(output_metadata or {})}),
                queue_name,
                job_id,
                WorkState.ACTIVE.value,
                run_owner,
                run_attempt_id,
                WorkState.FAILED.value,
            )
        elif terminal_status == "STOPPED":
            transition_ctes = f"""
                transitioned AS (
                    UPDATE {schema}.job AS job
                    SET completed_on = NOW(),
                        state = %s::{schema}.job_state,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        run_owner = NULL,
                        run_attempt_id = NULL,
                        run_lease_expires_at = NULL
                    WHERE job.name = %s
                      AND job.id = %s::uuid
                      AND job.state = %s::{schema}.job_state
                      AND job.run_owner = %s
                      AND job.run_attempt_id = %s::uuid
                    RETURNING job.state::text AS final_state
                )
            """
            transition_params = (
                WorkState.CANCELLED.value,
                queue_name,
                job_id,
                WorkState.ACTIVE.value,
                run_owner,
                run_attempt_id,
            )
        else:
            raise ValueError(f"Unsupported terminal status: {terminal_status}")

        query = f"""
            WITH {transition_ctes},
            outcome AS (
                SELECT
                    EXISTS(SELECT 1 FROM transitioned) AS accepted,
                    (SELECT final_state FROM transitioned LIMIT 1) AS final_state
            ),
            audited AS (
                INSERT INTO {schema}.job_attempt AS existing (
                    run_attempt_id,
                    job_id,
                    job_name,
                    dag_id,
                    run_owner,
                    scheduler_lease_owner,
                    gateway_instance_id,
                    attempt_state,
                    terminal_at,
                    terminal_status,
                    terminal_work_state,
                    terminal_source,
                    terminal_gateway_instance_id,
                    terminal_scheduler_lease_owner,
                    terminal_accepted,
                    terminal_reject_reason,
                    metadata,
                    updated_on
                )
                SELECT
                    %s::uuid,
                    %s::uuid,
                    %s,
                    %s::uuid,
                    %s,
                    %s,
                    %s,
                    CASE
                        WHEN outcome.accepted THEN outcome.final_state
                        ELSE 'terminal_rejected'
                    END,
                    NOW(),
                    %s,
                    CASE
                        WHEN outcome.accepted THEN outcome.final_state
                        ELSE NULL
                    END,
                    %s,
                    %s,
                    %s,
                    outcome.accepted,
                    CASE
                        WHEN outcome.accepted THEN NULL
                        ELSE 'db_update_zero_rows'
                    END,
                    jsonb_build_object('missing_activation_audit', TRUE),
                    NOW()
                FROM outcome
                ON CONFLICT (run_attempt_id) DO UPDATE
                SET attempt_state = CASE
                        WHEN (
                            existing.terminal_accepted IS TRUE
                            OR existing.recovery_at IS NOT NULL
                        )
                        AND EXCLUDED.terminal_accepted IS FALSE
                        THEN existing.attempt_state
                        ELSE EXCLUDED.attempt_state
                    END,
                    terminal_at = COALESCE(
                        existing.terminal_at,
                        EXCLUDED.terminal_at
                    ),
                    terminal_status = CASE
                        WHEN existing.terminal_accepted IS TRUE
                         AND EXCLUDED.terminal_accepted IS FALSE
                        THEN existing.terminal_status
                        ELSE EXCLUDED.terminal_status
                    END,
                    terminal_work_state = CASE
                        WHEN existing.terminal_accepted IS TRUE
                         AND EXCLUDED.terminal_accepted IS FALSE
                        THEN existing.terminal_work_state
                        ELSE EXCLUDED.terminal_work_state
                    END,
                    terminal_source = CASE
                        WHEN existing.terminal_accepted IS TRUE
                         AND EXCLUDED.terminal_accepted IS FALSE
                        THEN existing.terminal_source
                        ELSE EXCLUDED.terminal_source
                    END,
                    terminal_gateway_instance_id = CASE
                        WHEN existing.terminal_accepted IS TRUE
                         AND EXCLUDED.terminal_accepted IS FALSE
                        THEN existing.terminal_gateway_instance_id
                        ELSE EXCLUDED.terminal_gateway_instance_id
                    END,
                    terminal_scheduler_lease_owner = CASE
                        WHEN existing.terminal_accepted IS TRUE
                         AND EXCLUDED.terminal_accepted IS FALSE
                        THEN existing.terminal_scheduler_lease_owner
                        ELSE EXCLUDED.terminal_scheduler_lease_owner
                    END,
                    terminal_accepted = CASE
                        WHEN existing.terminal_accepted IS TRUE
                         AND EXCLUDED.terminal_accepted IS FALSE
                        THEN TRUE
                        ELSE EXCLUDED.terminal_accepted
                    END,
                    terminal_reject_reason = CASE
                        WHEN existing.terminal_accepted IS TRUE
                         AND EXCLUDED.terminal_accepted IS FALSE
                        THEN existing.terminal_reject_reason
                        ELSE EXCLUDED.terminal_reject_reason
                    END,
                    updated_on = NOW()
                RETURNING existing.run_attempt_id
            )
            SELECT outcome.accepted, outcome.final_state
            FROM outcome
            JOIN audited ON TRUE
        """
        audit_params = (
            run_attempt_id,
            job_id,
            queue_name,
            dag_id,
            run_owner,
            scheduler_lease_owner,
            gateway_instance_id,
            terminal_status,
            source,
            gateway_instance_id,
            scheduler_lease_owner,
        )
        operation_started = time.perf_counter()
        pool_wait_started = time.perf_counter()
        async with self._pool.acquire() as conn:
            acquired_at = time.perf_counter()
            sql_started = time.perf_counter()
            row = await conn.fetchrow(
                query,
                *transition_params,
                *audit_params,
            )
            sql_ms = (time.perf_counter() - sql_started) * 1000.0

        accepted = bool(row[0])
        final_state = str(row[1]) if row[1] is not None else None
        scheduler_trace(
            "terminal_db_operation_completed",
            operation="job_terminal_transition",
            job_id=job_id,
            dag_id=dag_id,
            run_attempt_id=run_attempt_id,
            terminal_status=terminal_status,
            accepted=accepted,
            final_state=final_state,
            pool_wait_ms=(acquired_at - pool_wait_started) * 1000.0,
            sql_ms=sql_ms,
            commit_ms=None,
            autocommit=True,
            total_ms=(time.perf_counter() - operation_started) * 1000.0,
        )
        return accepted, final_state

    async def cancel_job_attempt(
        self,
        job_id: str,
        queue_name: str,
        run_owner: str,
        run_attempt_id: str,
        schema: str = DEFAULT_SCHEMA,
    ) -> set[str]:
        operation_started = time.perf_counter()
        pool_wait_started = time.perf_counter()
        async with self._pool.acquire() as conn:
            acquired_at = time.perf_counter()
            sql_started = time.perf_counter()
            row = await conn.fetchrow(
                f"""
                UPDATE {schema}.job
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
            sql_ms = (time.perf_counter() - sql_started) * 1000.0
        scheduler_trace(
            "terminal_db_operation_completed",
            operation="job_cancel",
            job_id=job_id,
            run_attempt_id=run_attempt_id,
            pool_wait_ms=(acquired_at - pool_wait_started) * 1000.0,
            sql_ms=sql_ms,
            commit_ms=None,
            autocommit=True,
            total_ms=(time.perf_counter() - operation_started) * 1000.0,
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
        operation_started = time.perf_counter()
        pool_wait_started = time.perf_counter()
        async with self._pool.acquire() as conn:
            acquired_at = time.perf_counter()
            sql_started = time.perf_counter()
            count = await conn.fetchval(query)
            sql_ms = (time.perf_counter() - sql_started) * 1000.0
        scheduler_trace(
            "terminal_db_operation_completed",
            operation="job_complete",
            job_id=job_id,
            run_attempt_id=run_attempt_id,
            pool_wait_ms=(acquired_at - pool_wait_started) * 1000.0,
            sql_ms=sql_ms,
            commit_ms=None,
            autocommit=True,
            total_ms=(time.perf_counter() - operation_started) * 1000.0,
        )
        return int(count or 0)

    async def complete_job_attempts(
        self,
        attempts: dict[str, tuple[str, str]],
        *,
        run_owner: str,
        output_metadata: Optional[dict] = None,
        schema: str = DEFAULT_SCHEMA,
    ) -> set[str]:
        """Complete several fenced job attempts in one statement."""
        if not attempts:
            return set()

        job_ids = list(attempts)
        queue_names = [attempts[job_id][0] for job_id in job_ids]
        run_attempt_ids = [attempts[job_id][1] for job_id in job_ids]
        output = {"on_complete": "done", **(output_metadata or {})}
        operation_started = time.perf_counter()
        pool_wait_started = time.perf_counter()
        async with self._pool.acquire() as conn:
            acquired_at = time.perf_counter()
            sql_started = time.perf_counter()
            rows = await conn.fetch(
                f"""
                WITH requested(job_id, queue_name, run_attempt_id) AS (
                    SELECT *
                    FROM unnest(%s::uuid[], %s::text[], %s::uuid[])
                ), completed AS (
                    UPDATE {schema}.job AS job
                    SET completed_on = NOW(),
                        state = %s::{schema}.job_state,
                        output = %s::jsonb,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        run_owner = NULL,
                        run_lease_expires_at = NULL
                    FROM requested
                    WHERE job.id = requested.job_id
                      AND job.name = requested.queue_name
                      AND job.state = %s::{schema}.job_state
                      AND job.run_owner = %s
                      AND job.run_attempt_id = requested.run_attempt_id
                    RETURNING job.id
                )
                SELECT id FROM completed
                """,
                job_ids,
                queue_names,
                run_attempt_ids,
                WorkState.COMPLETED.value,
                Jsonb(output),
                WorkState.ACTIVE.value,
                run_owner,
            )
            sql_ms = (time.perf_counter() - sql_started) * 1000.0

        completed_ids = {str(row[0]) for row in rows}
        scheduler_trace(
            "terminal_db_operation_completed",
            operation="control_flow_batch_complete",
            jobs=len(job_ids),
            completed=len(completed_ids),
            pool_wait_ms=(acquired_at - pool_wait_started) * 1000.0,
            sql_ms=sql_ms,
            commit_ms=None,
            autocommit=True,
            total_ms=(time.perf_counter() - operation_started) * 1000.0,
        )
        return completed_ids

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
        operation_started = time.perf_counter()
        pool_wait_started = time.perf_counter()
        async with self._pool.acquire() as conn:
            acquired_at = time.perf_counter()
            sql_started = time.perf_counter()
            row = await conn.fetchrow(query)
            sql_ms = (time.perf_counter() - sql_started) * 1000.0
        scheduler_trace(
            "terminal_db_operation_completed",
            operation="job_fail",
            job_id=job_id,
            run_attempt_id=run_attempt_id,
            pool_wait_ms=(acquired_at - pool_wait_started) * 1000.0,
            sql_ms=sql_ms,
            commit_ms=None,
            autocommit=True,
            total_ms=(time.perf_counter() - operation_started) * 1000.0,
        )
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
                UPDATE {schema}.job
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
        operation_started = time.perf_counter()
        pool_wait_started = time.perf_counter()
        async with self._pool.acquire() as conn:
            acquired_at = time.perf_counter()
            sql_started = time.perf_counter()
            value = await conn.fetchval(
                f"SELECT {DEFAULT_SCHEMA}.resolve_dag_state(%s::uuid)", dag_id
            )
            sql_ms = (time.perf_counter() - sql_started) * 1000.0
        scheduler_trace(
            "terminal_db_operation_completed",
            operation="dag_resolve",
            dag_id=dag_id,
            pool_wait_ms=(acquired_at - pool_wait_started) * 1000.0,
            sql_ms=sql_ms,
            commit_ms=None,
            autocommit=True,
            total_ms=(time.perf_counter() - operation_started) * 1000.0,
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
        metadata = work_info.data.get("metadata", {})
        planner = metadata.get("planner")
        search_documents = build_job_search_documents(
            plan=plan, dag_nodes=dag_nodes, planner=planner
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    insert_dag(DEFAULT_SCHEMA),
                    dag_id,
                    dag_name,
                    Jsonb(plan.model_dump()),
                    work_info.soft_sla,
                    work_info.hard_sla,
                    planner,
                    work_info.priority,
                    work_info.name,
                    metadata.get("project_id"),
                    metadata.get("ref_type"),
                    metadata.get("ref_id"),
                    work_info.policy,
                    len(dag_nodes),
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
                FROM {DEFAULT_SCHEMA}.job
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
                       attempt.attempt_state, attempt.terminal_source,
                       attempt.terminal_work_state,
                       attempt.recovery_reason, attempt.gateway_instance_id,
                       attempt.executor, attempt.terminal_gateway_instance_id,
                       attempt.terminal_reject_reason
                FROM {DEFAULT_SCHEMA}.job j
                LEFT JOIN LATERAL (
                    SELECT a.attempt_state, a.terminal_source,
                           a.terminal_work_state,
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
                "terminal_source": blocker[9],
                "terminal_work_state": blocker[10],
                "recovery_reason": blocker[11],
                "gateway_instance_id": blocker[12],
                "executor": blocker[13],
                "terminal_gateway_instance_id": blocker[14],
                "terminal_reject_reason": blocker[15],
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
                    UPDATE {schema}.job
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
                        UPDATE {schema}.job
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
                "monitoring/throughput_analysis.sql",
                create_sql_from_file(
                    schema,
                    os.path.join(schema_dir, "monitoring", "throughput_analysis.sql"),
                ),
            )
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
            (
                """
                SELECT COUNT(*) = 14
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name IN ('dag', 'dag_history')
                  AND column_name = ANY(%s)
                """,
                (
                    schema,
                    [
                        "priority",
                        "submission_name",
                        "project_id",
                        "ref_type",
                        "ref_id",
                        "policy",
                        "task_count",
                    ],
                ),
                f"{schema} DAG admission projection is incomplete",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (f"{schema}.admission_candidate_dags(integer,integer,uuid[])",),
                f"{schema}.admission_candidate_dags is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (f"{schema}.monitor_system_throughput(integer,text)",),
                f"{schema}.monitor_system_throughput is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (f"{schema}.monitor_planner_throughput(integer,text)",),
                f"{schema}.monitor_planner_throughput is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (f"{schema}.monitor_task_throughput(integer,text)",),
                f"{schema}.monitor_task_throughput is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (f"{schema}.get_operational_dag(uuid,integer,integer,integer)",),
                f"{schema}.get_operational_dag is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (
                    f"{schema}.list_operational_jobs(integer,integer,text[],text,"
                    "text,text,text,uuid,integer,integer,integer)",
                ),
                f"{schema}.list_operational_jobs is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (
                    f"{schema}.list_operational_attempts(integer,integer,text[],"
                    "text,text,text,text,text,integer,integer)",
                ),
                f"{schema}.list_operational_attempts is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (
                    f"{schema}.list_operational_events(integer,timestamptz,text,"
                    "integer,text,text,text)",
                ),
                f"{schema}.list_operational_events is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (f"{schema}.get_operational_flow(integer,text,integer)",),
                f"{schema}.get_operational_flow is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (f"{schema}.get_operational_database_health()",),
                f"{schema}.get_operational_database_health is missing",
            ),
            (
                "SELECT to_regprocedure(%s) IS NOT NULL",
                (
                    f"{schema}.list_operational_execution_history(uuid,uuid,"
                    "integer,integer)",
                ),
                f"{schema}.list_operational_execution_history is missing",
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
                  AND p.pronargs = 5
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
                or "_run_attempt_ids" not in activation
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
                SELECT id FROM {schema}.job
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
