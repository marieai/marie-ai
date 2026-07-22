#!/usr/bin/env python3
"""Generate and measure persistent scheduler corpora in PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

GENERATOR_VERSION = 1
UUID_DERIVATION_VERSION = 1
ROOT_NAMESPACE = uuid.UUID("e629fc5d-3c9f-5be8-aeed-0615f3481723")
SCHEDULER_SCHEMA = "marie_scheduler"
STRESS_SCHEMA = "marie_stress"

GRAPH_SHAPES = {"single", "chain", "fanout", "diamond", "mixed"}
WORKLOAD_PROFILES = {"ready", "active", "completed", "failed", "mixed"}
PROJECTION_MODES = {"scheduler", "full"}
MIXED_STATES = (
    "created",
    "retry",
    "active",
    "completed",
    "skipped",
    "expired",
    "cancelled",
    "failed",
)
TERMINAL_STATES = {"completed", "skipped", "expired", "cancelled", "failed"}
DATABASE_KEYS = {
    "host",
    "port",
    "dbname",
    "user",
    "sslmode",
    "connect_timeout",
}


MANIFEST_DDL = f"""
CREATE SCHEMA IF NOT EXISTS {STRESS_SCHEMA};

CREATE TABLE IF NOT EXISTS {STRESS_SCHEMA}.run_manifest (
    run_id TEXT PRIMARY KEY,
    generator_version INTEGER NOT NULL,
    uuid_derivation_version INTEGER NOT NULL,
    seed BIGINT NOT NULL,
    graph_shape TEXT NOT NULL,
    nodes_per_dag INTEGER NOT NULL,
    queue_name TEXT NOT NULL,
    workload_profile TEXT NOT NULL,
    projection_mode TEXT NOT NULL,
    executor TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    active_lease_seconds INTEGER NOT NULL DEFAULT 2592000,
    config_hash TEXT NOT NULL,
    scheduler_schema_version INTEGER NOT NULL,
    target_dag_count BIGINT NOT NULL,
    high_water_mark BIGINT NOT NULL DEFAULT 0,
    schema_transitions JSONB NOT NULL DEFAULT '[]'::JSONB,
    checkpoints JSONB NOT NULL DEFAULT '[]'::JSONB,
    created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checkpoint_started_on TIMESTAMPTZ,
    checkpoint_completed_on TIMESTAMPTZ
);

ALTER TABLE {STRESS_SCHEMA}.run_manifest
    ADD COLUMN IF NOT EXISTS active_lease_seconds INTEGER NOT NULL DEFAULT 2592000;
ALTER TABLE {STRESS_SCHEMA}.run_manifest
    ADD COLUMN IF NOT EXISTS checkpoints JSONB NOT NULL DEFAULT '[]'::JSONB;
ALTER TABLE {STRESS_SCHEMA}.run_manifest
    ADD COLUMN IF NOT EXISTS checkpoint_started_on TIMESTAMPTZ;
"""


STAGING_DDL = """
CREATE TEMP TABLE IF NOT EXISTS stress_dag_stage (
    dag_index BIGINT NOT NULL,
    id UUID NOT NULL,
    name TEXT NOT NULL,
    state TEXT NOT NULL,
    serialized_dag JSONB NOT NULL,
    started_on TIMESTAMPTZ,
    completed_on TIMESTAMPTZ,
    soft_sla TIMESTAMPTZ,
    hard_sla TIMESTAMPTZ,
    planner TEXT NOT NULL
) ON COMMIT DELETE ROWS;

CREATE TEMP TABLE IF NOT EXISTS stress_job_stage (
    dag_index BIGINT NOT NULL,
    node_index INTEGER NOT NULL,
    id UUID NOT NULL,
    dag_id UUID NOT NULL,
    name TEXT NOT NULL,
    priority INTEGER NOT NULL,
    state TEXT NOT NULL,
    data JSONB NOT NULL,
    start_after TIMESTAMPTZ NOT NULL,
    started_on TIMESTAMPTZ,
    completed_on TIMESTAMPTZ,
    expire_seconds INTEGER NOT NULL,
    keep_until TIMESTAMPTZ NOT NULL,
    output JSONB,
    dependencies JSONB NOT NULL,
    job_level INTEGER NOT NULL,
    soft_sla TIMESTAMPTZ,
    hard_sla TIMESTAMPTZ,
    run_owner TEXT,
    run_attempt_id UUID,
    run_lease_expires_at TIMESTAMPTZ
) ON COMMIT DELETE ROWS;

CREATE TEMP TABLE IF NOT EXISTS stress_attempt_stage (
    run_attempt_id UUID NOT NULL,
    job_id UUID NOT NULL,
    job_name TEXT NOT NULL,
    dag_id UUID NOT NULL,
    run_owner TEXT NOT NULL,
    scheduler_lease_owner TEXT NOT NULL,
    gateway_instance_id TEXT NOT NULL,
    executor TEXT NOT NULL,
    activated_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL
) ON COMMIT DELETE ROWS;

CREATE TEMP TABLE IF NOT EXISTS stress_search_stage (
    job_id UUID NOT NULL,
    queue_name TEXT NOT NULL,
    dag_id UUID NOT NULL,
    planner TEXT NOT NULL,
    job_name TEXT NOT NULL,
    node_label TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    ref_type TEXT NOT NULL,
    asset_uri TEXT NOT NULL,
    method TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    executor TEXT NOT NULL,
    search_text TEXT NOT NULL
) ON COMMIT DELETE ROWS;
"""


INSERT_DAGS_SQL = f"""
INSERT INTO {SCHEDULER_SCHEMA}.dag (
    id, name, state, serialized_dag, started_on, completed_on,
    soft_sla, hard_sla, planner
)
SELECT
    id, name, state, serialized_dag, started_on, completed_on,
    soft_sla, hard_sla, planner
FROM pg_temp.stress_dag_stage
ORDER BY dag_index
ON CONFLICT DO NOTHING
"""


INSERT_JOBS_SQL = f"""
INSERT INTO {SCHEDULER_SCHEMA}.job (
    id, dag_id, name, priority, state, data, start_after, started_on,
    completed_on, expire_in, keep_until, output, dependencies, job_level,
    soft_sla, hard_sla, run_owner, run_attempt_id, run_lease_expires_at
)
SELECT
    id,
    dag_id,
    name,
    priority,
    state::{SCHEDULER_SCHEMA}.job_state,
    data,
    start_after,
    started_on,
    completed_on,
    make_interval(secs => expire_seconds),
    keep_until,
    output,
    dependencies,
    job_level,
    soft_sla,
    hard_sla,
    run_owner,
    run_attempt_id,
    run_lease_expires_at
FROM pg_temp.stress_job_stage
ORDER BY dag_index, node_index
ON CONFLICT DO NOTHING
"""


INSERT_ATTEMPTS_SQL = f"""
INSERT INTO {SCHEDULER_SCHEMA}.job_attempt (
    run_attempt_id, job_id, job_name, dag_id, run_owner,
    scheduler_lease_owner, gateway_instance_id, executor,
    attempt_state, activated_at, metadata
)
SELECT
    run_attempt_id, job_id, job_name, dag_id, run_owner,
    scheduler_lease_owner, gateway_instance_id, executor,
    'activated', activated_at, metadata
FROM pg_temp.stress_attempt_stage
ON CONFLICT DO NOTHING
"""


INSERT_SEARCH_SQL = f"""
INSERT INTO {SCHEDULER_SCHEMA}.job_search_document (
    job_id, queue_name, dag_id, planner, job_name, node_label,
    ref_id, ref_type, asset_uri, method, endpoint, executor, search_text
)
SELECT
    job_id, queue_name, dag_id, planner, job_name, node_label,
    ref_id, ref_type, asset_uri, method, endpoint, executor, search_text
FROM pg_temp.stress_search_stage
ON CONFLICT (queue_name, job_id) DO NOTHING
"""


@dataclass(frozen=True)
class StressConfig:
    run_id: str
    target_dag_count: int
    nodes_per_dag: int
    graph_shape: str
    workload_profile: str
    queue_name: str
    batch_size: int
    seed: int
    projection_mode: str
    analyze_after_seed: bool
    executor: str
    endpoint: str
    active_lease_seconds: int
    report: str | None
    database: Mapping[str, Any]
    allow_schema_transition: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StressConfig:
        database = values.get("database") or {}
        if not isinstance(database, Mapping):
            raise ValueError("database must be a JSON object")
        if "password" in database:
            raise ValueError("Put the database password in PGPASSWORD, not the config")
        unknown_database_keys = set(database) - DATABASE_KEYS
        if unknown_database_keys:
            names = ", ".join(sorted(unknown_database_keys))
            raise ValueError(f"Unsupported database config fields: {names}")

        config = cls(
            run_id=str(values.get("run_id", "")).strip(),
            target_dag_count=int(values.get("target_dag_count", 0)),
            nodes_per_dag=int(values.get("nodes_per_dag", 1)),
            graph_shape=str(values.get("graph_shape", "single")),
            workload_profile=str(values.get("workload_profile", "ready")),
            queue_name=str(values.get("queue_name", "scheduler_stress_v1")).strip(),
            batch_size=int(values.get("batch_size", 10_000)),
            seed=int(values.get("seed", 20260721)),
            projection_mode=str(values.get("projection_mode", "scheduler")),
            analyze_after_seed=bool(values.get("analyze_after_seed", True)),
            executor=str(values.get("executor", "mock_executor")).strip(),
            endpoint=str(values.get("endpoint", "/document/extract")).strip(),
            active_lease_seconds=int(
                values.get("active_lease_seconds", 30 * 24 * 60 * 60)
            ),
            report=(str(values["report"]) if values.get("report") else None),
            database=dict(database),
            allow_schema_transition=bool(values.get("allow_schema_transition", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if len(self.run_id) > 160:
            raise ValueError("run_id must be at most 160 characters")
        if self.target_dag_count <= 0:
            raise ValueError("target_dag_count must be greater than zero")
        if self.nodes_per_dag <= 0:
            raise ValueError("nodes_per_dag must be greater than zero")
        if self.graph_shape not in GRAPH_SHAPES:
            raise ValueError(f"Unsupported graph_shape: {self.graph_shape}")
        if self.graph_shape == "single" and self.nodes_per_dag != 1:
            raise ValueError("graph_shape=single requires nodes_per_dag=1")
        if self.graph_shape == "diamond" and self.nodes_per_dag < 4:
            raise ValueError("graph_shape=diamond requires at least four nodes")
        if self.workload_profile not in WORKLOAD_PROFILES:
            raise ValueError(f"Unsupported workload_profile: {self.workload_profile}")
        if self.projection_mode not in PROJECTION_MODES:
            raise ValueError(f"Unsupported projection_mode: {self.projection_mode}")
        if not self.queue_name:
            raise ValueError("queue_name is required")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not self.executor:
            raise ValueError("executor is required")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")
        if self.active_lease_seconds <= 0:
            raise ValueError("active_lease_seconds must be greater than zero")

    @property
    def planner(self) -> str:
        return f"stress:{self.run_id}"

    @property
    def cohort_hash(self) -> str:
        payload = {
            "run_id": self.run_id,
            "seed": self.seed,
            "nodes_per_dag": self.nodes_per_dag,
            "graph_shape": self.graph_shape,
            "workload_profile": self.workload_profile,
            "queue_name": self.queue_name,
            "projection_mode": self.projection_mode,
            "executor": self.executor,
            "endpoint": self.endpoint,
            "active_lease_seconds": self.active_lease_seconds,
            "uuid_derivation_version": UUID_DERIVATION_VERSION,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class GeneratedJob:
    dag_index: int
    node_index: int
    id: uuid.UUID
    dag_id: uuid.UUID
    priority: int
    state: str
    data: dict[str, Any]
    dependencies: tuple[uuid.UUID, ...]
    job_level: int
    start_after: datetime
    started_on: datetime | None
    completed_on: datetime | None
    soft_sla: datetime
    hard_sla: datetime
    run_owner: str | None
    run_attempt_id: uuid.UUID | None
    run_lease_expires_at: datetime | None


@dataclass(frozen=True)
class GeneratedDag:
    index: int
    id: uuid.UUID
    state: str
    jobs: tuple[GeneratedJob, ...]
    serialized_dag: dict[str, Any]
    started_on: datetime | None
    completed_on: datetime | None
    soft_sla: datetime
    hard_sla: datetime


@cache
def run_namespace(run_id: str) -> uuid.UUID:
    return uuid.uuid5(ROOT_NAMESPACE, run_id)


def dag_id_for(run_id: str, dag_index: int) -> uuid.UUID:
    return uuid.uuid5(run_namespace(run_id), f"dag:{dag_index}")


def job_id_for(run_id: str, dag_index: int, node_index: int) -> uuid.UUID:
    return uuid.uuid5(run_namespace(run_id), f"dag:{dag_index}:node:{node_index}")


def attempt_id_for(run_id: str, dag_index: int, node_index: int) -> uuid.UUID:
    return uuid.uuid5(
        run_namespace(run_id), f"dag:{dag_index}:node:{node_index}:attempt:0"
    )


def graph_shape_for(config: StressConfig, dag_index: int) -> str:
    if config.graph_shape != "mixed":
        return config.graph_shape
    available = ["chain"]
    if config.nodes_per_dag >= 2:
        available.append("fanout")
    if config.nodes_per_dag >= 4:
        available.append("diamond")
    if config.nodes_per_dag == 1:
        return "single"
    return available[(config.seed + dag_index) % len(available)]


def dependencies_for(
    shape: str, job_ids: Sequence[uuid.UUID], node_index: int
) -> tuple[uuid.UUID, ...]:
    if shape == "single" or node_index == 0:
        return ()
    if shape == "chain":
        return (job_ids[node_index - 1],)
    if shape == "fanout":
        return (job_ids[0],)
    if shape == "diamond":
        if node_index == len(job_ids) - 1:
            return tuple(job_ids[1:-1])
        return (job_ids[0],)
    raise ValueError(f"Unsupported graph shape: {shape}")


def level_for(shape: str, node_index: int, node_count: int) -> int:
    if shape == "chain":
        return node_count - node_index - 1
    if shape == "diamond" and node_index == node_count - 1:
        return 0
    if shape == "diamond":
        return 2 if node_index == 0 else 1
    return 1 if node_index == 0 and node_count > 1 else 0


def leaf_nodes(shape: str, node_count: int) -> set[int]:
    if shape == "chain" or shape == "diamond":
        return {node_count - 1}
    if shape == "fanout":
        return set(range(1, node_count)) or {0}
    return {0}


def expected_ready_jobs(config: StressConfig, dag_count: int) -> int | None:
    if config.workload_profile != "ready":
        return None
    if config.graph_shape != "mixed":
        return dag_count * len(leaf_nodes(config.graph_shape, config.nodes_per_dag))

    shapes = ["chain"]
    if config.nodes_per_dag >= 2:
        shapes.append("fanout")
    if config.nodes_per_dag >= 4:
        shapes.append("diamond")
    cycles, remainder = divmod(dag_count, len(shapes))
    total = cycles * sum(
        len(leaf_nodes(shape, config.nodes_per_dag)) for shape in shapes
    )
    for offset in range(remainder):
        shape = shapes[(config.seed + cycles * len(shapes) + offset) % len(shapes)]
        total += len(leaf_nodes(shape, config.nodes_per_dag))
    return total


def profile_state(config: StressConfig, dag_index: int) -> str:
    if config.workload_profile != "mixed":
        return config.workload_profile
    return MIXED_STATES[(config.seed + dag_index) % len(MIXED_STATES)]


def build_dag(
    config: StressConfig, dag_index: int, generated_at: datetime
) -> GeneratedDag:
    shape = graph_shape_for(config, dag_index)
    dag_id = dag_id_for(config.run_id, dag_index)
    job_ids = tuple(
        job_id_for(config.run_id, dag_index, node_index)
        for node_index in range(config.nodes_per_dag)
    )
    selected_state = profile_state(config, dag_index)
    leaves = leaf_nodes(shape, config.nodes_per_dag)
    started_on = generated_at - timedelta(seconds=config.nodes_per_dag * 2)
    completed_on = generated_at - timedelta(seconds=1)
    soft_sla = generated_at + timedelta(minutes=5 + (dag_index % 4) * 5)
    hard_sla = soft_sla + timedelta(minutes=10)
    jobs: list[GeneratedJob] = []
    plan_nodes: list[dict[str, Any]] = []

    for node_index, job_id in enumerate(job_ids):
        dependencies = dependencies_for(shape, job_ids, node_index)
        state = "created" if selected_state == "ready" else selected_state
        if selected_state in {"ready", "active"} and node_index not in leaves:
            state = "completed"

        is_active = state == "active"
        is_terminal = state in TERMINAL_STATES
        timeline_started_on = generated_at - timedelta(
            seconds=(config.nodes_per_dag - node_index) * 2
        )
        job_started_on = timeline_started_on if is_active or is_terminal else None
        job_completed_on = (
            timeline_started_on + timedelta(seconds=1) if is_terminal else None
        )
        run_owner = f"stress:{config.run_id}" if is_active else None
        run_attempt_id = (
            attempt_id_for(config.run_id, dag_index, node_index) if is_active else None
        )
        metadata = {
            "stress_run_id": config.run_id,
            "stress_dag_index": dag_index,
            "stress_node_index": node_index,
            "planner": config.planner,
            "ref_id": f"{config.run_id}:{dag_index}",
            "ref_type": "stress",
            "uri": f"stress://{config.run_id}/{dag_index}",
            "on": f"{config.executor}://{config.endpoint.lstrip('/')}",
        }
        job = GeneratedJob(
            dag_index=dag_index,
            node_index=node_index,
            id=job_id,
            dag_id=dag_id,
            priority=(config.seed + dag_index * 17 + node_index * 31) % 101,
            state=state,
            data={"metadata": metadata},
            dependencies=dependencies,
            job_level=level_for(shape, node_index, config.nodes_per_dag),
            start_after=generated_at - timedelta(seconds=1),
            started_on=job_started_on,
            completed_on=job_completed_on,
            soft_sla=soft_sla,
            hard_sla=hard_sla,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            run_lease_expires_at=(
                generated_at + timedelta(seconds=config.active_lease_seconds)
                if is_active
                else None
            ),
        )
        jobs.append(job)
        plan_nodes.append(
            {
                "task_id": str(job.id),
                "query_str": f"stress-node-{node_index}",
                "dependencies": [str(value) for value in dependencies],
                "node_type": "COMPUTE",
                "definition": {
                    "method": "EXECUTOR_ENDPOINT",
                    "endpoint": config.endpoint,
                    "params": {"layout": None, "function": None},
                },
            }
        )

    dag_state = {
        "ready": "created",
        "created": "created",
        "retry": "created",
        "active": "active",
        "completed": "completed",
        "skipped": "completed",
        "expired": "failed",
        "cancelled": "failed",
        "failed": "failed",
    }[selected_state]
    dag_is_started = dag_state in {"active", "completed", "failed"}
    dag_is_terminal = dag_state in {"completed", "failed"}
    return GeneratedDag(
        index=dag_index,
        id=dag_id,
        state=dag_state,
        jobs=tuple(jobs),
        serialized_dag={"nodes": plan_nodes},
        started_on=started_on if dag_is_started else None,
        completed_on=completed_on if dag_is_terminal else None,
        soft_sla=soft_sla,
        hard_sla=hard_sla,
    )


def build_plan(config: StressConfig, current_count: int = 0) -> dict[str, Any]:
    if current_count > config.target_dag_count:
        raise ValueError(
            f"Refusing to shrink cohort from {current_count} to "
            f"{config.target_dag_count} DAGs"
        )
    remaining = config.target_dag_count - current_count
    chunks = (remaining + config.batch_size - 1) // config.batch_size
    return {
        "run_id": config.run_id,
        "current_dag_count": current_count,
        "target_dag_count": config.target_dag_count,
        "dags_to_add": remaining,
        "jobs_to_add": remaining * config.nodes_per_dag,
        "batch_size": config.batch_size,
        "chunk_count": chunks,
        "graph_shape": config.graph_shape,
        "nodes_per_dag": config.nodes_per_dag,
        "workload_profile": config.workload_profile,
        "projection_mode": config.projection_mode,
        "queue_name": config.queue_name,
        "executor": config.executor,
        "active_lease_seconds": config.active_lease_seconds,
        "uuid_derivation_version": UUID_DERIVATION_VERSION,
        "config_hash": config.cohort_hash,
        "destructive_actions": [],
    }


def statistics_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, dict[str, int | float | Decimal]]:
    delta: dict[str, dict[str, int | float | Decimal]] = {}
    for section in ("database", "wal"):
        before_values = before.get(section)
        after_values = after.get(section)
        if not isinstance(before_values, Mapping) or not isinstance(
            after_values, Mapping
        ):
            continue
        section_delta: dict[str, int | float | Decimal] = {}
        for name, after_value in after_values.items():
            before_value = before_values.get(name)
            if (
                isinstance(after_value, (int, float, Decimal))
                and not isinstance(after_value, bool)
                and isinstance(before_value, (int, float, Decimal))
                and not isinstance(before_value, bool)
            ):
                section_delta[name] = after_value - before_value
        delta[section] = section_delta
    return delta


def statement_delta(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    identity_fields = ("userid", "dbid", "toplevel", "queryid")
    before_by_id = {
        tuple(row.get(field) for field in identity_fields): row for row in before
    }
    fields = (
        "calls",
        "rows",
        "total_plan_time",
        "total_exec_time",
        "shared_blks_hit",
        "shared_blks_read",
        "shared_blks_dirtied",
        "shared_blks_written",
        "temp_blks_read",
        "temp_blks_written",
        "wal_records",
        "wal_fpi",
        "wal_bytes",
    )
    rows: list[dict[str, Any]] = []
    for current in after:
        identity = tuple(current.get(field) for field in identity_fields)
        previous = before_by_id.get(identity, {})
        row: dict[str, Any] = {
            "queryid": str(current.get("queryid")),
            "query": current.get("query"),
        }
        for field in identity_fields[:-1]:
            if current.get(field) is not None:
                row[field] = current[field]
        for field in fields:
            current_value = current.get(field)
            previous_value = previous.get(field, 0)
            if isinstance(current_value, (int, float, Decimal)) and isinstance(
                previous_value, (int, float, Decimal)
            ):
                row[field] = current_value - previous_value
        if row.get("calls", 0) > 0:
            rows.append(row)
    rows.sort(key=lambda row: row.get("total_exec_time", 0), reverse=True)
    return rows


class CorpusGenerator:
    def __init__(self, connection: psycopg.Connection[dict[str, Any]]) -> None:
        self.connection = connection

    def initialize(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT to_regclass('{SCHEDULER_SCHEMA}.version') AS version_table"
            )
            row = cursor.fetchone()
            if not row or row["version_table"] is None:
                raise RuntimeError("marie_scheduler schema is not installed")
            cursor.execute(
                f"SELECT MAX(version) AS version FROM {SCHEDULER_SCHEMA}.version"
            )
            version_row = cursor.fetchone()
            if not version_row or version_row["version"] is None:
                raise RuntimeError("marie_scheduler.version has no installed version")
            cursor.execute(MANIFEST_DDL)
        self.connection.commit()
        return int(version_row["version"])

    def acquire_lock(self, run_id: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS locked",
                (f"marie-stress:{run_id}",),
            )
            row = cursor.fetchone()
        if not row or not row["locked"]:
            raise RuntimeError(f"Another generator holds the lock for run_id={run_id}")

    def release_lock(self, run_id: str) -> None:
        self.connection.rollback()
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (f"marie-stress:{run_id}",),
            )
        self.connection.commit()

    def prepare_manifest(
        self, config: StressConfig, scheduler_schema_version: int
    ) -> dict[str, Any]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {STRESS_SCHEMA}.run_manifest (
                    run_id, generator_version, uuid_derivation_version, seed,
                    graph_shape, nodes_per_dag, queue_name, workload_profile,
                    projection_mode, executor, endpoint, active_lease_seconds,
                    config_hash, scheduler_schema_version, target_dag_count
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s
                )
                ON CONFLICT (run_id) DO NOTHING
                """,
                (
                    config.run_id,
                    GENERATOR_VERSION,
                    UUID_DERIVATION_VERSION,
                    config.seed,
                    config.graph_shape,
                    config.nodes_per_dag,
                    config.queue_name,
                    config.workload_profile,
                    config.projection_mode,
                    config.executor,
                    config.endpoint,
                    config.active_lease_seconds,
                    config.cohort_hash,
                    scheduler_schema_version,
                    config.target_dag_count,
                ),
            )
            cursor.execute(
                f"SELECT * FROM {STRESS_SCHEMA}.run_manifest WHERE run_id = %s",
                (config.run_id,),
            )
            manifest = cursor.fetchone()
            if manifest is None:
                raise RuntimeError("Failed to create or load stress manifest")

            self._validate_manifest(config, manifest)
            high_water_mark = int(manifest["high_water_mark"])
            if high_water_mark > config.target_dag_count:
                raise RuntimeError(
                    f"Refusing to shrink cohort from {high_water_mark} to "
                    f"{config.target_dag_count} DAGs"
                )
            stored_version = int(manifest["scheduler_schema_version"])
            if stored_version != scheduler_schema_version:
                if not config.allow_schema_transition:
                    raise RuntimeError(
                        "Scheduler schema changed from "
                        f"{stored_version} to {scheduler_schema_version}; "
                        "start a new run or pass --allow-schema-transition"
                    )
                cursor.execute(
                    f"""
                    UPDATE {STRESS_SCHEMA}.run_manifest
                    SET scheduler_schema_version = %s,
                        schema_transitions = schema_transitions || jsonb_build_array(
                            jsonb_build_object(
                                'from', %s,
                                'to', %s,
                                'recorded_on', NOW()
                            )
                        ),
                        updated_on = NOW()
                    WHERE run_id = %s
                    """,
                    (
                        scheduler_schema_version,
                        stored_version,
                        scheduler_schema_version,
                        config.run_id,
                    ),
                )
                manifest["scheduler_schema_version"] = scheduler_schema_version

            cursor.execute(
                f"""
                UPDATE {STRESS_SCHEMA}.run_manifest
                SET target_dag_count = %s, updated_on = NOW()
                WHERE run_id = %s
                """,
                (config.target_dag_count, config.run_id),
            )
            manifest["target_dag_count"] = config.target_dag_count
        self.connection.commit()
        return manifest

    @staticmethod
    def _validate_manifest(config: StressConfig, manifest: Mapping[str, Any]) -> None:
        expected = {
            "generator_version": GENERATOR_VERSION,
            "uuid_derivation_version": UUID_DERIVATION_VERSION,
            "seed": config.seed,
            "graph_shape": config.graph_shape,
            "nodes_per_dag": config.nodes_per_dag,
            "queue_name": config.queue_name,
            "workload_profile": config.workload_profile,
            "projection_mode": config.projection_mode,
            "executor": config.executor,
            "endpoint": config.endpoint,
            "active_lease_seconds": config.active_lease_seconds,
            "config_hash": config.cohort_hash,
        }
        mismatches = [
            name for name, value in expected.items() if manifest.get(name) != value
        ]
        if mismatches:
            raise RuntimeError(
                "Run manifest does not match immutable fields: " + ", ".join(mismatches)
            )

    def ensure_queue(self, config: StressConfig) -> None:
        options = {
            "retry_limit": 2,
            "retry_delay": 0,
            "retry_backoff": False,
            "expire_in_seconds": 900,
            "retention_minutes": 20_160,
        }
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {SCHEDULER_SCHEMA}.create_queue(%s, %s::json)",
                (config.queue_name, Jsonb(options)),
            )
            cursor.execute(
                f"""
                SELECT partition_name
                FROM {SCHEDULER_SCHEMA}.queue
                WHERE name = %s
                """,
                (config.queue_name,),
            )
            row = cursor.fetchone()
            if not row or not row["partition_name"]:
                raise RuntimeError(
                    f"Queue partition was not created: {config.queue_name}"
                )
        self.connection.commit()

    def create_staging_tables(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(STAGING_DDL)
        self.connection.commit()

    def seed(self, config: StressConfig, current_count: int) -> dict[str, Any]:
        if current_count > config.target_dag_count:
            raise RuntimeError(
                f"Refusing to shrink cohort from {current_count} to "
                f"{config.target_dag_count} DAGs"
            )
        started = time.perf_counter()
        inserted_dags = 0
        inserted_jobs = 0

        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {STRESS_SCHEMA}.run_manifest
                SET checkpoint_started_on = NOW(),
                    checkpoint_completed_on = NULL,
                    updated_on = NOW()
                WHERE run_id = %s
                """,
                (config.run_id,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Unknown stress run_id: {config.run_id}")
        self.connection.commit()

        for start in range(current_count, config.target_dag_count, config.batch_size):
            stop = min(start + config.batch_size, config.target_dag_count)
            generated_at = datetime.now(timezone.utc)
            dags = tuple(
                build_dag(config, index, generated_at) for index in range(start, stop)
            )
            chunk_jobs = sum(len(dag.jobs) for dag in dags)
            chunk_attempts = sum(
                1 for dag in dags for job in dag.jobs if job.run_attempt_id is not None
            )

            try:
                self._copy_chunk(config, dags)
                with self.connection.cursor() as cursor:
                    cursor.execute(INSERT_DAGS_SQL)
                    if cursor.rowcount != len(dags):
                        raise RuntimeError(
                            f"DAG insert conflict in range [{start}, {stop})"
                        )
                    cursor.execute(INSERT_JOBS_SQL)
                    if cursor.rowcount != chunk_jobs:
                        raise RuntimeError(
                            f"Job insert conflict in range [{start}, {stop})"
                        )
                    if chunk_attempts:
                        cursor.execute(INSERT_ATTEMPTS_SQL)
                        if cursor.rowcount != chunk_attempts:
                            raise RuntimeError(
                                f"Attempt insert conflict in range [{start}, {stop})"
                            )
                    if config.projection_mode == "full":
                        cursor.execute(INSERT_SEARCH_SQL)
                        if cursor.rowcount != chunk_jobs:
                            raise RuntimeError(
                                f"Search insert conflict in range [{start}, {stop})"
                            )
                    cursor.execute(
                        f"""
                        UPDATE {STRESS_SCHEMA}.run_manifest
                        SET high_water_mark = %s, updated_on = NOW()
                        WHERE run_id = %s AND high_water_mark = %s
                        """,
                        (stop, config.run_id, start),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            f"Manifest high-water mark changed during range [{start}, {stop})"
                        )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

            inserted_dags += len(dags)
            inserted_jobs += chunk_jobs
            print(
                f"seeded DAGs {stop:,}/{config.target_dag_count:,} "
                f"for run_id={config.run_id}",
                file=sys.stderr,
            )

        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {SCHEDULER_SCHEMA}.job
                SET run_lease_expires_at = NOW() + make_interval(secs => %s)
                WHERE data->'metadata'->>'stress_run_id' = %s
                  AND name = %s
                  AND state = 'active'
                  AND run_lease_expires_at <= NOW() + INTERVAL '1 hour'
                """,
                (
                    config.active_lease_seconds,
                    config.run_id,
                    config.queue_name,
                ),
            )
            cursor.execute(
                f"""
                UPDATE {STRESS_SCHEMA}.run_manifest
                SET checkpoint_completed_on = NOW(),
                    checkpoints = checkpoints || jsonb_build_array(
                        jsonb_build_object(
                            'target_dag_count', %s,
                            'high_water_mark', high_water_mark,
                            'started_on', checkpoint_started_on,
                            'completed_on', NOW(),
                            'requested_report', %s
                        )
                    ),
                    updated_on = NOW()
                WHERE run_id = %s
                """,
                (config.target_dag_count, config.report, config.run_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Unknown stress run_id: {config.run_id}")
        self.connection.commit()
        elapsed = time.perf_counter() - started
        return {
            "inserted_dags": inserted_dags,
            "inserted_jobs": inserted_jobs,
            "elapsed_seconds": elapsed,
            "dags_per_second": inserted_dags / elapsed if elapsed else 0.0,
            "jobs_per_second": inserted_jobs / elapsed if elapsed else 0.0,
        }

    def _copy_chunk(self, config: StressConfig, dags: Sequence[GeneratedDag]) -> None:
        jobs = tuple(job for dag in dags for job in dag.jobs)
        with self.connection.cursor() as cursor:
            with cursor.copy(
                """
                COPY pg_temp.stress_dag_stage (
                    dag_index, id, name, state, serialized_dag, started_on,
                    completed_on, soft_sla, hard_sla, planner
                ) FROM STDIN
                """
            ) as copy:
                for dag in dags:
                    copy.write_row(
                        (
                            dag.index,
                            dag.id,
                            f"stress-{dag.id}",
                            dag.state,
                            Jsonb(dag.serialized_dag),
                            dag.started_on,
                            dag.completed_on,
                            dag.soft_sla,
                            dag.hard_sla,
                            config.planner,
                        )
                    )

            with cursor.copy(
                """
                COPY pg_temp.stress_job_stage (
                    dag_index, node_index, id, dag_id, name, priority, state,
                    data, start_after, started_on, completed_on, expire_seconds,
                    keep_until, output, dependencies, job_level, soft_sla,
                    hard_sla, run_owner, run_attempt_id, run_lease_expires_at
                ) FROM STDIN
                """
            ) as copy:
                for job in jobs:
                    copy.write_row(
                        (
                            job.dag_index,
                            job.node_index,
                            job.id,
                            job.dag_id,
                            config.queue_name,
                            job.priority,
                            job.state,
                            Jsonb(job.data),
                            job.start_after,
                            job.started_on,
                            job.completed_on,
                            900,
                            job.start_after + timedelta(days=14),
                            (
                                Jsonb({"stress": True})
                                if job.state in TERMINAL_STATES
                                else None
                            ),
                            Jsonb([str(value) for value in job.dependencies]),
                            job.job_level,
                            job.soft_sla,
                            job.hard_sla,
                            job.run_owner,
                            job.run_attempt_id,
                            job.run_lease_expires_at,
                        )
                    )

            active_jobs = [job for job in jobs if job.run_attempt_id is not None]
            if active_jobs:
                with cursor.copy(
                    """
                    COPY pg_temp.stress_attempt_stage (
                        run_attempt_id, job_id, job_name, dag_id, run_owner,
                        scheduler_lease_owner, gateway_instance_id, executor,
                        activated_at, metadata
                    ) FROM STDIN
                    """
                ) as copy:
                    for job in active_jobs:
                        copy.write_row(
                            (
                                job.run_attempt_id,
                                job.id,
                                config.queue_name,
                                job.dag_id,
                                job.run_owner,
                                job.run_owner,
                                f"stress-gateway:{config.run_id}",
                                config.executor,
                                job.started_on,
                                Jsonb({"stress_run_id": config.run_id}),
                            )
                        )

            if config.projection_mode == "full":
                with cursor.copy(
                    """
                    COPY pg_temp.stress_search_stage (
                        job_id, queue_name, dag_id, planner, job_name,
                        node_label, ref_id, ref_type, asset_uri, method,
                        endpoint, executor, search_text
                    ) FROM STDIN
                    """
                ) as copy:
                    for job in jobs:
                        ref_id = f"{config.run_id}:{job.dag_index}"
                        copy.write_row(
                            (
                                job.id,
                                config.queue_name,
                                job.dag_id,
                                config.planner,
                                config.queue_name,
                                f"stress-node-{job.node_index}",
                                ref_id,
                                "stress",
                                f"stress://{config.run_id}/{job.dag_index}",
                                "EXECUTOR_ENDPOINT",
                                config.endpoint,
                                config.executor,
                                " ".join(
                                    (
                                        config.planner,
                                        config.queue_name,
                                        ref_id,
                                        config.endpoint,
                                        config.executor,
                                    )
                                ).lower(),
                            )
                        )

    def manifest(self, run_id: str) -> dict[str, Any]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM {STRESS_SCHEMA}.run_manifest WHERE run_id = %s",
                (run_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"Unknown stress run_id: {run_id}")
        return row

    def verify(
        self, config: StressConfig, *, require_target: bool = True
    ) -> dict[str, Any]:
        manifest = self.manifest(config.run_id)
        high_water_mark = int(manifest["high_water_mark"])
        expected_jobs = high_water_mark * config.nodes_per_dag
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    (SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.dag
                     WHERE planner = %(planner)s) AS dags,
                    (SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.job
                     WHERE data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND name = %(queue_name)s) AS jobs,
                    (SELECT COUNT(DISTINCT (data->'metadata'->>'stress_dag_index')::BIGINT)
                     FROM {SCHEDULER_SCHEMA}.job
                     WHERE data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND name = %(queue_name)s)
                        AS dag_indexes,
                    (SELECT COALESCE(SUM(jsonb_array_length(dependencies)), 0)
                     FROM {SCHEDULER_SCHEMA}.job
                     WHERE data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND name = %(queue_name)s)
                        AS dependency_json,
                    (SELECT COUNT(*)
                     FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
                     JOIN {SCHEDULER_SCHEMA}.job job
                       ON job.name = dependency.job_name
                      AND job.id = dependency.job_id
                     WHERE job.data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND job.name = %(queue_name)s)
                        AS normalized_dependencies,
                    (SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.job
                     WHERE data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND name = %(queue_name)s
                       AND state = 'active') AS active_jobs,
                    (SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.ready_jobs_view
                     WHERE data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND name = %(queue_name)s)
                        AS ready_jobs,
                    (SELECT COUNT(*)
                     FROM {SCHEDULER_SCHEMA}.job active_job
                     JOIN {SCHEDULER_SCHEMA}.job_attempt attempt
                       ON attempt.job_id = active_job.id
                      AND attempt.run_attempt_id = active_job.run_attempt_id
                     WHERE active_job.data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND active_job.name = %(queue_name)s
                       AND active_job.state = 'active'
                       AND active_job.run_owner IS NOT NULL
                       AND active_job.run_lease_expires_at > NOW()
                       AND active_job.lease_owner IS NULL
                       AND active_job.lease_expires_at IS NULL
                       AND attempt.job_name = active_job.name
                       AND attempt.dag_id = active_job.dag_id
                       AND attempt.run_owner = active_job.run_owner
                       AND attempt.scheduler_lease_owner = active_job.run_owner
                       AND attempt.gateway_instance_id = %(gateway_instance_id)s
                       AND attempt.executor = %(executor)s
                       AND attempt.attempt_state = 'activated') AS active_contracts,
                    (SELECT COUNT(*)
                     FROM {SCHEDULER_SCHEMA}.job run_job
                     JOIN {SCHEDULER_SCHEMA}.job_attempt attempt
                       ON attempt.job_id = run_job.id
                     WHERE run_job.data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND run_job.name = %(queue_name)s)
                        AS attempts,
                    (SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.job_search_document
                     WHERE planner = %(planner)s
                       AND queue_name = %(queue_name)s) AS search_documents,
                    (SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.dag_history
                     WHERE planner = %(planner)s) AS dag_history,
                    (SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.job_history
                     WHERE data->'metadata'->>'stress_run_id' = %(run_id)s
                       AND name = %(queue_name)s)
                        AS job_history
                """,
                {
                    "planner": config.planner,
                    "run_id": config.run_id,
                    "queue_name": config.queue_name,
                    "gateway_instance_id": f"stress-gateway:{config.run_id}",
                    "executor": config.executor,
                },
            )
            counts = cursor.fetchone()
        if counts is None:
            raise RuntimeError("Verification query returned no result")

        expected_search = expected_jobs if config.projection_mode == "full" else 0
        expected_ready = expected_ready_jobs(config, high_water_mark)
        checks = {
            "target_reached": (
                not require_target or high_water_mark == config.target_dag_count
            ),
            "manifest_matches_dags": counts["dags"] == high_water_mark,
            "job_count": counts["jobs"] == expected_jobs,
            "dag_index_count": counts["dag_indexes"] == high_water_mark,
            "normalized_dependencies": (
                counts["normalized_dependencies"] == counts["dependency_json"]
            ),
            "active_contract_count": (
                counts["active_contracts"] == counts["active_jobs"]
            ),
            "ready_frontier_count": (
                expected_ready is None or counts["ready_jobs"] == expected_ready
            ),
            "search_projection_count": counts["search_documents"] == expected_search,
            "dag_history_present": counts["dag_history"] >= high_water_mark,
            "job_history_present": counts["job_history"] >= expected_jobs,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "expected": {
                "dags": high_water_mark,
                "target_dags": config.target_dag_count,
                "jobs": expected_jobs,
                "ready_jobs": expected_ready,
                "search_documents": expected_search,
            },
            "observed": counts,
        }

    def analyze(self, config: StressConfig) -> None:
        tables = ["dag", "job", "job_dependencies", "job_attempt"]
        if config.projection_mode == "full":
            tables.append("job_search_document")
        with self.connection.cursor() as cursor:
            for table in tables:
                cursor.execute(f"ANALYZE {SCHEDULER_SCHEMA}.{table}")
        self.connection.commit()

    def database_snapshot(self, config: StressConfig) -> dict[str, Any]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_jsonb(stats) AS stats FROM pg_stat_database stats "
                "WHERE datname = current_database()"
            )
            database = cursor.fetchone()
            cursor.execute(
                "SELECT CASE WHEN to_regclass('pg_catalog.pg_stat_wal') IS NULL "
                "THEN NULL ELSE (SELECT to_jsonb(stats) FROM pg_stat_wal stats) END "
                "AS stats"
            )
            wal = cursor.fetchone()
            cursor.execute(
                f"""
                WITH queue_partition AS (
                    SELECT format(
                        '{SCHEDULER_SCHEMA}.%I', partition_name
                    ) AS relation_name
                    FROM {SCHEDULER_SCHEMA}.queue
                    WHERE name = %s
                ), relations(logical_name, relation_name) AS (
                    VALUES
                        ('dag', '{SCHEDULER_SCHEMA}.dag'),
                        ('job_root', '{SCHEDULER_SCHEMA}.job'),
                        ('job_history', '{SCHEDULER_SCHEMA}.job_history'),
                        ('dag_history', '{SCHEDULER_SCHEMA}.dag_history'),
                        ('job_attempt', '{SCHEDULER_SCHEMA}.job_attempt'),
                        ('job_search_document',
                            '{SCHEDULER_SCHEMA}.job_search_document'),
                        ('run_manifest', '{STRESS_SCHEMA}.run_manifest'),
                        ('queue_partition',
                            (SELECT relation_name FROM queue_partition))
                )
                SELECT
                    logical_name,
                    relation_name,
                    pg_relation_size(to_regclass(relation_name)) AS heap_bytes,
                    pg_indexes_size(to_regclass(relation_name)) AS index_bytes,
                    GREATEST(
                        pg_total_relation_size(to_regclass(relation_name))
                            - pg_relation_size(to_regclass(relation_name))
                            - pg_indexes_size(to_regclass(relation_name)),
                        0
                    ) AS toast_bytes,
                    pg_total_relation_size(to_regclass(relation_name)) AS total_bytes
                FROM relations
                WHERE to_regclass(relation_name) IS NOT NULL
                ORDER BY logical_name
                """,
                (config.queue_name,),
            )
            relations = cursor.fetchall()
            cursor.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM pg_inherits
                WHERE inhparent = '{SCHEDULER_SCHEMA}.job'::regclass
                """
            )
            partition_count = cursor.fetchone()
            cursor.execute(
                """
                SELECT
                    schemaname,
                    relname,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    idx_tup_fetch,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del,
                    n_live_tup,
                    n_dead_tup,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze,
                    vacuum_count,
                    autovacuum_count,
                    analyze_count,
                    autoanalyze_count
                FROM pg_stat_user_tables
                WHERE schemaname IN (%s, %s)
                ORDER BY schemaname, relname
                """,
                (SCHEDULER_SCHEMA, STRESS_SCHEMA),
            )
            table_activity = cursor.fetchall()
            cursor.execute(
                """
                SELECT
                    COALESCE(namespace.nspname, 'database') AS schemaname,
                    COALESCE(relation.relname, 'database') AS relname,
                    locks.mode,
                    locks.granted,
                    COUNT(*) AS count
                FROM pg_locks locks
                LEFT JOIN pg_class relation ON relation.oid = locks.relation
                LEFT JOIN pg_namespace namespace
                  ON namespace.oid = relation.relnamespace
                WHERE locks.database = (
                    SELECT oid FROM pg_database WHERE datname = current_database()
                )
                  AND (
                    locks.relation IS NULL
                    OR namespace.nspname IN (%s, %s)
                  )
                GROUP BY
                    namespace.nspname,
                    relation.relname,
                    locks.mode,
                    locks.granted
                ORDER BY schemaname, relname, locks.mode, locks.granted
                """,
                (SCHEDULER_SCHEMA, STRESS_SCHEMA),
            )
            locks = cursor.fetchall()
        return {
            "database": database["stats"] if database else None,
            "wal": wal["stats"] if wal else None,
            "relations": relations,
            "job_partition_count": (
                partition_count["count"] if partition_count else None
            ),
            "table_activity": table_activity,
            "locks": locks,
        }

    def statement_snapshot(self) -> dict[str, Any]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    EXISTS (
                        SELECT 1 FROM pg_extension
                        WHERE extname = 'pg_stat_statements'
                    ) AS installed,
                    current_setting('shared_preload_libraries', TRUE) AS preload,
                    current_setting('pg_stat_statements.track', TRUE) AS track
                """
            )
            settings = cursor.fetchone()
            statements: list[dict[str, Any]] = []
            preload = settings["preload"] if settings else None
            preloaded_libraries = {
                name.strip() for name in str(preload or "").split(",")
            }
            available = bool(
                settings
                and settings["installed"]
                and "pg_stat_statements" in preloaded_libraries
            )
            if available:
                cursor.execute(
                    """
                    SELECT to_jsonb(stats) AS stats
                    FROM pg_stat_statements stats
                    WHERE dbid = (
                        SELECT oid FROM pg_database
                        WHERE datname = current_database()
                    )
                      AND query ILIKE '%marie_scheduler%'
                      AND query NOT ILIKE '%pg_stat_statements%'
                    """
                )
                statements = [row["stats"] for row in cursor.fetchall()]
        return {
            "installed": bool(settings and settings["installed"]),
            "available": available,
            "shared_preload_libraries": preload,
            "track": settings["track"] if settings else None,
            "statements": statements,
        }

    def benchmark(self, config: StressConfig) -> dict[str, Any]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id FROM {SCHEDULER_SCHEMA}.dag
                WHERE planner = %s
                  AND state IN ('created', 'active')
                ORDER BY id
                LIMIT 1000
                """,
                (config.planner,),
            )
            dag_ids = [row["id"] for row in cursor.fetchall()]
            cursor.execute(
                f"""
                SELECT id FROM {SCHEDULER_SCHEMA}.job
                WHERE data->'metadata'->>'stress_run_id' = %s
                  AND name = %s
                ORDER BY id
                LIMIT 1000
                """,
                (config.run_id, config.queue_name),
            )
            job_ids = [row["id"] for row in cursor.fetchall()]

        queries = {
            "run_dag_count": (
                f"SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.dag WHERE planner = %s",
                (config.planner,),
            ),
            "run_job_count": (
                f"""
                SELECT COUNT(*) FROM {SCHEDULER_SCHEMA}.job
                WHERE data->'metadata'->>'stress_run_id' = %s
                  AND name = %s
                """,
                (config.run_id, config.queue_name),
            ),
            "ready_sample": (
                f"""
                SELECT id FROM {SCHEDULER_SCHEMA}.ready_jobs_view
                WHERE data->'metadata'->>'stress_run_id' = %s
                  AND name = %s
                LIMIT 1000
                """,
                (config.run_id, config.queue_name),
            ),
            "hydrate_frontier_dags": (
                f"""
                SELECT hydrated.dag_id, hydrated.serialized_dag
                FROM {SCHEDULER_SCHEMA}.hydrate_frontier_dags() hydrated
                JOIN {SCHEDULER_SCHEMA}.dag run_dag
                  ON run_dag.id = hydrated.dag_id
                WHERE run_dag.planner = %s
                LIMIT 1000
                """,
                (config.planner,),
            ),
            "hydrate_frontier_jobs": (
                f"""
                SELECT dag_id, job
                FROM {SCHEDULER_SCHEMA}.hydrate_frontier_jobs(%s::uuid[])
                """,
                (dag_ids,),
            ),
            "priority_lookup": (
                f"""
                SELECT id, priority FROM {SCHEDULER_SCHEMA}.job
                WHERE name = %s
                  AND id = ANY(%s::uuid[])
                """,
                (config.queue_name, job_ids),
            ),
        }
        plans: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        before = self.statement_snapshot()
        with self.connection.cursor() as cursor:
            for name, (query, params) in queries.items():
                started = time.perf_counter()
                cursor.execute(query, params)
                returned_rows = len(cursor.fetchall())
                timings[name] = {
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "returned_rows": returned_rows,
                }
                cursor.execute(
                    "EXPLAIN (ANALYZE, BUFFERS, WAL, FORMAT JSON) " + query,
                    params,
                )
                row = cursor.fetchone()
                plans[name] = row["QUERY PLAN"] if row else None
        after = self.statement_snapshot()
        return {
            "plans": plans,
            "timings": timings,
            "pg_stat_statements": {
                "installed": after["installed"],
                "available": after["available"],
                "shared_preload_libraries": after["shared_preload_libraries"],
                "track": after["track"],
                "reset_performed": False,
                "deltas": statement_delta(before["statements"], after["statements"]),
            },
        }


def connect(config: StressConfig) -> psycopg.Connection[dict[str, Any]]:
    kwargs = dict(config.database)
    kwargs.setdefault("connect_timeout", 10)
    kwargs["application_name"] = f"scheduler-db-stresser:{config.run_id[:40]}"
    return psycopg.connect(**kwargs, row_factory=dict_row)


def load_config(path: str, overrides: Mapping[str, Any]) -> StressConfig:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Stress config must contain a JSON object")
    values = dict(raw)
    values.update({key: value for key, value in overrides.items() if value is not None})
    return StressConfig.from_mapping(values)


def source_revision() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def json_value(value: Any) -> Any:
    if isinstance(value, (datetime, uuid.UUID, Decimal, Path)):
        return str(value)
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def emit_report(report: Mapping[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, default=json_value)
    print(rendered)
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")


def public_config(config: StressConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "target_dag_count": config.target_dag_count,
        "nodes_per_dag": config.nodes_per_dag,
        "graph_shape": config.graph_shape,
        "workload_profile": config.workload_profile,
        "queue_name": config.queue_name,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "projection_mode": config.projection_mode,
        "analyze_after_seed": config.analyze_after_seed,
        "executor": config.executor,
        "endpoint": config.endpoint,
        "active_lease_seconds": config.active_lease_seconds,
        "report": config.report,
        "allow_schema_transition": config.allow_schema_transition,
        "config_hash": config.cohort_hash,
    }


def command_report(
    command: str, config: StressConfig, payload: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "command": command,
        "generated_at": datetime.now(timezone.utc),
        "generator_version": GENERATOR_VERSION,
        "source": source_revision(),
        "config": public_config(config),
        **payload,
    }


def run_database_command(command: str, config: StressConfig) -> dict[str, Any]:
    with connect(config) as connection:
        generator = CorpusGenerator(connection)
        scheduler_schema_version = generator.initialize()
        generator.acquire_lock(config.run_id)
        try:
            if command == "seed":
                manifest = generator.prepare_manifest(config, scheduler_schema_version)
            else:
                manifest = generator.manifest(config.run_id)
                generator._validate_manifest(config, manifest)
                if (
                    int(manifest["scheduler_schema_version"])
                    != scheduler_schema_version
                ):
                    raise RuntimeError(
                        "Scheduler schema version does not match the run manifest"
                    )
            current_count = int(manifest["high_water_mark"])
            if current_count > config.target_dag_count:
                raise RuntimeError(
                    f"Refusing to shrink cohort from {current_count} to "
                    f"{config.target_dag_count} DAGs"
                )

            if command == "seed":
                generator.ensure_queue(config)
                generator.create_staging_tables()
                if current_count:
                    existing = generator.verify(config, require_target=False)
                    if not existing["passed"]:
                        raise RuntimeError(
                            "Existing cohort failed verification before growth"
                        )
                before = generator.database_snapshot(config)
                seed_result = generator.seed(config, current_count)
                if config.analyze_after_seed:
                    generator.analyze(config)
                verification = generator.verify(config)
                after = generator.database_snapshot(config)
                return command_report(
                    command,
                    config,
                    {
                        "scheduler_schema_version": scheduler_schema_version,
                        "plan": build_plan(config, current_count),
                        "seed": seed_result,
                        "verification": verification,
                        "database_before": before,
                        "database_after": after,
                        "database_delta": statistics_delta(before, after),
                    },
                )

            if command == "verify":
                verification = generator.verify(config)
                return command_report(
                    command,
                    config,
                    {
                        "scheduler_schema_version": scheduler_schema_version,
                        "verification": verification,
                    },
                )

            if command == "benchmark":
                if current_count != config.target_dag_count:
                    raise RuntimeError(
                        f"Checkpoint has {current_count} DAGs; seed target "
                        f"{config.target_dag_count} before benchmarking"
                    )
                return command_report(
                    command,
                    config,
                    {
                        "scheduler_schema_version": scheduler_schema_version,
                        "database": generator.database_snapshot(config),
                        "benchmark": generator.benchmark(config),
                    },
                )
        finally:
            generator.release_lock(config.run_id)
    raise ValueError(f"Unsupported command: {command}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to the JSON config")
    parser.add_argument("--run-id")
    parser.add_argument("--target-dags", dest="target_dag_count", type=int)
    parser.add_argument("--nodes-per-dag", type=int)
    parser.add_argument("--graph-shape", choices=sorted(GRAPH_SHAPES))
    parser.add_argument("--workload-profile", choices=sorted(WORKLOAD_PROFILES))
    parser.add_argument("--queue-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--projection-mode", choices=sorted(PROJECTION_MODES))
    parser.add_argument("--executor")
    parser.add_argument("--endpoint")
    parser.add_argument("--active-lease-seconds", type=int)
    parser.add_argument("--report")
    parser.add_argument("--allow-schema-transition", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and measure persistent scheduler PostgreSQL corpora"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "seed", "verify", "benchmark"):
        command_parser = subparsers.add_parser(command)
        add_common_arguments(command_parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "run_id": args.run_id,
        "target_dag_count": args.target_dag_count,
        "nodes_per_dag": args.nodes_per_dag,
        "graph_shape": args.graph_shape,
        "workload_profile": args.workload_profile,
        "queue_name": args.queue_name,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "projection_mode": args.projection_mode,
        "executor": args.executor,
        "endpoint": args.endpoint,
        "active_lease_seconds": args.active_lease_seconds,
        "report": args.report,
        "allow_schema_transition": args.allow_schema_transition,
    }
    try:
        config = load_config(args.config, overrides)
        if args.command == "plan" or args.dry_run:
            report = command_report(args.command, config, {"plan": build_plan(config)})
        else:
            report = run_database_command(args.command, config)
        emit_report(report, config.report)
        if (
            not args.dry_run
            and "verification" in report
            and not report["verification"]["passed"]
        ):
            return 1
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        psycopg.Error,
        json.JSONDecodeError,
    ) as error:
        print(f"scheduler_db_stresser: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
