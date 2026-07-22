#!/usr/bin/env python3
"""Verify one scheduler stress cohort against PostgreSQL state."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row

SCHEDULER_SCHEMA = "marie_scheduler"
STRESS_SCHEMA = "marie_stress"
DATABASE_KEYS = {
    "host",
    "port",
    "dbname",
    "user",
    "sslmode",
    "connect_timeout",
}
STATUSES = {"pass", "fail", "skipped", "error"}
TERMINAL_STATES = {"completed", "failed", "cancelled", "expired", "skipped"}
ATTEMPT_CHECK_CATEGORIES = {
    "active_missing_attempt_identity": "attempts",
    "expired_active_run_leases": "attempts",
    "active_attempt_identity": "attempts",
    "attempt_identity_scope": "attempts",
    "dispatched_missing_terminal_or_recovery": "terminals",
    "duplicate_accepted_completed_terminal_by_job": "terminals",
    "accepted_terminal_outcome_conflict": "terminals",
    "stale_terminal_accepted": "terminals",
    "terminal_job_retains_lease": "terminals",
    "dispatched_without_gateway_instance": "attempts",
    "accepted_terminal_missing_terminal_gateway": "terminals",
}
ATTEMPT_CHECK_NAMES = tuple(ATTEMPT_CHECK_CATEGORIES)


SCOPE_CTES = f"""
params AS (
    SELECT
        %s::TEXT AS run_id,
        %s::INTEGER AS sample_limit,
        %s::TIMESTAMPTZ AS settle_deadline
),
manifest AS MATERIALIZED (
    SELECT manifest.*
    FROM {STRESS_SCHEMA}.run_manifest manifest
    JOIN params ON params.run_id = manifest.run_id
),
scoped_dags AS MATERIALIZED (
    SELECT dag.*
    FROM {SCHEDULER_SCHEMA}.dag dag
    JOIN params ON dag.planner = 'stress:' || params.run_id
),
scoped_jobs AS MATERIALIZED (
    SELECT job.*
    FROM {SCHEDULER_SCHEMA}.job job
    JOIN params
      ON job.data->'metadata'->>'stress_run_id' = params.run_id
)
"""


@dataclass(frozen=True)
class CheckSpec:
    name: str
    category: str
    expectation: str
    violations_sql: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str
    status: str
    observed: Any
    expected: Any
    bad_rows: int
    sample: list[Any]
    query_duration_ms: float
    reason: str | None = None
    mandatory: bool = True

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Unsupported check status: {self.status}")


def _spec(
    name: str,
    category: str,
    expectation: str,
    violations_sql: str,
) -> CheckSpec:
    return CheckSpec(name, category, expectation, violations_sql.strip())


CHECKS = (
    _spec(
        "manifest_checkpoint",
        "structure",
        "Manifest target, high-water mark, and committed DAG count agree.",
        """
        SELECT format(
            'target=%s high_water=%s dags=%s',
            manifest.target_dag_count,
            manifest.high_water_mark,
            counts.dags
        ) AS id
        FROM manifest
        CROSS JOIN (SELECT COUNT(*) AS dags FROM scoped_dags) counts
        WHERE manifest.target_dag_count <> manifest.high_water_mark
           OR manifest.high_water_mark <> counts.dags
        """,
    ),
    _spec(
        "job_cardinality",
        "structure",
        "The run has nodes_per_dag jobs for every committed DAG.",
        """
        SELECT format('expected=%s observed=%s', expected_jobs, observed_jobs) AS id
        FROM (
            SELECT
                manifest.high_water_mark * manifest.nodes_per_dag AS expected_jobs,
                (SELECT COUNT(*) FROM scoped_jobs) AS observed_jobs
            FROM manifest
        ) counts
        WHERE expected_jobs <> observed_jobs
        """,
    ),
    _spec(
        "logical_identity_unique",
        "structure",
        "Every logical DAG/node coordinate is present once and within the manifest range.",
        """
        WITH job_identity AS (
            SELECT
                job.id,
                job.dag_id,
                job.data->'metadata'->>'stress_dag_index' AS dag_index,
                job.data->'metadata'->>'stress_node_index' AS node_index
            FROM scoped_jobs job
        )
        SELECT 'invalid:' || id::TEXT AS id
        FROM job_identity, manifest
        WHERE CASE
            WHEN dag_index ~ '^[0-9]+$' AND node_index ~ '^[0-9]+$'
            THEN dag_index::BIGINT < 0
              OR dag_index::BIGINT >= manifest.high_water_mark
              OR node_index::INTEGER < 0
              OR node_index::INTEGER >= manifest.nodes_per_dag
            ELSE TRUE
        END
        UNION ALL
        SELECT format('duplicate-coordinate:%s:%s', dag_index, node_index)
        FROM job_identity
        GROUP BY dag_index, node_index
        HAVING COUNT(*) <> 1
        UNION ALL
        SELECT 'dag-index-maps-to-multiple-dags:' || dag_index
        FROM job_identity
        GROUP BY dag_index
        HAVING COUNT(DISTINCT dag_id) <> 1
        """,
    ),
    _spec(
        "job_dag_run_scope",
        "structure",
        "Every tagged job belongs to a DAG in the same run.",
        """
        SELECT job.id::TEXT AS id
        FROM scoped_jobs job
        LEFT JOIN scoped_dags dag ON dag.id = job.dag_id
        WHERE dag.id IS NULL
        UNION ALL
        SELECT 'untagged:' || job.id::TEXT
        FROM {SCHEDULER_SCHEMA}.job job
        JOIN scoped_dags dag ON dag.id = job.dag_id
        JOIN params ON TRUE
        WHERE job.data->'metadata'->>'stress_run_id' IS DISTINCT FROM params.run_id
        """,
    ),
    _spec(
        "serialized_graph_matches_jobs",
        "structure",
        "Serialized nodes and dependencies match persisted jobs exactly.",
        """
        WITH nodes AS (
            SELECT
                dag.id AS dag_id,
                node.value->>'task_id' AS task_id,
                node.value->'dependencies' AS dependencies
            FROM scoped_dags dag
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(dag.serialized_dag->'nodes') = 'array'
                    THEN dag.serialized_dag->'nodes'
                    ELSE '[]'::JSONB
                END
            ) node(value)
        ), invalid_serialized AS (
            SELECT dag.id
            FROM scoped_dags dag
            WHERE jsonb_typeof(dag.serialized_dag->'nodes') IS DISTINCT FROM 'array'
        )
        SELECT 'invalid-serialized:' || id::TEXT AS id FROM invalid_serialized
        UNION ALL
        SELECT format('duplicate-node:%s:%s', dag_id, task_id)
        FROM nodes
        GROUP BY dag_id, task_id
        HAVING task_id IS NULL OR COUNT(*) <> 1
        UNION ALL
        SELECT format('node-without-job:%s:%s', node.dag_id, node.task_id)
        FROM nodes node
        LEFT JOIN scoped_jobs job
          ON job.dag_id = node.dag_id
         AND job.id::TEXT = node.task_id
        WHERE job.id IS NULL
        UNION ALL
        SELECT 'job-without-node:' || job.id::TEXT
        FROM scoped_jobs job
        LEFT JOIN nodes node
          ON node.dag_id = job.dag_id
         AND node.task_id = job.id::TEXT
        WHERE node.task_id IS NULL
        UNION ALL
        SELECT 'serialized-dependencies:' || job.id::TEXT
        FROM scoped_jobs job
        JOIN nodes node
          ON node.dag_id = job.dag_id
         AND node.task_id = job.id::TEXT
        WHERE COALESCE(node.dependencies, '[]'::JSONB)
              IS DISTINCT FROM COALESCE(job.dependencies, '[]'::JSONB)
        """,
    ),
    _spec(
        "normalized_dependencies_match",
        "dependencies",
        "Normalized and JSON dependencies agree and remain inside each DAG.",
        """
        WITH json_dependencies AS (
            SELECT job.id AS job_id, job.dag_id, dependency.value AS depends_on_id
            FROM scoped_jobs job
            CROSS JOIN LATERAL jsonb_array_elements_text(
                CASE
                    WHEN jsonb_typeof(job.dependencies) = 'array'
                    THEN job.dependencies
                    ELSE '[]'::JSONB
                END
            ) dependency(value)
        ), normalized AS (
            SELECT
                dependency.job_id,
                child.dag_id,
                dependency.depends_on_id::TEXT AS depends_on_id,
                parent.dag_id AS parent_dag_id
            FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
            JOIN scoped_jobs child
              ON child.name = dependency.job_name
             AND child.id = dependency.job_id
            LEFT JOIN {SCHEDULER_SCHEMA}.job parent
              ON parent.name = dependency.depends_on_name
             AND parent.id = dependency.depends_on_id
        )
        SELECT 'invalid-json:' || job.id::TEXT AS id
        FROM scoped_jobs job
        WHERE jsonb_typeof(job.dependencies) IS DISTINCT FROM 'array'
        UNION ALL
        SELECT format('json-only:%s:%s', job_id, depends_on_id) AS id
        FROM (
            SELECT job_id, depends_on_id FROM json_dependencies
            EXCEPT
            SELECT job_id, depends_on_id FROM normalized
        ) missing
        UNION ALL
        SELECT format('normalized-only:%s:%s', job_id, depends_on_id)
        FROM (
            SELECT job_id, depends_on_id FROM normalized
            EXCEPT
            SELECT job_id, depends_on_id FROM json_dependencies
        ) extra
        UNION ALL
        SELECT format('cross-dag:%s:%s', job_id, depends_on_id)
        FROM normalized
        WHERE parent_dag_id IS NULL OR parent_dag_id <> dag_id
        """,
    ),
    _spec(
        "dependency_levels_acyclic",
        "dependencies",
        "Every dependency edge moves from a higher parent level to a lower child level.",
        f"""
        SELECT format('%s->%s', parent.id, child.id) AS id
        FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
        JOIN scoped_jobs child
          ON child.name = dependency.job_name
         AND child.id = dependency.job_id
        JOIN {SCHEDULER_SCHEMA}.job parent
          ON parent.name = dependency.depends_on_name
         AND parent.id = dependency.depends_on_id
        WHERE parent.dag_id <> child.dag_id
           OR parent.job_level <= child.job_level
        """,
    ),
    _spec(
        "graph_root_leaf_shape",
        "dependencies",
        "Each DAG has the root and leaf counts required by its configured shape.",
        f"""
        WITH topology AS (
            SELECT
                dag.id,
                COUNT(*) FILTER (
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
                        WHERE dependency.job_name = job.name
                          AND dependency.job_id = job.id
                    )
                ) AS roots,
                COUNT(*) FILTER (
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
                        WHERE dependency.depends_on_name = job.name
                          AND dependency.depends_on_id = job.id
                    )
                ) AS leaves
            FROM scoped_dags dag
            JOIN scoped_jobs job ON job.dag_id = dag.id
            GROUP BY dag.id
        )
        SELECT format('%s:roots=%s:leaves=%s', topology.id, roots, leaves) AS id
        FROM topology, manifest
        WHERE roots <> 1
           OR leaves <> CASE manifest.graph_shape
                WHEN 'fanout' THEN GREATEST(manifest.nodes_per_dag - 1, 1)
                ELSE 1
              END
        """,
    ),
    _spec(
        "search_projection_cardinality",
        "structure",
        "Full projection has one search row per job; scheduler projection has none.",
        f"""
        WITH projected AS (
            SELECT
                document.job_id,
                document.dag_id,
                COUNT(*) AS rows
            FROM {SCHEDULER_SCHEMA}.job_search_document document
            WHERE document.planner = (SELECT 'stress:' || run_id FROM params)
            GROUP BY document.job_id, document.dag_id
        )
        SELECT 'job:' || job.id::TEXT AS id
        FROM scoped_jobs job
        LEFT JOIN projected
          ON projected.job_id = job.id
         AND projected.dag_id = job.dag_id
        CROSS JOIN manifest
        WHERE (manifest.projection_mode = 'full' AND COALESCE(projected.rows, 0) <> 1)
           OR (manifest.projection_mode = 'scheduler' AND projected.rows IS NOT NULL)
        UNION ALL
        SELECT 'extra:' || projected.job_id::TEXT
        FROM projected
        LEFT JOIN scoped_jobs job
          ON job.id = projected.job_id
         AND job.dag_id = projected.dag_id
        WHERE job.id IS NULL
        """,
    ),
    _spec(
        "dependency_start_order",
        "dependencies",
        "A child starts only after every required parent succeeds.",
        f"""
        SELECT format('%s->%s', parent.id, child.id) AS id
        FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
        JOIN scoped_jobs child
          ON child.name = dependency.job_name
         AND child.id = dependency.job_id
        JOIN {SCHEDULER_SCHEMA}.job parent
          ON parent.name = dependency.depends_on_name
         AND parent.id = dependency.depends_on_id
        WHERE child.started_on IS NOT NULL
          AND (
              parent.state::TEXT NOT IN ('completed', 'skipped')
              OR parent.completed_on IS NULL
              OR parent.completed_on > child.started_on
          )
        """,
    ),
    _spec(
        "failed_dependency_not_started",
        "dependencies",
        "Failed dependencies do not expose ordinary descendants.",
        f"""
        SELECT format('%s->%s', parent.id, child.id) AS id
        FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
        JOIN scoped_jobs child
          ON child.name = dependency.job_name
         AND child.id = dependency.job_id
        JOIN {SCHEDULER_SCHEMA}.job parent
          ON parent.name = dependency.depends_on_name
         AND parent.id = dependency.depends_on_id
        WHERE parent.state::TEXT IN ('failed', 'cancelled', 'expired')
          AND (child.started_on IS NOT NULL OR child.state::TEXT = 'active')
          AND COALESCE(child.branch_metadata->>'skipped', 'false') <> 'true'
        """,
    ),
    _spec(
        "terminal_dag_consistency",
        "terminals",
        "DAG state agrees with all member job states.",
        """
        SELECT dag.id::TEXT AS id
        FROM scoped_dags dag
        WHERE (
            dag.state = 'completed'
            AND EXISTS (
                SELECT 1 FROM scoped_jobs job
                WHERE job.dag_id = dag.id
                  AND job.state::TEXT NOT IN ('completed', 'skipped')
            )
        ) OR (
            dag.state IN ('failed', 'cancelled')
            AND (
                EXISTS (
                    SELECT 1 FROM scoped_jobs job
                    WHERE job.dag_id = dag.id
                      AND job.state::TEXT NOT IN ('completed', 'skipped', 'failed', 'cancelled', 'expired')
                )
                OR NOT EXISTS (
                    SELECT 1 FROM scoped_jobs job
                    WHERE job.dag_id = dag.id
                      AND job.state::TEXT IN ('failed', 'cancelled', 'expired')
                )
            )
        ) OR (
            dag.state = 'active'
            AND (
                NOT EXISTS (
                    SELECT 1 FROM scoped_jobs job
                    WHERE job.dag_id = dag.id AND job.state::TEXT = 'active'
                )
                OR EXISTS (
                    SELECT 1 FROM scoped_jobs job
                    WHERE job.dag_id = dag.id
                      AND job.state::TEXT IN ('failed', 'cancelled', 'expired')
                )
            )
        ) OR (
            dag.state = 'created'
            AND (
                NOT EXISTS (
                    SELECT 1 FROM scoped_jobs job
                    WHERE job.dag_id = dag.id
                      AND job.state::TEXT IN ('created', 'retry')
                )
                OR EXISTS (
                    SELECT 1 FROM scoped_jobs job
                    WHERE job.dag_id = dag.id
                      AND job.state::TEXT IN ('active', 'failed', 'cancelled', 'expired')
                )
            )
        )
        """,
    ),
)


def inspect_serialized_plan(plan: Mapping[str, Any] | Any) -> dict[str, Any]:
    payload = plan.model_dump() if hasattr(plan, "model_dump") else plan
    nodes = payload.get("nodes", [])
    node_ids = [str(node.get("task_id", "")) for node in nodes]
    counts = Counter(node_ids)
    duplicates = sorted(node_id for node_id, count in counts.items() if count > 1)
    node_set = set(node_ids)
    missing = sorted(
        {
            str(dependency)
            for node in nodes
            for dependency in node.get("dependencies", []) or []
            if str(dependency) not in node_set
        }
    )
    dependents: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for node in nodes:
        node_id = str(node.get("task_id", ""))
        for dependency in node.get("dependencies", []) or []:
            dependency_id = str(dependency)
            if dependency_id in dependents:
                dependents[dependency_id].append(node_id)
                indegree[node_id] += 1
    pending = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while pending:
        node_id = pending.popleft()
        visited += 1
        for child_id in dependents[node_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                pending.append(child_id)
    edge_count = sum(len(node.get("dependencies", []) or []) for node in nodes)
    return {
        "node_count": len(nodes),
        "edge_count": edge_count,
        "root_count": sum(not (node.get("dependencies", []) or []) for node in nodes),
        "leaf_count": sum(not dependents[node_id] for node_id in node_ids),
        "duplicate_node_ids": duplicates,
        "missing_dependencies": missing,
        "cyclic": bool(nodes) and visited != len(nodes),
    }


def load_database_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Stress config must contain a JSON object")
    database = payload.get("database") or {}
    if not isinstance(database, dict):
        raise ValueError("database must be a JSON object")
    if "password" in database:
        raise ValueError("Put the database password in PGPASSWORD, not the config")
    unknown = set(database) - DATABASE_KEYS
    if unknown:
        raise ValueError(
            "Unsupported database config fields: " + ", ".join(sorted(unknown))
        )
    return dict(database)


def connect(
    run_id: str, database: Mapping[str, Any]
) -> psycopg.Connection[dict[str, Any]]:
    kwargs = dict(database)
    kwargs.setdefault("connect_timeout", 10)
    kwargs["application_name"] = f"scheduler-correctness:{run_id[:40]}"
    return psycopg.connect(**kwargs, row_factory=dict_row)


def _query_for(spec: CheckSpec) -> str:
    return f"""
    /* scheduler-correctness:{spec.name} */
    WITH {SCOPE_CTES},
    violations AS MATERIALIZED (
        {spec.violations_sql}
    )
    SELECT
        COUNT(*)::BIGINT AS bad_rows,
        jsonb_build_object('bad_rows', COUNT(*)) AS observed,
        jsonb_build_object('bad_rows', 0) AS expected,
        COALESCE((
            SELECT jsonb_agg(sample.id ORDER BY sample.id)
            FROM (
                SELECT id
                FROM violations
                ORDER BY id
                LIMIT (SELECT sample_limit FROM params)
            ) sample
        ), '[]'::JSONB) AS sample
    FROM violations
    """


def skipped_result(
    name: str, category: str, reason: str, *, mandatory: bool = False
) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        status="skipped",
        observed=None,
        expected=None,
        bad_rows=0,
        sample=[],
        query_duration_ms=0.0,
        reason=reason,
        mandatory=mandatory,
    )


class SchedulerCorrectnessVerifier:
    def __init__(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
        sample_limit: int,
        settle_deadline: datetime,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id is required")
        if sample_limit <= 0 or sample_limit > 1_000:
            raise ValueError("sample_limit must be between 1 and 1000")
        self.connection = connection
        self.run_id = run_id
        self.sample_limit = sample_limit
        self.settle_deadline = settle_deadline

    def manifest(self) -> dict[str, Any]:
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(
                    f"SELECT * FROM {STRESS_SCHEMA}.run_manifest WHERE run_id = %s",
                    (self.run_id,),
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"Unknown stress run_id: {self.run_id}")
        return row

    def run_check(self, spec: CheckSpec) -> CheckResult:
        started = time.perf_counter()
        try:
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(
                        _query_for(spec),
                        (self.run_id, self.sample_limit, self.settle_deadline),
                    )
                    row = cursor.fetchone()
            if row is None:
                raise RuntimeError("check query returned no result")
            bad_rows = int(row["bad_rows"])
            return CheckResult(
                name=spec.name,
                category=spec.category,
                status="pass" if bad_rows == 0 else "fail",
                observed=row["observed"],
                expected=row["expected"],
                bad_rows=bad_rows,
                sample=list(row["sample"] or [])[: self.sample_limit],
                query_duration_ms=(time.perf_counter() - started) * 1000,
                reason=None if bad_rows == 0 else spec.expectation,
            )
        except (psycopg.Error, RuntimeError, KeyError, TypeError, ValueError) as error:
            return CheckResult(
                name=spec.name,
                category=spec.category,
                status="error",
                observed=None,
                expected={"bad_rows": 0},
                bad_rows=0,
                sample=[],
                query_duration_ms=(time.perf_counter() - started) * 1000,
                reason=str(error),
            )

    def run_attempt_checks(self) -> list[CheckResult]:
        started = time.perf_counter()
        query = f"""
            /* scheduler-correctness:shared-attempt-invariants */
            SELECT check_name, category, bad_rows, sample, expectation
            FROM {SCHEDULER_SCHEMA}.scheduler_attempt_invariant_checks(
                %s,
                NULL,
                NULL,
                %s,
                %s
            )
        """
        try:
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(
                        query,
                        (
                            f"stress:{self.run_id}",
                            self.settle_deadline,
                            self.sample_limit,
                        ),
                    )
                    rows = cursor.fetchall()
            duration_ms = (time.perf_counter() - started) * 1000
            by_name = {str(row["check_name"]): row for row in rows}
            missing = set(ATTEMPT_CHECK_NAMES) - set(by_name)
            unexpected = set(by_name) - set(ATTEMPT_CHECK_NAMES)
            if missing or unexpected:
                raise RuntimeError(
                    "Shared attempt invariant contract mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )

            results = []
            before_settle_deadline = datetime.now(timezone.utc) < self.settle_deadline
            for name in ATTEMPT_CHECK_NAMES:
                row = by_name[name]
                if name == "dispatched_missing_terminal_or_recovery" and (
                    before_settle_deadline
                ):
                    results.append(
                        skipped_result(
                            name,
                            str(row["category"]),
                            "The settle deadline has not been reached.",
                        )
                    )
                    continue
                bad_rows = int(row["bad_rows"])
                results.append(
                    CheckResult(
                        name=name,
                        category=str(row["category"]),
                        status="pass" if bad_rows == 0 else "fail",
                        observed={"bad_rows": bad_rows},
                        expected={"bad_rows": 0},
                        bad_rows=bad_rows,
                        sample=list(row["sample"] or [])[: self.sample_limit],
                        query_duration_ms=duration_ms,
                        reason=None if bad_rows == 0 else str(row["expectation"]),
                    )
                )
            return results
        except (psycopg.Error, RuntimeError, KeyError, TypeError, ValueError) as error:
            duration_ms = (time.perf_counter() - started) * 1000
            return [
                CheckResult(
                    name=name,
                    category=ATTEMPT_CHECK_CATEGORIES[name],
                    status="error",
                    observed=None,
                    expected={"bad_rows": 0},
                    bad_rows=0,
                    sample=[],
                    query_duration_ms=duration_ms,
                    reason=str(error),
                )
                for name in ATTEMPT_CHECK_NAMES
            ]

    def verify(self, gateway_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
        manifest = self.manifest()
        results: list[CheckResult] = []
        for spec in CHECKS:
            if (
                spec.name == "graph_root_leaf_shape"
                and manifest["graph_shape"] == "mixed"
            ):
                results.append(
                    skipped_result(
                        spec.name,
                        spec.category,
                        "Mixed cohorts validate each serialized graph but have no single root/leaf formula.",
                    )
                )
                continue
            results.append(self.run_check(spec))
        results.extend(self.run_attempt_checks())
        results.extend(self._gateway_checks(gateway_report))
        counts = Counter(result.status for result in results)
        passed = not any(
            result.mandatory and result.status in {"fail", "error"}
            for result in results
        )
        return {
            "run_id": self.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "settle_deadline": self.settle_deadline.isoformat(),
            "sample_limit": self.sample_limit,
            "manifest": {
                "target_dag_count": manifest["target_dag_count"],
                "high_water_mark": manifest["high_water_mark"],
                "nodes_per_dag": manifest["nodes_per_dag"],
                "graph_shape": manifest["graph_shape"],
                "workload_profile": manifest["workload_profile"],
                "projection_mode": manifest["projection_mode"],
                "scheduler_schema_version": manifest["scheduler_schema_version"],
            },
            "passed": passed,
            "status_counts": dict(counts),
            "checks": [asdict(result) for result in results],
        }

    def _gateway_checks(
        self, gateway_report: Mapping[str, Any] | None
    ) -> list[CheckResult]:
        if gateway_report is None:
            reason = "No gateway report was supplied."
            return [
                skipped_result("gateway_scheduler_identity", "events", reason),
                skipped_result("gateway_event_order", "events", reason),
                skipped_result("gateway_terminal_agreement", "events", reason),
                skipped_result("post_drain_capacity", "capacity", reason),
            ]
        jobs = gateway_report.get("jobs")
        if not isinstance(jobs, list):
            return [
                CheckResult(
                    name="gateway_report_shape",
                    category="events",
                    status="error",
                    observed=type(jobs).__name__,
                    expected="jobs array",
                    bad_rows=0,
                    sample=[],
                    query_duration_ms=0.0,
                    reason="Gateway report does not contain a jobs array.",
                )
            ]
        return [
            self._gateway_identity(jobs),
            self._gateway_event_order(jobs),
            self._gateway_terminal_agreement(jobs),
            self._post_drain_capacity(gateway_report),
        ]

    def _gateway_identity(self, jobs: list[Any]) -> CheckResult:
        job_ids = [
            str(job["job_id"])
            for job in jobs
            if isinstance(job, dict) and job.get("job_id")
        ]
        duplicates = sorted(
            job_id for job_id, count in Counter(job_ids).items() if count > 1
        )
        if not job_ids:
            return skipped_result(
                "gateway_scheduler_identity",
                "events",
                "The gateway report contains no accepted scheduler identities.",
            )
        started = time.perf_counter()
        query = f"""
            SELECT submitted.id, COUNT(job.id) AS matches
            FROM unnest(%s::TEXT[]) submitted(id)
            LEFT JOIN {SCHEDULER_SCHEMA}.job job
              ON job.id::TEXT = submitted.id
             AND job.data->'metadata'->>'stress_run_id' = %s
            GROUP BY submitted.id
        """
        try:
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(query, (job_ids, self.run_id))
                    rows = cursor.fetchall()
            bad = duplicates + [str(row["id"]) for row in rows if row["matches"] != 1]
            return CheckResult(
                name="gateway_scheduler_identity",
                category="events",
                status="pass" if not bad else "fail",
                observed={"submitted_ids": len(job_ids), "bad_rows": len(bad)},
                expected={"matches_per_id": 1, "duplicates": 0},
                bad_rows=len(bad),
                sample=bad[: self.sample_limit],
                query_duration_ms=(time.perf_counter() - started) * 1000,
                reason=(
                    None if not bad else "Gateway job IDs must map once inside the run."
                ),
            )
        except psycopg.Error as error:
            return CheckResult(
                name="gateway_scheduler_identity",
                category="events",
                status="error",
                observed=None,
                expected={"matches_per_id": 1},
                bad_rows=0,
                sample=[],
                query_duration_ms=(time.perf_counter() - started) * 1000,
                reason=str(error),
            )

    def _gateway_event_order(self, jobs: list[Any]) -> CheckResult:
        bad: list[str] = []
        observed_events = 0
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                bad.append(f"index:{index}")
                continue
            events = job.get("raw_events")
            if not isinstance(events, list) or not events:
                continue
            observed_events += len(events)
            kinds = [str(event).rsplit(".", 1)[-1] for event in events]
            positions = {
                kind: kinds.index(kind)
                for kind in ("scheduled", "started", "completed", "failed")
                if kind in kinds
            }
            terminal_positions = [
                positions[kind] for kind in ("completed", "failed") if kind in positions
            ]
            invalid = (
                (
                    "scheduled" in positions
                    and "started" in positions
                    and positions["scheduled"] > positions["started"]
                )
                or (
                    "started" in positions
                    and terminal_positions
                    and positions["started"] > min(terminal_positions)
                )
                or (
                    "scheduled" in positions
                    and terminal_positions
                    and positions["scheduled"] > min(terminal_positions)
                )
                or ("completed" in positions and "failed" in positions)
                or sum(kind in {"completed", "failed"} for kind in kinds) > 1
            )
            if invalid:
                bad.append(str(job.get("job_id") or job.get("request_id") or index))
        if observed_events == 0:
            return skipped_result(
                "gateway_event_order",
                "events",
                "The gateway report contains no scheduler event evidence.",
            )
        return CheckResult(
            name="gateway_event_order",
            category="events",
            status="pass" if not bad else "fail",
            observed={"events": observed_events, "bad_rows": len(bad)},
            expected={"bad_rows": 0},
            bad_rows=len(bad),
            sample=bad[: self.sample_limit],
            query_duration_ms=0.0,
            reason=None if not bad else "Gateway events are out of order or conflict.",
        )

    def _gateway_terminal_agreement(self, jobs: list[Any]) -> CheckResult:
        terminal_jobs = {
            str(job["job_id"]): str(job["terminal_status"])
            for job in jobs
            if isinstance(job, dict)
            and job.get("job_id")
            and job.get("terminal_status")
        }
        if not terminal_jobs:
            return skipped_result(
                "gateway_terminal_agreement",
                "events",
                "The gateway report contains no terminal outcomes.",
            )
        started = time.perf_counter()
        query = f"""
            SELECT submitted.id, dag.state
            FROM unnest(%s::TEXT[]) submitted(id)
            LEFT JOIN {SCHEDULER_SCHEMA}.job job
              ON job.id::TEXT = submitted.id
             AND job.data->'metadata'->>'stress_run_id' = %s
            LEFT JOIN {SCHEDULER_SCHEMA}.dag dag ON dag.id = job.dag_id
        """
        try:
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(query, (list(terminal_jobs), self.run_id))
                    rows = cursor.fetchall()
            bad = []
            for row in rows:
                gateway_state = terminal_jobs[str(row["id"])]
                database_state = row["state"]
                agrees = gateway_state == "completed" and database_state == "completed"
                agrees = agrees or (
                    gateway_state == "failed"
                    and database_state in {"failed", "cancelled"}
                )
                if not agrees:
                    bad.append(
                        f"{row['id']}:gateway={gateway_state}:database={database_state}"
                    )
            return CheckResult(
                name="gateway_terminal_agreement",
                category="events",
                status="pass" if not bad else "fail",
                observed={"terminal_jobs": len(terminal_jobs), "bad_rows": len(bad)},
                expected={"bad_rows": 0},
                bad_rows=len(bad),
                sample=bad[: self.sample_limit],
                query_duration_ms=(time.perf_counter() - started) * 1000,
                reason=(
                    None
                    if not bad
                    else "Gateway outcomes disagree with authoritative DAG state."
                ),
            )
        except psycopg.Error as error:
            return CheckResult(
                name="gateway_terminal_agreement",
                category="events",
                status="error",
                observed=None,
                expected={"bad_rows": 0},
                bad_rows=0,
                sample=[],
                query_duration_ms=(time.perf_counter() - started) * 1000,
                reason=str(error),
            )

    def _post_drain_capacity(self, report: Mapping[str, Any]) -> CheckResult:
        snapshot = report.get("post_drain_capacity")
        if not isinstance(snapshot, Mapping):
            return skipped_result(
                "post_drain_capacity",
                "capacity",
                "The gateway report has no post_drain_capacity snapshot.",
            )
        used = int(snapshot.get("used", 0))
        holders = int(snapshot.get("holder_count", 0))
        bad_rows = int(used != 0) + int(holders != 0)
        return CheckResult(
            name="post_drain_capacity",
            category="capacity",
            status="pass" if bad_rows == 0 else "fail",
            observed={"used": used, "holder_count": holders},
            expected={"used": 0, "holder_count": 0},
            bad_rows=bad_rows,
            sample=[],
            query_duration_ms=0.0,
            reason=(
                None
                if bad_rows == 0
                else "Capacity usage or holders remain after drain."
            ),
        )


def parse_deadline(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    deadline = datetime.fromisoformat(normalized)
    if deadline.tzinfo is None:
        raise ValueError("settle_deadline must include a timezone")
    return deadline.astimezone(timezone.utc)


def load_gateway_report(
    path: str | None,
) -> tuple[Mapping[str, Any] | None, str | None]:
    if path is None:
        return None, None
    try:
        payload = json.loads(Path(path).read_text())
    except OSError as error:
        return None, str(error)
    if not isinstance(payload, dict):
        raise ValueError("Gateway report must contain a JSON object")
    return payload, None


def emit_report(report: Mapping[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(rendered)
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")


def print_summary(report: Mapping[str, Any]) -> None:
    print(
        f"Scheduler correctness: {'PASS' if report['passed'] else 'FAIL'} "
        f"(run_id={report['run_id']})",
        file=sys.stderr,
    )
    for check in report["checks"]:
        suffix = f" - {check['reason']}" if check.get("reason") else ""
        print(
            f"[{check['status'].upper():7}] {check['name']} "
            f"bad_rows={check['bad_rows']}{suffix}",
            file=sys.stderr,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify one scheduler stress cohort against PostgreSQL"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", help="Optional scheduler stresser JSON config")
    parser.add_argument("--gateway-report")
    parser.add_argument("--settle-deadline", help="ISO-8601 timestamp; defaults to now")
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--report", help="Optional JSON report path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        database = load_database_config(args.config)
        gateway_report, gateway_error = load_gateway_report(args.gateway_report)
        with connect(args.run_id, database) as connection:
            verifier = SchedulerCorrectnessVerifier(
                connection,
                args.run_id,
                args.sample_limit,
                parse_deadline(args.settle_deadline),
            )
            report = verifier.verify(gateway_report)
        if gateway_error:
            report["checks"].append(
                asdict(
                    skipped_result(
                        "gateway_report_available",
                        "events",
                        gateway_error,
                    )
                )
            )
            report["status_counts"]["skipped"] = (
                report["status_counts"].get("skipped", 0) + 1
            )
        print_summary(report)
        emit_report(report, args.report)
        return 0 if report["passed"] else 1
    except (
        OSError,
        ValueError,
        RuntimeError,
        psycopg.Error,
        json.JSONDecodeError,
    ) as error:
        print(f"scheduler_correctness: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
