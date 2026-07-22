#!/usr/bin/env python3
"""Verify a generated scheduler corpus or a live gateway run."""

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
SCOPES = {"corpus", "gateway"}
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

GATEWAY_SCOPE_CTES = f"""
params AS (
    SELECT
        %s::TEXT AS run_id,
        %s::INTEGER AS sample_limit,
        %s::TIMESTAMPTZ AS settle_deadline,
        %s::TEXT[] AS dag_ids,
        %s::TEXT[] AS forced_dag_ids
),
requested_dags AS MATERIALIZED (
    SELECT DISTINCT requested.id
    FROM params
    CROSS JOIN LATERAL unnest(params.dag_ids) AS requested(id)
),
forced_dags AS MATERIALIZED (
    SELECT DISTINCT forced.id
    FROM params
    CROSS JOIN LATERAL unnest(params.forced_dag_ids) AS forced(id)
),
scoped_dags AS MATERIALIZED (
    SELECT dag.*
    FROM {SCHEDULER_SCHEMA}.dag dag
    JOIN requested_dags requested ON requested.id = dag.id::TEXT
),
scoped_jobs AS MATERIALIZED (
    SELECT job.*
    FROM {SCHEDULER_SCHEMA}.job job
    JOIN scoped_dags dag ON dag.id = job.dag_id
)
"""


@dataclass(frozen=True)
class CheckSpec:
    name: str
    category: str
    expectation: str
    violations_sql: str
    context_sql: str | None = None
    observed_sql: str | None = None


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
    *,
    context_sql: str | None = None,
    observed_sql: str | None = None,
) -> CheckSpec:
    return CheckSpec(
        name,
        category,
        expectation,
        violations_sql.strip(),
        context_sql.strip() if context_sql else None,
        observed_sql.strip() if observed_sql else None,
    )


CHECKS = (
    _spec(
        "manifest_checkpoint",
        "structure",
        "Manifest target, high-water mark, and committed DAG count agree.",
        """
        SELECT concat(
            'target=', manifest.target_dag_count,
            ' high_water=', manifest.high_water_mark,
            ' dags=', counts.dags
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
        SELECT concat('expected=', expected_jobs, ' observed=', observed_jobs) AS id
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
        SELECT concat('duplicate-coordinate:', dag_index, ':', node_index)
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
        f"""
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
        SELECT concat('duplicate-node:', dag_id, ':', task_id)
        FROM nodes
        GROUP BY dag_id, task_id
        HAVING task_id IS NULL OR COUNT(*) <> 1
        UNION ALL
        SELECT concat('node-without-job:', node.dag_id, ':', node.task_id)
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
        f"""
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
        SELECT concat('json-only:', job_id, ':', depends_on_id) AS id
        FROM (
            SELECT job_id, depends_on_id FROM json_dependencies
            EXCEPT
            SELECT job_id, depends_on_id FROM normalized
        ) missing
        UNION ALL
        SELECT concat('normalized-only:', job_id, ':', depends_on_id)
        FROM (
            SELECT job_id, depends_on_id FROM normalized
            EXCEPT
            SELECT job_id, depends_on_id FROM json_dependencies
        ) extra
        UNION ALL
        SELECT concat('cross-dag:', job_id, ':', depends_on_id)
        FROM normalized
        WHERE parent_dag_id IS NULL OR parent_dag_id <> dag_id
        """,
    ),
    _spec(
        "dependency_levels_acyclic",
        "dependencies",
        "Every dependency edge moves from a higher parent level to a lower child level.",
        f"""
        SELECT parent.id::TEXT || '->' || child.id::TEXT AS id
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
        SELECT concat(topology.id, ':roots=', roots, ':leaves=', leaves) AS id
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
        SELECT parent.id::TEXT || '->' || child.id::TEXT AS id
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
        SELECT parent.id::TEXT || '->' || child.id::TEXT AS id
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

GATEWAY_DAG_SCOPE_CHECK = _spec(
    "gateway_dag_scope",
    "structure",
    "Every accepted gateway ID resolves to one fully run-tagged DAG.",
    """
    SELECT 'missing-dag:' || requested.id AS id
    FROM requested_dags requested
    LEFT JOIN scoped_dags dag ON dag.id::TEXT = requested.id
    WHERE dag.id IS NULL
    UNION ALL
    SELECT 'run-tag:' || job.id::TEXT
    FROM scoped_jobs job, params
    WHERE job.data->'metadata'->>'stress_run_id' IS DISTINCT FROM params.run_id
    UNION ALL
    SELECT 'planner-tag:' || job.id::TEXT
    FROM scoped_jobs job
    JOIN scoped_dags dag ON dag.id = job.dag_id
    WHERE job.data->'metadata'->>'stress_planner' IS DISTINCT FROM dag.planner
    """,
)

PARALLEL_GRAPH_CONTEXT_SQL = f"""
    node_degrees AS MATERIALIZED (
        SELECT
            job.dag_id,
            job.id,
            (
                SELECT COUNT(*)
                FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
                WHERE dependency.job_name = job.name
                  AND dependency.job_id = job.id
            ) AS indegree,
            (
                SELECT COUNT(*)
                FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
                WHERE dependency.depends_on_name = job.name
                  AND dependency.depends_on_id = job.id
            ) AS outdegree
        FROM scoped_jobs job
    ), topology AS MATERIALIZED (
        SELECT
            dag.id,
            COUNT(node.id) AS nodes,
            COALESCE(SUM(node.indegree), 0) AS edges,
            COUNT(*) FILTER (WHERE node.indegree = 0) AS roots,
            COUNT(*) FILTER (WHERE node.outdegree = 0) AS leaves,
            COUNT(*) FILTER (WHERE node.outdegree > 1) AS fanout_nodes,
            COUNT(*) FILTER (WHERE node.indegree > 1) AS fanin_nodes
        FROM scoped_dags dag
        LEFT JOIN node_degrees node ON node.dag_id = dag.id
        GROUP BY dag.id
    )
"""

PARALLEL_GRAPH_OBSERVED_SQL = """
    jsonb_build_object(
        'bad_rows', COUNT(*),
        'dag_count', (SELECT COUNT(*) FROM topology),
        'nodes_min', (SELECT MIN(nodes) FROM topology),
        'nodes_max', (SELECT MAX(nodes) FROM topology),
        'edges_min', (SELECT MIN(edges) FROM topology),
        'edges_max', (SELECT MAX(edges) FROM topology),
        'roots_min', (SELECT MIN(roots) FROM topology),
        'roots_max', (SELECT MAX(roots) FROM topology),
        'leaves_min', (SELECT MIN(leaves) FROM topology),
        'leaves_max', (SELECT MAX(leaves) FROM topology),
        'fanout_nodes_min', (SELECT MIN(fanout_nodes) FROM topology),
        'fanout_nodes_max', (SELECT MAX(fanout_nodes) FROM topology),
        'fanin_nodes_min', (SELECT MIN(fanin_nodes) FROM topology),
        'fanin_nodes_max', (SELECT MAX(fanin_nodes) FROM topology),
        'dag_sample_truncated', (
            SELECT COUNT(*) FROM topology
        ) > (SELECT sample_limit FROM params),
        'dag_sample', COALESCE((
            SELECT jsonb_agg(
                jsonb_build_object(
                    'dag_id', topology_sample.id::TEXT,
                    'nodes', topology_sample.nodes,
                    'edges', topology_sample.edges,
                    'roots', topology_sample.roots,
                    'leaves', topology_sample.leaves,
                    'fanout_nodes', topology_sample.fanout_nodes,
                    'fanin_nodes', topology_sample.fanin_nodes
                ) ORDER BY topology_sample.id
            )
            FROM (
                SELECT *
                FROM topology
                ORDER BY id
                LIMIT (SELECT sample_limit FROM params)
            ) topology_sample
        ), '[]'::JSONB)
    )
"""

PARALLEL_GRAPH_CHECK = _spec(
    "parallel_graph_topology",
    "dependencies",
    "Every live DAG is multi-node and contains both fan-out and fan-in.",
    """
    SELECT concat(
        id, ':nodes=', nodes, ':edges=', edges, ':roots=', roots,
        ':leaves=', leaves, ':fanout=', fanout_nodes, ':fanin=', fanin_nodes
    ) AS id
    FROM topology
    WHERE nodes < 2
       OR edges < 1
       OR roots <> 1
       OR leaves < 1
       OR fanout_nodes < 1
       OR fanin_nodes < 1
    """,
    context_sql=PARALLEL_GRAPH_CONTEXT_SQL,
    observed_sql=PARALLEL_GRAPH_OBSERVED_SQL,
)

FAILED_DESCENDANTS_CHECK = _spec(
    "failed_descendants_blocked",
    "dependencies",
    "No transitive descendant starts after a required ancestor fails.",
    f"""
    WITH RECURSIVE edges AS (
        SELECT
            child.dag_id,
            parent.id AS parent_id,
            child.id AS child_id
        FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
        JOIN scoped_jobs child
          ON child.name = dependency.job_name
         AND child.id = dependency.job_id
        JOIN scoped_jobs parent
          ON parent.name = dependency.depends_on_name
         AND parent.id = dependency.depends_on_id
    ), descendants AS (
        SELECT dag_id, parent_id AS ancestor_id, child_id AS descendant_id
        FROM edges
        UNION
        SELECT
            descendants.dag_id,
            descendants.ancestor_id,
            edges.child_id
        FROM descendants
        JOIN edges
          ON edges.dag_id = descendants.dag_id
         AND edges.parent_id = descendants.descendant_id
    )
    SELECT ancestor.id::TEXT || '->' || descendant.id::TEXT AS id
    FROM descendants
    JOIN scoped_jobs ancestor ON ancestor.id = descendants.ancestor_id
    JOIN scoped_jobs descendant ON descendant.id = descendants.descendant_id
    WHERE ancestor.state::TEXT IN ('failed', 'cancelled', 'expired')
      AND (descendant.started_on IS NOT NULL OR descendant.state::TEXT = 'active')
      AND COALESCE(descendant.branch_metadata->>'skipped', 'false') <> 'true'
    """,
)

FORCED_FAILURE_CHECK = _spec(
    "forced_failure_propagation",
    "terminals",
    "Every force-failed DAG fails and cancels its unstarted downstream work.",
    f"""
    WITH RECURSIVE forced_edges AS (
        SELECT
            child.dag_id,
            parent.id AS parent_id,
            child.id AS child_id
        FROM {SCHEDULER_SCHEMA}.job_dependencies dependency
        JOIN scoped_jobs child
          ON child.name = dependency.job_name
         AND child.id = dependency.job_id
        JOIN scoped_jobs parent
          ON parent.name = dependency.depends_on_name
         AND parent.id = dependency.depends_on_id
        JOIN forced_dags forced ON forced.id = child.dag_id::TEXT
    ), forced_descendants AS (
        SELECT dag_id, parent_id AS ancestor_id, child_id AS descendant_id
        FROM forced_edges
        UNION
        SELECT
            descendants.dag_id,
            descendants.ancestor_id,
            edge.child_id
        FROM forced_descendants descendants
        JOIN forced_edges edge
          ON edge.dag_id = descendants.dag_id
         AND edge.parent_id = descendants.descendant_id
    )
    SELECT 'no-forced-dags' AS id
    FROM params
    WHERE cardinality(params.forced_dag_ids) = 0
    UNION ALL
    SELECT 'missing-forced-dag:' || forced.id
    FROM forced_dags forced
    LEFT JOIN scoped_dags dag ON dag.id::TEXT = forced.id
    WHERE dag.id IS NULL
    UNION ALL
    SELECT 'forced-dag-state:' || dag.id::TEXT || ':' || dag.state::TEXT
    FROM forced_dags forced
    JOIN scoped_dags dag ON dag.id::TEXT = forced.id
    WHERE dag.state::TEXT NOT IN ('failed', 'cancelled')
    UNION ALL
    SELECT 'forced-nonterminal-job:' || job.id::TEXT || ':' || job.state::TEXT
    FROM forced_dags forced
    JOIN scoped_dags dag ON dag.id::TEXT = forced.id
    JOIN scoped_jobs job ON job.dag_id = dag.id
    WHERE job.state::TEXT NOT IN (
        'completed', 'skipped', 'failed', 'cancelled', 'expired'
    )
    UNION ALL
    SELECT 'forced-dag-without-failed-job:' || dag.id::TEXT
    FROM forced_dags forced
    JOIN scoped_dags dag ON dag.id::TEXT = forced.id
    JOIN scoped_jobs job ON job.dag_id = dag.id
    GROUP BY dag.id
    HAVING COUNT(*) FILTER (WHERE job.state::TEXT = 'failed') = 0
    UNION ALL
    SELECT 'forced-dag-without-cancelled-downstream:' || dag.id::TEXT
    FROM forced_dags forced
    JOIN scoped_dags dag ON dag.id::TEXT = forced.id
    JOIN scoped_jobs job ON job.dag_id = dag.id
    GROUP BY dag.id
    HAVING NOT EXISTS (
        SELECT 1
        FROM forced_descendants descendants
        JOIN scoped_jobs ancestor ON ancestor.id = descendants.ancestor_id
        JOIN scoped_jobs descendant ON descendant.id = descendants.descendant_id
        WHERE descendants.dag_id = dag.id
          AND ancestor.state::TEXT = 'failed'
          AND descendant.state::TEXT = 'cancelled'
          AND descendant.output->>'cancel_reason' = 'dag_failed'
    )
    """,
)

GATEWAY_REUSED_CHECK_NAMES = (
    "job_dag_run_scope",
    "serialized_graph_matches_jobs",
    "normalized_dependencies_match",
    "dependency_levels_acyclic",
    "dependency_start_order",
    "failed_dependency_not_started",
    "terminal_dag_consistency",
)
GATEWAY_REUSED_CHECKS = tuple(
    spec for spec in CHECKS if spec.name in GATEWAY_REUSED_CHECK_NAMES
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
    fanout_nodes = sum(len(children) > 1 for children in dependents.values())
    fanin_nodes = sum(degree > 1 for degree in indegree.values())
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
        "fanout_nodes": fanout_nodes,
        "fanin_nodes": fanin_nodes,
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


def _query_for(spec: CheckSpec, scope_ctes: str = SCOPE_CTES) -> str:
    ctes = [scope_ctes]
    if spec.context_sql:
        ctes.append(spec.context_sql)
    ctes.append(
        f"""
        violations AS MATERIALIZED (
            {spec.violations_sql}
        )
        """
    )
    observed_sql = spec.observed_sql or "jsonb_build_object('bad_rows', COUNT(*))"
    rendered_ctes = ",\n".join(ctes)
    return f"""
    /* scheduler-correctness:{spec.name} */
    WITH {rendered_ctes}
    SELECT
        COUNT(*)::BIGINT AS bad_rows,
        {observed_sql} AS observed,
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
        *,
        scope: str = "corpus",
        dag_ids: Sequence[str] = (),
        forced_dag_ids: Sequence[str] = (),
        require_parallel_graph: bool = False,
        require_failure_propagation: bool = False,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id is required")
        if sample_limit <= 0 or sample_limit > 1_000:
            raise ValueError("sample_limit must be between 1 and 1000")
        if scope not in SCOPES:
            raise ValueError(f"Unsupported correctness scope: {scope}")
        if scope == "gateway" and not dag_ids:
            raise ValueError("Gateway correctness scope requires accepted DAG IDs")
        if scope != "gateway" and (
            require_parallel_graph or require_failure_propagation
        ):
            raise ValueError("Live graph requirements need --scope gateway")
        self.connection = connection
        self.run_id = run_id
        self.sample_limit = sample_limit
        self.settle_deadline = settle_deadline
        self.scope = scope
        self.dag_ids = tuple(dag_ids)
        self.forced_dag_ids = tuple(forced_dag_ids)
        self.require_parallel_graph = require_parallel_graph
        self.require_failure_propagation = require_failure_propagation

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
        if self.scope == "gateway":
            scope_ctes = GATEWAY_SCOPE_CTES
            params = (
                self.run_id,
                self.sample_limit,
                self.settle_deadline,
                list(self.dag_ids),
                list(self.forced_dag_ids),
            )
        else:
            scope_ctes = SCOPE_CTES
            params = (self.run_id, self.sample_limit, self.settle_deadline)
        try:
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(_query_for(spec, scope_ctes), params)
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
        results: list[CheckResult] = []
        manifest_payload: dict[str, Any] | None
        if self.scope == "gateway":
            if gateway_report is None:
                raise ValueError("Gateway correctness scope requires --gateway-report")
            results.append(self.run_check(GATEWAY_DAG_SCOPE_CHECK))
            results.extend(self.run_check(spec) for spec in GATEWAY_REUSED_CHECKS)
            if self.require_parallel_graph:
                results.append(self.run_check(PARALLEL_GRAPH_CHECK))
            results.append(self.run_check(FAILED_DESCENDANTS_CHECK))
            if self.require_failure_propagation:
                results.append(self.run_check(FORCED_FAILURE_CHECK))
            manifest_payload = None
        else:
            manifest = self.manifest()
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
            manifest_payload = {
                "target_dag_count": manifest["target_dag_count"],
                "high_water_mark": manifest["high_water_mark"],
                "nodes_per_dag": manifest["nodes_per_dag"],
                "graph_shape": manifest["graph_shape"],
                "workload_profile": manifest["workload_profile"],
                "projection_mode": manifest["projection_mode"],
            }
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
            "scope": self.scope,
            "manifest": manifest_payload,
            "gateway_scope": (
                {
                    "accepted_dag_count": len(self.dag_ids),
                    "forced_failure_dag_count": len(self.forced_dag_ids),
                    "require_parallel_graph": self.require_parallel_graph,
                    "require_failure_propagation": self.require_failure_propagation,
                }
                if self.scope == "gateway"
                else None
            ),
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
            if self.scope == "gateway":
                return CheckResult(
                    name="gateway_scheduler_identity",
                    category="events",
                    status="fail",
                    observed={"accepted_dag_ids": 0},
                    expected={"accepted_dag_ids": ">0"},
                    bad_rows=1,
                    sample=[],
                    query_duration_ms=0.0,
                    reason="The gateway report contains no accepted DAG IDs.",
                )
            return skipped_result(
                "gateway_scheduler_identity",
                "events",
                "The gateway report contains no accepted scheduler identities.",
            )
        started = time.perf_counter()
        query = f"""
            SELECT
                submitted.id,
                COUNT(DISTINCT dag.id) AS dag_matches,
                COUNT(job.id) FILTER (
                    WHERE job.data->'metadata'->>'stress_run_id' = %s
                ) AS tagged_jobs
            FROM unnest(%s::TEXT[]) submitted(id)
            LEFT JOIN {SCHEDULER_SCHEMA}.dag dag ON dag.id::TEXT = submitted.id
            LEFT JOIN {SCHEDULER_SCHEMA}.job job ON job.dag_id = dag.id
            GROUP BY submitted.id
        """
        try:
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(query, (self.run_id, job_ids))
                    rows = cursor.fetchall()
            bad = duplicates + [
                str(row["id"])
                for row in rows
                if row["dag_matches"] != 1 or row["tagged_jobs"] < 1
            ]
            return CheckResult(
                name="gateway_scheduler_identity",
                category="events",
                status="pass" if not bad else "fail",
                observed={"accepted_dag_ids": len(job_ids), "bad_rows": len(bad)},
                expected={
                    "dag_matches_per_id": 1,
                    "tagged_jobs": ">=1",
                    "duplicates": 0,
                },
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
                expected={"dag_matches_per_id": 1},
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
            if not job.get("job_id"):
                continue
            events = job.get("raw_events")
            if not isinstance(events, list) or not events:
                if self.scope == "gateway":
                    bad.append(f"{job['job_id']}:missing-events")
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
            invalid = self.scope == "gateway" and (
                kinds.count("scheduled") < 1
                or kinds.count("started") < 1
                or sum(kind in {"completed", "failed"} for kind in kinds) != 1
            )
            invalid = invalid or (
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
            if self.scope == "gateway":
                return CheckResult(
                    name="gateway_event_order",
                    category="events",
                    status="fail",
                    observed={"events": 0, "bad_rows": max(1, len(bad))},
                    expected={"events": ">0", "bad_rows": 0},
                    bad_rows=max(1, len(bad)),
                    sample=bad[: self.sample_limit],
                    query_duration_ms=0.0,
                    reason="The gateway report contains no scheduler event evidence.",
                )
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
        accepted_job_ids = {
            str(job["job_id"])
            for job in jobs
            if isinstance(job, dict) and job.get("job_id")
        }
        terminal_jobs = {
            str(job["job_id"]): str(job["terminal_status"])
            for job in jobs
            if isinstance(job, dict)
            and job.get("job_id")
            and job.get("terminal_status")
        }
        if not terminal_jobs:
            if self.scope == "gateway":
                return CheckResult(
                    name="gateway_terminal_agreement",
                    category="events",
                    status="fail",
                    observed={"terminal_jobs": 0},
                    expected={"terminal_jobs": len(accepted_job_ids)},
                    bad_rows=max(1, len(accepted_job_ids)),
                    sample=sorted(accepted_job_ids)[: self.sample_limit],
                    query_duration_ms=0.0,
                    reason="The gateway report contains no terminal outcomes.",
                )
            return skipped_result(
                "gateway_terminal_agreement",
                "events",
                "The gateway report contains no terminal outcomes.",
            )
        started = time.perf_counter()
        query = f"""
            SELECT
                submitted.id,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM {SCHEDULER_SCHEMA}.job job
                        WHERE job.dag_id = dag.id
                          AND job.data->'metadata'->>'stress_run_id' = %s
                    ) THEN dag.state
                END AS state
            FROM unnest(%s::TEXT[]) submitted(id)
            LEFT JOIN {SCHEDULER_SCHEMA}.dag dag ON dag.id::TEXT = submitted.id
        """
        try:
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(query, (self.run_id, list(terminal_jobs)))
                    rows = cursor.fetchall()
            bad = [
                f"{job_id}:missing-terminal"
                for job_id in sorted(accepted_job_ids - terminal_jobs.keys())
            ]
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
            if self.scope == "gateway":
                return CheckResult(
                    name="post_drain_capacity",
                    category="capacity",
                    status="fail",
                    observed=None,
                    expected={"ok": True, "used": 0, "holder_count": 0},
                    bad_rows=1,
                    sample=[],
                    query_duration_ms=0.0,
                    reason="The gateway report has no post_drain_capacity snapshot.",
                )
            return skipped_result(
                "post_drain_capacity",
                "capacity",
                "The gateway report has no post_drain_capacity snapshot.",
            )
        ok = snapshot.get("ok")
        used = snapshot.get("used")
        holders = snapshot.get("holder_count")
        bad_rows = int(ok is not True) + int(used != 0) + int(holders != 0)
        return CheckResult(
            name="post_drain_capacity",
            category="capacity",
            status="pass" if bad_rows == 0 else "fail",
            observed={"ok": ok, "used": used, "holder_count": holders},
            expected={"ok": True, "used": 0, "holder_count": 0},
            bad_rows=bad_rows,
            sample=[],
            query_duration_ms=0.0,
            reason=(
                None
                if bad_rows == 0
                else "Capacity snapshot is unhealthy, incomplete, or not drained."
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


def gateway_scope_ids(
    report: Mapping[str, Any], run_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    run_identity = report.get("run_identity")
    if not isinstance(run_identity, Mapping) or run_identity.get("run_id") != run_id:
        raise ValueError("Gateway report run_id does not match --run-id")

    jobs = report.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Gateway report does not contain a jobs array")

    accepted = [
        job
        for job in jobs
        if isinstance(job, Mapping) and job.get("job_id") is not None
    ]
    job_ids = tuple(str(job["job_id"]) for job in accepted)
    if not job_ids:
        raise ValueError("Gateway report contains no accepted DAG IDs")

    summary = report.get("summary")
    submitted_jobs = (
        summary.get("submitted_jobs") if isinstance(summary, Mapping) else None
    )
    if submitted_jobs is not None and int(submitted_jobs) != len(job_ids):
        raise ValueError(
            "Gateway report does not retain every accepted job; set "
            "--max-retained-jobs at least as high as --job-count"
        )

    if any(job.get("stress_run_id") != run_id for job in accepted):
        raise ValueError("Gateway job records do not all match --run-id")

    forced_ids = tuple(str(job["job_id"]) for job in accepted if job.get("force_fail"))
    return job_ids, forced_ids


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
        description="Verify a scheduler corpus or live gateway run against PostgreSQL"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scope", choices=sorted(SCOPES), default="corpus")
    parser.add_argument("--config", help="Optional scheduler stresser JSON config")
    parser.add_argument("--gateway-report")
    parser.add_argument("--require-parallel-graph", action="store_true")
    parser.add_argument("--require-failure-propagation", action="store_true")
    parser.add_argument("--settle-deadline", help="ISO-8601 timestamp; defaults to now")
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--report", help="Optional JSON report path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        database = load_database_config(args.config)
        gateway_report, gateway_error = load_gateway_report(args.gateway_report)
        if args.scope == "gateway":
            if gateway_error:
                raise ValueError(f"Gateway report is unavailable: {gateway_error}")
            if gateway_report is None:
                raise ValueError("--scope gateway requires --gateway-report")
            dag_ids, forced_dag_ids = gateway_scope_ids(gateway_report, args.run_id)
        else:
            dag_ids, forced_dag_ids = (), ()
        with connect(args.run_id, database) as connection:
            verifier = SchedulerCorrectnessVerifier(
                connection,
                args.run_id,
                args.sample_limit,
                parse_deadline(args.settle_deadline),
                scope=args.scope,
                dag_ids=dag_ids,
                forced_dag_ids=forced_dag_ids,
                require_parallel_graph=args.require_parallel_graph,
                require_failure_propagation=args.require_failure_propagation,
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
