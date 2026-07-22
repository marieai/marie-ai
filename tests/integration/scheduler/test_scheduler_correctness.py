from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg
import pytest

from tools.stress.scheduler_correctness import SchedulerCorrectnessVerifier
from tools.stress.scheduler_db_stresser import CorpusGenerator, StressConfig, connect

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("MARIE_SCHEDULER_DB_STRESS_INTEGRATION") != "1",
        reason=(
            "Set MARIE_SCHEDULER_DB_STRESS_INTEGRATION=1 for PostgreSQL integration"
        ),
    ),
]


@dataclass(frozen=True)
class CorruptionCase:
    name: str
    expected_check: str
    sql: str
    workload_profile: str = "completed"
    params: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeededCorpus:
    connection: psycopg.Connection[dict[str, object]]
    verifier: SchedulerCorrectnessVerifier
    run_id: str


CORRUPTION_CASES = (
    CorruptionCase(
        "manifest-count",
        "manifest_checkpoint",
        """
        UPDATE marie_stress.run_manifest
        SET target_dag_count = target_dag_count + 1
        WHERE run_id = %s
        """,
        params=("run_id",),
    ),
    CorruptionCase(
        "untagged-job",
        "job_dag_run_scope",
        """
        UPDATE marie_scheduler.job
        SET data = jsonb_set(
            data,
            '{metadata,stress_run_id}',
            to_jsonb(%s::text)
        )
        WHERE data->'metadata'->>'stress_run_id' = %s
          AND data->'metadata'->>'stress_node_index' = '0'
        """,
        params=("foreign_run_id", "run_id"),
    ),
    CorruptionCase(
        "serialized-graph",
        "serialized_graph_matches_jobs",
        """
        UPDATE marie_scheduler.dag
        SET serialized_dag = jsonb_set(serialized_dag, '{nodes}', '[]'::jsonb)
        WHERE planner = %s
        """,
        params=("planner",),
    ),
    CorruptionCase(
        "missing-normalized-dependency",
        "normalized_dependencies_match",
        """
        DELETE FROM marie_scheduler.job_dependencies dependency
        USING marie_scheduler.job child
        WHERE child.name = dependency.job_name
          AND child.id = dependency.job_id
          AND child.data->'metadata'->>'stress_run_id' = %s
          AND child.data->'metadata'->>'stress_node_index' = '1'
        """,
        params=("run_id",),
    ),
    CorruptionCase(
        "dependency-cycle",
        "dependency_levels_acyclic",
        """
        UPDATE marie_scheduler.job child
        SET dependencies = jsonb_build_array(parent.id::text)
        FROM marie_scheduler.job parent
        WHERE child.data->'metadata'->>'stress_run_id' = %s
          AND child.data->'metadata'->>'stress_node_index' = '0'
          AND parent.data->'metadata'->>'stress_run_id' = %s
          AND parent.data->'metadata'->>'stress_node_index' = '2'
          AND parent.dag_id = child.dag_id
        """,
        params=("run_id", "run_id"),
    ),
    CorruptionCase(
        "cross-dag-dependency",
        "normalized_dependencies_match",
        """
        UPDATE marie_scheduler.job child
        SET dependencies = jsonb_build_array(parent.id::text)
        FROM marie_scheduler.job parent
        WHERE child.data->'metadata'->>'stress_run_id' = %s
          AND child.data->'metadata'->>'stress_dag_index' = '0'
          AND child.data->'metadata'->>'stress_node_index' = '1'
          AND parent.data->'metadata'->>'stress_run_id' = %s
          AND parent.data->'metadata'->>'stress_dag_index' = '1'
          AND parent.data->'metadata'->>'stress_node_index' = '0'
        """,
        params=("run_id", "run_id"),
    ),
    CorruptionCase(
        "missing-attempt",
        "active_attempt_identity",
        """
        DELETE FROM marie_scheduler.job_attempt
        WHERE dag_id IN (
            SELECT id FROM marie_scheduler.dag WHERE planner = %s
        )
        """,
        workload_profile="active",
        params=("planner",),
    ),
    CorruptionCase(
        "active-attempt-id-mismatch",
        "active_attempt_identity",
        """
        UPDATE marie_scheduler.job
        SET run_attempt_id = gen_random_uuid()
        WHERE data->'metadata'->>'stress_run_id' = %s
          AND state::text = 'active'
        """,
        workload_profile="active",
        params=("run_id",),
    ),
    CorruptionCase(
        "attempt-job-mismatch",
        "attempt_identity_scope",
        """
        UPDATE marie_scheduler.job_attempt
        SET job_id = gen_random_uuid()
        WHERE dag_id IN (
            SELECT id FROM marie_scheduler.dag WHERE planner = %s
        )
        """,
        workload_profile="active",
        params=("planner",),
    ),
    CorruptionCase(
        "attempt-queue-mismatch",
        "attempt_identity_scope",
        """
        UPDATE marie_scheduler.job_attempt
        SET job_name = 'incorrect-queue'
        WHERE dag_id IN (
            SELECT id FROM marie_scheduler.dag WHERE planner = %s
        )
        """,
        workload_profile="active",
        params=("planner",),
    ),
    CorruptionCase(
        "attempt-dag-mismatch",
        "attempt_identity_scope",
        """
        UPDATE marie_scheduler.job_attempt
        SET dag_id = gen_random_uuid()
        WHERE job_id IN (
            SELECT id
            FROM marie_scheduler.job
            WHERE data->'metadata'->>'stress_run_id' = %s
        )
        """,
        workload_profile="active",
        params=("run_id",),
    ),
    CorruptionCase(
        "expired-run-lease",
        "expired_active_run_leases",
        """
        UPDATE marie_scheduler.job
        SET run_lease_expires_at = NOW() - INTERVAL '1 second'
        WHERE data->'metadata'->>'stress_run_id' = %s
          AND state::text = 'active'
        """,
        workload_profile="active",
        params=("run_id",),
    ),
    CorruptionCase(
        "attempt-owner-mismatch",
        "attempt_identity_scope",
        """
        UPDATE marie_scheduler.job_attempt
        SET scheduler_lease_owner = 'incorrect-owner'
        WHERE dag_id IN (
            SELECT id FROM marie_scheduler.dag WHERE planner = %s
        )
        """,
        workload_profile="active",
        params=("planner",),
    ),
    CorruptionCase(
        "duplicate-accepted-completion",
        "duplicate_accepted_completed_terminal_by_job",
        """
        INSERT INTO marie_scheduler.job_attempt (
            run_attempt_id, job_id, job_name, dag_id, run_owner,
            scheduler_lease_owner, gateway_instance_id, attempt_state,
            activated_at, terminal_at, terminal_status, terminal_work_state,
            terminal_source, terminal_gateway_instance_id, terminal_accepted
        )
        SELECT
            gen_random_uuid(), job.id, job.name, job.dag_id, 'stress-owner',
            'stress-owner', 'stress-gateway', 'completed',
            NOW() - make_interval(secs => series.value), NOW(), 'completed',
            'completed', 'job_event', 'stress-gateway', TRUE
        FROM marie_scheduler.job job
        CROSS JOIN generate_series(1, 2) AS series(value)
        WHERE job.data->'metadata'->>'stress_run_id' = %s
          AND job.data->'metadata'->>'stress_node_index' = '0'
        """,
        params=("run_id",),
    ),
    CorruptionCase(
        "stale-terminal",
        "stale_terminal_accepted",
        """
        INSERT INTO marie_scheduler.job_attempt (
            run_attempt_id, job_id, job_name, dag_id, run_owner,
            scheduler_lease_owner, gateway_instance_id, attempt_state,
            activated_at, terminal_at, terminal_status, terminal_work_state,
            terminal_source, terminal_gateway_instance_id, terminal_accepted
        )
        SELECT
            gen_random_uuid(), job.id, job.name, job.dag_id, 'stress-owner',
            'stress-owner', 'stress-gateway', attempt.attempt_state,
            NOW() + attempt.activated_offset,
            CASE WHEN attempt.terminal_accepted THEN NOW() ELSE NULL END,
            CASE WHEN attempt.terminal_accepted THEN 'completed' ELSE NULL END,
            CASE WHEN attempt.terminal_accepted THEN 'completed' ELSE NULL END,
            CASE WHEN attempt.terminal_accepted THEN 'job_event' ELSE NULL END,
            CASE WHEN attempt.terminal_accepted THEN 'stress-gateway' ELSE NULL END,
            attempt.terminal_accepted
        FROM marie_scheduler.job job
        CROSS JOIN (
            VALUES
                ('completed', INTERVAL '-2 minutes', TRUE),
                ('activated', INTERVAL '-1 minute', NULL::boolean)
        ) AS attempt(attempt_state, activated_offset, terminal_accepted)
        WHERE job.data->'metadata'->>'stress_run_id' = %s
          AND job.data->'metadata'->>'stress_node_index' = '0'
        """,
        params=("run_id",),
    ),
    CorruptionCase(
        "terminal-lease",
        "terminal_job_retains_lease",
        """
        UPDATE marie_scheduler.job
        SET run_owner = 'leaked-owner',
            run_attempt_id = gen_random_uuid(),
            run_lease_expires_at = NOW() + INTERVAL '1 hour'
        WHERE data->'metadata'->>'stress_run_id' = %s
          AND data->'metadata'->>'stress_node_index' = '0'
        """,
        params=("run_id",),
    ),
    CorruptionCase(
        "dependency-start-order",
        "dependency_start_order",
        """
        UPDATE marie_scheduler.job parent
        SET completed_on = child.started_on + INTERVAL '1 second'
        FROM marie_scheduler.job child
        WHERE parent.data->'metadata'->>'stress_run_id' = %s
          AND parent.data->'metadata'->>'stress_node_index' = '0'
          AND child.data->'metadata'->>'stress_run_id' = %s
          AND child.data->'metadata'->>'stress_node_index' = '1'
          AND child.dag_id = parent.dag_id
        """,
        params=("run_id", "run_id"),
    ),
)


def _case_params(case: CorruptionCase, run_id: str) -> tuple[str, ...]:
    values = {
        "foreign_run_id": f"foreign-{run_id}",
        "planner": f"stress:{run_id}",
        "run_id": run_id,
    }
    return tuple(values[name] for name in case.params)


@pytest.fixture
def seeded_corpus(request: pytest.FixtureRequest) -> Iterator[SeededCorpus]:
    case = getattr(request, "param", None)
    workload_profile = (
        case.workload_profile if isinstance(case, CorruptionCase) else "completed"
    )
    suffix = uuid.uuid4().hex[:12]
    run_id = f"scheduler-correctness-{suffix}"
    queue_name = f"scheduler_correctness_{suffix}"
    config = StressConfig.from_mapping(
        {
            "run_id": run_id,
            "target_dag_count": 2,
            "nodes_per_dag": 3,
            "graph_shape": "chain",
            "workload_profile": workload_profile,
            "queue_name": queue_name,
            "batch_size": 2,
            "seed": 20260721,
            "projection_mode": "full",
            "analyze_after_seed": False,
            "executor": "mock_executor",
            "endpoint": "/document/extract",
            "database": {},
        }
    )

    with connect(config) as connection:
        generator = CorpusGenerator(connection)
        version = generator.initialize()
        generator.acquire_lock(run_id)
        try:
            manifest = generator.prepare_manifest(config, version)
            generator.ensure_queue(config)
            generator.create_staging_tables()
            generator.seed(config, int(manifest["high_water_mark"]))
            verifier = SchedulerCorrectnessVerifier(
                connection,
                run_id,
                sample_limit=10,
                settle_deadline=datetime.now(timezone.utc),
            )
            yield SeededCorpus(connection, verifier, run_id)
        finally:
            generator.release_lock(run_id)
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM marie_scheduler.job_attempt "
                    "WHERE dag_id IN ("
                    "SELECT id FROM marie_scheduler.dag WHERE planner = %s) "
                    "OR job_id IN ("
                    "SELECT id FROM marie_scheduler.job "
                    "WHERE data->'metadata'->>'stress_run_id' = %s)",
                    (f"stress:{run_id}", run_id),
                )
                cursor.execute(
                    "DELETE FROM marie_scheduler.dag WHERE planner = %s",
                    (f"stress:{run_id}",),
                )
                cursor.execute(
                    "DELETE FROM marie_scheduler.job_history "
                    "WHERE data->'metadata'->>'stress_run_id' = %s",
                    (run_id,),
                )
                cursor.execute(
                    "DELETE FROM marie_scheduler.dag_history WHERE planner = %s",
                    (f"stress:{run_id}",),
                )
                cursor.execute(
                    "DELETE FROM marie_stress.run_manifest WHERE run_id = %s",
                    (run_id,),
                )
                cursor.execute(
                    "SELECT marie_scheduler.delete_queue(%s)",
                    (queue_name,),
                )
            connection.commit()


@pytest.mark.parametrize(
    ("seeded_corpus", "case"),
    [(case, case) for case in CORRUPTION_CASES],
    indirect=["seeded_corpus"],
    ids=[case.name for case in CORRUPTION_CASES],
)
def test_each_database_corruption_fails_its_named_check(
    seeded_corpus: SeededCorpus,
    case: CorruptionCase,
) -> None:
    assert seeded_corpus.verifier.verify()["passed"] is True

    with seeded_corpus.connection.cursor() as cursor:
        cursor.execute(case.sql, _case_params(case, seeded_corpus.run_id))
        assert cursor.rowcount > 0
    seeded_corpus.connection.commit()

    report = seeded_corpus.verifier.verify()
    checks = {check["name"]: check for check in report["checks"]}

    assert report["passed"] is False
    assert checks[case.expected_check]["status"] == "fail"


def test_gateway_disagreement_and_leaked_capacity_fail(
    seeded_corpus: SeededCorpus,
) -> None:
    with seeded_corpus.connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id
            FROM marie_scheduler.job
            WHERE data->'metadata'->>'stress_run_id' = %s
            ORDER BY id
            LIMIT 1
            """,
            (seeded_corpus.run_id,),
        )
        row = cursor.fetchone()
    assert row is not None

    report = seeded_corpus.verifier.verify(
        {
            "jobs": [
                {
                    "job_id": str(row["id"]),
                    "terminal_status": "failed",
                    "raw_events": ["job.scheduled", "job.started", "job.failed"],
                }
            ],
            "post_drain_capacity": {"used": 1, "holder_count": 1},
        }
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["passed"] is False
    assert checks["gateway_terminal_agreement"]["status"] == "fail"
    assert checks["post_drain_capacity"]["status"] == "fail"
