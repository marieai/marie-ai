from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from tools.stress.scheduler_correctness import SchedulerCorrectnessVerifier
from tools.stress.scheduler_db_stresser import (
    CorpusGenerator,
    GeneratedDag,
    StressConfig,
    connect,
)

pytestmark = pytest.mark.slow


@pytest.mark.skipif(
    os.environ.get("MARIE_SCHEDULER_DB_STRESS_INTEGRATION") != "1",
    reason="Set MARIE_SCHEDULER_DB_STRESS_INTEGRATION=1 for PostgreSQL integration",
)
def test_seed_resume_and_verify_against_scheduler_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = uuid.uuid4().hex[:12]
    run_id = f"scheduler-db-integration-{suffix}"
    queue_name = f"scheduler_stress_test_{suffix}"
    base = {
        "run_id": run_id,
        "target_dag_count": 2,
        "nodes_per_dag": 2,
        "graph_shape": "chain",
        "workload_profile": "ready",
        "queue_name": queue_name,
        "batch_size": 1,
        "seed": 20260721,
        "projection_mode": "full",
        "analyze_after_seed": False,
        "executor": "mock_executor",
        "endpoint": "/document/extract",
        "database": {},
    }
    first = StressConfig.from_mapping(base)

    with connect(first) as connection:
        generator = CorpusGenerator(connection)
        version = generator.initialize()
        generator.acquire_lock(run_id)
        try:
            manifest = generator.prepare_manifest(first, version)
            generator.ensure_queue(first)
            generator.create_staging_tables()
            original_copy = generator._copy_chunk

            def fail_second_chunk(
                config: StressConfig, dags: Sequence[GeneratedDag]
            ) -> None:
                if dags[0].index == 1:
                    raise RuntimeError("injected chunk failure")
                original_copy(config, dags)

            monkeypatch.setattr(generator, "_copy_chunk", fail_second_chunk)
            with pytest.raises(RuntimeError, match="injected chunk failure"):
                generator.seed(first, int(manifest["high_water_mark"]))

            interrupted = generator.manifest(run_id)
            assert interrupted["high_water_mark"] == 1

            monkeypatch.setattr(generator, "_copy_chunk", original_copy)
            result = generator.seed(first, int(interrupted["high_water_mark"]))
            verification = generator.verify(first)

            assert result["inserted_dags"] == 1
            assert verification["passed"] is True

            manifest = generator.prepare_manifest(first, version)
            result = generator.seed(first, int(manifest["high_water_mark"]))
            assert result["inserted_dags"] == 0

            second = StressConfig.from_mapping({**base, "target_dag_count": 3})
            manifest = generator.prepare_manifest(second, version)
            result = generator.seed(second, int(manifest["high_water_mark"]))
            verification = generator.verify(second)

            assert result["inserted_dags"] == 1
            assert verification["expected"]["dags"] == 3
            assert verification["passed"] is True
            completed = generator.manifest(run_id)
            assert completed["high_water_mark"] == 3
            assert len(completed["checkpoints"]) == 3

            smaller = StressConfig.from_mapping({**base, "target_dag_count": 2})
            with pytest.raises(RuntimeError, match="Refusing to shrink"):
                generator.prepare_manifest(smaller, version)
            assert generator.manifest(run_id)["target_dag_count"] == 3

            with pytest.raises(RuntimeError, match="Scheduler schema changed"):
                generator.prepare_manifest(second, version + 1)
            transitioned_config = StressConfig.from_mapping(
                {**base, "target_dag_count": 3, "allow_schema_transition": True}
            )
            transitioned = generator.prepare_manifest(
                transitioned_config,
                version + 1,
            )
            assert transitioned["scheduler_schema_version"] == version + 1
            recorded = generator.manifest(run_id)
            assert recorded["schema_transitions"][-1]["from"] == version
            assert recorded["schema_transitions"][-1]["to"] == version + 1
        finally:
            generator.release_lock(run_id)
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM marie_scheduler.job_attempt "
                    "WHERE metadata->>'stress_run_id' = %s",
                    (run_id,),
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
                cursor.execute("SELECT marie_scheduler.delete_queue(%s)", (queue_name,))
            connection.commit()


@pytest.mark.skipif(
    os.environ.get("MARIE_SCHEDULER_DB_STRESS_INTEGRATION") != "1",
    reason="Set MARIE_SCHEDULER_DB_STRESS_INTEGRATION=1 for PostgreSQL integration",
)
def test_correctness_verifier_detects_corrupted_dependency_level() -> None:
    suffix = uuid.uuid4().hex[:12]
    run_id = f"scheduler-correctness-integration-{suffix}"
    queue_name = f"scheduler_correctness_test_{suffix}"
    config = StressConfig.from_mapping(
        {
            "run_id": run_id,
            "target_dag_count": 1,
            "nodes_per_dag": 3,
            "graph_shape": "chain",
            "workload_profile": "completed",
            "queue_name": queue_name,
            "batch_size": 1,
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

            clean = verifier.verify()

            assert clean["passed"] is True

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE marie_scheduler.job child
                    SET job_level = parent.job_level
                    FROM marie_scheduler.job parent
                    WHERE child.data->'metadata'->>'stress_run_id' = %s
                      AND child.data->'metadata'->>'stress_node_index' = '1'
                      AND parent.data->'metadata'->>'stress_run_id' = %s
                      AND parent.data->'metadata'->>'stress_node_index' = '0'
                    """,
                    (run_id, run_id),
                )
            connection.commit()

            corrupted = verifier.verify()
            checks = {check["name"]: check for check in corrupted["checks"]}

            assert corrupted["passed"] is False
            assert checks["dependency_levels_acyclic"]["status"] == "fail"
        finally:
            generator.release_lock(run_id)
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM marie_scheduler.job_attempt "
                    "WHERE dag_id IN (SELECT id FROM marie_scheduler.dag WHERE planner = %s)",
                    (f"stress:{run_id}",),
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
                cursor.execute("SELECT marie_scheduler.delete_queue(%s)", (queue_name,))
            connection.commit()


@pytest.mark.skipif(
    os.environ.get("MARIE_SCHEDULER_DB_STRESS_INTEGRATION") != "1",
    reason="Set MARIE_SCHEDULER_DB_STRESS_INTEGRATION=1 for PostgreSQL integration",
)
def test_active_profile_persists_and_verifies_durable_attempt() -> None:
    suffix = uuid.uuid4().hex[:12]
    run_id = f"scheduler-db-active-{suffix}"
    queue_name = f"scheduler_stress_active_{suffix}"
    config = StressConfig.from_mapping(
        {
            "run_id": run_id,
            "target_dag_count": 1,
            "nodes_per_dag": 1,
            "graph_shape": "single",
            "workload_profile": "active",
            "queue_name": queue_name,
            "batch_size": 1,
            "projection_mode": "scheduler",
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

            assert generator.verify(config)["passed"] is True

            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE marie_scheduler.job "
                    "SET run_lease_expires_at = NOW() - INTERVAL '1 second' "
                    "WHERE name = %s AND data->'metadata'->>'stress_run_id' = %s",
                    (queue_name, run_id),
                )
            connection.commit()
            expired = generator.verify(config)
            assert expired["checks"]["active_contract_count"] is False

            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE marie_scheduler.job "
                    "SET run_lease_expires_at = NOW() + INTERVAL '1 day' "
                    "WHERE name = %s AND data->'metadata'->>'stress_run_id' = %s",
                    (queue_name, run_id),
                )
                cursor.execute(
                    "DELETE FROM marie_scheduler.job_attempt "
                    "WHERE metadata->>'stress_run_id' = %s",
                    (run_id,),
                )
            connection.commit()
            missing_attempt = generator.verify(config)
            assert missing_attempt["checks"]["active_contract_count"] is False
        finally:
            generator.release_lock(run_id)
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM marie_scheduler.job_attempt "
                    "WHERE metadata->>'stress_run_id' = %s",
                    (run_id,),
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
                cursor.execute("SELECT marie_scheduler.delete_queue(%s)", (queue_name,))
            connection.commit()


@pytest.mark.skipif(
    os.environ.get("MARIE_SCHEDULER_DB_STRESS_INTEGRATION") != "1",
    reason="Set MARIE_SCHEDULER_DB_STRESS_INTEGRATION=1 for PostgreSQL integration",
)
def test_run_scoped_advisory_lock_rejects_second_generator() -> None:
    run_id = f"scheduler-db-lock-{uuid.uuid4().hex[:12]}"
    config = StressConfig.from_mapping(
        {
            "run_id": run_id,
            "target_dag_count": 1,
            "database": {},
        }
    )

    with connect(config) as first_connection, connect(config) as second_connection:
        first = CorpusGenerator(first_connection)
        second = CorpusGenerator(second_connection)
        first.initialize()
        second.initialize()
        first.acquire_lock(run_id)
        try:
            with pytest.raises(RuntimeError, match="Another generator holds the lock"):
                second.acquire_lock(run_id)
        finally:
            first.release_lock(run_id)
