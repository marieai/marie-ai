from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tools.stress.scheduler_db_stresser import (
    GENERATOR_VERSION,
    MIXED_STATES,
    UUID_DERIVATION_VERSION,
    CorpusGenerator,
    StressConfig,
    build_dag,
    build_plan,
    dag_id_for,
    expected_ready_jobs,
    job_id_for,
    load_config,
    main,
    statement_delta,
    statistics_delta,
)


def build_config(**overrides: object) -> StressConfig:
    values: dict[str, object] = {
        "run_id": "scheduler-scale-test",
        "target_dag_count": 1_000,
        "nodes_per_dag": 1,
        "graph_shape": "single",
        "workload_profile": "ready",
        "queue_name": "scheduler_stress_v1",
        "batch_size": 100,
        "seed": 20260721,
        "projection_mode": "scheduler",
        "analyze_after_seed": True,
        "executor": "mock_executor",
        "endpoint": "/document/extract",
        "database": {},
    }
    values.update(overrides)
    return StressConfig.from_mapping(values)


def test_ids_are_deterministic_and_run_scoped() -> None:
    assert dag_id_for("run-a", 42) == dag_id_for("run-a", 42)
    assert dag_id_for("run-a", 42) != dag_id_for("run-b", 42)
    assert job_id_for("run-a", 42, 0) == job_id_for("run-a", 42, 0)
    assert job_id_for("run-a", 42, 0) != job_id_for("run-a", 42, 1)


@pytest.mark.parametrize(
    ("shape", "node_count", "expected"),
    [
        ("chain", 4, [[], [0], [1], [2]]),
        ("fanout", 4, [[], [0], [0], [0]]),
        ("diamond", 4, [[], [0], [0], [1, 2]]),
    ],
)
def test_graph_dependencies_match_shape(
    shape: str, node_count: int, expected: list[list[int]]
) -> None:
    config = build_config(graph_shape=shape, nodes_per_dag=node_count)
    dag = build_dag(config, 7, datetime(2026, 7, 21, tzinfo=timezone.utc))
    indexes_by_id = {job.id: job.node_index for job in dag.jobs}

    observed = [
        [indexes_by_id[dependency] for dependency in job.dependencies]
        for job in dag.jobs
    ]

    assert observed == expected
    expected_levels = {
        "chain": [3, 2, 1, 0],
        "fanout": [1, 0, 0, 0],
        "diamond": [2, 1, 1, 0],
    }
    assert [job.job_level for job in dag.jobs] == expected_levels[shape]
    assert [node["task_id"] for node in dag.serialized_dag["nodes"]] == [
        str(job.id) for job in dag.jobs
    ]
    assert all(
        node["definition"]["method"] == "EXECUTOR_ENDPOINT"
        for node in dag.serialized_dag["nodes"]
    )


def test_active_profile_builds_valid_attempt_frontier() -> None:
    config = build_config(
        graph_shape="chain",
        nodes_per_dag=4,
        workload_profile="active",
    )

    dag = build_dag(config, 0, datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert [job.state for job in dag.jobs] == [
        "completed",
        "completed",
        "completed",
        "active",
    ]
    active = dag.jobs[-1]
    assert active.run_owner == "stress:scheduler-scale-test"
    assert active.run_attempt_id is not None
    assert active.run_lease_expires_at is not None
    assert active.run_lease_expires_at > datetime(2026, 7, 21, tzinfo=timezone.utc)
    assert active.started_on is not None
    parent = dag.jobs[-2]
    assert parent.completed_on is not None
    assert parent.completed_on <= active.started_on


def test_ready_profile_uses_created_state_and_due_start_time() -> None:
    now = datetime(2026, 7, 21, tzinfo=timezone.utc)
    config = build_config(
        graph_shape="chain",
        nodes_per_dag=3,
        workload_profile="ready",
    )

    dag = build_dag(config, 0, now)

    assert [job.state for job in dag.jobs] == ["completed", "completed", "created"]
    assert all(job.start_after < now for job in dag.jobs)
    assert dag.jobs[0].dependencies == ()
    assert dag.jobs[1].dependencies == (dag.jobs[0].id,)


def test_ready_profile_counts_only_unblocked_frontier_nodes() -> None:
    chain = build_config(
        graph_shape="chain",
        nodes_per_dag=4,
        workload_profile="ready",
    )
    fanout = build_config(
        graph_shape="fanout",
        nodes_per_dag=4,
        workload_profile="ready",
    )
    mixed = build_config(
        graph_shape="mixed",
        nodes_per_dag=4,
        workload_profile="ready",
        seed=0,
    )

    assert expected_ready_jobs(chain, 10) == 10
    assert expected_ready_jobs(fanout, 10) == 30
    assert expected_ready_jobs(mixed, 3) == 5


def test_mixed_profile_covers_every_persisted_state() -> None:
    config = build_config(workload_profile="mixed")
    now = datetime(2026, 7, 21, tzinfo=timezone.utc)

    states = {build_dag(config, index, now).jobs[0].state for index in range(8)}

    assert states == set(MIXED_STATES)


def test_plan_treats_target_as_total() -> None:
    config = build_config(target_dag_count=100_000, nodes_per_dag=1)

    plan = build_plan(config, current_count=1_000)

    assert plan["dags_to_add"] == 99_000
    assert plan["jobs_to_add"] == 99_000
    assert plan["chunk_count"] == 990
    assert plan["destructive_actions"] == []


def test_plan_refuses_to_shrink() -> None:
    with pytest.raises(ValueError, match="Refusing to shrink"):
        build_plan(build_config(target_dag_count=999), current_count=1_000)


def test_cohort_hash_ignores_checkpoint_settings() -> None:
    original = build_config()
    next_checkpoint = build_config(
        target_dag_count=100_000,
        batch_size=10_000,
        report="/tmp/next.json",
    )
    different_queue = build_config(queue_name="scheduler_stress_v2")

    assert original.cohort_hash == next_checkpoint.cohort_hash
    assert original.cohort_hash != different_queue.cohort_hash


def test_manifest_validation_rejects_immutable_config_changes() -> None:
    config = build_config()
    manifest = {
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

    CorpusGenerator._validate_manifest(config, manifest)
    manifest["queue_name"] = "different_queue"

    with pytest.raises(RuntimeError, match="queue_name"):
        CorpusGenerator._validate_manifest(config, manifest)


def test_statistics_delta_uses_database_counters_without_reset() -> None:
    before = {
        "database": {"xact_commit": 10, "temp_bytes": 20, "stats_reset": "old"},
        "wal": {"wal_bytes": 100},
    }
    after = {
        "database": {"xact_commit": 14, "temp_bytes": 35, "stats_reset": "old"},
        "wal": {"wal_bytes": 160},
    }

    assert statistics_delta(before, after) == {
        "database": {"xact_commit": 4, "temp_bytes": 15},
        "wal": {"wal_bytes": 60},
    }


def test_statement_delta_keeps_only_queries_executed_in_window() -> None:
    before = [
        {
            "queryid": 1,
            "query": "SELECT 1 FROM marie_scheduler.dag",
            "calls": 5,
            "rows": 5,
            "total_exec_time": 10.0,
        }
    ]
    after = [
        {
            "queryid": 1,
            "query": "SELECT 1 FROM marie_scheduler.dag",
            "calls": 7,
            "rows": 7,
            "total_exec_time": 14.0,
        },
        {
            "queryid": 2,
            "query": "SELECT 1 FROM marie_scheduler.job",
            "calls": 0,
            "rows": 0,
            "total_exec_time": 0.0,
        },
    ]

    assert statement_delta(before, after) == [
        {
            "queryid": "1",
            "query": "SELECT 1 FROM marie_scheduler.dag",
            "calls": 2,
            "rows": 2,
            "total_exec_time": 4.0,
        }
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_id": ""}, "run_id is required"),
        ({"target_dag_count": 0}, "target_dag_count"),
        ({"graph_shape": "single", "nodes_per_dag": 2}, "requires nodes_per_dag=1"),
        ({"graph_shape": "diamond", "nodes_per_dag": 3}, "at least four"),
        ({"endpoint": "document/extract"}, "must start with"),
        ({"active_lease_seconds": 0}, "active_lease_seconds"),
        ({"database": {"password": "secret"}}, "PGPASSWORD"),
    ],
)
def test_config_rejects_invalid_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_config(**overrides)


def test_load_config_applies_cli_overrides(tmp_path) -> None:
    path = tmp_path / "stress.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "from-file",
                "target_dag_count": 1_000,
                "nodes_per_dag": 1,
            }
        )
    )

    config = load_config(
        str(path),
        {"run_id": "from-cli", "target_dag_count": 100_000, "batch_size": None},
    )

    assert config.run_id == "from-cli"
    assert config.target_dag_count == 100_000
    assert config.batch_size == 10_000


def test_seed_dry_run_does_not_connect(tmp_path, capsys) -> None:
    path = tmp_path / "stress.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "dry-run",
                "target_dag_count": 1_000,
                "nodes_per_dag": 1,
                "database": {"host": "must-not-connect.invalid"},
            }
        )
    )

    result = main(["seed", "--config", str(path), "--dry-run"])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["plan"]["dags_to_add"] == 1_000
    assert "database" not in payload["config"]
