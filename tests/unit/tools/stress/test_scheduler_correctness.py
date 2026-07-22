from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

import pytest

from tools.stress.scheduler_correctness import (
    ATTEMPT_CHECK_CATEGORIES,
    ATTEMPT_CHECK_NAMES,
    CHECKS,
    FAILED_DESCENDANTS_CHECK,
    FORCED_FAILURE_CHECK,
    GATEWAY_DAG_SCOPE_CHECK,
    GATEWAY_REUSED_CHECKS,
    GATEWAY_SCOPE_CTES,
    PARALLEL_GRAPH_CHECK,
    SchedulerCorrectnessVerifier,
    _query_for,
    gateway_scope_ids,
    inspect_serialized_plan,
    load_database_config,
    load_gateway_report,
    parse_deadline,
)


@pytest.mark.parametrize(
    "planner_name",
    [
        "query_planner_mock_simple",
        "query_planner_mock_annotator_llm",
        "query_planner_mock_medium",
        "query_planner_mock_complex",
        "query_planner_mock_with_subgraphs",
        "query_planner_mock_parallel_subgraphs",
        "query_planner_mock_branch_simple",
        "query_planner_mock_switch_complexity",
        "query_planner_mock_branch_multi_condition",
        "query_planner_mock_nested_branches",
        "query_planner_mock_branch_python_function",
        "query_planner_mock_branch_jsonpath_advanced",
        "query_planner_mock_branch_all_match",
        "query_planner_mock_branch_regex_matching",
        "query_planner_mock_guardrail_simple",
        "query_planner_mock_guardrail_retry_loop",
        "query_planner_mock_guardrail_executor_metric",
        "query_planner_mock_guardrail_multi_metric",
        "query_planner_mock_hitl_approval",
        "query_planner_mock_hitl_correction",
        "query_planner_mock_hitl_router",
        "query_planner_mock_hitl_complete_workflow",
        "query_planner_mock_connector_tool",
        "query_planner_mock_plugin_tool",
        "query_planner_mock_plugin_model",
        "query_planner_mock_plugin_datasource",
        "query_planner_mock_plugin_trigger",
    ],
)
def test_representative_mock_plans_are_valid_correctness_fixtures(
    planner_name: str,
) -> None:
    from marie.job.job_manager import generate_job_id
    from marie.logging_core.log_bus import GLOBAL_LOG_BUS
    from marie.query_planner import mock_query_plans
    from marie.query_planner.base import PlannerInfo

    GLOBAL_LOG_BUS.flush()
    planner = getattr(mock_query_plans, planner_name)
    plan = planner(PlannerInfo(name="correctness", base_id=generate_job_id()))

    topology = inspect_serialized_plan(plan)

    assert topology["node_count"] > 0
    assert topology["root_count"] == 1
    assert topology["leaf_count"] >= 1
    assert topology["duplicate_node_ids"] == []
    assert topology["missing_dependencies"] == []
    assert topology["cyclic"] is False


def test_serialized_plan_inspection_reports_cycle_and_missing_dependency() -> None:
    topology = inspect_serialized_plan(
        {
            "nodes": [
                {"task_id": "a", "dependencies": ["b"]},
                {"task_id": "b", "dependencies": ["a", "missing"]},
            ]
        }
    )

    assert topology["cyclic"] is True
    assert topology["missing_dependencies"] == ["missing"]


def test_parallel_mock_plan_has_expected_24_node_topology() -> None:
    from marie.job.job_manager import generate_job_id
    from marie.logging_core.log_bus import GLOBAL_LOG_BUS
    from marie.query_planner.base import PlannerInfo
    from marie.query_planner.mock_query_plans import (
        query_planner_mock_parallel_subgraphs,
    )

    GLOBAL_LOG_BUS.flush()
    plan = query_planner_mock_parallel_subgraphs(
        PlannerInfo(name="correctness", base_id=generate_job_id())
    )

    topology = inspect_serialized_plan(plan)

    assert topology["node_count"] == 24
    assert topology["edge_count"] == 34
    assert topology["root_count"] == 1
    assert topology["leaf_count"] == 1
    assert topology["fanout_nodes"] == 4
    assert topology["fanin_nodes"] == 4


def test_check_names_are_unique() -> None:
    names = [check.name for check in CHECKS] + [
        GATEWAY_DAG_SCOPE_CHECK.name,
        PARALLEL_GRAPH_CHECK.name,
        FAILED_DESCENDANTS_CHECK.name,
        FORCED_FAILURE_CHECK.name,
        *ATTEMPT_CHECK_NAMES,
    ]

    assert len(names) == len(set(names))


def test_check_queries_use_only_scope_parameters() -> None:
    for spec in CHECKS:
        query = _query_for(spec)

        assert query.count("%s") == 3, spec.name
        assert "{SCHEDULER_SCHEMA}" not in query, spec.name


def test_gateway_check_queries_use_gateway_scope_parameters() -> None:
    specs = (
        GATEWAY_DAG_SCOPE_CHECK,
        *GATEWAY_REUSED_CHECKS,
        PARALLEL_GRAPH_CHECK,
        FAILED_DESCENDANTS_CHECK,
        FORCED_FAILURE_CHECK,
    )

    for spec in specs:
        query = _query_for(spec, GATEWAY_SCOPE_CTES)

        assert query.count("%s") == 5, spec.name
        assert "{SCHEDULER_SCHEMA}" not in query, spec.name


def test_parallel_graph_query_bounds_per_dag_evidence() -> None:
    query = _query_for(PARALLEL_GRAPH_CHECK, GATEWAY_SCOPE_CTES)

    assert "'dag_count'" in query
    assert "'nodes_min'" in query
    assert "'nodes_max'" in query
    assert "'edges_min'" in query
    assert "'edges_max'" in query
    assert "'fanout_nodes_min'" in query
    assert "'fanin_nodes_min'" in query
    assert "'dag_sample'" in query
    assert "LIMIT (SELECT sample_limit FROM params)" in query


class FakeCursor:
    def __init__(
        self,
        row: dict[str, Any] | None,
        error: Exception | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.row = row
        self.error = error
        self.rows = rows or []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, _params: object = None) -> None:
        if "scheduler-correctness:" in query and self.error is not None:
            raise self.error

    def fetchone(self) -> dict[str, Any] | None:
        return self.row

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class FakeConnection:
    def __init__(
        self,
        row: dict[str, Any] | None,
        error: Exception | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.row = row
        self.error = error
        self.rows = rows

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.row, self.error, self.rows)

    def transaction(self):
        return nullcontext()


def build_verifier(
    row: dict[str, Any] | None, *, sample_limit: int = 2, error: Exception | None = None
) -> SchedulerCorrectnessVerifier:
    return SchedulerCorrectnessVerifier(
        FakeConnection(row, error),  # type: ignore[arg-type]
        "correctness-run",
        sample_limit,
        datetime(2026, 7, 21, tzinfo=timezone.utc),
    )


def test_run_check_limits_failure_samples() -> None:
    verifier = build_verifier(
        {
            "bad_rows": 4,
            "observed": {"bad_rows": 4},
            "expected": {"bad_rows": 0},
            "sample": ["a", "b", "c", "d"],
        }
    )

    result = verifier.run_check(CHECKS[0])

    assert result.status == "fail"
    assert result.bad_rows == 4
    assert result.sample == ["a", "b"]


def test_run_check_reports_query_errors() -> None:
    verifier = build_verifier(None, error=RuntimeError("query failed"))

    result = verifier.run_check(CHECKS[0])

    assert result.status == "error"
    assert result.reason == "query failed"


def test_shared_attempt_checks_preserve_schema_contract() -> None:
    rows = [
        {
            "check_name": name,
            "category": category,
            "bad_rows": int(name == "terminal_job_retains_lease"),
            "sample": ["job-1"] if name == "terminal_job_retains_lease" else [],
            "expectation": "invariant expectation",
        }
        for name, category in ATTEMPT_CHECK_CATEGORIES.items()
    ]
    verifier = SchedulerCorrectnessVerifier(
        FakeConnection(None, rows=rows),  # type: ignore[arg-type]
        "correctness-run",
        2,
        datetime(2026, 7, 21, tzinfo=timezone.utc),
    )

    results = verifier.run_attempt_checks()

    assert [result.name for result in results] == list(ATTEMPT_CHECK_NAMES)
    failed = next(
        result for result in results if result.name == "terminal_job_retains_lease"
    )
    assert failed.status == "fail"
    assert failed.sample == ["job-1"]


def test_shared_attempt_check_contract_mismatch_errors_every_check() -> None:
    verifier = SchedulerCorrectnessVerifier(
        FakeConnection(None, rows=[]),  # type: ignore[arg-type]
        "correctness-run",
        2,
        datetime(2026, 7, 21, tzinfo=timezone.utc),
    )

    results = verifier.run_attempt_checks()

    assert len(results) == len(ATTEMPT_CHECK_NAMES)
    assert {result.status for result in results} == {"error"}
    assert all("contract mismatch" in (result.reason or "") for result in results)


def test_manifest_refuses_unknown_run() -> None:
    with pytest.raises(RuntimeError, match="Unknown stress run_id"):
        build_verifier(None).manifest()


def test_gateway_scope_extracts_accepted_and_forced_dag_ids() -> None:
    report = {
        "run_identity": {"run_id": "correctness-run"},
        "summary": {"submitted_jobs": 2},
        "jobs": [
            {
                "job_id": "dag-1",
                "stress_run_id": "correctness-run",
                "force_fail": False,
            },
            {
                "job_id": "dag-2",
                "stress_run_id": "correctness-run",
                "force_fail": True,
            },
        ],
    }

    dag_ids, forced_dag_ids = gateway_scope_ids(report, "correctness-run")

    assert dag_ids == ("dag-1", "dag-2")
    assert forced_dag_ids == ("dag-2",)


def test_gateway_scope_rejects_truncated_job_records() -> None:
    report = {
        "run_identity": {"run_id": "correctness-run"},
        "summary": {"submitted_jobs": 2},
        "jobs": [
            {
                "job_id": "dag-1",
                "stress_run_id": "correctness-run",
            }
        ],
    }

    with pytest.raises(ValueError, match="does not retain every accepted job"):
        gateway_scope_ids(report, "correctness-run")


def test_gateway_scope_requires_accepted_dag_ids() -> None:
    with pytest.raises(ValueError, match="requires accepted DAG IDs"):
        SchedulerCorrectnessVerifier(
            FakeConnection(None),  # type: ignore[arg-type]
            "correctness-run",
            2,
            datetime(2026, 7, 21, tzinfo=timezone.utc),
            scope="gateway",
        )


def test_gateway_event_check_detects_conflicting_terminal_events() -> None:
    verifier = build_verifier(None)

    result = verifier._gateway_event_order(
        [
            {
                "job_id": "job-1",
                "raw_events": [
                    "extract.scheduled",
                    "extract.started",
                    "extract.completed",
                    "extract.failed",
                ],
            }
        ]
    )

    assert result.status == "fail"
    assert result.sample == ["job-1"]


def test_gateway_scope_fails_when_required_report_evidence_is_missing() -> None:
    verifier = SchedulerCorrectnessVerifier(
        FakeConnection(None),  # type: ignore[arg-type]
        "correctness-run",
        2,
        datetime(2026, 7, 21, tzinfo=timezone.utc),
        scope="gateway",
        dag_ids=("dag-1",),
    )

    identity = verifier._gateway_identity([])
    event_order = verifier._gateway_event_order([{"job_id": "dag-1"}])
    terminal_agreement = verifier._gateway_terminal_agreement([{"job_id": "dag-1"}])
    capacity = verifier._post_drain_capacity({})

    assert identity.status == "fail"
    assert event_order.status == "fail"
    assert terminal_agreement.status == "fail"
    assert capacity.status == "fail"


def test_gateway_scope_requires_complete_lifecycle_for_every_accepted_dag() -> None:
    verifier = SchedulerCorrectnessVerifier(
        FakeConnection(None),  # type: ignore[arg-type]
        "correctness-run",
        2,
        datetime(2026, 7, 21, tzinfo=timezone.utc),
        scope="gateway",
        dag_ids=("dag-1",),
    )

    result = verifier._gateway_event_order(
        [{"job_id": "dag-1", "raw_events": ["job.scheduled", "job.completed"]}]
    )

    assert result.status == "fail"
    assert result.sample == ["dag-1"]


def test_gateway_identity_maps_public_job_id_to_dag() -> None:
    verifier = SchedulerCorrectnessVerifier(
        FakeConnection(
            None,
            rows=[{"id": "dag-1", "dag_matches": 1, "tagged_jobs": 23}],
        ),  # type: ignore[arg-type]
        "correctness-run",
        2,
        datetime(2026, 7, 21, tzinfo=timezone.utc),
    )

    result = verifier._gateway_identity([{"job_id": "dag-1"}])

    assert result.status == "pass"


@pytest.mark.parametrize(
    "snapshot",
    [
        {"ok": False, "used": 0, "holder_count": 0},
        {"ok": True, "used": 0},
    ],
)
def test_post_drain_capacity_fails_closed(snapshot: dict[str, Any]) -> None:
    result = build_verifier(None)._post_drain_capacity(
        {"post_drain_capacity": snapshot}
    )

    assert result.status == "fail"


def test_database_config_never_accepts_credentials(tmp_path) -> None:
    path = tmp_path / "stress.json"
    path.write_text(
        json.dumps({"database": {"host": "localhost", "password": "secret"}})
    )

    with pytest.raises(ValueError, match="PGPASSWORD"):
        load_database_config(str(path))


def test_unavailable_gateway_report_is_optional(tmp_path) -> None:
    report, error = load_gateway_report(str(tmp_path / "missing.json"))

    assert report is None
    assert error is not None


def test_settle_deadline_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        parse_deadline("2026-07-21T12:00:00")
