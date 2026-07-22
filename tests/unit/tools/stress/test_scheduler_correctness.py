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
    SchedulerCorrectnessVerifier,
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


def test_check_names_are_unique() -> None:
    names = [check.name for check in CHECKS] + list(ATTEMPT_CHECK_NAMES)

    assert len(names) == len(set(names))


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
