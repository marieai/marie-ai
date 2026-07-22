from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.stress.scheduler_qualification import (
    CAPACITY_FACTORS,
    SCALE_TARGETS,
    QualificationConfig,
    ResourceBudget,
    calculate_rates,
    classify_queue_age,
    classify_resource_trend,
    compare_trace_pair,
    evaluate_overload_recovery,
    evaluate_results,
    expand_matrix,
    main,
    match_trace_pairs,
    validate_result,
)


def build_config(**overrides: object) -> QualificationConfig:
    values: dict[str, object] = {
        "run_id": "scheduler-scale-v1",
        "database_identity": "scheduler-lab-primary",
        "source_identity": "commit-abc123",
        "configuration_identity": "scheduler-default-v1",
        "executor_capacity_jobs_per_second": 10.0,
        "capacity_duration_seconds": 600,
        "burst_duration_seconds": 60,
        "endurance_duration_seconds": [28_800, 86_400],
        "burst_targets": [100_000, 1_000_000, 10_000_000],
        "endurance_targets": [1_000_000, 10_000_000],
        "overload_factors": [2.0],
        "burst_factors": [5.0, 10.0],
        "recovery_max_queue_age_seconds": 30,
        "recovery_max_backlog_jobs": 0,
        "resource_budgets": {},
    }
    values.update(overrides)
    return QualificationConfig.from_mapping(values)


def build_result(trial: dict[str, object]) -> dict[str, object]:
    compact = trial["trace_mode"] == "compact"
    return {
        **trial,
        "submitted_jobs": 600,
        "accepted_jobs": 600,
        "completed_jobs": 600,
        "elapsed_seconds": 600,
        "preflight": {
            "passed": True,
            "queue": "scheduler_stress_v1",
            "required_executors": ["mock_executor"],
        },
        "database_checkpoint": {
            "target_dag_count": trial["target_dag_count"],
            "current_dag_count": trial["target_dag_count"],
            "same_corpus": True,
        },
        "correctness_result": {"passed": True},
        "capacity_holders_in_use_after_drain": 0,
        "queue_samples": [
            {
                "elapsed_seconds": 0,
                "input_rate": 8,
                "live_backlog": 4,
                "oldest_queue_age_seconds": 4,
            },
            {
                "elapsed_seconds": 10,
                "input_rate": 8,
                "live_backlog": 3,
                "oldest_queue_age_seconds": 3,
            },
            {
                "elapsed_seconds": 20,
                "input_rate": 8,
                "live_backlog": 2,
                "oldest_queue_age_seconds": 2,
            },
            {
                "elapsed_seconds": 30,
                "input_rate": 8,
                "live_backlog": 0,
                "oldest_queue_age_seconds": 0,
            },
        ],
        "resource_samples": [
            {"elapsed_seconds": 0, "live_backlog": 0, "gateway_rss_bytes": 100},
            {"elapsed_seconds": 10, "live_backlog": 0, "gateway_rss_bytes": 100},
            {"elapsed_seconds": 20, "live_backlog": 0, "gateway_rss_bytes": 100},
        ],
        "metrics": {
            "throughput_jobs_per_second": 1.0 if not compact else 0.95,
            "latency_p95_ms": 100 if not compact else 105,
            "latency_p99_ms": 120 if not compact else 126,
            "event_loop_lag_p99_ms": 5 if not compact else 6,
            "cpu_percent": 40 if not compact else 42,
            "peak_rss_bytes": 1_000 if not compact else 1_050,
            "trace_bytes": 100 if compact else 0,
        },
    }


def test_matrix_expands_required_scale_capacity_and_trace_pairs() -> None:
    trials = expand_matrix(build_config())

    assert len(trials) == 68
    assert {trial["target_dag_count"] for trial in trials} == set(SCALE_TARGETS)
    assert {
        trial["load_factor"] for trial in trials if trial["scenario"] == "capacity"
    } == set(CAPACITY_FACTORS)
    assert all(
        trial["database_identity"] == "scheduler-lab-primary" for trial in trials
    )

    pairs: dict[str, set[str]] = {}
    for trial in trials:
        pairs.setdefault(trial["trace_pair_id"], set()).add(trial["trace_mode"])
    assert all(modes == {"off", "compact"} for modes in pairs.values())


def test_matrix_calculates_load_from_discovered_capacity() -> None:
    trial = next(
        trial
        for trial in expand_matrix(build_config())
        if trial["scenario"] == "capacity"
        and trial["load_factor"] == 0.75
        and trial["trace_mode"] == "off"
    )

    assert trial["target_submit_rate"] == pytest.approx(7.5)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"scale_targets": [1_000]}, "scale_targets"),
        ({"trace_modes": ["off", "full"]}, "trace_modes"),
        ({"overload_factors": [1.5]}, "200%"),
        ({"burst_factors": [4.0]}, "5x-10x"),
        ({"burst_factors": [5.0, 5.0]}, "duplicates"),
    ],
)
def test_config_rejects_incomplete_qualification_contract(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_config(**override)


def test_rate_calculation_reports_acceptance_and_completion() -> None:
    rates = calculate_rates(
        submitted_jobs=100,
        accepted_jobs=90,
        completed_jobs=81,
        duration_seconds=10,
    )

    assert rates == {
        "submitted_jobs_per_second": 10.0,
        "accepted_jobs_per_second": 9.0,
        "completed_jobs_per_second": 8.1,
        "acceptance_pct": 90.0,
        "completion_pct": 90.0,
    }


def test_rate_calculation_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="completed_jobs"):
        calculate_rates(
            submitted_jobs=10,
            accepted_jobs=8,
            completed_jobs=9,
            duration_seconds=10,
        )


def test_resource_trend_removes_backlog_and_history_growth() -> None:
    samples = [
        {
            "elapsed_seconds": 0,
            "live_backlog": 0,
            "completed_history_count": 0,
            "rss": 100,
        },
        {
            "elapsed_seconds": 1_800,
            "live_backlog": 10,
            "completed_history_count": 20,
            "rss": 170,
        },
        {
            "elapsed_seconds": 3_600,
            "live_backlog": 20,
            "completed_history_count": 40,
            "rss": 240,
        },
    ]

    trend = classify_resource_trend(
        samples,
        "rss",
        ResourceBudget(
            max_residual_growth_per_hour=1,
            backlog_unit_cost=3,
            history_unit_cost=2,
        ),
    )

    assert trend["classification"] == "workload_correlated"
    assert trend["passed"] is True
    assert trend["residual_growth_per_hour"] == pytest.approx(0)


def test_resource_trend_fails_unexplained_growth() -> None:
    samples = [
        {"elapsed_seconds": 0, "live_backlog": 0, "threads": 10},
        {"elapsed_seconds": 1_800, "live_backlog": 0, "threads": 12},
        {"elapsed_seconds": 3_600, "live_backlog": 0, "threads": 14},
    ]

    trend = classify_resource_trend(
        samples,
        "threads",
        ResourceBudget(max_residual_growth_per_hour=1),
    )

    assert trend["classification"] == "unexplained_growth"
    assert trend["passed"] is False
    assert trend["residual_growth_per_hour"] == pytest.approx(4)


def test_queue_age_classifies_stable_or_draining_tail() -> None:
    result = classify_queue_age(
        [
            {"elapsed_seconds": 0, "oldest_queue_age_seconds": 0},
            {"elapsed_seconds": 10, "oldest_queue_age_seconds": 5},
            {"elapsed_seconds": 20, "oldest_queue_age_seconds": 4},
            {"elapsed_seconds": 30, "oldest_queue_age_seconds": 2},
        ]
    )

    assert result["classification"] == "stable_or_draining"
    assert result["passed"] is True


def test_overload_recovery_requires_queue_to_remain_drained() -> None:
    samples = [
        {
            "elapsed_seconds": 0,
            "input_rate": 20,
            "live_backlog": 50,
            "oldest_queue_age_seconds": 10,
        },
        {
            "elapsed_seconds": 10,
            "input_rate": 20,
            "live_backlog": 100,
            "oldest_queue_age_seconds": 20,
        },
        {
            "elapsed_seconds": 20,
            "input_rate": 5,
            "live_backlog": 50,
            "oldest_queue_age_seconds": 15,
        },
        {
            "elapsed_seconds": 30,
            "input_rate": 5,
            "live_backlog": 0,
            "oldest_queue_age_seconds": 1,
        },
        {
            "elapsed_seconds": 40,
            "input_rate": 5,
            "live_backlog": 0,
            "oldest_queue_age_seconds": 0,
        },
    ]

    recovery = evaluate_overload_recovery(
        samples,
        service_capacity=10,
        max_queue_age_seconds=2,
        max_backlog_jobs=0,
    )

    assert recovery["passed"] is True
    assert recovery["recovery_seconds"] == 10


def test_trace_pair_reports_matched_observer_effect() -> None:
    off_trial, compact_trial = expand_matrix(build_config())[:2]
    off = build_result(off_trial)
    compact = build_result(compact_trial)

    comparison = compare_trace_pair(off, compact)

    assert comparison["trace_bytes"] == 100
    assert comparison["deltas_pct"]["throughput_jobs_per_second_delta_pct"] == (
        pytest.approx(-5)
    )
    assert comparison["deltas_pct"]["peak_rss_bytes_delta_pct"] == pytest.approx(5)


def test_trace_pair_rejects_configuration_mismatch() -> None:
    off_trial, compact_trial = expand_matrix(build_config())[:2]
    off = build_result(off_trial)
    compact = build_result(compact_trial)
    compact["source_identity"] = "different-commit"

    with pytest.raises(ValueError, match="source_identity"):
        compare_trace_pair(off, compact)


def test_trace_pair_matching_rejects_unpaired_result() -> None:
    off_trial = expand_matrix(build_config())[0]

    with pytest.raises(ValueError, match="requires off and compact"):
        match_trace_pairs([build_result(off_trial)])


def test_result_contract_requires_same_database_and_zero_holders() -> None:
    trial = expand_matrix(build_config())[0]
    result = build_result(trial)
    result["database_checkpoint"] = {
        "target_dag_count": trial["target_dag_count"],
        "current_dag_count": trial["target_dag_count"],
        "same_corpus": False,
    }
    result["capacity_holders_in_use_after_drain"] = 1

    errors = validate_result(result, trial)

    assert "database checkpoint is not tied to the same corpus" in errors
    assert "capacity holders remain after drain" in errors


def test_complete_matrix_evaluation_passes_valid_results() -> None:
    config = build_config(burst_targets=[], endurance_targets=[])
    trials = expand_matrix(config)
    results = [build_result(trial) for trial in trials]
    for result in results:
        if result["scenario"] == "overload":
            result["queue_samples"] = [
                {
                    "elapsed_seconds": 0,
                    "input_rate": 20,
                    "live_backlog": 10,
                    "oldest_queue_age_seconds": 5,
                },
                {
                    "elapsed_seconds": 10,
                    "input_rate": 5,
                    "live_backlog": 2,
                    "oldest_queue_age_seconds": 2,
                },
                {
                    "elapsed_seconds": 20,
                    "input_rate": 5,
                    "live_backlog": 0,
                    "oldest_queue_age_seconds": 0,
                },
            ]

    evaluation = evaluate_results(config, results)

    assert evaluation["passed"] is True
    assert evaluation["planned_trial_count"] == 48
    assert len(evaluation["trace_pairs"]) == 24


def test_endurance_evaluation_requires_resource_budgets() -> None:
    config = build_config()
    endurance_trials = [
        trial for trial in expand_matrix(config) if trial["scenario"] == "endurance"
    ][:2]

    evaluation = evaluate_results(
        config, [build_result(trial) for trial in endurance_trials]
    )

    assert evaluation["trials"][0]["passed"] is False
    assert (
        "endurance resource budgets are not configured"
        in evaluation["trials"][0]["errors"]
    )


def test_failed_trace_pair_is_reported_without_aborting_evaluation() -> None:
    config = build_config(burst_targets=[], endurance_targets=[])
    off_trial, compact_trial = expand_matrix(config)[:2]
    off = build_result(off_trial)
    compact = build_result(compact_trial)
    compact["correctness_result"] = {"passed": False}

    evaluation = evaluate_results(config, [off, compact])

    assert evaluation["passed"] is False
    assert evaluation["trace_pair_errors"] == [
        "Compact-trace result failed correctness"
    ]


def test_plan_cli_writes_expanded_matrix(tmp_path: Path) -> None:
    config_path = tmp_path / "qualification.json"
    output_path = tmp_path / "matrix.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "scheduler-scale-v1",
                "database_identity": "scheduler-lab-primary",
                "source_identity": "commit-abc123",
                "configuration_identity": "scheduler-default-v1",
                "executor_capacity_jobs_per_second": 10,
                "capacity_duration_seconds": 600,
                "burst_duration_seconds": 60,
                "endurance_duration_seconds": [28_800, 86_400],
                "burst_targets": [100_000, 1_000_000, 10_000_000],
                "endurance_targets": [1_000_000, 10_000_000],
                "recovery_max_queue_age_seconds": 30,
                "recovery_max_backlog_jobs": 0,
            }
        )
    )

    exit_code = main(
        ["plan", "--config", str(config_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text())["trial_count"] == 68
