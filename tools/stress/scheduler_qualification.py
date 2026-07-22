from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

QUALIFICATION_VERSION = "1"
SCALE_TARGETS = (1_000, 100_000, 1_000_000, 10_000_000)
CAPACITY_FACTORS = (0.60, 0.75, 0.90, 1.00, 1.20)
TRACE_MODES = ("off", "compact")


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    max_residual_growth_per_hour: float
    backlog_unit_cost: float = 0.0
    history_unit_cost: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourceBudget":
        budget = cls(
            max_residual_growth_per_hour=float(value["max_residual_growth_per_hour"]),
            backlog_unit_cost=float(value.get("backlog_unit_cost", 0.0)),
            history_unit_cost=float(value.get("history_unit_cost", 0.0)),
        )
        if (
            min(
                budget.max_residual_growth_per_hour,
                budget.backlog_unit_cost,
                budget.history_unit_cost,
            )
            < 0
        ):
            raise ValueError("Resource budgets cannot be negative")
        return budget


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    run_id: str
    database_identity: str
    source_identity: str
    configuration_identity: str
    executor_capacity_jobs_per_second: float
    capacity_duration_seconds: float
    burst_duration_seconds: float
    endurance_duration_seconds: tuple[float, ...]
    burst_targets: tuple[int, ...]
    endurance_targets: tuple[int, ...]
    overload_factors: tuple[float, ...]
    burst_factors: tuple[float, ...]
    recovery_max_queue_age_seconds: float
    recovery_max_backlog_jobs: int
    resource_budgets: Mapping[str, ResourceBudget]
    scale_targets: tuple[int, ...] = SCALE_TARGETS
    capacity_factors: tuple[float, ...] = CAPACITY_FACTORS
    trace_modes: tuple[str, ...] = TRACE_MODES
    endurance_load_factor: float = 0.75

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QualificationConfig":
        resource_budgets = {
            name: ResourceBudget.from_mapping(budget)
            for name, budget in value.get("resource_budgets", {}).items()
        }
        config = cls(
            run_id=str(value["run_id"]),
            database_identity=str(value["database_identity"]),
            source_identity=str(value["source_identity"]),
            configuration_identity=str(value["configuration_identity"]),
            executor_capacity_jobs_per_second=float(
                value["executor_capacity_jobs_per_second"]
            ),
            capacity_duration_seconds=float(value["capacity_duration_seconds"]),
            burst_duration_seconds=float(value["burst_duration_seconds"]),
            endurance_duration_seconds=tuple(
                float(item) for item in value["endurance_duration_seconds"]
            ),
            burst_targets=tuple(int(item) for item in value["burst_targets"]),
            endurance_targets=tuple(int(item) for item in value["endurance_targets"]),
            overload_factors=tuple(
                float(item) for item in value.get("overload_factors", (2.0,))
            ),
            burst_factors=tuple(
                float(item) for item in value.get("burst_factors", (5.0, 10.0))
            ),
            recovery_max_queue_age_seconds=float(
                value["recovery_max_queue_age_seconds"]
            ),
            recovery_max_backlog_jobs=int(value["recovery_max_backlog_jobs"]),
            resource_budgets=resource_budgets,
            scale_targets=tuple(
                int(item) for item in value.get("scale_targets", SCALE_TARGETS)
            ),
            capacity_factors=tuple(
                float(item) for item in value.get("capacity_factors", CAPACITY_FACTORS)
            ),
            trace_modes=tuple(
                str(item) for item in value.get("trace_modes", TRACE_MODES)
            ),
            endurance_load_factor=float(value.get("endurance_load_factor", 0.75)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not all(
            (
                self.run_id,
                self.database_identity,
                self.source_identity,
                self.configuration_identity,
            )
        ):
            raise ValueError(
                "Run, database, source, and configuration identities are required"
            )
        if self.scale_targets != SCALE_TARGETS:
            raise ValueError(f"scale_targets must be {list(SCALE_TARGETS)}")
        if not set(CAPACITY_FACTORS).issubset(self.capacity_factors):
            raise ValueError(f"capacity_factors must include {list(CAPACITY_FACTORS)}")
        if 2.0 not in self.overload_factors:
            raise ValueError("overload_factors must include the selected 200% point")
        if any(factor < 5.0 or factor > 10.0 for factor in self.burst_factors):
            raise ValueError("burst_factors must stay within the planned 5x-10x range")
        if set(self.trace_modes) != set(TRACE_MODES):
            raise ValueError("trace_modes must contain exactly 'off' and 'compact'")
        if not set(self.burst_targets).issubset(self.scale_targets):
            raise ValueError("burst_targets must be scale checkpoints")
        if not set(self.endurance_targets).issubset(self.scale_targets):
            raise ValueError("endurance_targets must be scale checkpoints")
        sequences = (
            self.scale_targets,
            self.capacity_factors,
            self.overload_factors,
            self.burst_factors,
            self.endurance_duration_seconds,
            self.burst_targets,
            self.endurance_targets,
            self.trace_modes,
        )
        if any(len(values) != len(set(values)) for values in sequences):
            raise ValueError("Qualification matrix values cannot contain duplicates")
        if not self.burst_factors:
            raise ValueError("At least one burst factor is required")
        if self.endurance_targets and not self.endurance_duration_seconds:
            raise ValueError("Endurance targets require at least one duration")
        positive_values = (
            self.executor_capacity_jobs_per_second,
            self.capacity_duration_seconds,
            self.burst_duration_seconds,
            self.endurance_load_factor,
            *self.endurance_duration_seconds,
            *self.capacity_factors,
            *self.overload_factors,
            *self.burst_factors,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("Rates, factors, and durations must be positive")
        if self.recovery_max_queue_age_seconds < 0:
            raise ValueError("recovery_max_queue_age_seconds cannot be negative")
        if self.recovery_max_backlog_jobs < 0:
            raise ValueError("recovery_max_backlog_jobs cannot be negative")


def load_config(path: str | Path) -> QualificationConfig:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Qualification config must be a JSON object")
    return QualificationConfig.from_mapping(payload)


def calculate_rates(
    *,
    submitted_jobs: int,
    accepted_jobs: int,
    completed_jobs: int,
    duration_seconds: float,
) -> dict[str, float]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if min(submitted_jobs, accepted_jobs, completed_jobs) < 0:
        raise ValueError("Job counts cannot be negative")
    if accepted_jobs > submitted_jobs:
        raise ValueError("accepted_jobs cannot exceed submitted_jobs")
    if completed_jobs > accepted_jobs:
        raise ValueError("completed_jobs cannot exceed accepted_jobs")

    return {
        "submitted_jobs_per_second": submitted_jobs / duration_seconds,
        "accepted_jobs_per_second": accepted_jobs / duration_seconds,
        "completed_jobs_per_second": completed_jobs / duration_seconds,
        "acceptance_pct": (
            accepted_jobs / submitted_jobs * 100.0 if submitted_jobs else 0.0
        ),
        "completion_pct": (
            completed_jobs / accepted_jobs * 100.0 if accepted_jobs else 0.0
        ),
    }


def _trial_id(
    run_id: str,
    target: int,
    scenario: str,
    load_factor: float,
    duration_seconds: float,
    trace_mode: str,
) -> str:
    factor = f"{load_factor:.2f}".replace(".", "p")
    duration = int(duration_seconds)
    return f"{run_id}-{target}-{scenario}-{factor}-{duration}s-{trace_mode}"


def expand_matrix(config: QualificationConfig) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []

    def add_trials(
        *,
        target: int,
        scenario: str,
        load_factor: float,
        duration_seconds: float,
    ) -> None:
        pair_id = _trial_id(
            config.run_id,
            target,
            scenario,
            load_factor,
            duration_seconds,
            "trace-pair",
        )
        for trace_mode in config.trace_modes:
            trials.append(
                {
                    "qualification_version": QUALIFICATION_VERSION,
                    "trial_id": _trial_id(
                        config.run_id,
                        target,
                        scenario,
                        load_factor,
                        duration_seconds,
                        trace_mode,
                    ),
                    "trace_pair_id": pair_id,
                    "run_id": config.run_id,
                    "database_identity": config.database_identity,
                    "source_identity": config.source_identity,
                    "configuration_identity": config.configuration_identity,
                    "target_dag_count": target,
                    "scenario": scenario,
                    "load_factor": load_factor,
                    "target_submit_rate": (
                        config.executor_capacity_jobs_per_second * load_factor
                    ),
                    "executor_capacity_jobs_per_second": (
                        config.executor_capacity_jobs_per_second
                    ),
                    "duration_seconds": duration_seconds,
                    "trace_mode": trace_mode,
                }
            )

    for target in config.scale_targets:
        for factor in config.capacity_factors:
            add_trials(
                target=target,
                scenario="capacity",
                load_factor=factor,
                duration_seconds=config.capacity_duration_seconds,
            )
        for factor in config.overload_factors:
            add_trials(
                target=target,
                scenario="overload",
                load_factor=factor,
                duration_seconds=config.capacity_duration_seconds,
            )

    for target in config.burst_targets:
        for factor in config.burst_factors:
            add_trials(
                target=target,
                scenario="burst",
                load_factor=factor,
                duration_seconds=config.burst_duration_seconds,
            )

    for target in config.endurance_targets:
        for duration in config.endurance_duration_seconds:
            add_trials(
                target=target,
                scenario="endurance",
                load_factor=config.endurance_load_factor,
                duration_seconds=duration,
            )

    return trials


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0:
        raise ValueError("Sample times must span a nonzero interval")
    return (
        sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(xs, ys))
        / denominator
    )


def classify_resource_trend(
    samples: Sequence[Mapping[str, Any]],
    metric: str,
    budget: ResourceBudget,
) -> dict[str, Any]:
    if len(samples) < 3:
        raise ValueError("At least three resource samples are required")

    elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
    if any(later <= earlier for earlier, later in zip(elapsed, elapsed[1:])):
        raise ValueError("Resource sample times must increase")

    raw = [float(sample[metric]) for sample in samples]
    normalized = [
        value
        - float(sample.get("live_backlog", 0)) * budget.backlog_unit_cost
        - float(sample.get("completed_history_count", 0)) * budget.history_unit_cost
        for sample, value in zip(samples, raw)
    ]
    raw_slope = _linear_slope(elapsed, raw) * 3600.0
    residual_slope = _linear_slope(elapsed, normalized) * 3600.0

    if residual_slope > budget.max_residual_growth_per_hour:
        classification = "unexplained_growth"
        passed = False
    elif raw_slope > budget.max_residual_growth_per_hour:
        classification = "workload_correlated"
        passed = True
    else:
        classification = "stable"
        passed = True

    return {
        "metric": metric,
        "classification": classification,
        "passed": passed,
        "raw_growth_per_hour": raw_slope,
        "residual_growth_per_hour": residual_slope,
        "max_residual_growth_per_hour": budget.max_residual_growth_per_hour,
        "sample_count": len(samples),
    }


def evaluate_overload_recovery(
    samples: Sequence[Mapping[str, Any]],
    *,
    service_capacity: float,
    max_queue_age_seconds: float,
    max_backlog_jobs: int,
) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("At least two queue samples are required")
    elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
    if any(later <= earlier for earlier, later in zip(elapsed, elapsed[1:])):
        raise ValueError("Queue sample times must increase")

    overloaded_indexes = [
        index
        for index, sample in enumerate(samples)
        if float(sample["input_rate"]) > service_capacity
    ]
    if not overloaded_indexes:
        raise ValueError("Recovery samples do not contain an overload period")

    overload_end = overloaded_indexes[-1]
    below_capacity = next(
        (
            index
            for index in range(overload_end + 1, len(samples))
            if float(samples[index]["input_rate"]) < service_capacity
        ),
        None,
    )
    if below_capacity is None:
        return {
            "passed": False,
            "reason": "input_never_fell_below_capacity",
            "recovery_seconds": None,
        }

    recovered_at = None
    for index in range(below_capacity, len(samples)):
        remaining = samples[index:]
        if all(
            int(sample["live_backlog"]) <= max_backlog_jobs
            and float(sample["oldest_queue_age_seconds"]) <= max_queue_age_seconds
            for sample in remaining
        ):
            recovered_at = index
            break

    if recovered_at is None:
        return {
            "passed": False,
            "reason": "queue_did_not_recover",
            "recovery_seconds": None,
        }

    return {
        "passed": True,
        "reason": None,
        "recovery_seconds": elapsed[recovered_at] - elapsed[below_capacity],
        "below_capacity_at_seconds": elapsed[below_capacity],
        "recovered_at_seconds": elapsed[recovered_at],
    }


def classify_queue_age(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(samples) < 3:
        raise ValueError("At least three queue samples are required")
    steady_window = samples[len(samples) // 2 :]
    elapsed = [float(sample["elapsed_seconds"]) for sample in steady_window]
    queue_age = [float(sample["oldest_queue_age_seconds"]) for sample in steady_window]
    slope = _linear_slope(elapsed, queue_age)
    return {
        "classification": "stable_or_draining" if slope <= 0 else "growing",
        "passed": slope <= 0,
        "queue_age_growth_per_second": slope,
        "sample_count": len(steady_window),
    }


_RESULT_IDENTITY_FIELDS = (
    "qualification_version",
    "trial_id",
    "trace_pair_id",
    "run_id",
    "database_identity",
    "source_identity",
    "configuration_identity",
    "target_dag_count",
    "scenario",
    "load_factor",
    "duration_seconds",
    "trace_mode",
)


def validate_result(
    result: Mapping[str, Any], planned_trial: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    for field in _RESULT_IDENTITY_FIELDS:
        if field not in result:
            errors.append(f"missing {field}")
        elif result[field] != planned_trial[field]:
            errors.append(f"{field} does not match the planned trial")

    for field in (
        "submitted_jobs",
        "accepted_jobs",
        "completed_jobs",
        "elapsed_seconds",
        "preflight",
        "database_checkpoint",
        "correctness_result",
        "capacity_holders_in_use_after_drain",
        "queue_samples",
        "resource_samples",
        "metrics",
    ):
        if field not in result:
            errors.append(f"missing {field}")

    preflight = result.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("passed") is not True:
        errors.append("preflight did not pass")

    checkpoint = result.get("database_checkpoint")
    if not isinstance(checkpoint, Mapping):
        errors.append("database_checkpoint must be an object")
    else:
        if checkpoint.get("target_dag_count") != planned_trial["target_dag_count"]:
            errors.append("database checkpoint target does not match")
        current_count = checkpoint.get("current_dag_count")
        if (
            not isinstance(current_count, int)
            or current_count < planned_trial["target_dag_count"]
        ):
            errors.append("database checkpoint has not reached the planned target")
        if checkpoint.get("same_corpus") is not True:
            errors.append("database checkpoint is not tied to the same corpus")

    correctness = result.get("correctness_result")
    if not isinstance(correctness, Mapping) or correctness.get("passed") is not True:
        errors.append("correctness verification did not pass")
    if result.get("capacity_holders_in_use_after_drain") != 0:
        errors.append("capacity holders remain after drain")
    return errors


_TRACE_METRICS = (
    "throughput_jobs_per_second",
    "latency_p95_ms",
    "latency_p99_ms",
    "event_loop_lag_p99_ms",
    "cpu_percent",
    "peak_rss_bytes",
)


def _percent_delta(baseline: float, observed: float) -> float | None:
    if baseline == 0:
        return None
    return (observed - baseline) / baseline * 100.0


def compare_trace_pair(
    trace_off: Mapping[str, Any], trace_compact: Mapping[str, Any]
) -> dict[str, Any]:
    if trace_off.get("trace_mode") != "off":
        raise ValueError("Trace baseline must use trace_mode='off'")
    if trace_compact.get("trace_mode") != "compact":
        raise ValueError("Trace comparison must use trace_mode='compact'")

    for field in _RESULT_IDENTITY_FIELDS:
        if field in {"trial_id", "trace_mode"}:
            continue
        if trace_off.get(field) != trace_compact.get(field):
            raise ValueError(f"Trace pair mismatch: {field}")
    if trace_off.get("correctness_result", {}).get("passed") is not True:
        raise ValueError("Trace-off result failed correctness")
    if trace_compact.get("correctness_result", {}).get("passed") is not True:
        raise ValueError("Compact-trace result failed correctness")

    off_metrics = trace_off.get("metrics")
    compact_metrics = trace_compact.get("metrics")
    if not isinstance(off_metrics, Mapping) or not isinstance(compact_metrics, Mapping):
        raise ValueError("Trace results must include metrics")

    deltas: dict[str, float | None] = {}
    for metric in _TRACE_METRICS:
        if metric not in off_metrics or metric not in compact_metrics:
            raise ValueError(f"Trace results are missing {metric}")
        deltas[f"{metric}_delta_pct"] = _percent_delta(
            float(off_metrics[metric]), float(compact_metrics[metric])
        )
    if float(compact_metrics.get("trace_bytes", 0)) <= 0:
        raise ValueError("Compact trace result must report positive trace_bytes")

    return {
        "trace_pair_id": trace_off["trace_pair_id"],
        "trace_off_trial_id": trace_off["trial_id"],
        "trace_compact_trial_id": trace_compact["trial_id"],
        "trace_bytes": float(compact_metrics["trace_bytes"]),
        "deltas_pct": deltas,
    }


def match_trace_pairs(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for result in results:
        pair_id = str(result["trace_pair_id"])
        mode = str(result["trace_mode"])
        modes = grouped.setdefault(pair_id, {})
        if mode in modes:
            raise ValueError(f"Duplicate {mode} result for trace pair {pair_id}")
        modes[mode] = result

    comparisons = []
    for pair_id, modes in grouped.items():
        if set(modes) != set(TRACE_MODES):
            raise ValueError(f"Trace pair {pair_id} requires off and compact results")
        comparisons.append(compare_trace_pair(modes["off"], modes["compact"]))
    return comparisons


def evaluate_results(
    config: QualificationConfig, results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    planned_trials = expand_matrix(config)
    planned_by_id = {trial["trial_id"]: trial for trial in planned_trials}
    seen_ids: set[str] = set()
    evaluations: list[dict[str, Any]] = []

    for result in results:
        trial_id = str(result.get("trial_id", ""))
        if trial_id in seen_ids:
            raise ValueError(f"Duplicate result for trial {trial_id}")
        seen_ids.add(trial_id)
        planned = planned_by_id.get(trial_id)
        if planned is None:
            raise ValueError(
                f"Result does not belong to the planned matrix: {trial_id}"
            )

        errors = validate_result(result, planned)
        if planned["scenario"] == "endurance" and not config.resource_budgets:
            errors.append("endurance resource budgets are not configured")
        rates = None
        queue_age = None
        recovery = None
        resource_trends: list[dict[str, Any]] = []
        if not errors:
            try:
                rates = calculate_rates(
                    submitted_jobs=int(result["submitted_jobs"]),
                    accepted_jobs=int(result["accepted_jobs"]),
                    completed_jobs=int(result["completed_jobs"]),
                    duration_seconds=float(result["elapsed_seconds"]),
                )
                queue_samples = result["queue_samples"]
                if planned["scenario"] == "capacity":
                    queue_age = classify_queue_age(queue_samples)
                elif planned["scenario"] in {"overload", "burst"}:
                    recovery = evaluate_overload_recovery(
                        queue_samples,
                        service_capacity=config.executor_capacity_jobs_per_second,
                        max_queue_age_seconds=config.recovery_max_queue_age_seconds,
                        max_backlog_jobs=config.recovery_max_backlog_jobs,
                    )

                resource_samples = result["resource_samples"]
                resource_trends = [
                    classify_resource_trend(resource_samples, metric, budget)
                    for metric, budget in config.resource_budgets.items()
                ]
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"invalid measurement data: {exc}")

        passed = (
            not errors
            and (queue_age is None or queue_age["passed"])
            and (recovery is None or recovery["passed"])
            and all(trend["passed"] for trend in resource_trends)
        )
        evaluations.append(
            {
                "trial_id": trial_id,
                "passed": passed,
                "errors": errors,
                "rates": rates,
                "queue_age": queue_age,
                "recovery": recovery,
                "resource_trends": resource_trends,
            }
        )

    missing = sorted(set(planned_by_id) - seen_ids)
    trace_pair_errors: list[str] = []
    try:
        trace_pairs = match_trace_pairs(results) if results else []
    except (KeyError, TypeError, ValueError) as exc:
        trace_pairs = []
        trace_pair_errors.append(str(exc))
    return {
        "qualification_version": QUALIFICATION_VERSION,
        "run_id": config.run_id,
        "passed": (
            bool(evaluations)
            and not missing
            and not trace_pair_errors
            and all(evaluation["passed"] for evaluation in evaluations)
        ),
        "planned_trial_count": len(planned_trials),
        "result_count": len(results),
        "missing_trial_ids": missing,
        "trials": evaluations,
        "trace_pairs": trace_pairs,
        "trace_pair_errors": trace_pair_errors,
    }


def _load_results(path: str | Path) -> list[Mapping[str, Any]]:
    result_path = Path(path)
    if result_path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in result_path.read_text().splitlines()
            if line.strip()
        ]
    payload = json.loads(result_path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Qualification results must be a JSON list or JSONL")
    return payload


def _write_payload(payload: Mapping[str, Any], output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and evaluate scheduler scale qualification runs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Expand the qualification matrix")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output")

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate completed qualification results"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--results", required=True)
    evaluate.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.command == "plan":
        trials = expand_matrix(config)
        _write_payload(
            {
                "qualification_version": QUALIFICATION_VERSION,
                "run_id": config.run_id,
                "trial_count": len(trials),
                "trials": trials,
            },
            args.output,
        )
        return 0

    results = _load_results(args.results)
    evaluation = evaluate_results(config, results)
    _write_payload(evaluation, args.output)
    return 0 if evaluation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
