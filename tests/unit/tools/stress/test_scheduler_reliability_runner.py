from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from tools.stress.etcd_recovery_orchestration import (
    EtcdRecoveryRunner,
    EtcdReliabilityConfig,
    parse_capacity_snapshot,
)
from tools.stress.scheduler_reliability_runner import (
    REQUIRED_FAULTS,
    ActionResult,
    CommandSpec,
    ConfigurationError,
    GatewayDiagnosticSpec,
    ProbeResult,
    ProbeSpec,
    QueryBudgetSpec,
    Scenario,
    ScenarioContext,
    SchedulerReliabilityRunner,
    TargetSpec,
    evaluate_query_budget,
    load_config,
    validate_mutation_command,
)


class FakeClock:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self.started_at = datetime(2026, 7, 21, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.started_at + timedelta(seconds=self.elapsed)

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


class FakeActionAdapter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.results: dict[str, ActionResult] = {}
        self.errors: dict[str, BaseException] = {}

    def start(self, command: CommandSpec, context: ScenarioContext) -> object:
        del context
        self.events.append(f"action:start:{command.name}")
        return object()

    def run(self, command: CommandSpec, context: ScenarioContext) -> ActionResult:
        del context
        self.events.append(f"action:run:{command.name}")
        error = self.errors.get(command.name)
        if error is not None:
            raise error
        return self.results.get(command.name, ActionResult(ok=True))

    def finish(
        self, handle: Any, timeout_seconds: float, context: ScenarioContext
    ) -> ActionResult:
        del handle, timeout_seconds, context
        self.events.append("action:finish:workload")
        return self.results.get("workload-finish", ActionResult(ok=True))

    def stop(self, handle: Any, context: ScenarioContext) -> ActionResult:
        del handle, context
        self.events.append("action:stop:workload")
        return ActionResult(ok=True)


class FakeProbeAdapter:
    def __init__(
        self,
        events: list[str],
        results: Mapping[str, list[ProbeResult]] | None = None,
    ) -> None:
        self.events = events
        self.results = {
            name: deque(sequence) for name, sequence in (results or {}).items()
        }
        self.last: dict[str, ProbeResult] = {}

    def probe(self, spec: ProbeSpec, context: ScenarioContext) -> ProbeResult:
        del context
        self.events.append(f"probe:{spec.name}")
        queue = self.results.get(spec.name)
        if queue:
            result = queue.popleft()
            self.last[spec.name] = result
            return result
        if spec.name in self.last:
            return self.last[spec.name]
        return ProbeResult(status="pass")


def command(name: str, *argv: str, timeout: float = 10) -> CommandSpec:
    return CommandSpec(name=name, argv=argv or ("probe", name), timeout_seconds=timeout)


def probe(name: str, *, deadline: float = 2, mandatory: bool = True) -> ProbeSpec:
    return ProbeSpec(
        name=name,
        command=command(name),
        deadline_seconds=deadline,
        poll_interval_seconds=1,
        mandatory=mandatory,
    )


def scenario(**overrides: Any) -> Scenario:
    values: dict[str, Any] = {
        "name": "gateway-owner-kill-test",
        "fault": REQUIRED_FAULTS["gateway-owner-kill"],
        "enabled": True,
        "unavailable_reason": None,
        "seed": 17,
        "run_id": "ha-run-17",
        "queue": "0000-0000-0000-0000",
        "required_executors": ("extract_executor",),
        "load_profile": "one-active-attempt",
        "target": TargetSpec(kind="container", value="gateway-1"),
        "workload": command("workload", "run-workload", "ha-run-17", timeout=30),
        "preflight": (probe("healthy-preflight"),),
        "trigger": probe("dispatch-confirmed"),
        "injection": command("inject", "docker", "pause", "gateway-1"),
        "fault_probe": probe("gateway-paused"),
        "fault_hold_seconds": 0,
        "recovery": command("recover", "docker", "unpause", "gateway-1"),
        "recovery_probe": probe("gateway-recovered"),
        "settle_probes": (probe("settled"),),
        "verifiers": (probe("correctness"),),
        "required_verifier_checks": (
            "terminal_leases_released",
            "post_drain_capacity",
        ),
        "acceptable_outcomes": ("completed_once", "accepted_recovery_audit"),
    }
    values.update(overrides)
    return Scenario(**values)


def passing_correctness() -> ProbeResult:
    return ProbeResult(
        status="pass",
        observed={
            "checks": [
                {"name": "terminal_leases_released", "status": "pass"},
                {"name": "post_drain_capacity", "status": "pass"},
            ]
        },
        artifacts=({"name": "correctness", "path": "/tmp/correctness.json"},),
    )


def build_runner(
    selected: Scenario | None = None,
    *,
    results: Mapping[str, list[ProbeResult]] | None = None,
    allow_mutation: bool = True,
    confirmed_target: str | None = "gateway-1",
    dry_run: bool = False,
) -> tuple[SchedulerReliabilityRunner, FakeActionAdapter, FakeProbeAdapter, list[str]]:
    events: list[str] = []
    actions = FakeActionAdapter(events)
    probe_results = defaultdict(list)
    if results:
        probe_results.update(results)
    if "correctness" not in probe_results:
        probe_results["correctness"] = [passing_correctness()]
    probes = FakeProbeAdapter(events, probe_results)
    runner = SchedulerReliabilityRunner(
        selected or scenario(),
        action_adapter=actions,
        probe_adapter=probes,
        config_sha256="abc123",
        allow_mutation=allow_mutation,
        confirmed_target=confirmed_target,
        dry_run=dry_run,
        clock=FakeClock(),
    )
    return runner, actions, probes, events


def test_runner_orders_phases_and_consolidates_fault_artifacts() -> None:
    results = {
        "gateway-paused": [
            ProbeResult(
                status="pass",
                artifacts=({"name": "fault", "path": "/tmp/fault.json"},),
            )
        ]
    }
    runner, actions, _, events = build_runner(results=results)
    actions.results["inject"] = ActionResult(
        ok=True,
        effect_applied=True,
        artifacts=({"name": "inject", "path": "/tmp/inject.json"},),
    )

    report = runner.run()

    assert report["status"] == "passed"
    phase_names = [phase["name"] for phase in report["phases"]]
    assert phase_names == [
        "preflight",
        "workload_start",
        "trigger_wait",
        "injection",
        "fault_observation",
        "recovery",
        "recovery_wait",
        "settle",
        "workload_finish",
        "correctness_verification",
        "report_consolidation",
    ]
    assert events.index("action:run:inject") < events.index("action:run:recover")
    assert all(report["fault_timestamps"].values())
    assert {artifact["owner"] for artifact in report["artifacts"]} == {
        "inject",
        "gateway-paused",
        "correctness",
    }
    assert report["config_sha256"] == "abc123"


def test_trigger_timeout_fails_without_injecting() -> None:
    runner, _, _, events = build_runner(
        results={
            "dispatch-confirmed": [ProbeResult(status="fail", reason="no dispatch")]
        }
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert "did not pass within 2s" in report["reason"]
    assert "action:run:inject" not in events
    assert "action:run:recover" not in events
    assert events[-1] == "action:stop:workload"


def test_unobserved_injection_fails_and_still_recovers() -> None:
    runner, _, _, events = build_runner(
        results={
            "gateway-paused": [
                ProbeResult(status="fail", reason="target stayed running")
            ]
        }
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert report["fault_timestamps"]["injection_observed_at"] is None
    assert "action:run:recover" in events


def test_unobserved_recovery_cannot_pass() -> None:
    runner, _, _, _ = build_runner(
        results={
            "gateway-recovered": [
                ProbeResult(status="fail", reason="gateway remains unavailable")
            ]
        }
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert report["fault_timestamps"]["recovery_observed_at"] is None


@pytest.mark.parametrize(
    ("allow_mutation", "confirmed_target"),
    [(False, "gateway-1"), (True, "gateway-2")],
)
def test_exact_target_confirmation_and_opt_in_are_required(
    allow_mutation: bool, confirmed_target: str
) -> None:
    runner, _, _, events = build_runner(
        allow_mutation=allow_mutation,
        confirmed_target=confirmed_target,
    )

    report = runner.run()

    assert report["status"] == "skipped"
    assert events == []


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RuntimeError("injection broke"), "inject"),
        (KeyboardInterrupt(), "cancelled"),
    ],
)
def test_recovery_runs_after_injection_failure_or_cancellation(
    error: BaseException, reason: str
) -> None:
    runner, actions, _, events = build_runner()
    actions.errors["inject"] = error

    report = runner.run()

    assert report["status"] == "infrastructure_error"
    assert reason in report["reason"]
    assert "action:run:recover" in events
    assert report["fault_timestamps"]["recovery_completed_at"] is not None
    assert report["cancelled"] is isinstance(error, KeyboardInterrupt)


def test_failed_skipped_and_infrastructure_error_are_distinct() -> None:
    failed, _, _, _ = build_runner(
        results={"settled": [ProbeResult(status="fail", reason="open jobs")]}
    )
    skipped, _, _, _ = build_runner(allow_mutation=False)
    infrastructure, _, _, _ = build_runner(
        results={
            "healthy-preflight": [
                ProbeResult(status="error", reason="probe command missing")
            ]
        }
    )

    assert failed.run()["status"] == "failed"
    assert skipped.run()["status"] == "skipped"
    assert infrastructure.run()["status"] == "infrastructure_error"


def test_verifier_failure_overrides_successful_workload_and_recovery() -> None:
    runner, _, _, _ = build_runner(
        results={
            "correctness": [
                ProbeResult(status="fail", reason="duplicate accepted terminal")
            ]
        }
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert report["reason"] == "duplicate accepted terminal"


@pytest.mark.parametrize(
    "fault_name",
    ["partial-lease-jobs-by-id", "activation-failure-after-reservation"],
)
def test_cleanup_checks_are_mandatory_for_partial_lease_and_activation_faults(
    fault_name: str,
) -> None:
    selected = scenario(
        fault=REQUIRED_FAULTS[fault_name],
        required_verifier_checks=("terminal_leases_released", "post_drain_capacity"),
    )
    runner, _, _, _ = build_runner(
        selected,
        results={
            "correctness": [
                ProbeResult(
                    status="pass",
                    observed={
                        "checks": [
                            {"name": "terminal_leases_released", "status": "pass"},
                            {"name": "post_drain_capacity", "status": "fail"},
                        ]
                    },
                )
            ]
        },
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert "post_drain_capacity" in report["reason"]


def test_two_scheduler_shortfall_reports_per_missing_lease_budget() -> None:
    budget = QueryBudgetSpec(
        probe=probe("query-budget"),
        max_per_missing_lease={"get_job_by_id": 0.5, "postgresql_statements": 2},
    )
    selected = scenario(
        fault=REQUIRED_FAULTS["two-scheduler-lease-shortfall"],
        query_budget=budget,
    )
    runner, _, _, _ = build_runner(
        selected,
        results={
            "query-budget": [
                ProbeResult(
                    status="pass",
                    observed={
                        "missing_leases": 4,
                        "counters": {
                            "get_job_by_id": 2,
                            "postgresql_statements": 8,
                        },
                    },
                )
            ]
        },
    )

    report = runner.run()

    assert report["status"] == "passed"
    assert report["query_budget"] == {
        "missing_leases": 4,
        "passed": True,
        "counters": {
            "get_job_by_id": {
                "total": 2.0,
                "per_missing_lease": 0.5,
                "max_per_missing_lease": 0.5,
                "passed": True,
            },
            "postgresql_statements": {
                "total": 8.0,
                "per_missing_lease": 2.0,
                "max_per_missing_lease": 2,
                "passed": True,
            },
        },
    }


def test_linear_shortfall_query_amplification_fails() -> None:
    result = evaluate_query_budget(
        {
            "missing_leases": 10,
            "counters": {"get_job_by_id": 10, "postgresql_statements": 50},
        },
        {"get_job_by_id": 0.5, "postgresql_statements": 2},
    )

    assert result["passed"] is False
    assert result["counters"]["get_job_by_id"]["per_missing_lease"] == 1
    assert result["counters"]["postgresql_statements"]["per_missing_lease"] == 5


def test_dry_run_executes_no_commands() -> None:
    runner, _, _, events = build_runner(dry_run=True, allow_mutation=False)

    report = runner.run()

    assert report["status"] == "skipped"
    assert report["reason"] == "dry-run: no commands were executed"
    assert events == []


def test_dry_run_can_inspect_a_disabled_scenario() -> None:
    runner, _, _, events = build_runner(
        scenario(enabled=False, unavailable_reason="operator setup incomplete"),
        dry_run=True,
        allow_mutation=False,
    )

    report = runner.run()

    assert report["reason"] == "dry-run: no commands were executed"
    assert events == []


def test_active_active_diagnostics_remain_per_gateway() -> None:
    selected = scenario(
        active_active=True,
        gateway_diagnostics=(
            GatewayDiagnosticSpec("http://gateway-1:51000", probe("diagnostic-1")),
            GatewayDiagnosticSpec("http://gateway-2:51000", probe("diagnostic-2")),
        ),
    )
    runner, _, _, _ = build_runner(
        selected,
        results={
            "diagnostic-1": [
                ProbeResult(status="pass", query_counters={"dispatch": 10}),
                ProbeResult(status="pass", query_counters={"dispatch": 14}),
            ],
            "diagnostic-2": [
                ProbeResult(status="pass", query_counters={"dispatch": 20}),
                ProbeResult(status="pass", query_counters={"dispatch": 23}),
            ],
        },
    )

    report = runner.run()

    assert len(report["gateway_snapshots"]) == 4
    assert report["query_counter_deltas"] == {
        "http://gateway-1:51000": {"dispatch": 4.0},
        "http://gateway-2:51000": {"dispatch": 3.0},
    }


def test_mutations_reject_shells_broad_commands_and_implicit_targets() -> None:
    target = TargetSpec(kind="container", value="gateway-1")

    with pytest.raises(ConfigurationError, match="shell"):
        validate_mutation_command(
            command("shell", "bash", "-c", "docker pause gateway-1"), target
        )
    with pytest.raises(ConfigurationError, match="forbidden broad mutation"):
        validate_mutation_command(
            command("prune", "docker", "system", "prune", "gateway-1"), target
        )
    with pytest.raises(ConfigurationError, match="exact container target"):
        validate_mutation_command(
            command("wrong", "docker", "pause", "gateway-2"), target
        )


def test_required_fault_catalog_has_deadlines_outcomes_and_checks() -> None:
    assert len(REQUIRED_FAULTS) == 20
    for fault in REQUIRED_FAULTS.values():
        assert fault.trigger_deadline_seconds > 0
        assert fault.injection_deadline_seconds > 0
        assert fault.recovery_deadline_seconds > 0
        assert fault.settle_deadline_seconds > 0
        assert fault.acceptable_outcomes
        assert fault.required_checks


def test_example_config_loads_and_rejects_credentials(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[4]
    config = load_config(
        repository_root / "tools/stress/scheduler-reliability.config.example.json"
    )

    assert "gateway-owner-kill-example" in config.scenarios
    correctness_verifier = next(
        verifier
        for verifier in config.scenarios["gateway-owner-kill-example"].verifiers
        if verifier.name == "scheduler-correctness"
    )
    scope_index = correctness_verifier.command.argv.index("--scope")
    assert correctness_verifier.command.argv[scope_index + 1] == "gateway"
    assert len(config.config_sha256) == 64

    credential_config = tmp_path / "credential.json"
    credential_config.write_text(
        '{"version": 1, "password": "not-allowed", "scenarios": []}'
    )
    with pytest.raises(ConfigurationError, match="secured process environment"):
        load_config(credential_config)


def test_ha_sql_uses_session_parameters_and_safe_mutation_default() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    ha_root = repository_root / "config/psql/high-availability"
    scheduler_checks = (ha_root / "ha_scheduler_checks.sql").read_text()
    gateway_checks = (ha_root / "ha_inflight_gateway_kill_invariants.sql").read_text()
    lost_event = (ha_root / "ha_lost_terminal_event_reconciliation.sql").read_text()
    invariant_helper = (ha_root / "scheduler_attempt_invariant_checks.sql").read_text()

    assert "current_setting('marie.ha_run_start', TRUE)" in scheduler_checks
    assert "current_setting('marie.ha_run_end', TRUE)" in scheduler_checks
    assert "current_setting('marie.ha_run_start', TRUE)" in gateway_checks
    assert (
        "current_setting('marie.ha_killed_gateway_instance_id', TRUE)" in gateway_checks
    )
    assert "current_setting('marie.ha_target_job_id', TRUE)" in lost_event
    assert "current_setting('marie.ha_enable_mutation', TRUE)" in lost_event
    assert "FALSE\n    ) AS enable_mutation" in lost_event
    assert "AND p.enable_mutation IS FALSE" in lost_event
    assert (
        "CREATE OR REPLACE FUNCTION "
        "marie_scheduler.scheduler_attempt_invariant_checks" in invariant_helper
    )
    assert "{schema}" not in invariant_helper


def etcd_capacity(*slots: tuple[str, int, int]) -> dict[str, Any]:
    return {
        "status": "OK",
        "result": {
            "slots": [
                {"name": name, "capacity": configured, "available": available}
                for name, configured, available in slots
            ]
        },
    }


class EtcdFakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(seconds, 0.1)


class EtcdCapacitySequence:
    def __init__(self, clock: EtcdFakeClock, snapshots: list[dict[str, Any]]) -> None:
        self.clock = clock
        self.snapshots = snapshots
        self.index = 0

    def __call__(self) -> dict[str, Any]:
        snapshot = self.snapshots[min(self.index, len(self.snapshots) - 1)]
        self.index += 1
        self.clock.value += 1.0
        return snapshot


class EtcdFakeSimulator:
    def __init__(self) -> None:
        self.event_callback = None
        self.timeline: dict[str, Any] = {
            "container": "etcd-test",
            "status": "pending",
            "cleanup": {"status": "not_needed"},
        }

    def run(self) -> dict[str, Any]:
        self.timeline["status"] = "running"
        try:
            if self.event_callback is not None:
                self.event_callback("paused", {"cycle": 1})
                self.event_callback("recovered", {"cycle": 1})
        except Exception:
            self.timeline["status"] = "failed"
            raise
        self.timeline["status"] = "passed"
        return self.timeline


class EtcdFakeWorkload:
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.report_path = Path("/tmp/fake-gateway-report.json")
        self.job_jsonl_path = None
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def wait(self, _: float) -> dict[str, Any]:
        assert self.started
        return self.report

    def stop(self) -> None:
        self.stopped = True


def etcd_workload_report(
    *,
    open_jobs: int = 0,
    backlog_recovered: bool = True,
    dispatches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "run_identity": {"run_id": "etcd-run"},
        "summary": {
            "submitted_jobs": 5,
            "completed_jobs": 5 - open_jobs,
            "failed_jobs": 0,
            "open_jobs": open_jobs,
        },
        "reliability": {
            "accepted_during_outage": 5,
            "backlog_depth_at_zero": 5,
            "backlog_recovered": backlog_recovered,
            "dispatches": (
                dispatches
                if dispatches is not None
                else [
                    {
                        "confirmed_at": 101.5,
                        "consumes_executor_slot": False,
                    },
                    {
                        "confirmed_at": 102.5,
                        "consumes_executor_slot": True,
                    },
                ]
            ),
        },
    }


def build_etcd_runner(
    *,
    snapshots: list[dict[str, Any]],
    workload_report: dict[str, Any] | None = None,
    verifier_report: dict[str, Any] | None = None,
    scenario_name: str = "submission-during-outage",
) -> tuple[EtcdRecoveryRunner, EtcdFakeWorkload | None]:
    clock = EtcdFakeClock()
    workload = EtcdFakeWorkload(workload_report or etcd_workload_report())
    if scenario_name == "idle-reconnect":
        workload = None
    runner = EtcdRecoveryRunner(
        config=EtcdReliabilityConfig(
            scenario=scenario_name,
            run_id="etcd-run",
            required_executors=("extract",),
            preflight_timeout=2.0,
            zero_slot_timeout=2.0,
            recovery_timeout=2.0,
            poll_interval=0.5,
            workload_timeout=5.0,
        ),
        capacity_reader=EtcdCapacitySequence(clock, snapshots),
        simulator=EtcdFakeSimulator(),
        workload=workload,
        verifier=lambda: verifier_report or {"run_id": "etcd-run", "passed": True},
        clock=clock,
        sleep=clock.sleep,
    )
    return runner, workload


def test_etcd_capacity_parser_supports_gateway_shape() -> None:
    snapshot = parse_capacity_snapshot(
        etcd_capacity(("extract", 4, 3), ("classify", 2, 2)), 123.0
    )

    assert snapshot.slots["extract"].capacity == 4
    assert snapshot.slots["extract"].available == 3


def test_etcd_zero_slot_detection_and_recovery_pass() -> None:
    runner, workload = build_etcd_runner(
        snapshots=[
            etcd_capacity(("extract", 4, 4)),
            etcd_capacity(),
            etcd_capacity(("extract", 4, 2)),
        ]
    )

    report = runner.run()

    assert report["status"] == "passed"
    assert workload is not None and workload.started is True
    assert report["outages"][0]["zero_capacity_detected_at"] == 101.0
    assert report["outages"][0]["capacity_restored_at"] == 102.0
    assert report["dispatch"]["normal_during_suppression"] == 0
    assert report["dispatch"]["control_flow_during_suppression"] == 1


def test_etcd_runner_uses_streamed_job_records_for_dispatch_timing(
    tmp_path: Path,
) -> None:
    runner, workload = build_etcd_runner(
        snapshots=[
            etcd_capacity(("extract", 4, 4)),
            etcd_capacity(),
            etcd_capacity(("extract", 4, 2)),
        ],
        workload_report={
            "run_identity": {"run_id": "etcd-run"},
            "summary": {
                "submitted_jobs": 1,
                "completed_jobs": 1,
                "failed_jobs": 0,
                "retained_job_records": 0,
                "streamed_job_records": 1,
            },
            "reliability": {"observed": {"open_jobs": 0}},
            "jobs": [],
        },
    )
    assert workload is not None
    workload.job_jsonl_path = tmp_path / "gateway-jobs.jsonl"
    workload.job_jsonl_path.write_text(
        '{"stress_run_id":"etcd-run","request_id":"job-1",'
        '"started_at":102.5,"completed_at":103.0}\n'
    )

    report = runner.run()

    assert report["status"] == "passed"
    assert report["dispatch"]["source"] == "job_jsonl"
    assert report["dispatch"]["normal_after_recovery"] == 1


def test_etcd_zero_slot_timeout_stops_workload() -> None:
    runner, workload = build_etcd_runner(
        snapshots=[etcd_capacity(("extract", 4, 4))],
        scenario_name="active-execution-outage",
        workload_report={
            **etcd_workload_report(),
            "reliability": {
                **etcd_workload_report()["reliability"],
                "in_flight_at_outage": 2,
            },
        },
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert workload is not None and workload.stopped is True
    assert "did not reach zero" in report["failures"][0]["message"]


def test_etcd_registration_mismatch_fails_recovery() -> None:
    runner, _ = build_etcd_runner(
        snapshots=[
            etcd_capacity(("extract", 4, 4)),
            etcd_capacity(),
            etcd_capacity(("extract", 2, 2)),
        ]
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert "registration/capacity did not match" in report["failures"][0]["message"]


def test_etcd_backlog_failure_cannot_pass() -> None:
    runner, _ = build_etcd_runner(
        snapshots=[
            etcd_capacity(("extract", 4, 4)),
            etcd_capacity(),
            etcd_capacity(("extract", 4, 4)),
        ],
        workload_report=etcd_workload_report(open_jobs=2, backlog_recovered=False),
    )

    report = runner.run()
    codes = {failure["code"] for failure in report["failures"]}

    assert report["status"] == "failed"
    assert {"backlog-not-drained", "backlog-not-recovered"} <= codes


def test_etcd_dispatch_during_zero_capacity_cannot_pass() -> None:
    runner, _ = build_etcd_runner(
        snapshots=[
            etcd_capacity(("extract", 4, 4)),
            etcd_capacity(),
            etcd_capacity(("extract", 4, 4)),
        ],
        workload_report=etcd_workload_report(
            dispatches=[
                {
                    "confirmed_at": 101.5,
                    "consumes_executor_slot": True,
                }
            ]
        ),
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert report["dispatch"]["normal_during_suppression"] == 1


def test_etcd_final_verifier_failure_cannot_pass() -> None:
    runner, _ = build_etcd_runner(
        snapshots=[
            etcd_capacity(("extract", 4, 4)),
            etcd_capacity(),
            etcd_capacity(("extract", 4, 4)),
        ],
        verifier_report={"run_id": "etcd-run", "passed": False},
    )

    report = runner.run()

    assert report["status"] == "failed"
    assert any(
        failure["code"] == "correctness-failed" for failure in report["failures"]
    )
