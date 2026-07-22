#!/usr/bin/env python3
"""Run one explicitly selected scheduler fault scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

REPORT_STATUSES = {"passed", "failed", "skipped", "infrastructure_error"}
PROBE_STATUSES = {"pass", "fail", "unavailable", "error"}
SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "database_url",
    "password",
    "secret",
    "token",
)
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|authorization|password|secret|token)\s*[=:]\s*)([^\s,;]+)"
)
URI_CREDENTIAL_PATTERN = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.I)
FORBIDDEN_COMMAND_PATTERNS = (
    re.compile(r"\b(?:docker|podman)\s+(?:system\s+)?prune\b", re.I),
    re.compile(r"\b(?:killall|pkill)\b", re.I),
    re.compile(r"\brm\s+-[^\s]*r[^\s]*f\b", re.I),
    re.compile(r"\btruncate\s+(?:table\s+)?", re.I),
    re.compile(r"\bdrop\s+(?:database|schema)\b", re.I),
)


class ConfigurationError(ValueError):
    pass


class ScenarioFailure(RuntimeError):
    pass


class ScenarioUnavailable(RuntimeError):
    pass


class InfrastructureFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class FaultDefinition:
    name: str
    trigger_deadline_seconds: float
    injection_deadline_seconds: float
    recovery_deadline_seconds: float
    settle_deadline_seconds: float
    acceptable_outcomes: tuple[str, ...]
    required_checks: tuple[str, ...]


def _fault(
    name: str,
    *,
    trigger: float,
    injection: float,
    recovery: float,
    settle: float,
    outcomes: Sequence[str],
    checks: Sequence[str],
) -> FaultDefinition:
    return FaultDefinition(
        name=name,
        trigger_deadline_seconds=trigger,
        injection_deadline_seconds=injection,
        recovery_deadline_seconds=recovery,
        settle_deadline_seconds=settle,
        acceptable_outcomes=tuple(outcomes),
        required_checks=tuple(checks),
    )


TERMINAL_OUTCOMES = (
    "completed_once",
    "failed_durably_within_retry_policy",
    "accepted_recovery_audit",
    "intentionally_pending_with_documented_reason",
)
TERMINAL_CHECKS = (
    "stale_terminal_not_accepted",
    "duplicate_completed_terminal",
    "terminal_leases_released",
    "dispatched_terminal_or_recovery",
    "post_drain_capacity",
)


REQUIRED_FAULTS: dict[str, FaultDefinition] = {
    definition.name: definition
    for definition in (
        _fault(
            "gateway-owner-kill",
            trigger=120,
            injection=15,
            recovery=180,
            settle=300,
            outcomes=TERMINAL_OUTCOMES,
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "gateway-schedulers-restart",
            trigger=120,
            injection=60,
            recovery=180,
            settle=300,
            outcomes=TERMINAL_OUTCOMES,
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "scheduler-run-lease-expiry",
            trigger=120,
            injection=15,
            recovery=240,
            settle=360,
            outcomes=("accepted_recovery_audit", "failed_durably_within_retry_policy"),
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "lost-local-terminal-event",
            trigger=120,
            injection=15,
            recovery=180,
            settle=300,
            outcomes=("completed_once_via_storage_sync",),
            checks=(
                "duplicate_completed_terminal",
                "terminal_leases_released",
                "post_drain_capacity",
            ),
        ),
        _fault(
            "executor-kill-before-dispatch-confirmation",
            trigger=120,
            injection=15,
            recovery=180,
            settle=300,
            outcomes=("lease_and_capacity_released_without_confirmed_dispatch",),
            checks=("terminal_leases_released", "post_drain_capacity"),
        ),
        _fault(
            "executor-kill-after-dispatch-confirmation",
            trigger=120,
            injection=15,
            recovery=240,
            settle=360,
            outcomes=TERMINAL_OUTCOMES,
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "malformed-executor-event",
            trigger=120,
            injection=15,
            recovery=60,
            settle=180,
            outcomes=("malformed_event_rejected_without_state_change",),
            checks=("stale_terminal_not_accepted", "duplicate_completed_terminal"),
        ),
        _fault(
            "duplicate-executor-event",
            trigger=120,
            injection=15,
            recovery=60,
            settle=180,
            outcomes=("duplicate_terminal_rejected_or_idempotently_ignored",),
            checks=("duplicate_completed_terminal",),
        ),
        _fault(
            "delayed-executor-event",
            trigger=120,
            injection=15,
            recovery=180,
            settle=300,
            outcomes=("current_event_accepted_or_superseded_event_rejected",),
            checks=("stale_terminal_not_accepted", "duplicate_completed_terminal"),
        ),
        _fault(
            "stale-executor-event",
            trigger=120,
            injection=15,
            recovery=60,
            settle=180,
            outcomes=("stale_event_rejected_without_current_attempt_mutation",),
            checks=("stale_terminal_not_accepted", "duplicate_completed_terminal"),
        ),
        _fault(
            "partial-lease-jobs-by-id",
            trigger=120,
            injection=15,
            recovery=120,
            settle=240,
            outcomes=("missing_leases_released_without_capacity_leak",),
            checks=("terminal_leases_released", "post_drain_capacity"),
        ),
        _fault(
            "activation-failure-after-reservation",
            trigger=120,
            injection=15,
            recovery=120,
            settle=240,
            outcomes=("lease_semaphore_and_capacity_released",),
            checks=("terminal_leases_released", "post_drain_capacity"),
        ),
        _fault(
            "two-scheduler-lease-shortfall",
            trigger=180,
            injection=15,
            recovery=120,
            settle=300,
            outcomes=("shortfall_released_without_linear_point_lookup_amplification",),
            checks=("terminal_leases_released", "post_drain_capacity"),
        ),
        _fault(
            "postgres-submission-outage",
            trigger=120,
            injection=30,
            recovery=180,
            settle=300,
            outcomes=("submission_rejected_or_accepted_item_durably_recovered",),
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "postgres-dispatch-outage",
            trigger=120,
            injection=30,
            recovery=180,
            settle=300,
            outcomes=TERMINAL_OUTCOMES,
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "postgres-pool-exhaustion",
            trigger=120,
            injection=30,
            recovery=180,
            settle=300,
            outcomes=("bounded_failure_or_recovery_without_lost_accepted_work",),
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "postgres-latency",
            trigger=120,
            injection=30,
            recovery=180,
            settle=300,
            outcomes=("deadline_failure_or_recovery_without_duplicate_terminal",),
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "postgres-statement-timeout",
            trigger=120,
            injection=30,
            recovery=180,
            settle=300,
            outcomes=("statement_timeout_recovered_without_lease_or_capacity_leak",),
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "rabbitmq-pause",
            trigger=120,
            injection=30,
            recovery=180,
            settle=300,
            outcomes=("durable_terminal_reconciled_after_broker_recovery",),
            checks=TERMINAL_CHECKS,
        ),
        _fault(
            "rabbitmq-consumer-disconnect",
            trigger=120,
            injection=30,
            recovery=180,
            settle=300,
            outcomes=("consumer_reconnected_and_durable_terminal_reconciled",),
            checks=TERMINAL_CHECKS,
        ),
    )
}


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    timeout_seconds: float
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    command: CommandSpec
    deadline_seconds: float
    poll_interval_seconds: float = 1.0
    mandatory: bool = True


@dataclass(frozen=True)
class TargetSpec:
    kind: str
    value: str


@dataclass(frozen=True)
class GatewayDiagnosticSpec:
    endpoint: str
    probe: ProbeSpec


@dataclass(frozen=True)
class QueryBudgetSpec:
    probe: ProbeSpec
    max_per_missing_lease: Mapping[str, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    fault: FaultDefinition
    enabled: bool
    unavailable_reason: str | None
    seed: int
    run_id: str
    queue: str
    required_executors: tuple[str, ...]
    load_profile: str
    target: TargetSpec
    workload: CommandSpec
    preflight: tuple[ProbeSpec, ...]
    trigger: ProbeSpec
    injection: CommandSpec
    fault_probe: ProbeSpec
    fault_hold_seconds: float
    recovery: CommandSpec
    recovery_probe: ProbeSpec
    settle_probes: tuple[ProbeSpec, ...]
    verifiers: tuple[ProbeSpec, ...]
    required_verifier_checks: tuple[str, ...]
    acceptable_outcomes: tuple[str, ...]
    gateway_diagnostics: tuple[GatewayDiagnosticSpec, ...] = ()
    active_active: bool = False
    query_budget: QueryBudgetSpec | None = None


@dataclass(frozen=True)
class RunnerConfig:
    scenarios: Mapping[str, Scenario]
    config_sha256: str


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    effect_applied: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ProbeResult:
    status: str
    observed: Any = None
    reason: str | None = None
    artifacts: tuple[Mapping[str, Any], ...] = ()
    query_counters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in PROBE_STATUSES:
            raise ValueError(f"Unsupported probe status: {self.status}")


@dataclass(frozen=True)
class ScenarioContext:
    scenario_name: str
    run_id: str
    queue: str
    seed: int
    target_kind: str
    target: str


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class ActionAdapter(Protocol):
    def start(self, command: CommandSpec, context: ScenarioContext) -> Any: ...

    def run(self, command: CommandSpec, context: ScenarioContext) -> ActionResult: ...

    def finish(
        self, handle: Any, timeout_seconds: float, context: ScenarioContext
    ) -> ActionResult: ...

    def stop(self, handle: Any, context: ScenarioContext) -> ActionResult: ...


class ProbeAdapter(Protocol):
    def probe(self, spec: ProbeSpec, context: ScenarioContext) -> ProbeResult: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in SECRET_KEY_PARTS):
                sanitized[str(key)] = "<redacted>"
            else:
                sanitized[str(key)] = sanitize(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        redacted = URI_CREDENTIAL_PATTERN.sub(r"\g<scheme><redacted>@", value)
        return SECRET_VALUE_PATTERN.sub(r"\1<redacted>", redacted)
    return value


def _reject_credentials(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in SECRET_KEY_PARTS):
                raise ConfigurationError(
                    f"{path}.{key} is not allowed; use the secured process environment"
                )
            _reject_credentials(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_credentials(item, f"{path}[{index}]")
        return
    if isinstance(value, str):
        if URI_CREDENTIAL_PATTERN.search(value) or SECRET_VALUE_PATTERN.search(value):
            raise ConfigurationError(
                f"{path} contains credentials; use the secured process environment"
            )


def _positive(value: Any, path: str) -> float:
    number = float(value)
    if number <= 0:
        raise ConfigurationError(f"{path} must be greater than 0")
    return number


def _nonnegative(value: Any, path: str) -> float:
    number = float(value)
    if number < 0:
        raise ConfigurationError(f"{path} must be at least 0")
    return number


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{path} must be an array")
    return value


def _command(payload: Any, path: str, *, allow_empty: bool = False) -> CommandSpec:
    data = _mapping(payload, path)
    argv = tuple(str(item) for item in _sequence(data.get("argv", []), f"{path}.argv"))
    if not argv and not allow_empty:
        raise ConfigurationError(f"{path}.argv must not be empty")
    return CommandSpec(
        name=str(data.get("name") or path.rsplit(".", 1)[-1]),
        argv=argv,
        timeout_seconds=_positive(
            data.get("timeout_seconds", 30), f"{path}.timeout_seconds"
        ),
        artifacts=tuple(
            str(item)
            for item in _sequence(data.get("artifacts", []), f"{path}.artifacts")
        ),
    )


def _probe(payload: Any, path: str, *, allow_empty: bool = False) -> ProbeSpec:
    data = _mapping(payload, path)
    command = _command(
        data.get("command", {}), f"{path}.command", allow_empty=allow_empty
    )
    return ProbeSpec(
        name=str(data.get("name") or path.rsplit(".", 1)[-1]),
        command=command,
        deadline_seconds=_positive(
            data.get("deadline_seconds", command.timeout_seconds),
            f"{path}.deadline_seconds",
        ),
        poll_interval_seconds=_positive(
            data.get("poll_interval_seconds", 1), f"{path}.poll_interval_seconds"
        ),
        mandatory=bool(data.get("mandatory", True)),
    )


def _fault_definition(name: str, payload: Any) -> FaultDefinition:
    data = _mapping(payload, f"faults.{name}")
    deadlines = _mapping(data.get("deadlines", {}), f"faults.{name}.deadlines")
    outcomes = tuple(
        str(item)
        for item in _sequence(
            data.get("acceptable_outcomes", []),
            f"faults.{name}.acceptable_outcomes",
        )
    )
    checks = tuple(
        str(item)
        for item in _sequence(
            data.get("required_checks", []), f"faults.{name}.required_checks"
        )
    )
    if not outcomes:
        raise ConfigurationError(f"faults.{name}.acceptable_outcomes must not be empty")
    if not checks:
        raise ConfigurationError(f"faults.{name}.required_checks must not be empty")
    return _fault(
        name,
        trigger=_positive(
            deadlines.get("trigger_seconds"), f"faults.{name}.deadlines.trigger_seconds"
        ),
        injection=_positive(
            deadlines.get("injection_seconds"),
            f"faults.{name}.deadlines.injection_seconds",
        ),
        recovery=_positive(
            deadlines.get("recovery_seconds"),
            f"faults.{name}.deadlines.recovery_seconds",
        ),
        settle=_positive(
            deadlines.get("settle_seconds"), f"faults.{name}.deadlines.settle_seconds"
        ),
        outcomes=outcomes,
        checks=checks,
    )


def _scenario(
    payload: Any,
    index: int,
    faults: Mapping[str, FaultDefinition],
) -> Scenario:
    path = f"scenarios[{index}]"
    data = _mapping(payload, path)
    name = str(data.get("name", "")).strip()
    fault_name = str(data.get("fault", "")).strip()
    if not name or fault_name not in faults:
        raise ConfigurationError(f"{path} must name a configured fault")
    enabled = bool(data.get("enabled", True))
    allow_empty = not enabled
    fault = faults[fault_name]
    target_data = _mapping(data.get("target", {}), f"{path}.target")
    target = TargetSpec(
        kind=str(target_data.get("kind", "")).strip(),
        value=str(target_data.get("value", "")).strip(),
    )
    if not target.kind or not target.value:
        raise ConfigurationError(f"{path}.target requires kind and value")
    load_data = _mapping(data.get("load_profile", {}), f"{path}.load_profile")
    preflight = tuple(
        _probe(item, f"{path}.preflight[{probe_index}]", allow_empty=allow_empty)
        for probe_index, item in enumerate(
            _sequence(data.get("preflight", []), f"{path}.preflight")
        )
    )
    settle = tuple(
        _probe(item, f"{path}.settle[{probe_index}]", allow_empty=allow_empty)
        for probe_index, item in enumerate(
            _sequence(data.get("settle", []), f"{path}.settle")
        )
    )
    verifiers = tuple(
        _probe(item, f"{path}.verifiers[{probe_index}]", allow_empty=allow_empty)
        for probe_index, item in enumerate(
            _sequence(data.get("verifiers", []), f"{path}.verifiers")
        )
    )
    diagnostics = []
    for diagnostic_index, item in enumerate(
        _sequence(data.get("gateway_diagnostics", []), f"{path}.gateway_diagnostics")
    ):
        diagnostic_data = _mapping(
            item, f"{path}.gateway_diagnostics[{diagnostic_index}]"
        )
        diagnostics.append(
            GatewayDiagnosticSpec(
                endpoint=str(diagnostic_data.get("endpoint", "")).strip(),
                probe=_probe(
                    diagnostic_data.get("probe", {}),
                    f"{path}.gateway_diagnostics[{diagnostic_index}].probe",
                    allow_empty=allow_empty,
                ),
            )
        )
    if any(not diagnostic.endpoint for diagnostic in diagnostics):
        raise ConfigurationError(f"{path}.gateway_diagnostics requires exact endpoints")
    query_budget = None
    if data.get("query_budget") is not None:
        budget_data = _mapping(data["query_budget"], f"{path}.query_budget")
        limits_data = _mapping(
            budget_data.get("max_per_missing_lease", {}),
            f"{path}.query_budget.max_per_missing_lease",
        )
        if not limits_data:
            raise ConfigurationError(
                f"{path}.query_budget.max_per_missing_lease must not be empty"
            )
        query_budget = QueryBudgetSpec(
            probe=_probe(
                budget_data.get("probe", {}),
                f"{path}.query_budget.probe",
                allow_empty=allow_empty,
            ),
            max_per_missing_lease={
                str(key): _nonnegative(
                    value, f"{path}.query_budget.max_per_missing_lease.{key}"
                )
                for key, value in limits_data.items()
            },
        )
    required_checks = tuple(
        str(item)
        for item in _sequence(
            data.get("required_verifier_checks", fault.required_checks),
            f"{path}.required_verifier_checks",
        )
    )
    outcomes = tuple(
        str(item)
        for item in _sequence(
            data.get("acceptable_outcomes", fault.acceptable_outcomes),
            f"{path}.acceptable_outcomes",
        )
    )
    if not outcomes or not required_checks:
        raise ConfigurationError(f"{path} requires outcomes and verifier checks")
    if set(outcomes) != set(fault.acceptable_outcomes):
        raise ConfigurationError(
            f"{path}.acceptable_outcomes must match the selected fault contract"
        )
    missing_checks = set(fault.required_checks) - set(required_checks)
    if missing_checks:
        raise ConfigurationError(
            f"{path}.required_verifier_checks omits: "
            + ", ".join(sorted(missing_checks))
        )
    scenario = Scenario(
        name=name,
        fault=fault,
        enabled=enabled,
        unavailable_reason=(
            str(data.get("unavailable_reason"))
            if data.get("unavailable_reason") is not None
            else None
        ),
        seed=int(data.get("seed", 0)),
        run_id=str(data.get("run_id", "")).strip(),
        queue=str(data.get("queue", "")).strip(),
        required_executors=tuple(
            str(item)
            for item in _sequence(
                data.get("required_executors", []), f"{path}.required_executors"
            )
        ),
        load_profile=str(load_data.get("name", "")).strip(),
        target=target,
        workload=_command(
            load_data.get("command", {}),
            f"{path}.load_profile.command",
            allow_empty=allow_empty,
        ),
        preflight=preflight,
        trigger=_probe(
            data.get("trigger", {}), f"{path}.trigger", allow_empty=allow_empty
        ),
        injection=_command(
            data.get("injection", {}), f"{path}.injection", allow_empty=allow_empty
        ),
        fault_probe=_probe(
            data.get("fault_probe", {}), f"{path}.fault_probe", allow_empty=allow_empty
        ),
        fault_hold_seconds=_nonnegative(
            data.get("fault_hold_seconds", 0), f"{path}.fault_hold_seconds"
        ),
        recovery=_command(
            data.get("recovery", {}), f"{path}.recovery", allow_empty=allow_empty
        ),
        recovery_probe=_probe(
            data.get("recovery_probe", {}),
            f"{path}.recovery_probe",
            allow_empty=allow_empty,
        ),
        settle_probes=settle,
        verifiers=verifiers,
        required_verifier_checks=required_checks,
        acceptable_outcomes=outcomes,
        gateway_diagnostics=tuple(diagnostics),
        active_active=bool(data.get("active_active", False)),
        query_budget=query_budget,
    )
    if enabled:
        if not scenario.run_id or not scenario.queue or not scenario.load_profile:
            raise ConfigurationError(
                f"{path} requires run_id, queue, and load_profile.name"
            )
        if (
            not scenario.preflight
            or not scenario.settle_probes
            or not scenario.verifiers
        ):
            raise ConfigurationError(
                f"{path} requires preflight, settle, and verifiers"
            )
        if scenario.active_active and len(scenario.gateway_diagnostics) < 2:
            raise ConfigurationError(
                f"{path} active-active scenarios require at least two gateway diagnostics"
            )
        if (
            fault.name
            in {
                "gateway-owner-kill",
                "gateway-schedulers-restart",
                "two-scheduler-lease-shortfall",
            }
            and not scenario.active_active
        ):
            raise ConfigurationError(f"{path} must collect active-active diagnostics")
        if (
            fault.name == "two-scheduler-lease-shortfall"
            and scenario.query_budget is None
        ):
            raise ConfigurationError(
                f"{path} requires a two-scheduler query budget probe"
            )
        if scenario.trigger.deadline_seconds > fault.trigger_deadline_seconds:
            raise ConfigurationError(
                f"{path}.trigger exceeds the fault trigger deadline"
            )
        if scenario.fault_probe.deadline_seconds > fault.injection_deadline_seconds:
            raise ConfigurationError(
                f"{path}.fault_probe exceeds the injection deadline"
            )
        if scenario.recovery_probe.deadline_seconds > fault.recovery_deadline_seconds:
            raise ConfigurationError(
                f"{path}.recovery_probe exceeds the recovery deadline"
            )
        if any(
            probe.deadline_seconds > fault.settle_deadline_seconds
            for probe in scenario.settle_probes
        ):
            raise ConfigurationError(f"{path}.settle exceeds the settle deadline")
        if scenario.injection.timeout_seconds > fault.injection_deadline_seconds:
            raise ConfigurationError(
                f"{path}.injection timeout exceeds the injection deadline"
            )
        if scenario.recovery.timeout_seconds > fault.recovery_deadline_seconds:
            raise ConfigurationError(
                f"{path}.recovery timeout exceeds the recovery deadline"
            )
    return scenario


def load_config(path: str | Path) -> RunnerConfig:
    config_path = Path(path).expanduser()
    payload = json.loads(config_path.read_text())
    data = _mapping(payload, "config")
    _reject_credentials(data)
    if int(data.get("version", 0)) != 1:
        raise ConfigurationError("config.version must be 1")
    fault_payload = data.get("faults")
    if fault_payload is None:
        faults = REQUIRED_FAULTS
    else:
        raw_faults = _mapping(fault_payload, "faults")
        faults = {
            str(name): _fault_definition(str(name), definition)
            for name, definition in raw_faults.items()
        }
    scenarios = {}
    for index, item in enumerate(_sequence(data.get("scenarios", []), "scenarios")):
        scenario = _scenario(item, index, faults)
        if scenario.name in scenarios:
            raise ConfigurationError(f"Duplicate scenario name: {scenario.name}")
        scenarios[scenario.name] = scenario
    if not scenarios:
        raise ConfigurationError("config.scenarios must not be empty")
    canonical = json.dumps(sanitize(data), sort_keys=True, separators=(",", ":"))
    return RunnerConfig(
        scenarios=scenarios,
        config_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def resolve_command(command: CommandSpec, context: ScenarioContext) -> CommandSpec:
    values = {
        "scenario": context.scenario_name,
        "run_id": context.run_id,
        "queue": context.queue,
        "seed": str(context.seed),
        "target_kind": context.target_kind,
        "target": context.target,
    }
    try:
        argv = tuple(argument.format_map(values) for argument in command.argv)
        artifacts = tuple(path.format_map(values) for path in command.artifacts)
    except KeyError as error:
        raise ConfigurationError(
            f"Unsupported command placeholder in {command.name}: {error.args[0]}"
        ) from error
    return CommandSpec(command.name, argv, command.timeout_seconds, artifacts)


def validate_mutation_command(command: CommandSpec, target: TargetSpec) -> None:
    if not command.argv:
        raise ConfigurationError(f"{command.name} has no command")
    executable = Path(command.argv[0]).name.lower()
    if executable in {"bash", "dash", "sh", "zsh", "fish"}:
        raise ConfigurationError(f"{command.name} cannot execute a shell")
    rendered = " ".join(command.argv)
    for pattern in FORBIDDEN_COMMAND_PATTERNS:
        if pattern.search(rendered):
            raise ConfigurationError(
                f"{command.name} contains a forbidden broad mutation"
            )
    if target.value not in command.argv:
        raise ConfigurationError(
            f"{command.name} must include the exact {target.kind} target as one argv item"
        )


def _payload_from_stdout(stdout: str) -> Mapping[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _artifacts(
    command: CommandSpec, payload: Mapping[str, Any] | None
) -> tuple[Mapping[str, Any], ...]:
    collected: list[Mapping[str, Any]] = [
        {"name": Path(path).name, "path": path} for path in command.artifacts
    ]
    if payload is not None:
        raw_artifacts = payload.get("artifacts", [])
        if isinstance(raw_artifacts, Sequence) and not isinstance(
            raw_artifacts, (str, bytes)
        ):
            for item in raw_artifacts:
                if isinstance(item, Mapping):
                    collected.append(dict(item))
                elif isinstance(item, str):
                    collected.append({"name": Path(item).name, "path": item})
    return tuple(collected)


class CommandActionAdapter:
    def start(
        self, command: CommandSpec, context: ScenarioContext
    ) -> subprocess.Popen[Any]:
        del context
        return subprocess.Popen(
            command.argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def run(self, command: CommandSpec, context: ScenarioContext) -> ActionResult:
        del context
        try:
            completed = subprocess.run(
                command.argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=command.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise InfrastructureFailure(f"{command.name}: {error}") from error
        payload = _payload_from_stdout(completed.stdout)
        effect_applied = bool(payload.get("effect_applied", True)) if payload else True
        details: dict[str, Any] = {"return_code": completed.returncode}
        if payload and "observed" in payload:
            details["observed"] = payload["observed"]
        if completed.returncode != 0:
            details["error"] = (completed.stderr or completed.stdout).strip()[:1000]
        return ActionResult(
            ok=completed.returncode == 0,
            effect_applied=effect_applied,
            details=details,
            artifacts=_artifacts(command, payload),
        )

    def finish(
        self,
        handle: subprocess.Popen[Any],
        timeout_seconds: float,
        context: ScenarioContext,
    ) -> ActionResult:
        del context
        try:
            return_code = handle.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise InfrastructureFailure(
                "workload did not finish before its deadline"
            ) from error
        return ActionResult(ok=return_code == 0, details={"return_code": return_code})

    def stop(
        self, handle: subprocess.Popen[Any], context: ScenarioContext
    ) -> ActionResult:
        del context
        if handle.poll() is not None:
            return ActionResult(
                ok=True, effect_applied=False, details={"already_exited": True}
            )
        handle.terminate()
        try:
            handle.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.kill()
            handle.wait(timeout=10)
        return ActionResult(
            ok=True, details={"terminated_exact_workload_process": True}
        )


class CommandProbeAdapter:
    def probe(self, spec: ProbeSpec, context: ScenarioContext) -> ProbeResult:
        command = resolve_command(spec.command, context)
        try:
            completed = subprocess.run(
                command.argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=command.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return ProbeResult(status="error", reason=str(error))
        payload = _payload_from_stdout(completed.stdout)
        if payload is None:
            status = "pass" if completed.returncode == 0 else "fail"
            return ProbeResult(
                status=status,
                reason=(
                    None
                    if status == "pass"
                    else (completed.stderr or completed.stdout).strip()[:1000]
                ),
                artifacts=_artifacts(command, payload),
            )
        raw_status = payload.get("status")
        if raw_status in {"skipped", "unavailable"}:
            status = "unavailable"
        elif raw_status in PROBE_STATUSES:
            status = str(raw_status)
        elif "passed" in payload:
            status = "pass" if bool(payload["passed"]) else "fail"
        elif "ready" in payload:
            status = "pass" if bool(payload["ready"]) else "fail"
        else:
            status = "pass" if completed.returncode == 0 else "error"
        counters = payload.get("query_counters", {})
        if not isinstance(counters, Mapping):
            counters = {}
        return ProbeResult(
            status=status,
            observed=payload.get("observed", payload),
            reason=str(payload.get("reason")) if payload.get("reason") else None,
            artifacts=_artifacts(command, payload),
            query_counters={str(key): float(value) for key, value in counters.items()},
        )


def evaluate_query_budget(
    observed: Mapping[str, Any], limits: Mapping[str, float]
) -> dict[str, Any]:
    missing_leases = int(observed.get("missing_leases", 0))
    raw_counters = observed.get("counters", observed.get("query_counters", {}))
    counters = _mapping(raw_counters, "query_budget.observed.counters")
    result: dict[str, Any] = {
        "missing_leases": missing_leases,
        "passed": missing_leases > 0,
        "counters": {},
    }
    for name, maximum in limits.items():
        total = float(counters.get(name, 0))
        per_missing = total / missing_leases if missing_leases else None
        passed = per_missing is not None and per_missing <= maximum
        result["counters"][name] = {
            "total": total,
            "per_missing_lease": per_missing,
            "max_per_missing_lease": maximum,
            "passed": passed,
        }
        result["passed"] = result["passed"] and passed
    return result


class SchedulerReliabilityRunner:
    def __init__(
        self,
        scenario: Scenario,
        *,
        action_adapter: ActionAdapter,
        probe_adapter: ProbeAdapter,
        config_sha256: str,
        allow_mutation: bool,
        confirmed_target: str | None,
        dry_run: bool,
        clock: Clock | None = None,
        event_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.scenario = scenario
        self.action_adapter = action_adapter
        self.probe_adapter = probe_adapter
        self.config_sha256 = config_sha256
        self.allow_mutation = allow_mutation
        self.confirmed_target = confirmed_target
        self.dry_run = dry_run
        self.clock = clock or SystemClock()
        self.event_callback = event_callback
        self.context = ScenarioContext(
            scenario_name=scenario.name,
            run_id=scenario.run_id,
            queue=scenario.queue,
            seed=scenario.seed,
            target_kind=scenario.target.kind,
            target=scenario.target.value,
        )
        self.report = self._base_report()
        self._workload_handle: Any = None
        self._workload_finished = False
        self._injection_attempted = False
        self._injection_succeeded = False

    def _base_report(self) -> dict[str, Any]:
        fault = self.scenario.fault
        return {
            "schema_version": 1,
            "scenario": self.scenario.name,
            "fault": fault.name,
            "seed": self.scenario.seed,
            "run_id": self.scenario.run_id,
            "queue": self.scenario.queue,
            "required_executors": list(self.scenario.required_executors),
            "load_profile": self.scenario.load_profile,
            "target": asdict(self.scenario.target),
            "config_sha256": self.config_sha256,
            "dry_run": self.dry_run,
            "cancelled": False,
            "status": "infrastructure_error",
            "reason": None,
            "started_at": _utc_text(self.clock.now()),
            "completed_at": None,
            "deadlines": {
                "trigger_seconds": fault.trigger_deadline_seconds,
                "injection_seconds": fault.injection_deadline_seconds,
                "recovery_seconds": fault.recovery_deadline_seconds,
                "settle_seconds": fault.settle_deadline_seconds,
            },
            "acceptable_outcomes": list(self.scenario.acceptable_outcomes),
            "required_verifier_checks": list(self.scenario.required_verifier_checks),
            "phases": [],
            "fault_timestamps": {
                "injection_started_at": None,
                "injection_completed_at": None,
                "injection_observed_at": None,
                "recovery_started_at": None,
                "recovery_completed_at": None,
                "recovery_observed_at": None,
            },
            "artifacts": [],
            "gateway_snapshots": [],
            "query_counter_deltas": {},
            "query_budget": None,
            "errors": [],
        }

    def _timestamp(self) -> str:
        return _utc_text(self.clock.now())

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event, sanitize(payload))

    def _phase_start(self, name: str) -> dict[str, Any]:
        phase = {
            "name": name,
            "started_at": self._timestamp(),
            "completed_at": None,
            "status": "running",
        }
        self.report["phases"].append(phase)
        self._emit("phase_started", {"phase": name, "at": phase["started_at"]})
        return phase

    def _phase_finish(
        self,
        phase: dict[str, Any],
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        phase["completed_at"] = self._timestamp()
        phase["status"] = status
        if details:
            phase["details"] = sanitize(details)
        self._emit("phase_completed", phase)

    def _record_artifacts(
        self, phase: str, owner: str, artifacts: Sequence[Mapping[str, Any]]
    ) -> None:
        for artifact in artifacts:
            self.report["artifacts"].append(
                {
                    "phase": phase,
                    "owner": owner,
                    "captured_at": self._timestamp(),
                    "artifact": sanitize(dict(artifact)),
                }
            )

    def _run_action(self, phase_name: str, command: CommandSpec) -> ActionResult:
        phase = self._phase_start(phase_name)
        resolved = resolve_command(command, self.context)
        try:
            result = self.action_adapter.run(resolved, self.context)
        except KeyboardInterrupt:
            self._phase_finish(phase, "cancelled")
            raise
        except Exception as error:
            self._phase_finish(phase, "error", {"error": str(error)})
            raise InfrastructureFailure(f"{command.name}: {error}") from error
        self._record_artifacts(phase_name, command.name, result.artifacts)
        status = "passed" if result.ok else "error"
        self._phase_finish(phase, status, result.details)
        if not result.ok:
            raise InfrastructureFailure(f"{command.name} returned a nonzero result")
        return result

    def _probe_once(self, spec: ProbeSpec, phase_name: str) -> ProbeResult:
        try:
            result = self.probe_adapter.probe(spec, self.context)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            raise InfrastructureFailure(f"{spec.name}: {error}") from error
        self._record_artifacts(phase_name, spec.name, result.artifacts)
        return result

    def _wait_for_probe(self, spec: ProbeSpec, phase_name: str) -> ProbeResult:
        deadline = self.clock.monotonic() + spec.deadline_seconds
        attempts = 0
        last_result: ProbeResult | None = None
        while True:
            attempts += 1
            last_result = self._probe_once(spec, phase_name)
            if last_result.status == "pass":
                return last_result
            if last_result.status == "unavailable":
                raise ScenarioUnavailable(
                    last_result.reason or f"{spec.name} is unavailable"
                )
            if last_result.status == "error":
                raise InfrastructureFailure(
                    last_result.reason or f"{spec.name} probe failed"
                )
            remaining = deadline - self.clock.monotonic()
            if remaining <= 0:
                raise ScenarioFailure(
                    f"{spec.name} did not pass within {spec.deadline_seconds:g}s "
                    f"after {attempts} attempts: {last_result.reason or 'not ready'}"
                )
            self.clock.sleep(min(spec.poll_interval_seconds, remaining))

    def _run_probe_phase(self, name: str, probes: Sequence[ProbeSpec]) -> None:
        phase = self._phase_start(name)
        try:
            for probe in probes:
                self._wait_for_probe(probe, name)
        except (ScenarioFailure, ScenarioUnavailable, InfrastructureFailure) as error:
            self._phase_finish(phase, "failed", {"error": str(error)})
            raise
        self._phase_finish(
            phase, "passed", {"probes": [probe.name for probe in probes]}
        )

    def _run_diagnostics(self, stage: str) -> None:
        if not self.scenario.gateway_diagnostics:
            return
        phase_name = f"{stage}_gateway_diagnostics"
        phase = self._phase_start(phase_name)
        try:
            for diagnostic in self.scenario.gateway_diagnostics:
                result = self._probe_once(diagnostic.probe, phase_name)
                if result.status == "unavailable":
                    raise ScenarioUnavailable(
                        result.reason or f"gateway {diagnostic.endpoint} unavailable"
                    )
                if result.status == "error":
                    raise InfrastructureFailure(
                        result.reason
                        or f"gateway {diagnostic.endpoint} diagnostics failed"
                    )
                if result.status != "pass":
                    raise ScenarioFailure(
                        result.reason
                        or f"gateway {diagnostic.endpoint} diagnostics failed"
                    )
                self.report["gateway_snapshots"].append(
                    {
                        "stage": stage,
                        "endpoint": diagnostic.endpoint,
                        "captured_at": self._timestamp(),
                        "observed": sanitize(result.observed),
                        "query_counters": sanitize(result.query_counters),
                    }
                )
        except (ScenarioFailure, ScenarioUnavailable, InfrastructureFailure) as error:
            self._phase_finish(phase, "failed", {"error": str(error)})
            raise
        self._phase_finish(
            phase, "passed", {"gateway_count": len(self.scenario.gateway_diagnostics)}
        )

    def _start_workload(self) -> None:
        phase = self._phase_start("workload_start")
        command = resolve_command(self.scenario.workload, self.context)
        try:
            self._workload_handle = self.action_adapter.start(command, self.context)
        except Exception as error:
            self._phase_finish(phase, "error", {"error": str(error)})
            raise InfrastructureFailure(f"workload start failed: {error}") from error
        self._record_artifacts(
            "workload_start",
            command.name,
            tuple(
                {"name": Path(path).name, "path": path} for path in command.artifacts
            ),
        )
        self._phase_finish(phase, "passed")

    def _finish_workload(self) -> None:
        phase = self._phase_start("workload_finish")
        try:
            result = self.action_adapter.finish(
                self._workload_handle,
                self.scenario.workload.timeout_seconds,
                self.context,
            )
        except Exception as error:
            self._phase_finish(phase, "error", {"error": str(error)})
            raise InfrastructureFailure(f"workload finish failed: {error}") from error
        self._workload_finished = True
        self._record_artifacts(
            "workload_finish", self.scenario.workload.name, result.artifacts
        )
        self._phase_finish(phase, "passed" if result.ok else "failed", result.details)
        if not result.ok:
            raise ScenarioFailure("workload process failed")

    def _stop_workload(self) -> None:
        if self._workload_handle is None or self._workload_finished:
            return
        phase = self._phase_start("workload_cleanup")
        try:
            result = self.action_adapter.stop(self._workload_handle, self.context)
        except Exception as error:
            self._phase_finish(phase, "error", {"error": str(error)})
            self.report["errors"].append(f"workload cleanup failed: {error}")
            return
        self._phase_finish(phase, "passed" if result.ok else "error", result.details)
        if not result.ok:
            self.report["errors"].append("workload cleanup failed")

    def _run_verifiers(self) -> list[ProbeResult]:
        phase = self._phase_start("correctness_verification")
        results = []
        try:
            for verifier in self.scenario.verifiers:
                result = self._probe_once(verifier, "correctness_verification")
                results.append(result)
                if not verifier.mandatory:
                    continue
                if result.status == "fail":
                    raise ScenarioFailure(result.reason or f"{verifier.name} failed")
                if result.status in {"unavailable", "error"}:
                    raise InfrastructureFailure(
                        result.reason or f"{verifier.name} is unavailable"
                    )
            self._require_named_checks(results)
        except (ScenarioFailure, InfrastructureFailure) as error:
            self._phase_finish(phase, "failed", {"error": str(error)})
            raise
        self._phase_finish(
            phase,
            "passed",
            {"verifiers": [probe.name for probe in self.scenario.verifiers]},
        )
        return results

    def _require_named_checks(self, results: Sequence[ProbeResult]) -> None:
        statuses: dict[str, str] = {}
        for result in results:
            if not isinstance(result.observed, Mapping):
                continue
            checks = result.observed.get("checks", [])
            if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
                continue
            for check in checks:
                if isinstance(check, Mapping) and check.get("name"):
                    statuses[str(check["name"])] = str(check.get("status", ""))
        missing = [
            name
            for name in self.scenario.required_verifier_checks
            if name not in statuses
        ]
        failed = [
            name
            for name in self.scenario.required_verifier_checks
            if statuses.get(name) not in {None, "pass"}
        ]
        if missing or failed:
            parts = []
            if missing:
                parts.append("missing=" + ",".join(missing))
            if failed:
                parts.append("failed=" + ",".join(failed))
            raise ScenarioFailure(
                "mandatory verifier checks did not pass: " + " ".join(parts)
            )

    def _run_query_budget(self) -> None:
        if self.scenario.query_budget is None:
            return
        phase = self._phase_start("query_budget_verification")
        result = self._probe_once(
            self.scenario.query_budget.probe, "query_budget_verification"
        )
        if result.status in {"unavailable", "error"}:
            error = InfrastructureFailure(
                result.reason or "query budget probe unavailable"
            )
            self._phase_finish(phase, "failed", {"error": str(error)})
            raise error
        if result.status == "fail" or not isinstance(result.observed, Mapping):
            error = ScenarioFailure(result.reason or "query budget probe failed")
            self._phase_finish(phase, "failed", {"error": str(error)})
            raise error
        budget = evaluate_query_budget(
            result.observed, self.scenario.query_budget.max_per_missing_lease
        )
        self.report["query_budget"] = sanitize(budget)
        self._phase_finish(phase, "passed" if budget["passed"] else "failed", budget)
        if not budget["passed"]:
            raise ScenarioFailure(
                "per-missing-lease query amplification exceeded the configured budget"
            )

    def _calculate_query_counter_deltas(self) -> None:
        by_stage: dict[str, dict[str, Mapping[str, float]]] = {}
        for snapshot in self.report["gateway_snapshots"]:
            by_stage.setdefault(snapshot["stage"], {})[snapshot["endpoint"]] = snapshot[
                "query_counters"
            ]
        before = by_stage.get("pre_fault", {})
        after = by_stage.get("post_recovery", {})
        deltas = {}
        for endpoint in sorted(set(before) | set(after)):
            endpoint_before = before.get(endpoint, {})
            endpoint_after = after.get(endpoint, {})
            deltas[endpoint] = {
                key: float(endpoint_after.get(key, 0))
                - float(endpoint_before.get(key, 0))
                for key in sorted(set(endpoint_before) | set(endpoint_after))
            }
        self.report["query_counter_deltas"] = deltas

    def _authorize(self) -> None:
        if not self.allow_mutation:
            raise ScenarioUnavailable("mutation opt-in was not supplied")
        if self.confirmed_target != self.scenario.target.value:
            raise ScenarioUnavailable(
                "confirmed target does not exactly match the selected scenario target"
            )
        injection = resolve_command(self.scenario.injection, self.context)
        recovery = resolve_command(self.scenario.recovery, self.context)
        validate_mutation_command(injection, self.scenario.target)
        validate_mutation_command(recovery, self.scenario.target)

    def _require_deadline(self, started: float, limit: float, label: str) -> None:
        elapsed = self.clock.monotonic() - started
        if elapsed > limit:
            raise ScenarioFailure(
                f"{label} exceeded its {limit:g}s deadline: {elapsed:g}s"
            )

    def run(self) -> dict[str, Any]:
        status = "infrastructure_error"
        reason: str | None = None
        try:
            if self.dry_run:
                status = "skipped"
                reason = "dry-run: no commands were executed"
                return self._finalize(status, reason)
            if not self.scenario.enabled:
                raise ScenarioUnavailable(
                    self.scenario.unavailable_reason or "scenario is disabled"
                )
            self._authorize()
            self._run_probe_phase("preflight", self.scenario.preflight)
            self._run_diagnostics("pre_fault")
            self._start_workload()
            self._run_probe_phase("trigger_wait", (self.scenario.trigger,))

            injection_error: BaseException | None = None
            self._injection_attempted = True
            injection_started = self.clock.monotonic()
            recovery_started = injection_started
            self.report["fault_timestamps"]["injection_started_at"] = self._timestamp()
            try:
                result = self._run_action("injection", self.scenario.injection)
                self.report["fault_timestamps"][
                    "injection_completed_at"
                ] = self._timestamp()
                if not result.effect_applied:
                    raise ScenarioFailure(
                        "injection command reported that no effect was applied"
                    )
                self._run_probe_phase("fault_observation", (self.scenario.fault_probe,))
                self._injection_succeeded = True
                self.report["fault_timestamps"][
                    "injection_observed_at"
                ] = self._timestamp()
                self._require_deadline(
                    injection_started,
                    self.scenario.fault.injection_deadline_seconds,
                    "fault observation",
                )
                if self.scenario.fault_hold_seconds:
                    phase = self._phase_start("fault_hold")
                    self.clock.sleep(self.scenario.fault_hold_seconds)
                    self._phase_finish(
                        phase,
                        "passed",
                        {"seconds": self.scenario.fault_hold_seconds},
                    )
            except BaseException as error:
                injection_error = error
            finally:
                recovery_started = self.clock.monotonic()
                self.report["fault_timestamps"][
                    "recovery_started_at"
                ] = self._timestamp()
                try:
                    self._run_action("recovery", self.scenario.recovery)
                    self.report["fault_timestamps"][
                        "recovery_completed_at"
                    ] = self._timestamp()
                except BaseException as recovery_error:
                    if injection_error is not None:
                        self.report["errors"].append(
                            f"injection failed: {type(injection_error).__name__}: {injection_error}"
                        )
                    raise recovery_error
            if injection_error is not None:
                raise injection_error

            self._run_probe_phase("recovery_wait", (self.scenario.recovery_probe,))
            self.report["fault_timestamps"]["recovery_observed_at"] = self._timestamp()
            self._require_deadline(
                recovery_started,
                self.scenario.fault.recovery_deadline_seconds,
                "recovery observation",
            )
            settle_started = self.clock.monotonic()
            self._run_probe_phase("settle", self.scenario.settle_probes)
            self._require_deadline(
                settle_started,
                self.scenario.fault.settle_deadline_seconds,
                "settle",
            )
            self._run_diagnostics("post_recovery")
            self._finish_workload()
            self._run_verifiers()
            self._run_query_budget()
            self._calculate_query_counter_deltas()
            if not self._injection_succeeded:
                raise ScenarioFailure("fault injection was never observed")
            if self.report["fault_timestamps"]["recovery_observed_at"] is None:
                raise ScenarioFailure("recovery was never observed")
            status = "passed"
        except ScenarioUnavailable as error:
            status = (
                "skipped" if not self._injection_attempted else "infrastructure_error"
            )
            reason = str(error)
        except ScenarioFailure as error:
            status = "failed"
            reason = str(error)
        except KeyboardInterrupt:
            status = "infrastructure_error"
            reason = "scenario cancelled by operator"
            self.report["cancelled"] = True
        except (ConfigurationError, InfrastructureFailure) as error:
            status = "infrastructure_error"
            reason = str(error)
        except Exception as error:
            status = "infrastructure_error"
            reason = f"{type(error).__name__}: {error}"
        except BaseException as error:
            status = "infrastructure_error"
            reason = f"scenario cancelled: {type(error).__name__}: {error}"
            self.report["cancelled"] = True
        finally:
            self._stop_workload()
        return self._finalize(status, reason)

    def _finalize(self, status: str, reason: str | None) -> dict[str, Any]:
        if status not in REPORT_STATUSES:
            raise ValueError(f"Unsupported report status: {status}")
        phase = self._phase_start("report_consolidation")
        self.report["status"] = status
        self.report["reason"] = reason
        self.report["completed_at"] = self._timestamp()
        if reason:
            self.report["errors"].append(reason)
        self._phase_finish(
            phase,
            "passed",
            {"artifact_count": len(self.report["artifacts"]), "status": status},
        )
        return sanitize(self.report)


def write_report(report: Mapping[str, Any], output_path: str | Path | None) -> None:
    rendered = json.dumps(sanitize(report), indent=2, sort_keys=True)
    print(rendered)
    if output_path is None:
        return
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered + "\n")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one exact-target scheduler reliability fault scenario"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--list", action="store_true", dest="list_scenarios")
    parser.add_argument("--allow-mutation", action="store_true")
    parser.add_argument("--confirm-target")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args[:1] == ["etcd"]:
        from tools.stress.etcd_recovery_orchestration import main as etcd_main

        return etcd_main(raw_args[1:])

    args = parse_args(raw_args)
    try:
        config = load_config(args.config)
        if args.list_scenarios:
            for name, scenario in sorted(config.scenarios.items()):
                availability = "enabled" if scenario.enabled else "disabled"
                print(
                    f"{name}\t{scenario.fault.name}\t{availability}\t"
                    f"{scenario.target.kind}={scenario.target.value}"
                )
            return 0
        if not args.scenario:
            raise ConfigurationError("--scenario is required unless --list is used")
        try:
            scenario = config.scenarios[args.scenario]
        except KeyError as error:
            raise ConfigurationError(f"Unknown scenario: {args.scenario}") from error
        print(
            f"Resolved mutation target: scenario={scenario.name} "
            f"{scenario.target.kind}={scenario.target.value}",
            file=sys.stderr,
        )
        runner = SchedulerReliabilityRunner(
            scenario,
            action_adapter=CommandActionAdapter(),
            probe_adapter=CommandProbeAdapter(),
            config_sha256=config.config_sha256,
            allow_mutation=args.allow_mutation,
            confirmed_target=args.confirm_target,
            dry_run=args.dry_run,
        )
        report = runner.run()
        write_report(report, args.report)
        if report["cancelled"]:
            return 130
        if report["status"] == "passed":
            return 0
        if report["status"] in {"failed", "skipped"}:
            return 1
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"scheduler_reliability_runner: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
