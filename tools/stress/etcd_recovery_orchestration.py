#!/usr/bin/env python3
"""ETCD-specific orchestration used by the scheduler reliability runner."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tools.stress.etcd_outage_simulator import (
    EtcdOutageSimulator,
    _sanitize_report_value,
)

logger = logging.getLogger("EtcdRecoveryOrchestration")

ETCD_SCENARIOS = (
    "idle-reconnect",
    "submission-during-outage",
    "active-execution-outage",
    "repeated-flapping",
    "ttl-crossing-outage",
    "gateway-restart-during-outage",
)


class EtcdRecoveryGateError(RuntimeError):
    """Raised when an ETCD recovery invariant cannot be proven."""


@dataclass(frozen=True)
class CapacitySlot:
    name: str
    capacity: int
    available: int


@dataclass(frozen=True)
class CapacitySnapshot:
    captured_at: float
    slots: dict[str, CapacitySlot]


@dataclass(frozen=True)
class EtcdReliabilityConfig:
    scenario: str
    run_id: str
    required_executors: tuple[str, ...]
    preflight_timeout: float = 30.0
    zero_slot_timeout: float = 30.0
    recovery_timeout: float = 60.0
    poll_interval: float = 1.0
    workload_timeout: float = 900.0
    workload_warmup_seconds: float = 0.0
    capacity_sample_limit: int = 500
    dry_run: bool = False


def _unwrap_result(payload: Any) -> Any:
    if isinstance(payload, dict) and payload.get("status") in {"OK", "error"}:
        return payload.get("result")
    return payload


def parse_capacity_snapshot(payload: Any, captured_at: float) -> CapacitySnapshot:
    normalized = _unwrap_result(payload)
    if not isinstance(normalized, dict):
        raise ValueError("Capacity response must contain an object result")
    rows = normalized.get("slots")
    if not isinstance(rows, list):
        raise ValueError("Capacity response is missing result.slots")

    slots = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        name = str(row["name"])
        slots[name] = CapacitySlot(
            name=name,
            capacity=int(row.get("capacity", 0)),
            available=int(row.get("available", 0)),
        )
    return CapacitySnapshot(captured_at=captured_at, slots=slots)


def _parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()


def _extract_run_id(payload: dict[str, Any]) -> str | None:
    summary = payload.get("summary")
    identity = payload.get("run_identity")
    candidates = (
        payload.get("run_id"),
        payload.get("stress_run_id"),
        summary.get("run_id") if isinstance(summary, dict) else None,
        summary.get("stress_run_id") if isinstance(summary, dict) else None,
        identity.get("run_id") if isinstance(identity, dict) else None,
        identity.get("stress_run_id") if isinstance(identity, dict) else None,
    )
    return next((str(value) for value in candidates if value), None)


def _summary_value(payload: dict[str, Any], *names: str) -> Any:
    summary = payload.get("summary")
    sources = (payload, summary if isinstance(summary, dict) else {})
    for source in sources:
        for name in names:
            if name in source:
                return source[name]
    return None


def _reliability_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("reliability", "recovery"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _job_records(
    payload: dict[str, Any], jsonl_path: Path | None
) -> Iterator[dict[str, Any]]:
    if jsonl_path is not None:
        try:
            with jsonl_path.open() as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise EtcdRecoveryGateError(
                            f"Gateway job JSONL line {line_number} is not an object"
                        )
                    yield record
        except (OSError, json.JSONDecodeError) as exc:
            raise EtcdRecoveryGateError(
                f"Cannot read gateway job JSONL {jsonl_path}: {exc}"
            ) from exc
        return

    jobs = payload.get("jobs")
    if isinstance(jobs, list):
        yield from (job for job in jobs if isinstance(job, dict))


def _dispatches(
    payload: dict[str, Any], jsonl_path: Path | None
) -> Iterator[dict[str, Any]]:
    evidence = _reliability_evidence(payload)
    raw_dispatches = evidence.get("dispatches", payload.get("dispatches"))
    if isinstance(raw_dispatches, list):
        yield from (item for item in raw_dispatches if isinstance(item, dict))
        return

    for job in _job_records(payload, jsonl_path):
        confirmed_at = job.get("dispatch_confirmed_at")
        if confirmed_at is None:
            confirmed_at = job.get("started_at", job.get("scheduled_at"))
        if confirmed_at is None:
            continue
        yield {
            "confirmed_at": confirmed_at,
            "request_id": job.get("request_id"),
            "executor": job.get("executor", job.get("required_executor")),
            "consumes_executor_slot": job.get("consumes_executor_slot", True),
        }


class HttpCapacityReader:
    def __init__(
        self, *, capacity_url: str, api_key: str | None, timeout: float
    ) -> None:
        self.capacity_url = capacity_url
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.capacity_url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EtcdRecoveryGateError(
                f"Capacity endpoint unavailable: {exc}"
            ) from exc


class GatewayWorkloadCommand:
    def __init__(
        self,
        command: Sequence[str],
        report_path: str | Path,
        job_jsonl_path: str | Path | None = None,
    ) -> None:
        self.command = list(command)
        self.report_path = Path(report_path)
        self.job_jsonl_path = (
            Path(job_jsonl_path) if job_jsonl_path is not None else None
        )
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process is not None:
            return
        if self.report_path.exists():
            raise EtcdRecoveryGateError(
                f"Refusing stale gateway report path: {self.report_path}"
            )
        if self.job_jsonl_path is not None and self.job_jsonl_path.exists():
            raise EtcdRecoveryGateError(
                f"Refusing stale gateway job JSONL path: {self.job_jsonl_path}"
            )
        self.process = subprocess.Popen(self.command, text=True)

    def wait(self, timeout: float) -> dict[str, Any]:
        if self.process is None:
            raise EtcdRecoveryGateError("Gateway workload was not started")
        try:
            return_code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise EtcdRecoveryGateError(
                f"Gateway workload exceeded {timeout:.1f}s"
            ) from exc
        if not self.report_path.exists():
            raise EtcdRecoveryGateError(
                f"Gateway workload did not write {self.report_path}"
            )
        payload = json.loads(self.report_path.read_text())
        if not isinstance(payload, dict):
            raise EtcdRecoveryGateError("Gateway workload report must be an object")
        if return_code != 0:
            raise EtcdRecoveryGateError(
                f"Gateway workload exited with status {return_code}"
            )
        if self.job_jsonl_path is not None and not self.job_jsonl_path.exists():
            raise EtcdRecoveryGateError(
                f"Gateway workload did not write {self.job_jsonl_path}"
            )
        return payload

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10.0)


class CorrectnessVerifierCommand:
    def __init__(self, command: Sequence[str], report_path: str | Path) -> None:
        self.command = list(command)
        self.report_path = Path(report_path)

    def __call__(self) -> dict[str, Any]:
        if self.report_path.exists():
            raise EtcdRecoveryGateError(
                f"Refusing stale correctness report path: {self.report_path}"
            )
        result = subprocess.run(self.command, check=False, text=True)
        if not self.report_path.exists():
            raise EtcdRecoveryGateError(
                f"Correctness verifier did not write {self.report_path}"
            )
        payload = json.loads(self.report_path.read_text())
        if not isinstance(payload, dict):
            raise EtcdRecoveryGateError("Correctness report must be an object")
        if result.returncode not in {0, 1}:
            raise EtcdRecoveryGateError(
                f"Correctness verifier exited with status {result.returncode}"
            )
        return payload


class EtcdRecoveryRunner:
    def __init__(
        self,
        *,
        config: EtcdReliabilityConfig,
        capacity_reader: Callable[[], dict[str, Any]],
        simulator: Any,
        verifier: Callable[[], dict[str, Any]],
        workload: Any | None = None,
        restart_gateway: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.capacity_reader = capacity_reader
        self.simulator = simulator
        self.verifier = verifier
        self.workload = workload
        self.restart_gateway = restart_gateway
        self.clock = clock
        self.sleep = sleep
        self._baseline: CapacitySnapshot | None = None
        self._workload_started = False
        self._workload_collected = False
        self._gateway_restarted = False
        self._outages: dict[int, dict[str, Any]] = {}
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "scenario": config.scenario,
            "run_id": config.run_id,
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "required_executors": list(config.required_executors),
            "capacity": {"baseline": None, "samples": [], "dropped_samples": 0},
            "outages": [],
            "simulator": None,
            "workload": None,
            "dispatch": None,
            "correctness": None,
            "failures": [],
        }

    def _fail(self, code: str, message: str) -> None:
        self.report["failures"].append({"code": code, "message": message})

    def _append_capacity(self, sample: dict[str, Any]) -> None:
        samples = self.report["capacity"]["samples"]
        if len(samples) < self.config.capacity_sample_limit:
            samples.append(sample)
        else:
            self.report["capacity"]["dropped_samples"] += 1

    def _read_capacity(self, phase: str) -> CapacitySnapshot | None:
        captured_at = self.clock()
        try:
            snapshot = parse_capacity_snapshot(self.capacity_reader(), captured_at)
        except (EtcdRecoveryGateError, ValueError) as exc:
            self._append_capacity(
                {
                    "phase": phase,
                    "captured_at": captured_at,
                    "error": str(exc),
                    "slots": {},
                }
            )
            return None
        self._append_capacity(
            {
                "phase": phase,
                "captured_at": captured_at,
                "error": None,
                "slots": {name: asdict(slot) for name, slot in snapshot.slots.items()},
            }
        )
        return snapshot

    def _wait_capacity(
        self,
        *,
        phase: str,
        timeout: float,
        predicate: Callable[[CapacitySnapshot], bool],
        failure_message: str,
    ) -> CapacitySnapshot:
        deadline = self.clock() + timeout
        while True:
            snapshot = self._read_capacity(phase)
            if snapshot is not None and predicate(snapshot):
                return snapshot
            if self.clock() >= deadline:
                raise EtcdRecoveryGateError(failure_message)
            self.sleep(self.config.poll_interval)

    def _healthy(self, snapshot: CapacitySnapshot) -> bool:
        for executor in self.config.required_executors:
            slot = snapshot.slots.get(executor)
            if slot is None or slot.capacity <= 0 or slot.available <= 0:
                return False
        return True

    def _outage_visible(self, snapshot: CapacitySnapshot) -> bool:
        return any(
            executor not in snapshot.slots or snapshot.slots[executor].capacity <= 0
            for executor in self.config.required_executors
        )

    def _capacity_restored(self, snapshot: CapacitySnapshot) -> bool:
        if self._baseline is None:
            return False
        for executor in self.config.required_executors:
            restored = snapshot.slots.get(executor)
            baseline = self._baseline.slots[executor]
            if restored is None or restored.capacity != baseline.capacity:
                return False
        return True

    def _start_workload(self) -> None:
        if self.workload is None or self._workload_started:
            return
        self.workload.start()
        self._workload_started = True

    def _on_outage_event(self, event: str, details: dict[str, Any]) -> None:
        cycle = int(details["cycle"])
        outage = self._outages.setdefault(
            cycle,
            {
                "cycle": cycle,
                "paused_at": None,
                "zero_capacity_detected_at": None,
                "capacity_restored_at": None,
                "zero_detection_seconds": None,
                "recovery_seconds": None,
            },
        )
        event_at = self.clock()
        if event == "paused":
            outage["paused_at"] = event_at
            zero = self._wait_capacity(
                phase=f"cycle-{cycle}-outage",
                timeout=self.config.zero_slot_timeout,
                predicate=self._outage_visible,
                failure_message=(
                    f"Cycle {cycle}: required executor capacity did not reach zero "
                    f"or disappear within {self.config.zero_slot_timeout:.1f}s"
                ),
            )
            outage["zero_capacity_detected_at"] = zero.captured_at
            outage["zero_detection_seconds"] = zero.captured_at - event_at
            if self.config.scenario == "submission-during-outage":
                self._start_workload()
            if (
                self.config.scenario == "gateway-restart-during-outage"
                and not self._gateway_restarted
            ):
                if self.restart_gateway is None:
                    raise EtcdRecoveryGateError(
                        "Gateway restart scenario requires a supervisor command"
                    )
                self.restart_gateway()
                self._gateway_restarted = True
            return

        if event == "recovered":
            restored = self._wait_capacity(
                phase=f"cycle-{cycle}-recovery",
                timeout=self.config.recovery_timeout,
                predicate=self._capacity_restored,
                failure_message=(
                    f"Cycle {cycle}: required executor registration/capacity did not "
                    f"match the pre-outage snapshot within {self.config.recovery_timeout:.1f}s"
                ),
            )
            outage["capacity_restored_at"] = restored.captured_at
            outage["recovery_seconds"] = restored.captured_at - event_at

    def _active_at_outage(
        self, payload: dict[str, Any], jsonl_path: Path | None
    ) -> int:
        first_zero = min(
            (
                outage["zero_capacity_detected_at"]
                for outage in self._outages.values()
                if outage.get("zero_capacity_detected_at") is not None
            ),
            default=None,
        )
        if first_zero is None:
            return 0
        active = 0
        for job in _job_records(payload, jsonl_path):
            started_at = _parse_timestamp(job.get("started_at"))
            terminal_at = _parse_timestamp(
                job.get("completed_at") or job.get("failed_at")
            )
            if (
                started_at is not None
                and started_at <= first_zero
                and (terminal_at is None or terminal_at >= first_zero)
            ):
                active += 1
        return active

    def _evaluate_workload(self, payload: dict[str, Any]) -> None:
        report_run_id = _extract_run_id(payload)
        if report_run_id != self.config.run_id:
            self._fail(
                "workload-run-id-mismatch",
                f"Expected workload run_id {self.config.run_id!r}, got {report_run_id!r}",
            )

        accepted = _summary_value(payload, "accepted_jobs", "submitted_jobs")
        terminal = _summary_value(payload, "terminal_jobs")
        if terminal is None:
            terminal = int(_summary_value(payload, "completed_jobs") or 0) + int(
                _summary_value(payload, "failed_jobs") or 0
            )
        open_jobs = _summary_value(payload, "open_jobs")
        if open_jobs is None and accepted is not None:
            open_jobs = max(0, int(accepted) - int(terminal))
        if accepted is None or int(accepted) <= 0:
            self._fail("no-accepted-work", "Gateway report contains no accepted work")

        evidence = _reliability_evidence(payload)
        jsonl_path = getattr(self.workload, "job_jsonl_path", None)
        violations = 0
        control_flow = 0
        after_recovery = 0
        dispatch_count = 0
        for dispatch in _dispatches(payload, jsonl_path):
            confirmed_at = _parse_timestamp(dispatch.get("confirmed_at"))
            if confirmed_at is None:
                continue
            dispatch_count += 1
            consumes_slot = dispatch.get("consumes_executor_slot", True) is not False
            suppressed = False
            for outage in self._outages.values():
                zero_at = outage.get("zero_capacity_detected_at")
                restored_at = outage.get("capacity_restored_at")
                if zero_at is None or restored_at is None:
                    continue
                if zero_at <= confirmed_at < restored_at:
                    violations += int(consumes_slot)
                    control_flow += int(not consumes_slot)
                    suppressed = True
                    break
            if (
                not suppressed
                and consumes_slot
                and any(
                    outage.get("capacity_restored_at") is not None
                    and confirmed_at >= outage["capacity_restored_at"]
                    for outage in self._outages.values()
                )
            ):
                after_recovery += 1

        if violations:
            self._fail(
                "dispatch-during-zero-capacity",
                f"{violations} normal executor dispatches occurred while capacity was absent",
            )
        if open_jobs not in {None, 0}:
            self._fail(
                "backlog-not-drained", f"Gateway report contains {open_jobs} open jobs"
            )

        backlog_recovered = evidence.get("backlog_recovered")
        if backlog_recovered is False:
            self._fail(
                "backlog-not-recovered",
                "Gateway report states that backlog recovery failed",
            )
        backlog_depth = evidence.get(
            "backlog_depth_at_zero", evidence.get("backlog_depth_at_outage")
        )
        if backlog_depth is None and self.config.scenario != "idle-reconnect":
            backlog_depth = accepted
        if (
            backlog_depth is not None
            and int(backlog_depth) > 0
            and backlog_recovered is not True
            and after_recovery == 0
        ):
            self._fail(
                "backlog-resume-not-observed",
                "No executor dispatch resumed after capacity restoration",
            )

        if self.config.scenario == "submission-during-outage":
            accepted_during = evidence.get("accepted_during_outage", accepted)
            if accepted_during is None or int(accepted_during) <= 0:
                self._fail(
                    "outage-submission-not-observed",
                    "No accepted work was observed for the outage submission phase",
                )
        if self.config.scenario == "active-execution-outage":
            active = evidence.get("in_flight_at_outage")
            if active is None:
                active = self._active_at_outage(payload, jsonl_path)
            if int(active) <= 0:
                self._fail(
                    "active-attempt-not-observed",
                    "No in-flight work was proven at outage start",
                )

        self.report["dispatch"] = {
            "observed": dispatch_count,
            "source": "job_jsonl" if jsonl_path is not None else "gateway_report",
            "normal_during_suppression": violations,
            "control_flow_during_suppression": control_flow,
            "normal_after_recovery": after_recovery,
            "accepted_jobs": accepted,
            "terminal_jobs": terminal,
            "open_jobs": open_jobs,
            "backlog_recovered": backlog_recovered,
        }
        self.report["workload"] = {
            "run_id": report_run_id,
            "summary": _sanitize_report_value(payload.get("summary", {})),
            "report_path": str(getattr(self.workload, "report_path", "")) or None,
            "job_jsonl_path": str(jsonl_path) if jsonl_path is not None else None,
        }

    def _evaluate_correctness(self, payload: dict[str, Any]) -> None:
        report_run_id = _extract_run_id(payload)
        if report_run_id != self.config.run_id:
            self._fail(
                "correctness-run-id-mismatch",
                f"Expected verifier run_id {self.config.run_id!r}, got {report_run_id!r}",
            )
        summary = payload.get("summary")
        passed = payload.get("passed")
        if passed is None and isinstance(summary, dict):
            passed = summary.get("passed")
        if passed is not True:
            self._fail(
                "correctness-failed",
                "Final PostgreSQL correctness verification did not pass",
            )

    def run(self) -> dict[str, Any]:
        self.report["status"] = "running"
        self.report["started_at"] = self.clock()
        if self.config.dry_run:
            self.simulator.event_callback = None
            try:
                self.simulator.run()
                self.report["simulator"] = self.simulator.timeline
                self.report["status"] = "dry-run"
            except Exception as exc:
                self._fail("dry-run-failed", f"{type(exc).__name__}: {exc}")
                self.report["status"] = "failed"
            self.report["completed_at"] = self.clock()
            return self.report

        preflight_succeeded = False
        try:
            self._baseline = self._wait_capacity(
                phase="pre-outage",
                timeout=self.config.preflight_timeout,
                predicate=self._healthy,
                failure_message=(
                    "Required executors did not have positive configured and available "
                    "capacity before the outage"
                ),
            )
            self.report["capacity"]["baseline"] = {
                name: asdict(self._baseline.slots[name])
                for name in self.config.required_executors
            }
            preflight_succeeded = True
            if self.config.scenario not in {
                "idle-reconnect",
                "submission-during-outage",
            }:
                self._start_workload()
                self.sleep(self.config.workload_warmup_seconds)

            self.simulator.event_callback = self._on_outage_event
            self.simulator.run()
            self.report["simulator"] = self.simulator.timeline

            if self.workload is not None:
                if not self._workload_started:
                    raise EtcdRecoveryGateError("Gateway workload was never started")
                workload_report = self.workload.wait(self.config.workload_timeout)
                self._workload_collected = True
                self._evaluate_workload(workload_report)
        except KeyboardInterrupt as exc:
            self._fail("interrupted", f"{type(exc).__name__}: {exc}")
            self.report["simulator"] = getattr(self.simulator, "timeline", None)
        except Exception as exc:
            self._fail("orchestration-failed", f"{type(exc).__name__}: {exc}")
            self.report["simulator"] = getattr(self.simulator, "timeline", None)
        finally:
            if self._workload_started and not self._workload_collected:
                try:
                    self.workload.stop()
                except Exception as exc:
                    self._fail(
                        "workload-cleanup-failed", f"{type(exc).__name__}: {exc}"
                    )

        if preflight_succeeded:
            try:
                correctness = self.verifier()
                self.report["correctness"] = correctness
                self._evaluate_correctness(correctness)
            except Exception as exc:
                self._fail(
                    "correctness-invocation-failed", f"{type(exc).__name__}: {exc}"
                )

        self.report["outages"] = [
            self._outages[cycle] for cycle in sorted(self._outages)
        ]
        self.report["status"] = "passed" if not self.report["failures"] else "failed"
        self.report["completed_at"] = self.clock()
        return self.report


def write_etcd_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_sanitize_report_value(report), indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _parse_command(raw: str, option: str) -> list[str]:
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option} must be a JSON array of command arguments") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ValueError(f"{option} must be a non-empty JSON string array")
    return command


def _run_supervisor_command(command: Sequence[str]) -> None:
    result = subprocess.run(command, check=False, text=True)
    if result.returncode != 0:
        raise EtcdRecoveryGateError(
            f"Gateway supervisor command exited with status {result.returncode}"
        )


def build_etcd_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one ETCD scheduler reliability scenario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", choices=ETCD_SCENARIOS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capacity-url", required=True)
    parser.add_argument("--capacity-timeout", type=float, default=5.0)
    parser.add_argument("--api-key-env", default="MARIE_API_KEY")
    parser.add_argument(
        "--required-executor", action="append", required=True, dest="executors"
    )
    parser.add_argument("--container", required=True)
    parser.add_argument("--allow-container-mutation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--pause-seconds", type=float, default=10.0)
    parser.add_argument("--recover-seconds", type=float, default=20.0)
    parser.add_argument("--pause-jitter", type=float, default=0.0)
    parser.add_argument("--recover-jitter", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument("--preflight-timeout", type=float, default=30.0)
    parser.add_argument("--zero-slot-timeout", type=float, default=30.0)
    parser.add_argument("--recovery-timeout", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--workload-timeout", type=float, default=900.0)
    parser.add_argument("--workload-warmup-seconds", type=float, default=0.0)
    parser.add_argument("--service-ttl-seconds", type=float, default=None)
    parser.add_argument("--slot-ttl-seconds", type=float, default=None)
    parser.add_argument("--gateway-command", default=None)
    parser.add_argument("--gateway-report", default=None)
    parser.add_argument("--gateway-job-jsonl", default=None)
    parser.add_argument("--gateway-restart-command", default=None)
    parser.add_argument("--verifier-command", required=True)
    parser.add_argument("--verifier-report", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.dry_run and not args.allow_container_mutation:
        parser.error("--allow-container-mutation is required unless --dry-run is used")
    if args.scenario == "idle-reconnect":
        if args.gateway_command or args.gateway_report or args.gateway_job_jsonl:
            parser.error("idle-reconnect does not accept a gateway workload")
    elif not args.gateway_command or not args.gateway_report:
        parser.error(
            "non-idle scenarios require --gateway-command and --gateway-report"
        )
    if args.scenario == "repeated-flapping":
        if args.cycles < 3:
            parser.error("repeated-flapping requires --cycles >= 3")
        if args.seed is None:
            parser.error("repeated-flapping requires --seed")
    if (
        args.scenario == "gateway-restart-during-outage"
        and not args.gateway_restart_command
    ):
        parser.error("gateway restart scenario requires --gateway-restart-command")
    if args.scenario == "ttl-crossing-outage":
        ttls = (args.service_ttl_seconds, args.slot_ttl_seconds)
        if any(value is None for value in ttls):
            parser.error(
                "ttl-crossing-outage requires --service-ttl-seconds and --slot-ttl-seconds"
            )
        if args.pause_seconds <= max(ttls):
            parser.error("--pause-seconds must be longer than both configured TTLs")
    for name in (
        "cycles",
        "pause_seconds",
        "recover_seconds",
        "preflight_timeout",
        "zero_slot_timeout",
        "recovery_timeout",
        "workload_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.poll_interval < 0 or args.workload_warmup_seconds < 0:
        parser.error("poll interval and workload warmup must be nonnegative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_etcd_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    try:
        verifier_command = _parse_command(args.verifier_command, "--verifier-command")
        gateway_command = (
            _parse_command(args.gateway_command, "--gateway-command")
            if args.gateway_command
            else None
        )
        restart_command = (
            _parse_command(args.gateway_restart_command, "--gateway-restart-command")
            if args.gateway_restart_command
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    capacity_reader = HttpCapacityReader(
        capacity_url=args.capacity_url,
        api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
        timeout=args.capacity_timeout,
    )
    simulator = EtcdOutageSimulator(
        container=args.container,
        cycles=args.cycles,
        pause_seconds=args.pause_seconds,
        recover_seconds=args.recover_seconds,
        pause_jitter=args.pause_jitter,
        recover_jitter=args.recover_jitter,
        startup_delay=0.0,
        check_interval=min(0.5, args.poll_interval or 0.1),
        state_timeout=args.state_timeout,
        docker_bin=args.docker_bin,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    workload = (
        GatewayWorkloadCommand(
            gateway_command,
            args.gateway_report,
            job_jsonl_path=args.gateway_job_jsonl,
        )
        if gateway_command is not None
        else None
    )
    verifier = CorrectnessVerifierCommand(verifier_command, args.verifier_report)
    restart_gateway = (
        (lambda: _run_supervisor_command(restart_command))
        if restart_command is not None
        else None
    )
    runner = EtcdRecoveryRunner(
        config=EtcdReliabilityConfig(
            scenario=args.scenario,
            run_id=args.run_id,
            required_executors=tuple(args.executors),
            preflight_timeout=args.preflight_timeout,
            zero_slot_timeout=args.zero_slot_timeout,
            recovery_timeout=args.recovery_timeout,
            poll_interval=args.poll_interval,
            workload_timeout=args.workload_timeout,
            workload_warmup_seconds=args.workload_warmup_seconds,
            dry_run=args.dry_run,
        ),
        capacity_reader=capacity_reader,
        simulator=simulator,
        workload=workload,
        verifier=verifier,
        restart_gateway=restart_gateway,
    )
    report = runner.run()
    write_etcd_report(report, args.report)
    logger.info("ETCD scenario %s: %s", args.scenario, report["status"])
    return 0 if report["status"] in {"passed", "dry-run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
