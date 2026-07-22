#!/usr/bin/env python3
"""Inject bounded ETCD outages against one explicitly named Docker container."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("EtcdOutageSimulator")

REDACTED = "<redacted>"
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
)
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|authorization|password|secret|token)\s*[=:]\s*)([^\s,;]+)"
)


class DockerCommandError(RuntimeError):
    """Raised when a Docker command cannot be completed."""


@dataclass(frozen=True)
class DockerContainerState:
    status: str
    running: bool
    paused: bool
    restarting: bool
    dead: bool
    exit_code: int | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_report_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if any(part in normalized_key for part in _SECRET_KEY_PARTS):
                sanitized[key] = REDACTED
            else:
                sanitized[key] = _sanitize_report_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_report_value(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE_PATTERN.sub(rf"\1{REDACTED}", value)
    return value


class EtcdOutageSimulator:
    def __init__(
        self,
        *,
        container: str,
        cycles: int,
        pause_seconds: float,
        recover_seconds: float,
        pause_jitter: float,
        recover_jitter: float,
        startup_delay: float,
        check_interval: float,
        state_timeout: float,
        docker_bin: str,
        dry_run: bool,
        seed: int | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.container = container
        self.cycles = cycles
        self.pause_seconds = pause_seconds
        self.recover_seconds = recover_seconds
        self.pause_jitter = pause_jitter
        self.recover_jitter = recover_jitter
        self.startup_delay = startup_delay
        self.check_interval = check_interval
        self.state_timeout = state_timeout
        self.docker_bin = docker_bin
        self.dry_run = dry_run
        self.seed = seed
        self.event_callback = event_callback
        self._rng = random.Random(seed)
        self._paused_by_process = False
        self._pause_may_have_succeeded = False
        self._current_cycle: dict[str, Any] | None = None
        self._schedule = self._build_schedule()
        self.timeline: dict[str, Any] = {
            "schema_version": 1,
            "container": container,
            "seed": seed,
            "dry_run": dry_run,
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "planned_cycles": self._schedule,
            "cycles": [],
            "observed_states": [],
            "failures": [],
            "interrupted": False,
            "cleanup": {
                "needed": False,
                "attempted": False,
                "started_at": None,
                "command_started_at": None,
                "command_completed_at": None,
                "completed_at": None,
                "status": "pending",
                "inspection_error": None,
                "error": None,
            },
        }

    def _build_schedule(self) -> list[dict[str, Any]]:
        schedule = []
        for cycle in range(1, self.cycles + 1):
            pause_jitter = (
                self._rng.uniform(-self.pause_jitter, self.pause_jitter)
                if self.pause_jitter > 0
                else 0.0
            )
            recover_jitter = (
                self._rng.uniform(-self.recover_jitter, self.recover_jitter)
                if self.recover_jitter > 0
                else 0.0
            )
            schedule.append(
                {
                    "cycle": cycle,
                    "target_container": self.container,
                    "pause_seconds": max(0.0, self.pause_seconds + pause_jitter),
                    "recover_seconds": max(0.0, self.recover_seconds + recover_jitter),
                    "pause_jitter_seconds": pause_jitter,
                    "recover_jitter_seconds": recover_jitter,
                }
            )
        return schedule

    def _run_docker(self, *args: str) -> str:
        command = [self.docker_bin, *args]
        logger.debug("Running command: %s", shlex.join(command))

        if self.dry_run and args and args[0] in {"pause", "unpause"}:
            logger.info("[dry-run] %s", shlex.join(command))
            return ""

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.state_timeout,
            )
        except FileNotFoundError as exc:
            raise DockerCommandError(
                f"Docker binary not found: {self.docker_bin}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DockerCommandError(
                f"Docker command timed out after {self.state_timeout:.1f}s: "
                f"{shlex.join(command)}"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout or f"exit code {result.returncode}"
            raise DockerCommandError(
                f"Command failed: {shlex.join(command)}: {details}"
            )

        return (result.stdout or "").strip()

    def _inspect_state(
        self, *, phase: str, cycle: int | None = None
    ) -> DockerContainerState:
        raw = self._run_docker("inspect", "--format", "{{json .State}}", self.container)
        try:
            payload: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DockerCommandError(
                f"Docker inspect returned invalid state for {self.container}"
            ) from exc

        state = DockerContainerState(
            status=str(payload.get("Status", "unknown")),
            running=bool(payload.get("Running", False)),
            paused=bool(payload.get("Paused", False)),
            restarting=bool(payload.get("Restarting", False)),
            dead=bool(payload.get("Dead", False)),
            exit_code=payload.get("ExitCode"),
        )
        self.timeline["observed_states"].append(
            {
                "observed_at": _utc_now(),
                "phase": phase,
                "cycle": cycle,
                "target_container": self.container,
                "state": asdict(state),
            }
        )
        return state

    def _wait_for_state(
        self,
        *,
        phase: str,
        cycle: int | None = None,
        paused: bool | None = None,
        running: bool | None = None,
    ) -> DockerContainerState:
        deadline = time.monotonic() + self.state_timeout
        last_state = None

        while time.monotonic() < deadline:
            state = self._inspect_state(phase=phase, cycle=cycle)
            last_state = state
            paused_ok = paused is None or state.paused == paused
            running_ok = running is None or state.running == running
            if paused_ok and running_ok:
                return state
            self._sleep(self.check_interval)

        raise TimeoutError(
            f"Timed out waiting for container {self.container} to reach "
            f"paused={paused}, running={running}. Last state: {last_state}"
        )

    @staticmethod
    def _sleep(seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _notify(self, event: str, cycle: int) -> None:
        if self.event_callback is None:
            return
        self.event_callback(
            event,
            {
                "observed_at": _utc_now(),
                "cycle": cycle,
                "target_container": self.container,
            },
        )

    def prepare(self) -> None:
        if self.startup_delay > 0:
            logger.info("Initial startup delay: %.2fs", self.startup_delay)
            self._sleep(self.startup_delay)

        if self.dry_run:
            logger.info(
                "Dry run enabled; target container is %s and Docker mutation is disabled",
                self.container,
            )
            return

        state = self._inspect_state(phase="prepare")
        logger.info(
            "Validated target container %s: status=%s running=%s paused=%s",
            self.container,
            state.status,
            state.running,
            state.paused,
        )
        if state.paused:
            raise RuntimeError(
                f"Container {self.container} is already paused; refusing to take ownership"
            )
        if not state.running:
            raise RuntimeError(
                f"Container {self.container} is not running (status={state.status})"
            )

    def pause(self, cycle: dict[str, Any]) -> None:
        cycle_number = int(cycle["cycle"])
        actual = cycle["actual"]
        logger.info("Pausing container %s", self.container)
        actual["pause_command_started_at"] = _utc_now()
        self._pause_may_have_succeeded = not self.dry_run
        self._run_docker("pause", self.container)
        actual["pause_command_completed_at"] = _utc_now()
        actual["pause_command_suppressed"] = self.dry_run

        if not self.dry_run:
            self._paused_by_process = True
            self._wait_for_state(
                phase="pause_wait", cycle=cycle_number, paused=True, running=True
            )
        actual["pause_state_confirmed_at"] = _utc_now()

    def unpause(self, cycle: dict[str, Any]) -> None:
        cycle_number = int(cycle["cycle"])
        actual = cycle["actual"]
        logger.info("Unpausing container %s", self.container)
        actual["unpause_command_started_at"] = _utc_now()
        self._run_docker("unpause", self.container)
        actual["unpause_command_completed_at"] = _utc_now()
        actual["unpause_command_suppressed"] = self.dry_run

        if not self.dry_run:
            self._wait_for_state(
                phase="unpause_wait", cycle=cycle_number, paused=False, running=True
            )
            self._paused_by_process = False
            self._pause_may_have_succeeded = False
        actual["recovery_state_confirmed_at"] = _utc_now()

    def cleanup(self) -> bool:
        cleanup = self.timeline["cleanup"]
        cleanup["needed"] = bool(
            self._paused_by_process or self._pause_may_have_succeeded
        )
        if self.dry_run or not cleanup["needed"]:
            cleanup["status"] = "not_needed"
            cleanup["completed_at"] = _utc_now()
            return True

        cleanup["attempted"] = True
        cleanup["started_at"] = _utc_now()
        try:
            state = self._inspect_state(phase="cleanup_check")
        except Exception as exc:
            state = None
            cleanup["inspection_error"] = f"{type(exc).__name__}: {exc}"

        if state is not None and not state.paused and state.running:
            self._paused_by_process = False
            self._pause_may_have_succeeded = False
            cleanup["status"] = "already_recovered"
            cleanup["completed_at"] = _utc_now()
            return True

        try:
            logger.warning(
                "Cleanup: container %s may be paused; unpausing now", self.container
            )
            cleanup["command_started_at"] = _utc_now()
            self._run_docker("unpause", self.container)
            cleanup["command_completed_at"] = _utc_now()
            self._wait_for_state(phase="cleanup_wait", paused=False, running=True)
        except Exception as exc:
            cleanup["status"] = "failed"
            cleanup["error"] = f"{type(exc).__name__}: {exc}"
            cleanup["completed_at"] = _utc_now()
            self.timeline["status"] = "failed"
            logger.error("Cleanup failed while unpausing %s: %s", self.container, exc)
            return False

        self._paused_by_process = False
        self._pause_may_have_succeeded = False
        cleanup["status"] = "succeeded"
        cleanup["completed_at"] = _utc_now()
        return True

    def _record_failure(self, exc: BaseException, *, interrupted: bool) -> None:
        self.timeline["status"] = "interrupted" if interrupted else "failed"
        self.timeline["interrupted"] = interrupted
        self.timeline["failures"].append(
            {
                "recorded_at": _utc_now(),
                "cycle": (
                    self._current_cycle["cycle"]
                    if self._current_cycle is not None
                    else None
                ),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )

    def run(self) -> dict[str, Any]:
        self.timeline["status"] = "running"
        self.timeline["started_at"] = _utc_now()
        error: BaseException | None = None
        try:
            self.prepare()
            logger.info(
                "Starting ETCD outage simulation: container=%s cycles=%d pause=%.2fs recover=%.2fs seed=%s",
                self.container,
                self.cycles,
                self.pause_seconds,
                self.recover_seconds,
                self.seed,
            )

            for plan in self._schedule:
                cycle = {
                    "cycle": plan["cycle"],
                    "target_container": self.container,
                    "planned": dict(plan),
                    "actual": {
                        "started_at": _utc_now(),
                        "pause_command_started_at": None,
                        "pause_command_completed_at": None,
                        "pause_command_suppressed": False,
                        "pause_state_confirmed_at": None,
                        "pause_duration_seconds": None,
                        "unpause_command_started_at": None,
                        "unpause_command_completed_at": None,
                        "unpause_command_suppressed": False,
                        "recovery_state_confirmed_at": None,
                        "recovery_duration_seconds": None,
                        "completed_at": None,
                    },
                }
                self.timeline["cycles"].append(cycle)
                self._current_cycle = cycle
                cycle_number = int(cycle["cycle"])

                logger.info(
                    "Cycle %d/%d: pausing %s for %.2fs",
                    cycle_number,
                    self.cycles,
                    self.container,
                    plan["pause_seconds"],
                )
                self.pause(cycle)
                pause_started = time.monotonic()
                try:
                    self._notify("paused", cycle_number)
                    self._sleep(float(plan["pause_seconds"]))
                finally:
                    cycle["actual"]["pause_duration_seconds"] = (
                        time.monotonic() - pause_started
                    )

                logger.info(
                    "Cycle %d/%d: restoring %s",
                    cycle_number,
                    self.cycles,
                    self.container,
                )
                self.unpause(cycle)

                recovery_started = time.monotonic()
                try:
                    self._notify("recovered", cycle_number)
                    self._sleep(float(plan["recover_seconds"]))
                finally:
                    cycle["actual"]["recovery_duration_seconds"] = (
                        time.monotonic() - recovery_started
                    )
                cycle["actual"]["completed_at"] = _utc_now()

            self.timeline["status"] = "passed"
            logger.info("Finished ETCD outage simulation for %s", self.container)
        except KeyboardInterrupt as exc:
            self._record_failure(exc, interrupted=True)
            error = exc
        except Exception as exc:
            self._record_failure(exc, interrupted=False)
            error = exc

        cleanup_ok = self.cleanup()
        self.timeline["completed_at"] = _utc_now()
        self._current_cycle = None

        if error is not None:
            raise error
        if not cleanup_ok:
            raise DockerCommandError(
                f"Cleanup failed; container {self.container} may still be paused"
            )
        return self.timeline

    def write_timeline(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        payload = _sanitize_report_value(self.timeline)
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate ETCD outages using docker pause/unpause.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--container",
        required=True,
        help="Exact Docker container name for the ETCD instance to pause",
    )
    parser.add_argument(
        "--allow-container-mutation",
        action="store_true",
        help="Required opt-in for docker pause/unpause; omit only with --dry-run",
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--pause-seconds", type=float, default=10.0)
    parser.add_argument("--recover-seconds", type=float, default=20.0)
    parser.add_argument("--pause-jitter", type=float, default=0.0)
    parser.add_argument("--recover-jitter", type=float, default=0.0)
    parser.add_argument("--startup-delay", type=float, default=0.0)
    parser.add_argument("--check-interval", type=float, default=0.5)
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--timeline-json",
        default=None,
        help="Optional structured JSON timeline output path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and record the schedule without mutating Docker",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.dry_run and not args.allow_container_mutation:
        parser.error("--allow-container-mutation is required unless --dry-run is used")
    if args.cycles <= 0:
        parser.error("--cycles must be greater than 0")
    for name in (
        "pause_seconds",
        "recover_seconds",
        "pause_jitter",
        "recover_jitter",
        "startup_delay",
        "check_interval",
        "state_timeout",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    simulator = EtcdOutageSimulator(
        container=args.container,
        cycles=args.cycles,
        pause_seconds=args.pause_seconds,
        recover_seconds=args.recover_seconds,
        pause_jitter=args.pause_jitter,
        recover_jitter=args.recover_jitter,
        startup_delay=args.startup_delay,
        check_interval=args.check_interval,
        state_timeout=args.state_timeout,
        docker_bin=args.docker_bin,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    exit_code = 0
    try:
        simulator.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted; cleanup status=%s", simulator.timeline["cleanup"])
        exit_code = 130
    except Exception as exc:
        logger.error("ETCD outage simulation failed: %s", exc)
        exit_code = 1
    finally:
        if args.timeline_json:
            try:
                simulator.write_timeline(args.timeline_json)
            except OSError as exc:
                logger.error("Failed to write timeline %s: %s", args.timeline_json, exc)
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
