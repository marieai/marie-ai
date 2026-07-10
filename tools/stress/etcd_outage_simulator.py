#!/usr/bin/env python3
"""
Simulate ETCD outages by pausing and unpausing a Docker container.

Typical usage:
    python tools/stress/etcd_outage_simulator.py --cycles 3 --pause-seconds 10
    python etcd_outage_simulator.py --cycles 3 --pause-seconds 3   --recover-seconds 6 --recover-jitter 3

Run this alongside the gateway or worker stress tools to reproduce reconnect and
re-registration behavior while ETCD disappears and comes back.
"""

import argparse
import json
import logging
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("EtcdOutageSimulator")


class DockerCommandError(RuntimeError):
    """Raised when a Docker command fails."""


@dataclass
class DockerContainerState:
    status: str
    running: bool
    paused: bool
    restarting: bool
    dead: bool
    exit_code: Optional[int]


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
    ):
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
        self._paused_by_script = False

    def _run_docker(self, *args: str) -> str:
        cmd = [self.docker_bin, *args]
        logger.debug("Running command: %s", shlex.join(cmd))

        if self.dry_run and args and args[0] in {"pause", "unpause"}:
            logger.info("[dry-run] %s", shlex.join(cmd))
            return ""

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise DockerCommandError(
                f"Docker binary not found: {self.docker_bin}"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout or f"exit code {result.returncode}"
            raise DockerCommandError(f"Command failed: {shlex.join(cmd)}: {details}")

        return (result.stdout or "").strip()

    def _inspect_state(self) -> DockerContainerState:
        raw = self._run_docker("inspect", "--format", "{{json .State}}", self.container)
        state: dict[str, Any] = json.loads(raw)
        return DockerContainerState(
            status=str(state.get("Status", "unknown")),
            running=bool(state.get("Running", False)),
            paused=bool(state.get("Paused", False)),
            restarting=bool(state.get("Restarting", False)),
            dead=bool(state.get("Dead", False)),
            exit_code=state.get("ExitCode"),
        )

    def _wait_for_state(
        self,
        *,
        paused: Optional[bool] = None,
        running: Optional[bool] = None,
    ) -> DockerContainerState:
        deadline = time.monotonic() + self.state_timeout
        last_state = None

        while time.monotonic() < deadline:
            state = self._inspect_state()
            last_state = state

            paused_ok = paused is None or state.paused == paused
            running_ok = running is None or state.running == running
            if paused_ok and running_ok:
                return state

            time.sleep(self.check_interval)

        raise TimeoutError(
            f"Timed out waiting for container {self.container} to reach "
            f"paused={paused}, running={running}. Last state: {last_state}"
        )

    @staticmethod
    def _sleep(seconds: float):
        if seconds > 0:
            time.sleep(seconds)

    @staticmethod
    def _cycle_duration(base: float, jitter: float) -> float:
        if jitter <= 0:
            return base
        return max(0.0, base + random.uniform(-jitter, jitter))

    def prepare(self):
        if self.startup_delay > 0:
            logger.info("Initial startup delay: %.2fs", self.startup_delay)
            self._sleep(self.startup_delay)

        if self.dry_run:
            logger.info(
                "Dry run enabled; skipping docker inspection for container %s",
                self.container,
            )
            return

        state = self._inspect_state()
        logger.info(
            "Target container %s state: status=%s running=%s paused=%s",
            self.container,
            state.status,
            state.running,
            state.paused,
        )

        if state.paused:
            logger.warning(
                "Container %s is already paused; unpausing before starting test",
                self.container,
            )
            self.unpause()
            return

        if not state.running:
            raise RuntimeError(
                f"Container {self.container} is not running (status={state.status})"
            )

    def pause(self):
        logger.info("Pausing container %s", self.container)
        self._run_docker("pause", self.container)
        self._paused_by_script = True

        if not self.dry_run:
            self._wait_for_state(paused=True)

    def unpause(self):
        logger.info("Unpausing container %s", self.container)
        self._run_docker("unpause", self.container)
        self._paused_by_script = False

        if not self.dry_run:
            self._wait_for_state(paused=False, running=True)

    def cleanup(self):
        if not self._paused_by_script:
            return

        try:
            logger.warning(
                "Cleanup: container %s was left paused; unpausing now",
                self.container,
            )
            self.unpause()
        except Exception as exc:
            logger.error("Cleanup failed while unpausing %s: %s", self.container, exc)

    def run(self):
        self.prepare()
        logger.info(
            "Starting ETCD outage simulation: container=%s cycles=%d pause=%.2fs recover=%.2fs",
            self.container,
            self.cycles,
            self.pause_seconds,
            self.recover_seconds,
        )

        for cycle in range(1, self.cycles + 1):
            pause_for = self._cycle_duration(self.pause_seconds, self.pause_jitter)
            recover_for = self._cycle_duration(
                self.recover_seconds, self.recover_jitter
            )

            logger.info(
                "Cycle %d/%d: pausing %s for %.2fs",
                cycle,
                self.cycles,
                self.container,
                pause_for,
            )
            self.pause()
            self._sleep(pause_for)

            logger.info("Cycle %d/%d: restoring %s", cycle, self.cycles, self.container)
            self.unpause()

            if cycle < self.cycles or recover_for > 0:
                logger.info(
                    "Cycle %d/%d: recovery window %.2fs",
                    cycle,
                    self.cycles,
                    recover_for,
                )
                self._sleep(recover_for)

        logger.info("Finished ETCD outage simulation for %s", self.container)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate ETCD outages using docker pause/unpause.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--container",
        default="etcd-single",
        help="Docker container name for the ETCD instance to pause",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of outage cycles to inject",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=10.0,
        help="How long ETCD stays paused during each outage cycle",
    )
    parser.add_argument(
        "--recover-seconds",
        type=float,
        default=20.0,
        help="How long to wait after unpausing before the next cycle",
    )
    parser.add_argument(
        "--pause-jitter",
        type=float,
        default=0.0,
        help="Random jitter applied to pause duration (+/- seconds)",
    )
    parser.add_argument(
        "--recover-jitter",
        type=float,
        default=0.0,
        help="Random jitter applied to recovery duration (+/- seconds)",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=0.0,
        help="Optional delay before the first outage cycle starts",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=0.5,
        help="Polling interval while waiting for Docker state changes",
    )
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=10.0,
        help="Timeout while waiting for Docker pause/unpause state transitions",
    )
    parser.add_argument(
        "--docker-bin",
        default="docker",
        help="Docker CLI binary to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible jitter",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the outage schedule without calling docker pause/unpause",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.cycles <= 0:
        parser.error("--cycles must be greater than 0")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds must be >= 0")
    if args.recover_seconds < 0:
        parser.error("--recover-seconds must be >= 0")
    if args.pause_jitter < 0:
        parser.error("--pause-jitter must be >= 0")
    if args.recover_jitter < 0:
        parser.error("--recover-jitter must be >= 0")

    if args.seed is not None:
        random.seed(args.seed)

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
    )

    try:
        simulator.run()
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted; cleaning up")
        return 130
    except Exception as exc:
        logger.error("ETCD outage simulation failed: %s", exc)
        return 1
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    sys.exit(main())
