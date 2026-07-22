from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.stress.etcd_outage_simulator import (
    DockerCommandError,
    EtcdOutageSimulator,
)


class FakeDocker:
    def __init__(
        self,
        *,
        running: bool = True,
        paused: bool = False,
        pause_changes_state: bool = True,
        pause_returncode: int = 0,
        unpause_failures: int = 0,
    ) -> None:
        self.running = running
        self.paused = paused
        self.pause_changes_state = pause_changes_state
        self.pause_returncode = pause_returncode
        self.unpause_failures = unpause_failures
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> SimpleNamespace:
        self.calls.append(command)
        action = command[1]
        if action == "inspect":
            status = "running" if self.running else "exited"
            payload = {
                "Status": status,
                "Running": self.running,
                "Paused": self.paused,
                "Restarting": False,
                "Dead": False,
                "ExitCode": 0,
            }
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        if action == "pause":
            if self.pause_changes_state:
                self.paused = True
            return SimpleNamespace(
                returncode=self.pause_returncode,
                stdout="",
                stderr="pause failed" if self.pause_returncode else "",
            )
        if action == "unpause":
            if self.unpause_failures > 0:
                self.unpause_failures -= 1
                return SimpleNamespace(returncode=1, stdout="", stderr="unpause failed")
            self.paused = False
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected Docker command: {command}")


def build_simulator(**overrides: object) -> EtcdOutageSimulator:
    options = {
        "container": "etcd-test",
        "cycles": 1,
        "pause_seconds": 0.0,
        "recover_seconds": 0.0,
        "pause_jitter": 0.0,
        "recover_jitter": 0.0,
        "startup_delay": 0.0,
        "check_interval": 0.0001,
        "state_timeout": 0.01,
        "docker_bin": "docker",
        "dry_run": False,
        "seed": 7,
    }
    options.update(overrides)
    return EtcdOutageSimulator(**options)


def test_seed_produces_deterministic_jitter_schedule() -> None:
    first = build_simulator(
        cycles=3,
        pause_seconds=8.0,
        recover_seconds=15.0,
        pause_jitter=2.0,
        recover_jitter=3.0,
        seed=42,
    )
    second = build_simulator(
        cycles=3,
        pause_seconds=8.0,
        recover_seconds=15.0,
        pause_jitter=2.0,
        recover_jitter=3.0,
        seed=42,
    )
    different = build_simulator(
        cycles=3,
        pause_seconds=8.0,
        recover_seconds=15.0,
        pause_jitter=2.0,
        recover_jitter=3.0,
        seed=43,
    )

    assert first.timeline["planned_cycles"] == second.timeline["planned_cycles"]
    assert first.timeline["planned_cycles"] != different.timeline["planned_cycles"]


def test_commands_target_only_the_exact_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = FakeDocker()
    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", docker)

    timeline = build_simulator(container="etcd-exact-name").run()

    assert timeline["status"] == "passed"
    assert docker.calls
    assert all(command[-1] == "etcd-exact-name" for command in docker.calls)
    assert [command[1] for command in docker.calls].count("pause") == 1
    assert [command[1] for command in docker.calls].count("unpause") == 1


def test_dry_run_suppresses_all_docker_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_call(*_: object, **__: object) -> None:
        raise AssertionError("subprocess.run must not be called during a dry run")

    monkeypatch.setattr(
        "tools.stress.etcd_outage_simulator.subprocess.run", unexpected_call
    )

    timeline = build_simulator(dry_run=True, cycles=2).run()

    assert timeline["status"] == "passed"
    assert timeline["cleanup"]["status"] == "not_needed"
    assert all(
        cycle["actual"]["pause_command_suppressed"]
        and cycle["actual"]["unpause_command_suppressed"]
        for cycle in timeline["cycles"]
    )


def test_missing_docker_binary_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_: object, **__: object) -> None:
        raise FileNotFoundError("docker")

    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", missing)
    simulator = build_simulator(docker_bin="missing-docker")

    with pytest.raises(DockerCommandError, match="Docker binary not found"):
        simulator.run()

    assert simulator.timeline["status"] == "failed"
    assert simulator.timeline["failures"][0]["error_type"] == "DockerCommandError"


def test_docker_inspect_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(command: list[str], **_: object) -> None:
        raise subprocess.TimeoutExpired(command, 0.01)

    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", timeout)
    simulator = build_simulator(state_timeout=0.01)

    with pytest.raises(DockerCommandError, match="Docker command timed out"):
        simulator.run()

    assert simulator.timeline["failures"][0]["error_type"] == "DockerCommandError"


def test_nonzero_pause_command_is_reported_and_cleanup_checks_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = FakeDocker(pause_changes_state=False, pause_returncode=1)
    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", docker)
    simulator = build_simulator()

    with pytest.raises(DockerCommandError, match="pause failed"):
        simulator.run()

    assert simulator.timeline["cleanup"]["attempted"] is True
    assert simulator.timeline["cleanup"]["status"] == "already_recovered"


def test_pause_state_wait_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    docker = FakeDocker(pause_changes_state=False)
    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", docker)
    simulator = build_simulator(state_timeout=0.001)

    with pytest.raises(TimeoutError, match="Timed out waiting"):
        simulator.run()

    assert simulator.timeline["cleanup"]["status"] == "already_recovered"
    assert simulator.timeline["failures"][0]["error_type"] == "TimeoutError"


@pytest.mark.parametrize(
    ("running", "paused", "expected"),
    [
        (True, True, "already paused"),
        (False, False, "not running"),
    ],
)
def test_prepare_rejects_unsafe_initial_state(
    monkeypatch: pytest.MonkeyPatch,
    running: bool,
    paused: bool,
    expected: str,
) -> None:
    docker = FakeDocker(running=running, paused=paused)
    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", docker)

    with pytest.raises(RuntimeError, match=expected):
        build_simulator().run()

    assert not any(command[1] in {"pause", "unpause"} for command in docker.calls)


@pytest.mark.parametrize("interruption", [False, True])
def test_failure_or_interruption_after_pause_always_unpauses(
    monkeypatch: pytest.MonkeyPatch, interruption: bool
) -> None:
    docker = FakeDocker()
    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", docker)

    def stop_after_pause(_: str, __: dict[str, object]) -> None:
        if interruption:
            raise KeyboardInterrupt
        raise RuntimeError("observer failed")

    simulator = build_simulator(event_callback=stop_after_pause)
    expected = KeyboardInterrupt if interruption else RuntimeError

    with pytest.raises(expected):
        simulator.run()

    assert docker.paused is False
    assert [command[1] for command in docker.calls].count("unpause") == 1
    assert simulator.timeline["cleanup"]["status"] == "succeeded"
    assert simulator.timeline["interrupted"] is interruption


def test_unpause_failure_triggers_a_cleanup_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = FakeDocker(unpause_failures=1)
    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", docker)
    simulator = build_simulator()

    with pytest.raises(DockerCommandError, match="unpause failed"):
        simulator.run()

    assert docker.paused is False
    assert [command[1] for command in docker.calls].count("unpause") == 2
    assert simulator.timeline["cleanup"]["status"] == "succeeded"


def test_cleanup_attempts_unpause_when_inspection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = FakeDocker()
    inspect_count = 0

    def fail_cleanup_inspect(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal inspect_count
        if command[1] == "inspect":
            inspect_count += 1
            if inspect_count == 3:
                docker.calls.append(command)
                return SimpleNamespace(
                    returncode=1, stdout="", stderr="inspect unavailable"
                )
        return docker(command, **kwargs)

    monkeypatch.setattr(
        "tools.stress.etcd_outage_simulator.subprocess.run", fail_cleanup_inspect
    )

    def stop_after_pause(_: str, __: dict[str, object]) -> None:
        raise RuntimeError("observer failed")

    simulator = build_simulator(event_callback=stop_after_pause)
    with pytest.raises(RuntimeError):
        simulator.run()

    assert docker.paused is False
    assert [command[1] for command in docker.calls].count("unpause") == 1
    assert "inspect unavailable" in simulator.timeline["cleanup"]["inspection_error"]
    assert simulator.timeline["cleanup"]["status"] == "succeeded"


def test_timeline_is_complete_and_sanitizes_failure_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docker = FakeDocker()
    monkeypatch.setattr("tools.stress.etcd_outage_simulator.subprocess.run", docker)

    def fail_with_secret(_: str, __: dict[str, object]) -> None:
        raise RuntimeError("token=do-not-write-this")

    simulator = build_simulator(event_callback=fail_with_secret)
    with pytest.raises(RuntimeError):
        simulator.run()

    output_path = tmp_path / "etcd-timeline.json"
    simulator.write_timeline(output_path)
    payload = json.loads(output_path.read_text())
    serialized = json.dumps(payload)
    cycle = payload["cycles"][0]

    assert "do-not-write-this" not in serialized
    assert "token=<redacted>" in serialized
    assert cycle["target_container"] == "etcd-test"
    assert cycle["planned"]["pause_jitter_seconds"] == 0.0
    assert cycle["actual"]["pause_command_started_at"] is not None
    assert cycle["actual"]["pause_command_completed_at"] is not None
    assert cycle["actual"]["unpause_command_started_at"] is None
    assert payload["cleanup"]["attempted"] is True
    assert payload["cleanup"]["status"] == "succeeded"
