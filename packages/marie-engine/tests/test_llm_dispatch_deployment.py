from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_missing_fixture_preserves_failed_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "deployment"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.stress.llm_dispatch_deployment",
            "--output",
            str(output),
            "--fixture-manifest",
            str(tmp_path / "missing.json"),
        ],
        cwd=root,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    report_path = output / "report.json"
    assert report_path.exists(), "Startup failure must retain a safe failed report"
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failures"][0]["phase"] == "fixture_manifest"
    assert report["scenarios"] == []
    assert "Traceback" not in report_path.read_text()


def test_changed_output_cannot_be_reported_as_completed(tmp_path: Path) -> None:
    import hashlib

    from tools.stress.llm_dispatch_deployment import verify_output_files

    writes = []
    for index in range(9):
        path = tmp_path / f"{index}.md"
        path.write_text(f"output-{index}")
        writes.append(
            {
                "path": str(path),
                "task_id": str(index),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    verify_output_files(writes, tmp_path)
    (tmp_path / "0.md").write_text("changed after first completion")
    with pytest.raises(ValueError, match="output_hash"):
        verify_output_files(writes, tmp_path)


def test_parent_process_is_never_an_owned_worker() -> None:
    import psutil

    from tools.stress.llm_dispatch_deployment import verify_worker

    current = psutil.Process()
    with pytest.raises(ValueError, match="worker_parent"):
        verify_worker(
            {
                "pid": current.pid,
                "created": current.create_time(),
                "ppid": current.ppid(),
            },
            current.pid,
        )


def test_child_environment_does_not_copy_ambient_values(
    tmp_path: Path, monkeypatch
) -> None:
    from tools.stress.llm_dispatch_deployment import safe_environment

    monkeypatch.setenv("QUAL_UNRELATED_PRIVATE_VALUE", "never-copy-this")
    before = os.environ.get("HOME")
    result = safe_environment(tmp_path, Path(__file__).resolve().parents[3])
    assert "QUAL_UNRELATED_PRIVATE_VALUE" not in result
    assert "DUMP_ENV" not in result
    assert "HOME" not in result
    assert os.environ.get("HOME") == before
    assert result["MARIE_CACHE"].startswith(str(tmp_path))


def test_real_http_app_builds_endpoint_models_with_header_aliases() -> None:
    from types import SimpleNamespace

    from docarray import BaseDoc

    from marie.logging_core.logger import MarieLogger
    from marie.serve.runtimes.gateway.http_fastapi_app import get_fastapi_app

    streamer = SimpleNamespace(
        _endpoints_models_map={
            "/annotator/llm": {
                "input": BaseDoc,
                "output": BaseDoc,
                "is_generator": False,
                "parameters": None,
            }
        }
    )
    app = get_fastapi_app(
        streamer,
        "qualification",
        "",
        False,
        False,
        MarieLogger("qualification-http-test"),
    )
    route = next(route for route in app.routes if route.path == "/annotator/llm")
    model = route.endpoint.__annotations__["body"]
    request = model.model_validate(
        {
            "data": [{"id": "synthetic"}],
            "header": {"requestId": "request-1", "targetExecutor": "annotator_llm"},
        }
    )
    assert request.header.request_id == "request-1"
    assert request.header.target_executor == "annotator_llm"
    named = model.model_validate(
        {
            "data": [{"id": "synthetic"}],
            "header": {"request_id": "request-2", "target_executor": "annotator_llm"},
        }
    )
    assert named.header.request_id == "request-2"
    assert named.header.target_executor == "annotator_llm"
    assert "/annotator/llm" in app.openapi()["paths"]


def test_replacement_attempt_does_not_pass_same_attempt_recovery() -> None:
    from copy import deepcopy

    from tools.stress.llm_dispatch_deployment import assert_scheduler_recovery

    def snapshot(state, terminal=None):
        return {
            "attempts": {
                "body": {
                    "result": {
                        "items": [
                            {
                                "job_id": "job",
                                "run_attempt_id": "original",
                                "state": state,
                                "terminal_at": terminal,
                                "terminal_status": "SUCCEEDED" if terminal else None,
                                "terminal_accepted": bool(terminal),
                            }
                        ]
                    }
                }
            },
            "dags": {
                "body": {"result": {"items": [{"id": "dag", "state": "completed"}]}}
            },
        }

    row = {
        "before_history": snapshot("activated"),
        "during_outage": {"history": snapshot("activated")},
        "after_history": snapshot("completed", "terminal-time"),
    }
    assert_scheduler_recovery(row, "job", "dag")
    changed = deepcopy(row)
    changed["after_history"]["attempts"]["body"]["result"]["items"][0][
        "run_attempt_id"
    ] = "replacement"
    with pytest.raises(ValueError, match="same_scheduler_attempt"):
        assert_scheduler_recovery(changed, "job", "dag")


def test_aggregate_refuses_incomplete_deployment_cases(tmp_path: Path) -> None:
    from tools.stress.llm_dispatch_deployment_check import verify

    with pytest.raises(ValueError, match="three_case_reports"):
        verify([tmp_path / "s03-only.json"])


@pytest.mark.parametrize(
    "mutation", ["empty", "omitted", "duplicate", "invalid", "stale"]
)
def test_source_manifest_requires_exact_canonical_inventory(
    tmp_path: Path, mutation: str
) -> None:
    import hashlib

    from tools.stress.llm_dispatch_deployment import source_inventory
    from tools.stress.llm_dispatch_deployment_check import validate_manifest

    source = tmp_path / "source"
    (source / "marie").mkdir(parents=True)
    (source / "marie/first.py").write_text("first = 1\n")
    (source / "marie/second.py").write_text("second = 2\n")
    canonical = source_inventory(source)
    rows = [dict(row) for row in canonical]
    if mutation == "empty":
        rows = []
    elif mutation == "omitted":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "invalid":
        rows[0]["path"] = "../outside.py"
    else:
        rows[0]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "files": rows,
                "digest": hashlib.sha256(
                    json.dumps(rows, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    )
    with pytest.raises(ValueError):
        validate_manifest(path, canonical)


@pytest.mark.parametrize("replacement", ["missing", "stale", "valid"])
def test_used_replacement_manifest_must_match(tmp_path: Path, replacement: str) -> None:
    import hashlib

    from tools.stress.llm_dispatch_deployment import source_inventory
    from tools.stress.llm_dispatch_deployment_check import validate_launch_manifests

    source = tmp_path / "source"
    (source / "marie").mkdir(parents=True)
    (source / "marie/worker.py").write_text("worker = 1\n")
    canonical = source_inventory(source)
    initial = tmp_path / "initial.json"
    initial.write_text(
        json.dumps(
            {
                "files": canonical,
                "digest": hashlib.sha256(
                    json.dumps(canonical, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    )
    changed = tmp_path / "replacement.json"
    if replacement == "valid":
        changed.write_text(initial.read_text())
    if replacement == "stale":
        changed.write_text(initial.read_text())
        (source / "marie/worker.py").write_text("worker = 2\n")
        updated = source_inventory(source)
        initial.write_text(
            json.dumps(
                {
                    "files": updated,
                    "digest": hashlib.sha256(
                        json.dumps(updated, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
        )
        canonical = updated
    row = {
        "name": "SIGKILL",
        "producer": {"ppid": 1},
        "gateway_parent": {"source_manifest": str(initial)},
        "new_producer": {"ppid": 3},
        "replacement": {"pid": 3, "source_manifest": str(changed)},
    }
    report = {
        "startup_source": str(initial),
        "processes": [
            {"role": "executor", "pid": 1, "source_manifest": str(initial)},
            {"role": "gateway", "pid": 2, "source_manifest": str(initial)},
            {"role": "executor", "pid": 3, "source_manifest": str(changed)},
            {
                "role": "executor",
                "pid": 99,
                "source_manifest": str(tmp_path / "unused-history.json"),
            },
        ],
    }
    if replacement == "valid":
        assert validate_launch_manifests(report, row, canonical)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            validate_launch_manifests(report, row, canonical)


class PartialCommands:
    def __init__(self, root: Path, failure: str) -> None:
        self.root = root
        self.failure = failure
        self.created = {}
        self.inspected = 0
        self.discovery_fails = False

    def run(self, label, argv, **kwargs):
        if label.startswith("image-"):
            return "sha256:" + label.removeprefix("image-")
        if label == "compose-up":
            runtime = json.loads((self.root / "private-runtime.json").read_text())
            services = (
                ["postgres"]
                if self.failure == "compose"
                else ["postgres", "etcd", "minio"]
            )
            for name in services:
                planned = runtime["intended_services"][name]
                self.created[name] = {
                    "id": name + "-id",
                    "image": planned["image"],
                    "running": True,
                    "labels": {
                        "marie.test.owner": runtime["project"],
                        "com.docker.compose.project": runtime["project"],
                        "com.docker.compose.service": name,
                    },
                    "bindings": planned["bindings"],
                }
            if self.failure == "compose":
                raise ValueError("compose partially failed")
        if label == "compose-id":
            return self.created[argv[-1]]["id"]
        if label == "discover-owned":
            if self.discovery_fails:
                raise ValueError("discovery failed")
            return "\n".join(
                row["id"] if "--no-trunc" in argv else row["id"][:4]
                for row in self.created.values()
            )
        if label.startswith("stop-owned-"):
            for row in self.created.values():
                if row["id"] == argv[-1]:
                    row["running"] = False
        return ""

    def inspect(self, identity):
        self.inspected += 1
        if self.failure == "inspection" and self.inspected == 2:
            raise ValueError("intermediate inspection failed")
        return dict(next(row for row in self.created.values() if row["id"] == identity))


@pytest.fixture
def setup_ports(monkeypatch) -> None:
    from tools.stress import llm_dispatch_deployment

    ports = iter(range(45000, 45006))
    monkeypatch.setattr(llm_dispatch_deployment, "free_port", lambda: next(ports))


@pytest.mark.parametrize("failure", ["compose", "inspection"])
def test_partial_create_is_recovered_through_actual_cleanup(
    tmp_path: Path, failure: str, setup_ports
) -> None:
    from tools.stress.llm_dispatch_deployment import prepare, stop_owned

    commands = PartialCommands(tmp_path, failure)
    with pytest.raises(ValueError):
        prepare(tmp_path, Path(__file__).resolve().parents[3], {}, commands)
    runtime = json.loads((tmp_path / "private-runtime.json").read_text())
    assert set(runtime["intended_services"]) == {"postgres", "etcd", "minio"}
    assert set(runtime["containers"]) == (
        {"postgres"} if failure == "inspection" else set()
    )
    report = {"failures": []}
    stop_owned(tmp_path, runtime, commands, report)
    assert commands.created and all(
        not row["running"] for row in commands.created.values()
    )
    assert report["container_discovery_complete"] is True
    assert report["running_owned_containers"] == []


def test_partial_cleanup_never_succeeds_when_discovery_fails(
    tmp_path: Path, setup_ports
) -> None:
    from tools.stress.llm_dispatch_deployment import prepare, stop_owned

    commands = PartialCommands(tmp_path, "compose")
    with pytest.raises(ValueError):
        prepare(tmp_path, Path(__file__).resolve().parents[3], {}, commands)
    runtime = json.loads((tmp_path / "private-runtime.json").read_text())
    commands.discovery_fails = True
    report = {"failures": []}
    with pytest.raises(ValueError, match="complete_owned_cleanup"):
        stop_owned(tmp_path, runtime, commands, report)
    assert report["failures"] and commands.created["postgres"]["running"]


def test_cleanup_rejects_foreign_owner_and_continues_other_stops(
    tmp_path: Path,
    setup_ports,
) -> None:
    from tools.stress.llm_dispatch_deployment import prepare, stop_owned

    commands = PartialCommands(tmp_path, "inspection")
    with pytest.raises(ValueError):
        prepare(tmp_path, Path(__file__).resolve().parents[3], {}, commands)
    runtime = json.loads((tmp_path / "private-runtime.json").read_text())
    commands.created["etcd"]["labels"]["marie.test.owner"] = "another-owner"
    report = {"failures": []}
    with pytest.raises(ValueError, match="complete_owned_cleanup"):
        stop_owned(tmp_path, runtime, commands, report)
    assert commands.created["etcd"]["running"] is True
    assert commands.created["postgres"]["running"] is False
    assert commands.created["minio"]["running"] is False


@pytest.mark.parametrize(
    "probe_change", [None, "missing", "fabric", "parent", "stale", "created"]
)
def test_gateway_readiness_requires_current_dispatch_start(probe_change) -> None:
    from tools.stress.llm_dispatch_deployment import gateway_ready

    health = {
        "result": {
            "overall_state": "ok",
            "partial": False,
            "dependencies": [
                {"name": "discovery", "details": {"registered": 1, "ready": 1}}
            ],
        }
    }
    runtime = {
        "fabric": "owned-fabric",
        "processes": [{"role": "gateway", "pid": 100, "created": 10.0}],
    }
    probe = {
        "event": "gateway_dispatch",
        "fabric": "owned-fabric",
        "ppid": 100,
        "pid": 101,
        "at": 11.0,
        "created": 10.5,
    }
    changes = {
        "fabric": ("fabric", "other"),
        "parent": ("ppid", 99),
        "stale": ("at", 9.0),
        "created": ("created", 9.0),
    }
    if probe_change in changes:
        key, value = changes[probe_change]
        probe[key] = value
    assert gateway_ready(
        health, runtime, [] if probe_change == "missing" else [probe]
    ) is (probe_change is None)
