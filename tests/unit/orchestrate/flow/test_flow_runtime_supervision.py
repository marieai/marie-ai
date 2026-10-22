from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import marie.orchestrate.flow.base as flow_module
from marie.enums import FlowBuildLevel
from marie.excepts import RuntimeRunForeverEarlyError
from marie.orchestrate.deployments import Deployment
from marie.orchestrate.flow.base import Flow
from marie.orchestrate.pods import Pod


class _Process:
    def __init__(self, pid: int, exitcode: int | None) -> None:
        self.pid = pid
        self.exitcode = exitcode
        self.joined: list[float | None] = []

    def join(self, timeout: float | None = None) -> None:
        self.joined.append(timeout)


def _pod(process: _Process, raft_process: _Process | None = None) -> Pod:
    pod = Pod.__new__(Pod)
    pod.name = 'annotator/rep-0'
    pod.worker = process
    pod.raft_worker = raft_process
    return pod


def _flow(pod: Pod) -> Flow:
    deployment = Deployment.__new__(Deployment)
    deployment.args = SimpleNamespace(external=False, reload=False)
    deployment.uses_before_pod = None
    deployment.uses_after_pod = None
    deployment.head_pod = None
    deployment.gateway_pod = None
    deployment.shards = {0: SimpleNamespace(_pods=[pod])}

    flow = Flow()
    flow._deployment_nodes = {'annotator_llm': deployment}
    flow._build_level = FlowBuildLevel.RUNNING
    return flow


def test_block_raises_when_worker_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flow_module, '_RUNTIME_PROCESS_POLL_INTERVAL', 0.01)
    process = _Process(pid=4242, exitcode=-9)
    flow = _flow(_pod(process))

    with pytest.raises(RuntimeRunForeverEarlyError) as exc_info:
        flow.block()

    message = str(exc_info.value)
    assert 'deployment=annotator_llm' in message
    assert 'pod=annotator/rep-0' in message
    assert 'role=worker' in message
    assert 'pid=4242' in message
    assert 'exitcode=-9' in message
    assert 'signal=SIGKILL' in message
    assert process.joined == [0]
    assert all(
        thread.name != 'flow-runtime-supervisor' for thread in threading.enumerate()
    )


def test_block_returns_cleanly_for_external_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flow_module, '_RUNTIME_PROCESS_POLL_INTERVAL', 0.01)
    process = _Process(pid=4242, exitcode=None)
    flow = _flow(_pod(process))
    stop_event = threading.Event()
    timer = threading.Timer(0.03, stop_event.set)

    timer.start()
    try:
        flow.block(stop_event)
    finally:
        timer.cancel()
        timer.join()

    assert process.joined == []
    assert all(
        thread.name != 'flow-runtime-supervisor' for thread in threading.enumerate()
    )


def test_stopped_flow_does_not_report_dead_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flow_module, '_RUNTIME_PROCESS_POLL_INTERVAL', 0.01)
    process = _Process(pid=4242, exitcode=-9)
    flow = _flow(_pod(process))
    stop_event = threading.Event()
    stop_event.set()

    flow.block(stop_event)

    assert process.joined == []


def test_pod_exposes_worker_and_raft_processes() -> None:
    worker = _Process(pid=4242, exitcode=None)
    raft = _Process(pid=4243, exitcode=None)

    assert list(_pod(worker, raft)._iter_managed_processes()) == [
        ('worker', worker),
        ('raft', raft),
    ]
