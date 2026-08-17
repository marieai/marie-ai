import os
import signal
import threading

import pytest

import marie.orchestrate.flow.base as flow_module
from marie.excepts import RuntimeRunForeverEarlyError
from marie.runtime import Flow
from marie.serve.executors import BaseExecutor


def test_flow_exits_when_worker_is_sigkilled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flow_module, '_RUNTIME_PROCESS_POLL_INTERVAL', 0.05)
    flow = Flow().add(name='worker', uses=BaseExecutor, replicas=2)
    worker = None
    killer = None

    with pytest.raises(RuntimeRunForeverEarlyError, match='signal=SIGKILL'):
        with flow:
            worker = flow._deployment_nodes['worker'].shards[0]._pods[0].worker
            killer = threading.Timer(
                0.1,
                os.kill,
                args=(worker.pid, signal.SIGKILL),
            )
            killer.start()
            flow.block()

    assert killer is not None
    killer.join()
    assert worker is not None
    assert worker.exitcode == -signal.SIGKILL
    assert not worker.is_alive()
