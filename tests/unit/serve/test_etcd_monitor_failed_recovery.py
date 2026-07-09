import threading
from unittest.mock import Mock

import marie.serve.discovery.etcd_client as etcd_client_module
from marie.serve.discovery.base import ConnectionState
from marie.serve.discovery.etcd_client import EtcdClient
from marie.serve.discovery.timeout_utils import OperationTimeoutError


def _make_client(state: ConnectionState) -> EtcdClient:
    # mirrors tests/unit/serve/test_etcd_client_reconnect.py::_make_client
    client = EtcdClient.__new__(EtcdClient)
    client._state_lock = threading.RLock()
    client._connection_state = state
    client._connection_event_handlers = {
        ConnectionState.CONNECTED: [],
        ConnectionState.DISCONNECTED: [],
        ConnectionState.RECONNECTING: [],
        ConnectionState.FAILED: [],
    }
    client.state_tracker = Mock()
    client._shutting_down = False
    client._reconnect_attempts = 0
    client._max_reconnect_attempts = 10
    client._reconnect_timer = None
    client._monitor_ready = threading.Event()
    client._connection_monitor_running = True
    client._monitor_stop = threading.Event()
    client._last_successful_operation = 0
    client._in_recovery = False
    client._is_multi_endpoint = False
    client._client_idx = 0
    client.client = Mock()
    client._attempt_immediate_reconnection = Mock()
    client._schedule_recovery_attempt = Mock()
    return client


class CountedStop:
    """Stops the monitor loop after `limit` iterations."""

    def __init__(self, limit: int):
        self.calls = 0
        self.limit = limit

    def is_set(self) -> bool:
        return self.calls >= self.limit

    def wait(self, _timeout) -> bool:
        self.calls += 1
        return self.calls >= self.limit


def test_failed_state_schedules_recovery_when_health_checks_time_out(monkeypatch):
    client = _make_client(ConnectionState.FAILED)
    client._monitor_stop = CountedStop(5)
    monkeypatch.setattr(
        etcd_client_module,
        "run_with_timeout",
        Mock(side_effect=OperationTimeoutError("health_check", 2.0)),
    )

    client._monitor_connection_health()

    assert client._schedule_recovery_attempt.called
