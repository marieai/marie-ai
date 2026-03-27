import threading
from unittest.mock import Mock

import marie.serve.discovery.etcd_client as etcd_client_module
from marie.serve.discovery.base import ConnectionState
from marie.serve.discovery.etcd_client import EtcdClient
from marie.serve.discovery.timeout_utils import OperationTimeoutError


def _make_client(state: ConnectionState) -> EtcdClient:
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
    client._connection_monitor_running = False
    client._monitor_stop = threading.Event()
    client._last_successful_operation = 0
    client._in_recovery = False
    client._is_multi_endpoint = False
    client._client_idx = 0
    client.client = Mock()
    client._attempt_immediate_reconnection = Mock()
    client._schedule_recovery_attempt = Mock()
    return client


def test_reconnect_keeps_recovery_guard_enabled_during_connected_handlers():
    client = _make_client(ConnectionState.DISCONNECTED)
    observed = []

    def on_connected(_event):
        observed.append(client._in_recovery)

    client._connection_event_handlers[ConnectionState.CONNECTED].append(on_connected)
    client.connect = Mock(return_value=True)
    client.get = Mock(return_value=None)

    assert client.reconnect() is True
    assert observed == [True]
    assert client._in_recovery is False


def test_monitor_pauses_during_reconnect_and_drops_stale_failures(monkeypatch):
    client = _make_client(ConnectionState.CONNECTED)
    client._connection_monitor_running = True

    class SequencedMonitorStop:
        def __init__(self, owner):
            self.owner = owner
            self.calls = 0

        def is_set(self):
            return self.calls >= 3

        def wait(self, _timeout):
            self.calls += 1
            if self.calls == 1:
                self.owner._in_recovery = True
            elif self.calls == 2:
                self.owner._in_recovery = False
            return self.calls >= 3

    stop = SequencedMonitorStop(client)
    client._monitor_stop = stop

    def get_connection_state():
        states = {
            0: ConnectionState.CONNECTED,
            1: ConnectionState.RECONNECTING,
            2: ConnectionState.CONNECTED,
        }
        return states.get(stop.calls, ConnectionState.CONNECTED)

    client.get_connection_state = Mock(side_effect=get_connection_state)
    state_changes = []
    client._set_connection_state = Mock(
        side_effect=lambda state, error=None: state_changes.append(state)
    )

    probe_calls = []

    def fake_run_with_timeout(*_args, **_kwargs):
        probe_calls.append(stop.calls)
        raise OperationTimeoutError("health_check", 2.0)

    monkeypatch.setattr(etcd_client_module, "run_with_timeout", fake_run_with_timeout)

    client._monitor_connection_health()

    assert probe_calls == [0, 2]
    assert ConnectionState.DISCONNECTED not in state_changes
