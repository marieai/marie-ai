import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.serve.discovery.base import ConnectionState
from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler


def _handler(connection_state: ConnectionState) -> WorkerRequestHandler:
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MagicMock()
    handler._node = "worker-1:5000"
    handler._deployment = "mock_executor"
    handler._worker_id = "mock_executor/rep-0@worker-1:5000"
    handler._worker_state = HealthCheckResponse.ServingStatus.NOT_SERVING
    handler._etcd_client = MagicMock()
    handler._etcd_client.get_connection_state.return_value = connection_state
    handler._status_store = MagicMock()
    handler._status_lease_invalidator = MagicMock()
    handler._claim_and_mark_ready = MagicMock(return_value=True)
    handler._claim_and_mark_serving = MagicMock(return_value=True)
    handler._sem_renew_all_if_due = MagicMock()
    handler._status_hb_stop = threading.Event()
    handler._hb_supervisor_stop = threading.Event()
    handler._status_hb_thread = None
    handler._heartbeat_time = 60.0
    handler._base_heartbeat = 60.0
    return handler


def test_reconnect_invalidates_status_lease_before_reclaim() -> None:
    handler = _handler(ConnectionState.CONNECTED)
    calls: list[str] = []
    handler._status_lease_invalidator.side_effect = lambda: calls.append("invalidate")
    handler._claim_and_mark_ready.side_effect = lambda: calls.append("reclaim") or True

    handler._on_etcd_connected(SimpleNamespace())

    assert calls == ["invalidate", "reclaim"]


def test_disconnect_invalidates_status_lease() -> None:
    handler = _handler(ConnectionState.DISCONNECTED)

    handler._on_etcd_disconnected(SimpleNamespace(error=RuntimeError("offline")))

    handler._status_lease_invalidator.assert_called_once_with()


def test_reconnect_does_not_log_success_when_reclaim_is_rejected() -> None:
    handler = _handler(ConnectionState.CONNECTED)
    handler._claim_and_mark_ready.return_value = False

    handler._on_etcd_connected(SimpleNamespace())

    success_messages = [
        call.args[0]
        for call in handler.logger.info.call_args_list
        if call.args and "Re-claimed status" in call.args[0]
    ]
    assert success_messages == []


@pytest.mark.parametrize(
    "connection_state",
    [
        ConnectionState.DISCONNECTED,
        ConnectionState.RECONNECTING,
        ConnectionState.FAILED,
    ],
)
def test_disconnected_worker_skips_status_heartbeat_and_reclaim(
    connection_state: ConnectionState,
) -> None:
    handler = _handler(connection_state)

    assert handler._status_heartbeat_once() is None
    handler._status_store.heartbeat.assert_not_called()
    handler._claim_and_mark_ready.assert_not_called()
    handler._claim_and_mark_serving.assert_not_called()
    handler._sem_renew_all_if_due.assert_not_called()


def test_missing_status_is_reclaimed_after_connectivity_returns() -> None:
    handler = _handler(ConnectionState.DISCONNECTED)

    assert handler._status_heartbeat_once() is None

    handler._etcd_client.get_connection_state.return_value = ConnectionState.CONNECTED
    handler._status_store.heartbeat.return_value = False

    assert handler._status_heartbeat_once() is True
    handler._status_store.heartbeat.assert_called_once_with(
        handler._node, handler._deployment, handler._worker_id
    )
    handler._claim_and_mark_ready.assert_called_once_with()
    handler._sem_renew_all_if_due.assert_called_once_with()


def test_status_reclaim_is_skipped_if_connection_drops_during_heartbeat() -> None:
    handler = _handler(ConnectionState.CONNECTED)

    def disconnect_during_heartbeat(*_args) -> bool:
        handler._etcd_client.get_connection_state.return_value = (
            ConnectionState.DISCONNECTED
        )
        return False

    handler._status_store.heartbeat.side_effect = disconnect_during_heartbeat

    assert handler._status_heartbeat_once() is False
    handler._claim_and_mark_ready.assert_not_called()
    handler._sem_renew_all_if_due.assert_not_called()


def test_shutdown_interrupts_heartbeat_wait() -> None:
    handler = _handler(ConnectionState.CONNECTED)

    handler.setup_heartbeat()
    assert handler._status_hb_thread is not None
    assert handler._status_hb_thread.is_alive()

    handler.shutdown_heartbeat()

    assert not handler._status_hb_thread.is_alive()
