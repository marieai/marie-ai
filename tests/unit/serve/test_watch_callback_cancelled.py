import logging
import threading
from unittest.mock import Mock

import grpc

from marie.logging_core.predefined import default_logger as marie_logger
from marie.serve.discovery.base import ConnectionState
from marie.serve.discovery.etcd_client import EtcdClient


class _FakeRendezvous(grpc.RpcError):
    """Mimics grpc._channel._MultiThreadedRendezvous after channel teardown."""

    def code(self):
        return grpc.StatusCode.CANCELLED

    def details(self):
        return "Channel closed!"

    def __str__(self):
        return "<_MultiThreadedRendezvous CANCELLED Channel closed!>"


def _make_client() -> EtcdClient:
    client = EtcdClient.__new__(EtcdClient)
    client._state_lock = threading.RLock()
    client._connection_state = ConnectionState.RECONNECTING
    client._connection_event_handlers = {s: [] for s in ConnectionState}
    client.state_tracker = Mock()
    client._shutting_down = False
    client.encoding = "utf8"
    client._set_connection_state = Mock()
    return client


def test_cancelled_rendezvous_is_quiet_and_does_not_flip_state(caplog):
    client = _make_client()
    cb = client._create_watch_callback(b"/marie/gateway/marie", Mock())

    # MarieLogger sets propagate=False on its underlying stdlib logger, so
    # caplog's root-attached handler never sees its records; attach directly.
    marie_logger.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger=marie_logger.logger.name):
            cb(_FakeRendezvous())
    finally:
        marie_logger.logger.removeHandler(caplog.handler)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == []  # today: 2 ERROR records
    client._set_connection_state.assert_not_called()  # today: sets DISCONNECTED
    # and it says what actually happened, once, quietly
    assert any("re-establish" in r.message for r in caplog.records)
