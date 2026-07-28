from unittest.mock import Mock

import pytest

from marie.serve.discovery.base import ConnectionState
from marie.serve.discovery.etcd_client import EtcdClient
from marie.serve.discovery.registry import EtcdServiceRegistry

SVC = "gateway/marie"
ADDR = "192.168.1.10:60817"


@pytest.fixture
def client(mocker):
    c = mocker.Mock(spec=EtcdClient)
    c.get.return_value = None                      # register(): key not present
    c.put.return_value = ("put-key", None)         # register() unpacks 2-tuple
    c.get_connection_state.return_value = ConnectionState.CONNECTED
    return c


def _mk_lease(lease_id):
    lease = Mock()
    lease.id = lease_id
    lease.remaining_ttl = 6
    lease.ttl = 6
    lease.refresh.return_value = [Mock(TTL=6)]
    return lease


@pytest.fixture
def registry(client):
    reg = EtcdServiceRegistry(etcd_host=None, etcd_port=None, etcd_client=client)
    return reg


def test_heartbeat_reregisters_when_lease_not_found(registry, client):
    lease1, lease2 = _mk_lease(1), _mk_lease(2)
    client.lease.side_effect = [lease1, lease2]

    registry.register([SVC], ADDR, service_ttl=6)
    assert client.put.call_count == 1

    # the wipe scenario: lease is gone entirely -> refresh raises
    lease1.refresh.side_effect = Exception(
        "etcdserver: requested lease not found"
    )
    registry.heartbeat()

    # re-registered with a FRESH lease
    assert client.put.call_count == 2
    assert registry._leases[ADDR] is lease2


def test_heartbeat_reregister_preserves_metadata(registry, client):
    from marie.serve.discovery.address import JsonAddress

    lease1, lease2 = _mk_lease(1), _mk_lease(2)
    client.lease.side_effect = [lease1, lease2]

    registry.register(
        [SVC], ADDR, service_ttl=6, addr_cls=JsonAddress, metadata='{"x": 1}'
    )
    lease1.refresh.side_effect = Exception(
        "etcdserver: requested lease not found"
    )
    registry.heartbeat()

    assert client.put.call_count == 2
    # both puts carry the JsonAddress-encoded value (metadata preserved)
    first_val = client.put.call_args_list[0].args[1]
    second_val = client.put.call_args_list[1].args[1]
    assert first_val == second_val


def test_heartbeat_healthy_lease_does_not_reregister(registry, client):
    lease1 = _mk_lease(1)
    client.lease.side_effect = [lease1]

    registry.register([SVC], ADDR, service_ttl=6)
    client.get.return_value = "present"  # for Task 2's key check
    registry.heartbeat()

    assert client.put.call_count == 1


def test_heartbeat_reregisters_when_key_deleted_but_lease_alive(registry, client):
    lease1, lease2 = _mk_lease(1), _mk_lease(2)
    client.lease.side_effect = [lease1, lease2]

    registry.register([SVC], ADDR, service_ttl=6)
    assert client.put.call_count == 1

    # lease refreshes fine, but the key was deleted out from under it
    client.get.return_value = None
    registry.heartbeat()

    assert client.put.call_count == 2


def test_heartbeat_key_present_no_reregister(registry, client):
    lease1 = _mk_lease(1)
    client.lease.side_effect = [lease1]

    registry.register([SVC], ADDR, service_ttl=6)
    client.get.return_value = "value-bytes"
    registry.heartbeat()

    assert client.put.call_count == 1
    # the key was actually checked
    from marie.serve.discovery.util import form_service_key
    client.get.assert_called_with(form_service_key(SVC, ADDR))


def test_heartbeat_closed_channel_is_quiet_transient(registry, client, caplog):
    """A lease.refresh() racing our own reconnect raises a closed-channel
    ValueError — that must land in the known-connection-error warning branch
    (one line, no traceback), not the generic ERROR handler. Recovery is
    owned by _on_etcd_connected + the next beat's key check.

    NOTE: MarieLogger's stdlib logger has propagate=False — attach
    caplog.handler to the underlying logger directly or caplog captures
    nothing (vacuous pass)."""
    import logging

    from marie.logging_core.predefined import default_logger as marie_logger

    lease1 = _mk_lease(1)
    client.lease.side_effect = [lease1]
    registry.register([SVC], ADDR, service_ttl=6)

    lease1.refresh.side_effect = ValueError("Cannot invoke RPC: Channel closed!")

    marie_logger.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger=marie_logger.logger.name):
            registry.heartbeat()
    finally:
        marie_logger.logger.removeHandler(caplog.handler)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == []
    # and no re-registration attempt from THIS branch (put count unchanged)
    assert client.put.call_count == 1


def test_shutdown_detaches_handlers_without_closing_injected_client(registry, client):
    registry.shutdown()

    assert client.remove_connection_event_handler.call_count == 4
    client.close.assert_not_called()


def test_shutdown_closes_registry_owned_client(mocker):
    client = mocker.Mock(spec=EtcdClient)
    mocker.patch(
        "marie.serve.discovery.registry.get_etcd_client", return_value=client
    )
    close_client = mocker.patch("marie.serve.discovery.registry.close_etcd_client")
    registry = EtcdServiceRegistry(etcd_host="localhost", etcd_port=2379)

    registry.shutdown()

    close_client.assert_called_once_with(client)
