from types import SimpleNamespace

from marie.serve.discovery import (
    DEFAULT_DISCOVERY_HEARTBEAT_SEC,
    DEFAULT_DISCOVERY_LEASE_SEC,
    DiscoveryServiceMixin,
    _discovery_lease_params,
)


def test_uses_runtime_args_when_present():
    args = SimpleNamespace(discovery_lease_sec=12, discovery_heartbeat_sec=3.5)
    assert _discovery_lease_params(args) == (12, 3.5)


def test_falls_back_to_defaults():
    # The default VALUES are owner-tuned policy (see the module constants);
    # the contract under test is only that absent args fall back to them.
    args = SimpleNamespace()  # args absent entirely
    ttl, beat = _discovery_lease_params(args)
    assert (ttl, beat) == (DEFAULT_DISCOVERY_LEASE_SEC, DEFAULT_DISCOVERY_HEARTBEAT_SEC)
    assert ttl > 0 and beat > 0
    # hard invariant only (values are owner-tuned): a lease must outlive at
    # least one heartbeat interval or it can expire between healthy beats
    assert ttl > beat


def test_none_values_fall_back():
    args = SimpleNamespace(discovery_lease_sec=None, discovery_heartbeat_sec=None)
    assert _discovery_lease_params(args) == (
        DEFAULT_DISCOVERY_LEASE_SEC,
        DEFAULT_DISCOVERY_HEARTBEAT_SEC,
    )


def test_teardown_stops_owned_registry():
    registry = SimpleNamespace(shutdown=lambda: None)
    owner = SimpleNamespace(
        _etcd_registry=registry,
        sd_state="ready",
        logger=SimpleNamespace(warning=lambda *_args: None),
    )
    registry.shutdown = lambda: setattr(registry, "stopped", True)

    DiscoveryServiceMixin._teardown_service_discovery(owner)

    assert registry.stopped is True
    assert owner._etcd_registry is None
    assert owner.sd_state == "stopped"
