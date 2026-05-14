from marie.serve.discovery.container import EtcdConfig
from marie.serve.runtimes.worker.request_handling import _status_lease_timings


def test_status_lease_timings_use_configured_safe_values():
    config = EtcdConfig(lease_sec=12, heartbeat_sec=2.5)

    assert _status_lease_timings(config) == (12, 2.5)


def test_status_lease_timings_clamp_heartbeat_to_half_lease():
    config = EtcdConfig(lease_sec=6, heartbeat_sec=10)

    assert _status_lease_timings(config) == (6, 3.0)


def test_status_lease_timings_enforce_minimum_safe_values():
    config = EtcdConfig(lease_sec=1, heartbeat_sec=0)

    assert _status_lease_timings(config) == (2, 1.0)
