from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from marie.job.lease_cache import LeaseCache


class FakeLease:
    def __init__(self, lease_id: int, ttl: int, refresh_ttl: int, refresh_error=None):
        self.id = lease_id
        self.remaining_ttl = ttl
        self._refresh_ttl = refresh_ttl
        self._refresh_error = refresh_error
        self.refresh_calls = 0

    def refresh(self):
        self.refresh_calls += 1
        if self._refresh_error:
            raise self._refresh_error
        self.remaining_ttl = self._refresh_ttl
        return [SimpleNamespace(TTL=self._refresh_ttl)]


class FakeEtcd:
    def __init__(self, refresh_ttl: int = 10, refresh_error=None):
        self.refresh_ttl = refresh_ttl
        self.refresh_error = refresh_error
        self.leases: list[FakeLease] = []

    def lease(self, ttl: int):
        lease = FakeLease(
            len(self.leases) + 1,
            ttl,
            self.refresh_ttl,
            self.refresh_error,
        )
        self.leases.append(lease)
        return lease


class FakeLogger:
    def __init__(self):
        self.records = []

    def debug(self, *args, **kwargs):
        self.records.append(("debug", args, kwargs))

    def info(self, *args, **kwargs):
        self.records.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.records.append(("warning", args, kwargs))


def test_get_or_refresh_returns_comfortably_valid_cached_lease():
    etcd = FakeEtcd(refresh_ttl=9)
    cache = LeaseCache(etcd, ttl=10, margin=1.0)

    first = cache.get_or_refresh("node/depl")
    second = cache.get_or_refresh("node/depl")

    assert second is first
    assert first.refresh_calls == 0
    assert len(etcd.leases) == 1


def test_get_or_refresh_renews_cached_lease_near_expiry(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("marie.job.lease_cache.time.monotonic", lambda: now[0])
    etcd = FakeEtcd(refresh_ttl=9)
    logger = FakeLogger()
    cache = LeaseCache(etcd, ttl=10, margin=1.0, logger=logger)

    first = cache.get_or_refresh("node/depl")
    now[0] = 109.0
    second = cache.get_or_refresh("node/depl")

    assert second is first
    assert first.refresh_calls == 1
    assert len(etcd.leases) == 1
    assert logger.records[-1][2]["extra"] == {
        "event_type": "etcd_status_lease_renewal",
        "cache_key": "node/depl",
        "lease_id": 1,
        "lease_ttl": 9,
        "renewal_result": "refreshed",
    }
    assert logger.records[-1][1] == (
        "Status lease renewal state: result=refreshed cache_key=node/depl "
        "lease_id=1 lease_ttl=9",
    )


def test_get_or_refresh_replaces_zero_ttl_lease(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("marie.job.lease_cache.time.monotonic", lambda: now[0])
    etcd = FakeEtcd(refresh_ttl=0)
    logger = FakeLogger()
    cache = LeaseCache(etcd, ttl=10, margin=1.0, logger=logger)

    first = cache.get_or_refresh("node/depl")
    now[0] = 109.0
    second = cache.get_or_refresh("node/depl")

    assert second is not first
    assert second.id == 2
    assert len(etcd.leases) == 2
    replacement_record = next(
        record for record in logger.records if record[0] == "info"
    )
    assert replacement_record[1] == (
        "Status lease renewal state: result=expired_replaced cache_key=node/depl "
        "lease_id=1 lease_ttl=0",
    )
    assert replacement_record[2]["extra"]["renewal_result"] == "expired_replaced"


def test_get_or_refresh_replaces_missing_lease(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("marie.job.lease_cache.time.monotonic", lambda: now[0])
    etcd = FakeEtcd(refresh_error=RuntimeError("requested lease not found"))
    cache = LeaseCache(etcd, ttl=10, margin=1.0)

    first = cache.get_or_refresh("node/depl")
    now[0] = 109.0
    etcd.refresh_error = None
    second = cache.get_or_refresh("node/depl")

    assert second is not first
    assert second.id == 2
    assert len(etcd.leases) == 2


def test_get_or_refresh_does_not_replace_lease_after_transient_error(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("marie.job.lease_cache.time.monotonic", lambda: now[0])
    etcd = FakeEtcd(refresh_error=RuntimeError("temporarily unavailable"))
    logger = FakeLogger()
    cache = LeaseCache(etcd, ttl=10, margin=1.0, logger=logger)

    cache.get_or_refresh("node/depl")
    now[0] = 109.0

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        cache.get_or_refresh("node/depl")

    assert len(etcd.leases) == 1
    warning_record = next(record for record in logger.records if record[0] == "warning")
    assert warning_record[1] == (
        "Status lease renewal state: result=refresh_failed cache_key=node/depl "
        "lease_id=1 lease_ttl=None error=RuntimeError: temporarily unavailable",
    )
    assert warning_record[2]["extra"]["renewal_result"] == "refresh_failed"


def test_get_or_refresh_replaces_closed_channel_lease_after_client_recovery(
    monkeypatch,
):
    now = [100.0]
    monkeypatch.setattr("marie.job.lease_cache.time.monotonic", lambda: now[0])
    etcd = FakeEtcd(refresh_error=RuntimeError("Cannot invoke RPC: Channel closed!"))
    cache = LeaseCache(etcd, ttl=10, margin=1.0)

    first = cache.get_or_refresh("node/depl")
    now[0] = 109.0
    etcd.refresh_error = None
    second = cache.get_or_refresh("node/depl")

    assert second is not first
    assert second.id == 2
    assert len(etcd.leases) == 2


def test_get_or_refresh_grants_one_lease_under_concurrency():
    etcd = FakeEtcd()
    cache = LeaseCache(etcd, ttl=10, margin=1.0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        leases = list(executor.map(cache.get_or_refresh, ["node/depl"] * 32))

    assert len(etcd.leases) == 1
    assert all(lease is leases[0] for lease in leases)
