from types import SimpleNamespace

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

    def warning(self, *args, **kwargs):
        self.records.append(("warning", args, kwargs))


def test_get_or_refresh_renews_cached_lease():
    etcd = FakeEtcd(refresh_ttl=9)
    logger = FakeLogger()
    cache = LeaseCache(etcd, ttl=10, margin=1.0, logger=logger)

    first = cache.get_or_refresh("node/depl")
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


def test_get_or_refresh_replaces_zero_ttl_lease():
    etcd = FakeEtcd(refresh_ttl=0)
    cache = LeaseCache(etcd, ttl=10, margin=1.0)

    first = cache.get_or_refresh("node/depl")
    second = cache.get_or_refresh("node/depl")

    assert second is not first
    assert second.id == 2
    assert len(etcd.leases) == 2


def test_get_or_refresh_replaces_missing_lease():
    etcd = FakeEtcd(refresh_error=RuntimeError("requested lease not found"))
    cache = LeaseCache(etcd, ttl=10, margin=1.0)

    first = cache.get_or_refresh("node/depl")
    etcd.refresh_error = None
    second = cache.get_or_refresh("node/depl")

    assert second is not first
    assert second.id == 2
    assert len(etcd.leases) == 2
