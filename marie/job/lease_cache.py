import time
from typing import Any, Dict, Optional, Tuple

import etcd3


class LeaseCache:
    def __init__(self, etcd_client, ttl=5, margin=1.0):
        self.etcd = etcd_client
        self.ttl: int = int(ttl)
        self.margin: float = float(margin)
        self._cache: Dict[str, Tuple[Any, float]] = (
            {}
        )  # cache_key -> (lease, expiry_ts)

    def get_or_refresh(self, cache_key: str, ttl: Optional[int] = None) -> etcd3.Lease:
        """
        Get a cached lease if still valid; otherwise acquire a new one.
        TTL can be overridden per call; falls back to the default from the constructor.

        :param cache_key: Cache bucket key (e.g., "<addr>/<deployment>")
        :param ttl: Optional override TTL for this fetch/refresh
        :return: A lease object
        """
        lease: Optional[etcd3.Lease]
        exp: float
        lease, exp = self._cache.get(cache_key, (None, 0.0))  # type: ignore[assignment]
        now = time.monotonic()

        effective_ttl = int(ttl) if ttl and ttl > 0 else self.ttl

        # Reuse if still safely valid
        if lease is not None and now < (exp - self.margin):
            try:
                # remaining_ttl is guaranteed present in our lease implementation
                if lease.remaining_ttl > 0:
                    return lease
            except Exception:
                # stale/invalid -> refresh below
                pass

        # Refresh with effective TTL
        new_lease = self.etcd.lease(effective_ttl)
        self._cache[cache_key] = (new_lease, now + effective_ttl)
        return new_lease

    def invalidate(self, cache_key):
        self._cache.pop(cache_key, None)


def put_with_cached_lease(etcd, lease_cache, key, val, cache_key):
    lease = lease_cache.get_or_refresh(cache_key)
    try:
        etcd.put(key, val, lease=lease)
        return True
    except Exception as e:
        # Detect lease-not-found (message varies by client/driver)
        msg = str(e).lower()
        if "lease not found" in msg or "requested lease not found" in msg:
            lease_cache.invalidate(cache_key)
            # Re-acquire and retry once
            lease = lease_cache.get_or_refresh(cache_key)
            etcd.put(key, val, lease=lease)
            return True
        raise
