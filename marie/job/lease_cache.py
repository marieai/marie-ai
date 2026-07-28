import threading
import time
from typing import Any, Dict, Optional, Tuple

import etcd3


class LeaseCache:
    def __init__(self, etcd_client, ttl=5, margin=1.0, logger=None):
        self.etcd = etcd_client
        self.ttl: int = int(ttl)
        self.margin: float = float(margin)
        self.logger = logger
        self._cache: Dict[
            str, Tuple[Any, float]
        ] = {}  # cache_key -> (lease, expiry_ts)
        self._lock = threading.RLock()

    def _log_renewal(
        self,
        level: str,
        cache_key: str,
        lease: Any,
        lease_ttl: Optional[int],
        result: str,
        error: Optional[Exception] = None,
    ) -> None:
        if not self.logger:
            return
        extra = {
            "event_type": "etcd_status_lease_renewal",
            "cache_key": cache_key,
            "lease_id": getattr(lease, "id", None),
            "lease_ttl": lease_ttl,
            "renewal_result": result,
        }
        if error:
            extra["error_type"] = type(error).__name__
            extra["error_message"] = str(error)
        message = (
            "Status lease renewal state: "
            f"result={result} cache_key={cache_key} "
            f"lease_id={extra['lease_id']} lease_ttl={lease_ttl}"
        )
        if error:
            message = f"{message} error={type(error).__name__}: {error}"
        getattr(self.logger, level)(
            message,
            extra=extra,
        )

    def _ttl_from_refresh(self, result: Any) -> Optional[int]:
        try:
            result = result[0]
        except (TypeError, IndexError):
            pass
        for attr in ("TTL", "ttl"):
            try:
                ttl = int(getattr(result, attr))
                return ttl if ttl > 0 else None
            except (AttributeError, TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _is_missing_lease_error(error: Exception) -> bool:
        message = str(error).lower()
        return "lease not found" in message or "requested lease not found" in message

    def get_or_refresh(self, cache_key: str, ttl: Optional[int] = None) -> etcd3.Lease:
        """Return a live cached lease, refreshing it only near expiry.

        TTL can be overridden per call; it falls back to the constructor default.

        :param cache_key: Cache bucket key (e.g., "<addr>/<deployment>")
        :param ttl: Optional override TTL for this fetch/refresh
        :return: A lease object
        """
        with self._lock:
            lease: Optional[etcd3.Lease]
            exp: float
            lease, exp = self._cache.get(cache_key, (None, 0.0))  # type: ignore[assignment]
            now = time.monotonic()
            effective_ttl = int(ttl) if ttl and ttl > 0 else self.ttl

            if lease is not None and now < (exp - self.margin):
                return lease

            expired_lease: Optional[etcd3.Lease] = None
            expired_ttl: Optional[int] = None
            if lease is not None:
                try:
                    refreshed_ttl = self._ttl_from_refresh(lease.refresh())
                    if refreshed_ttl is None:
                        refreshed_ttl = int(getattr(lease, "remaining_ttl", 0) or 0)
                    if refreshed_ttl > 0:
                        self._cache[cache_key] = (lease, now + refreshed_ttl)
                        self._log_renewal(
                            "debug",
                            cache_key,
                            lease,
                            refreshed_ttl,
                            "refreshed",
                        )
                        return lease
                    expired_lease = lease
                    expired_ttl = refreshed_ttl
                except Exception as error:
                    self._log_renewal(
                        "warning",
                        cache_key,
                        lease,
                        None,
                        "refresh_failed",
                        error,
                    )
                    if not self._is_missing_lease_error(error):
                        raise
                self._cache.pop(cache_key, None)

            new_lease = self.etcd.lease(effective_ttl)
            self._cache[cache_key] = (new_lease, now + effective_ttl)
            if expired_lease is not None:
                self._log_renewal(
                    "info",
                    cache_key,
                    expired_lease,
                    expired_ttl,
                    "expired_replaced",
                )
            self._log_renewal("debug", cache_key, new_lease, effective_ttl, "created")
            return new_lease

    def invalidate(self, cache_key: str) -> None:
        with self._lock:
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
