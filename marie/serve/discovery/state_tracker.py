import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StateTracker:
    def __init__(self, ttl: float = 300.0):
        """
        Initialize StateTracker with global TTL.

        Args:
            ttl: Time to live in seconds for all cached states (default: 5 minutes)
        """
        self._state_cache: Dict[str, Any] = {}
        self._state_hashes: Dict[str, str] = {}
        self._last_clear_time = time.time()
        self._ttl = ttl
        self._lock = threading.RLock()

    def has_state_changed(self, key: str, new_value: Any) -> bool:
        """
        Check if the state has actually changed for a given key.
        Automatically clears all state if TTL has expired.

        Args:
            key: The state key
            new_value: The new value to compare

        Returns:
            True if state changed or cache was expired and cleared
        """
        with self._lock:
            # Check if we need to clear the entire cache due to TTL
            current_time = time.time()
            if current_time - self._last_clear_time > self._ttl:
                self._clear_all_internal()
                self._last_clear_time = current_time
                self._update_state(key, new_value)
                return True

            # Normal state change detection
            new_hash = self._generate_hash(new_value)
            previous_hash = self._state_hashes.get(key)

            if previous_hash is None or previous_hash != new_hash:
                self._update_state(key, new_value)
                return True

            return False

    def force_clear_all(self):
        """Force clear all cached states immediately."""
        with self._lock:
            self._clear_all_internal()
            self._last_clear_time = time.time()

    def get_current_state(self, key: str) -> Optional[Any]:
        """Get the current cached state for a key."""
        with self._lock:
            return self._state_cache.get(key)

    def get_cache_age(self) -> float:
        """Get the age of the current cache in seconds."""
        return time.time() - self._last_clear_time

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the state tracker."""
        with self._lock:
            return {
                'total_entries': len(self._state_cache),
                'cache_age_seconds': self.get_cache_age(),
                'ttl_seconds': self._ttl,
                'time_until_clear': max(0, self._ttl - self.get_cache_age()),
            }

    def update_ttl(self, new_ttl: float):
        """Update the TTL for the cache."""
        with self._lock:
            self._ttl = new_ttl

    def _update_state(self, key: str, value: Any):
        """Internal method to update state and hash."""
        self._state_cache[key] = value
        self._state_hashes[key] = self._generate_hash(value)

    def _clear_all_internal(self):
        """Internal method to clear all cached data."""
        entries_cleared = len(self._state_cache)
        self._state_cache.clear()
        self._state_hashes.clear()
        if entries_cleared > 0:
            logger.info(f"StateTracker: Cleared {entries_cleared} entries")

    def _generate_hash(self, value: Any) -> str:
        """Generate a consistent hash for any value."""
        if isinstance(value, dict):
            serialized = json.dumps(value, sort_keys=True)
        elif isinstance(value, (list, tuple)):
            serialized = json.dumps(list(value), sort_keys=True)
        else:
            serialized = str(value)

        return hashlib.md5(serialized.encode()).hexdigest()


if __name__ == "__main__":
    # Basic usage - cache expires every 5 minutes
    tracker = StateTracker(ttl=300.0)
    if tracker.has_state_changed("deployment/service1", "SERVING"):
        print("State changed or cache expired - process event")

    tracker = StateTracker(ttl=120.0)  # 2 minutes

    tracker.force_clear_all()

    stats = tracker.get_stats()
    print(
        f"Cache has {stats['total_entries']} entries, {stats['time_until_clear']} seconds until clear"
    )
