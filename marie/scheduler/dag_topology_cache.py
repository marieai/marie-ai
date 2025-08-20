import threading
from typing import Dict, List, Tuple

from cachetools import LRUCache

from marie.query_planner.base import QueryPlan
from marie.query_planner.planner import compute_job_levels, topological_sort


class DagTopologyCache:
    """
    Bounded cache for (sorted_nodes, job_levels) by dag_id using LRU.

    Concurrency:
      - LRU operations are guarded by a short global lock (LRU mutates on every hit).
      - Misses compute once per dag_id via a per-key lock to avoid duplicate work.
    """

    def __init__(self, maxsize: int = 4096):
        # LRU of dag_id -> (sorted_nodes, job_levels)
        self._cache: LRUCache[str, Tuple[List[str], Dict[str, int]]] = LRUCache(
            maxsize=maxsize
        )
        self._lru_lock = threading.RLock()

        # Per-dag build locks registry
        self._locks_guard = threading.Lock()
        self._build_locks: Dict[str, threading.Lock] = {}

    def _build_lock_for(self, dag_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._build_locks.get(dag_id)
            if lock is None:
                lock = threading.Lock()
                self._build_locks[dag_id] = lock
            return lock

    def get_sorted_nodes_and_levels(
        self, plan: QueryPlan, dag_id: str
    ) -> Tuple[List[str], Dict[str, int]]:
        # Fast path: try LRU under a short lock (required because LRU updates recency on hits)
        with self._lru_lock:
            val = self._cache.get(dag_id)
            if val is not None:
                return val

        # Miss: compute once for this dag_id
        lock = self._build_lock_for(dag_id)
        with lock:
            # Double-check after acquiring per-key lock in case another thread filled it
            with self._lru_lock:
                val = self._cache.get(dag_id)
                if val is not None:
                    return val

            # Compute outside the LRU lock (heavy work)
            sorted_nodes = topological_sort(plan)
            job_levels = compute_job_levels(sorted_nodes, plan)
            result = (sorted_nodes, job_levels)

            # Store under LRU lock
            with self._lru_lock:
                self._cache[dag_id] = result
            return result

    def clear_for(self, dag_id: str) -> None:
        with self._lru_lock:
            self._cache.pop(dag_id, None)

    def clear(self) -> None:
        with self._lru_lock:
            self._cache.clear()
