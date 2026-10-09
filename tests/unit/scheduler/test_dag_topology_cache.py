from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import pytest

import marie.scheduler.dag_topology_cache as topology_module
from marie.query_planner.base import Query, QueryPlan
from marie.scheduler.dag_topology_cache import DagTopologyCache


@pytest.fixture
def plan() -> QueryPlan:
    return QueryPlan(
        nodes=[
            Query(task_id='root', query_str='root'),
            Query(task_id='leaf', query_str='leaf', dependencies=['root']),
        ]
    )


def test_completed_build_locks_do_not_accumulate_with_dag_churn(plan: QueryPlan) -> None:
    cache = DagTopologyCache(maxsize=2)

    for index in range(128):
        assert cache.get_sorted_nodes_and_levels(plan, f'dag-{index}') == (
            ['root', 'leaf'],
            {'root': 1, 'leaf': 0},
        )
    gc.collect()

    assert len(cache._cache) == 2
    assert len(cache._build_locks) == 0


@pytest.mark.parametrize('clear_all', [True, False])
def test_cache_clearing_does_not_retain_unused_build_locks(
    plan: QueryPlan, clear_all: bool
) -> None:
    cache = DagTopologyCache(maxsize=2)
    cache.get_sorted_nodes_and_levels(plan, 'dag-1')
    cache.get_sorted_nodes_and_levels(plan, 'dag-2')

    if clear_all:
        cache.clear()
    else:
        cache.clear_for('dag-1')
    gc.collect()

    assert len(cache._cache) == (0 if clear_all else 1)
    assert len(cache._build_locks) == 0


@pytest.mark.parametrize('clear_all', [True, False])
def test_concurrent_builds_share_live_lock_during_cache_clearing(
    plan: QueryPlan, clear_all: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = DagTopologyCache(maxsize=1)
    release = threading.Event()
    lock_refs: Queue[weakref.ReferenceType] = Queue()
    sort_calls = 0
    count_guard = threading.Lock()
    original_sort = topology_module.topological_sort
    original_build_lock_for = cache._build_lock_for

    def observe_build_lock(dag_id: str) -> threading.Lock:
        lock = original_build_lock_for(dag_id)
        lock_refs.put(weakref.ref(lock))
        return lock

    def slow_sort(query_plan: QueryPlan) -> list[str]:
        nonlocal sort_calls
        with count_guard:
            sort_calls += 1
        assert release.wait(timeout=2), 'Test did not release topology computation'
        return original_sort(query_plan)

    monkeypatch.setattr(cache, '_build_lock_for', observe_build_lock)
    monkeypatch.setattr(topology_module, 'topological_sort', slow_sort)

    with ThreadPoolExecutor(max_workers=3) as workers:
        first = workers.submit(cache.get_sorted_nodes_and_levels, plan, 'dag-1')
        try:
            first_lock = lock_refs.get(timeout=1)
            if clear_all:
                cache.clear()
            else:
                cache.clear_for('dag-1')
            waiters = [
                workers.submit(cache.get_sorted_nodes_and_levels, plan, 'dag-1')
                for _ in range(2)
            ]
            waiter_locks = [lock_refs.get(timeout=1) for _ in waiters]
            assert all(lock_ref() is first_lock() for lock_ref in waiter_locks)
        finally:
            release.set()
        results = [future.result(timeout=2) for future in [first, *waiters]]

    assert results == [(['root', 'leaf'], {'root': 1, 'leaf': 0})] * 3
    assert sort_calls == 1
    gc.collect()
    assert len(cache._build_locks) == 0


def test_failed_build_does_not_retain_lock() -> None:
    cache = DagTopologyCache(maxsize=1)
    cyclic_plan = QueryPlan(
        nodes=[Query(task_id='cycle', query_str='cycle', dependencies=['cycle'])]
    )

    with pytest.raises(ValueError, match='cycle'):
        cache.get_sorted_nodes_and_levels(cyclic_plan, 'invalid-dag')
    gc.collect()

    assert len(cache._cache) == 0
    assert len(cache._build_locks) == 0
