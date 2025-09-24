import asyncio
from types import SimpleNamespace as NS

import pytest

from marie.scheduler.memory_frontier import MemoryFrontier


def wi_factory(
        jid: str,
        *,
        dag_id: str = "D1",
        name: str = "job",
        job_level: int = 0,
        priority: int = 1,
        deps=None,
        executor: str = "exe://default",
):
    """Duck-typed WorkInfo with only the fields MemoryFrontier touches."""
    return NS(
        id=jid,
        name=name,
        dag_id=dag_id,
        job_level=job_level,
        priority=priority,
        dependencies=list(deps or []),
        data={"metadata": {"on": executor}},
    )


@pytest.fixture
def frontier():
    return MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)


async def add_ready_jobs(frontier: MemoryFrontier, *jobs):
    # add into a fake "dag" (the function doesn't actually use the dag object)
    await frontier.add_dag(None, list(jobs))


@pytest.mark.asyncio
async def test_ordering_level_priority_age(frontier: MemoryFrontier):
    """
    Ensures peek_ready returns true heap order by (-level, -priority, added_at, seq).
    With same priority, higher job_level wins; with exact ties, older (smaller added_at) wins.
    """
    a = wi_factory("A", job_level=0, priority=1)
    c = wi_factory("C", job_level=2, priority=1)  # insert C first → older than B
    b = wi_factory("B", job_level=2, priority=1)
    d = wi_factory("D", job_level=1, priority=1)

    # Add roots in an order that makes C older than B
    await add_ready_jobs(frontier, a, c, b, d)

    out = await frontier.peek_ready(4)
    ids = [wi.id for wi in out]
    assert ids == ["C", "B", "D", "A"]


@pytest.mark.asyncio
async def test_ordering_with_priority_tie_break(frontier: MemoryFrontier):
    """
    If levels equal, higher priority wins; same level+priority -> age/FIFO.
    """
    x1 = wi_factory("X1", job_level=3, priority=1)
    x2 = wi_factory("X2", job_level=3, priority=5)
    x3 = wi_factory("X3", job_level=3, priority=5)
    await add_ready_jobs(frontier, x1, x2, x3)

    # Make X2 older than X3 to break tie within priority=5
    frontier._added_at["X2"] -= 5.0

    out = await frontier.peek_ready(3)
    assert [wi.id for wi in out] == ["X2", "X3", "X1"]


@pytest.mark.asyncio
async def test_compact_ready_heap_removes_stale(frontier: MemoryFrontier):
    j1 = wi_factory("S1")
    j2 = wi_factory("S2")
    j3 = wi_factory("S3")
    await add_ready_jobs(frontier, j1, j2, j3)

    # Simulate staleness: drop S2 from ready_set (e.g., removed/leased elsewhere)
    frontier._remove_from_ready_set("S2")

    before = len(frontier._ready_heap)
    removed = await frontier.compact_ready_heap(max_scan=10000)
    after = len(frontier._ready_heap)

    assert removed >= 1
    assert after <= before - removed
    # Remaining heap items correspond to still-ready ids
    heap_ids = {t[3] for t in frontier._ready_heap}
    assert "S2" not in heap_ids


@pytest.mark.asyncio
async def test_soft_lease_excludes_until_expiry(frontier: MemoryFrontier):
    j1 = wi_factory("L1")
    j2 = wi_factory("L2")
    await add_ready_jobs(frontier, j1, j2)

    # Soft-lease L1 for ~0.2s
    await frontier.mark_leased("L1", ttl_s=0.2)

    # While leased, peek should not return L1
    out1 = await frontier.peek_ready(2)
    ids1 = [wi.id for wi in out1]
    assert "L1" not in ids1
    assert "L2" in ids1

    # Wait for lease to expire and reap
    await asyncio.sleep(0.25)
    readded = await frontier.reap_expired_soft_leases()
    assert readded >= 1

    out2 = await frontier.peek_ready(2)
    ids2 = [wi.id for wi in out2]
    assert "L1" in ids2
    assert "L2" in ids2


@pytest.mark.asyncio
async def test_release_lease_local_preserves_added_at(frontier: MemoryFrontier):
    j = wi_factory("R1")
    await add_ready_jobs(frontier, j)

    # Take it out of ready via soft lease
    await frontier.mark_leased("R1", ttl_s=1.0)
    t_before = frontier._added_at["R1"]

    # Release the lease locally (should push back preserving added_at)
    await frontier.release_lease_local("R1")
    assert frontier._added_at["R1"] == pytest.approx(t_before)

    # And it should be peekable again
    out = await frontier.peek_ready(1)
    assert out and out[0].id == "R1"


@pytest.mark.asyncio
async def test_select_ready_scan_budget_skips_blocked_heads(frontier: MemoryFrontier):
    """
    Ensure select_ready can skip non-eligible heads (via filter_fn) up to scan_budget,
    returning deeper eligible items and restoring skipped ones.
    """
    # First 5 jobs are for an executor with 0 slots (blocked), last 2 are runnable
    blocked = [wi_factory(f"B{i}", executor="exe://blocked") for i in range(5)]
    runnable = [wi_factory("OK1", executor="exe://ok"), wi_factory("OK2", executor="exe://ok")]
    await add_ready_jobs(frontier, *(blocked + runnable))

    # Filter that rejects blocked executor
    def filter_fn(wi):
        ep = wi.data.get("metadata", {}).get("on", "")
        exe = ep.split("://", 1)[0] if "://" in ep else ep
        return exe != "exe" or (ep.endswith("ok"))

    picked = await frontier.select_ready(
        2, filter_fn=filter_fn, lease_ttl=0.2, scan_budget=64
    )
    assert [wi.id for wi in picked] == ["OK1", "OK2"]

    # The blocked heads must be restored to the heap (still present, not selected)
    heap_ids = {t[4] for t in frontier._ready_heap}
    for bj in blocked:
        assert bj.id in heap_ids


@pytest.mark.asyncio
async def test_take_only_ready_subset_and_order(frontier: MemoryFrontier):
    """
    take(ids) should only lease and return those still ready (and in the given order).
    """
    a = wi_factory("TA")
    b = wi_factory("TB")
    c = wi_factory("TC")
    await add_ready_jobs(frontier, a, b, c)

    # Make B non-ready by removing from ready_set (simulate race)
    frontier._remove_from_ready_set("TB")

    got = await frontier.take(["TB", "TC", "TA"], lease_ttl=0.1)
    assert [wi.id for wi in got] == ["TC", "TA"]  # TB skipped; order preserved

    # The leased ones should not appear in peek until lease expires
    out = await frontier.peek_ready(5)
    ids = [wi.id for wi in out]
    assert "TC" not in ids and "TA" not in ids
    assert "TB" not in ids  # TB remains not ready

    await asyncio.sleep(0.15)
    await frontier.reap_expired_soft_leases()
    out2 = await frontier.peek_ready(5)
    ids2 = [wi.id for wi in out2]
    # Only the previously leased ones reappear (TB still not ready)
    assert "TA" in ids2 and "TC" in ids2 and "TB" not in ids2
