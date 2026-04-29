import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

import marie.scheduler.memory_frontier as memory_frontier_module
from marie.scheduler.memory_frontier import MemoryFrontier


def wi_factory(
        jid: str,
        *,
        dag_id: str = "D1",
        name: str = "job",
        job_level: int = 0,
        priority: int = 1,
        state: str = "created",
        soft_sla=None,
        hard_sla=None,
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
        state=state,
        soft_sla=soft_sla,
        hard_sla=hard_sla,
        start_after=None,
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
async def test_zero_priority_is_stable_in_frontier(frontier: MemoryFrontier):
    """
    Default priority=0 should behave like a neutral value, not a special case.
    Level still wins, then age/FIFO.
    """
    low = wi_factory("Z1", job_level=1, priority=0)
    high = wi_factory("Z2", job_level=2, priority=0)
    tie = wi_factory("Z3", job_level=2, priority=0)
    await add_ready_jobs(frontier, low, high, tie)

    out = await frontier.peek_ready(3)
    assert [wi.id for wi in out] == ["Z2", "Z3", "Z1"]


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


@pytest.mark.asyncio
async def test_summary_includes_sla_snapshot(frontier: MemoryFrontier):
    now = datetime.now(timezone.utc)
    overdue_hard = wi_factory(
        "H1",
        priority=1,
        soft_sla=now - timedelta(hours=2),
        hard_sla=now - timedelta(minutes=31),
    )
    overdue_soft = wi_factory(
        "S1",
        priority=20,
        soft_sla=now - timedelta(minutes=16),
        hard_sla=now + timedelta(hours=2),
    )
    approaching = wi_factory(
        "A1",
        priority=5,
        soft_sla=now + timedelta(minutes=10),
        hard_sla=now + timedelta(hours=1),
    )
    completed = wi_factory(
        "C1",
        state="completed",
        soft_sla=now - timedelta(days=1),
        hard_sla=now - timedelta(hours=1),
    )

    await add_ready_jobs(frontier, overdue_hard, overdue_soft, approaching, completed)

    summary = frontier.summary(detail=True, top_n=3)
    sla = summary["sla"]

    assert sla["tracked"] == 3
    assert sla["approaching_soft"] == 1
    assert sla["soft_missed"] == 1
    assert sla["hard_missed"] == 1
    assert sla["highest_bucket"] >= 1002
    assert [row["id"] for row in sla["top_urgent"]] == ["H1", "S1", "A1"]


@pytest.mark.asyncio
async def test_refresh_priorities_rebuilds_ready_heap(frontier: MemoryFrontier):
    low = wi_factory("P1", job_level=1, priority=1)
    high = wi_factory("P2", job_level=1, priority=5)
    await add_ready_jobs(frontier, low, high)

    before = await frontier.peek_ready(2)
    assert [wi.id for wi in before] == ["P2", "P1"]

    changed = await frontier.refresh_priorities({"P1": 100, "P2": 5})
    assert changed == 1

    after = await frontier.peek_ready(2)
    assert [wi.id for wi in after] == ["P1", "P2"]


@pytest.mark.asyncio
async def test_refresh_priorities_keeps_zero_when_db_has_zero(frontier: MemoryFrontier):
    first = wi_factory("Q1", job_level=2, priority=0)
    second = wi_factory("Q2", job_level=1, priority=0)
    await add_ready_jobs(frontier, first, second)

    changed = await frontier.refresh_priorities({"Q1": 0, "Q2": 0})
    assert changed == 0

    out = await frontier.peek_ready(2)
    assert [wi.id for wi in out] == ["Q1", "Q2"]


@pytest.mark.asyncio
async def test_completed_job_not_resurrected_after_lease_expires(frontier: MemoryFrontier):
    """Completed jobs must not re-enter ready queue when soft lease expires."""
    job = wi_factory("resurrection_test", priority=1)
    await add_ready_jobs(frontier, job)

    # Take the job (applies soft lease)
    taken = await frontier.take(["resurrection_test"], lease_ttl=0.05)
    assert len(taken) == 1

    # Complete the job
    await frontier.on_job_completed("resurrection_test")

    # Wait for lease to expire
    await asyncio.sleep(0.1)

    # Reap expired leases - should NOT resurrect the completed job
    await frontier.reap_expired_soft_leases()

    # Job should NOT be in ready queue
    ready = await frontier.peek_ready(10)
    assert "resurrection_test" not in [wi.id for wi in ready]


@pytest.mark.asyncio
async def test_failed_job_not_resurrected_after_lease_expires(frontier: MemoryFrontier):
    """Failed jobs must not re-enter ready queue when soft lease expires."""
    job = wi_factory("failed_resurrection", priority=1)
    await add_ready_jobs(frontier, job)

    # Take the job (applies soft lease)
    taken = await frontier.take(["failed_resurrection"], lease_ttl=0.05)
    assert len(taken) == 1

    # Fail the job
    await frontier.on_job_failed("failed_resurrection")

    # Wait for lease to expire
    await asyncio.sleep(0.1)

    # Reap expired leases - should NOT resurrect the failed job
    await frontier.reap_expired_soft_leases()

    # Job should NOT be in ready queue
    ready = await frontier.peek_ready(10)
    assert "failed_resurrection" not in [wi.id for wi in ready]


@pytest.mark.asyncio
async def test_terminal_job_not_returned_by_still_ready(frontier: MemoryFrontier):
    """Terminal jobs should not pass the _still_ready check."""
    job = wi_factory("terminal_test", state="completed", priority=1)
    await add_ready_jobs(frontier, job)

    # Even though job is in ready set, it should not be peeked because it's terminal
    ready = await frontier.peek_ready(10)
    assert "terminal_test" not in [wi.id for wi in ready]


@pytest.mark.asyncio
async def test_release_lease_does_not_requeue_terminal_job(frontier: MemoryFrontier):
    """Releasing lease on terminal job should not re-add to ready queue."""
    job = wi_factory("release_test", priority=1)
    await add_ready_jobs(frontier, job)

    # Take and complete
    await frontier.take(["release_test"], lease_ttl=60.0)
    await frontier.on_job_completed("release_test")

    # Explicitly release lease (simulating a code path that releases manually)
    await frontier.release_lease_local("release_test")

    # Job should NOT be in ready queue
    ready = await frontier.peek_ready(10)
    assert "release_test" not in [wi.id for wi in ready]


@pytest.mark.asyncio
async def test_active_job_not_requeued_after_lease_expires(frontier: MemoryFrontier):
    job = wi_factory("active_reap_test", priority=1)
    await add_ready_jobs(frontier, job)

    taken = await frontier.take(["active_reap_test"], lease_ttl=0.05)
    assert len(taken) == 1

    await frontier.update_job_state("active_reap_test", "active")
    await asyncio.sleep(0.1)
    await frontier.reap_expired_soft_leases()

    ready = await frontier.peek_ready(10)
    assert "active_reap_test" not in [wi.id for wi in ready]


@pytest.mark.asyncio
async def test_release_lease_does_not_requeue_active_job(frontier: MemoryFrontier):
    job = wi_factory("active_release_test", priority=1)
    await add_ready_jobs(frontier, job)

    await frontier.take(["active_release_test"], lease_ttl=60.0)
    await frontier.update_job_state("active_release_test", "active")
    await frontier.release_lease_local("active_release_test")

    ready = await frontier.peek_ready(10)
    assert "active_release_test" not in [wi.id for wi in ready]


@pytest.mark.asyncio
async def test_on_job_skipped_sets_state_and_removes_from_ready(frontier: MemoryFrontier):
    """on_job_skipped() should set state to SKIPPED and remove from ready queue."""
    job = wi_factory("skip_test", priority=1)
    await add_ready_jobs(frontier, job)

    # Job should be ready initially
    ready = await frontier.peek_ready(10)
    assert "skip_test" in [wi.id for wi in ready]

    # Skip the job
    await frontier.on_job_skipped("skip_test")

    # Job should NOT be in ready queue
    ready = await frontier.peek_ready(10)
    assert "skip_test" not in [wi.id for wi in ready]

    # State should be SKIPPED (could be WorkState enum or string)
    wi = frontier.jobs_by_id["skip_test"]
    state_val = wi.state.value if hasattr(wi.state, "value") else wi.state
    assert state_val == "skipped"


@pytest.mark.asyncio
async def test_on_job_skipped_does_not_unblock_children(frontier: MemoryFrontier):
    """on_job_skipped() must NOT decrement dependency counts or unblock children."""
    # Create parent and child with dependency
    parent = wi_factory("parent", priority=1)
    child = wi_factory("child", priority=1, deps=["parent"])

    await add_ready_jobs(frontier, parent, child)

    # Child should not be ready (blocked by parent)
    ready = await frontier.peek_ready(10)
    ready_ids = [wi.id for wi in ready]
    assert "parent" in ready_ids
    assert "child" not in ready_ids

    # Skip the parent (should NOT unblock child)
    await frontier.on_job_skipped("parent")

    # Child should STILL not be ready (skip doesn't unblock)
    ready = await frontier.peek_ready(10)
    ready_ids = [wi.id for wi in ready]
    assert "parent" not in ready_ids  # Parent removed from ready
    assert "child" not in ready_ids   # Child still blocked

    # Verify unmet_count was NOT decremented
    assert frontier.unmet_count.get("child", 0) > 0


@pytest.mark.asyncio
async def test_on_job_completed_does_unblock_children(frontier: MemoryFrontier):
    """on_job_completed() SHOULD unblock children (contrast with on_job_skipped)."""
    # Create parent and child with dependency
    parent = wi_factory("parent2", priority=1)
    child = wi_factory("child2", priority=1, deps=["parent2"])

    await add_ready_jobs(frontier, parent, child)

    # Child should not be ready (blocked by parent)
    ready = await frontier.peek_ready(10)
    ready_ids = [wi.id for wi in ready]
    assert "parent2" in ready_ids
    assert "child2" not in ready_ids

    # Complete the parent (SHOULD unblock child)
    await frontier.on_job_completed("parent2")

    # Child should now be ready
    ready = await frontier.peek_ready(10)
    ready_ids = [wi.id for wi in ready]
    assert "parent2" not in ready_ids  # Parent removed from ready
    assert "child2" in ready_ids       # Child unblocked

    # Verify unmet_count was decremented to 0
    assert frontier.unmet_count.get("child2", 0) == 0


@pytest.mark.asyncio
async def test_skipped_job_not_resurrected_after_lease_expires(frontier: MemoryFrontier):
    """Skipped jobs must not re-enter ready queue when soft lease expires."""
    job = wi_factory("skip_resurrection", priority=1)
    await add_ready_jobs(frontier, job)

    # Take the job (applies soft lease)
    taken = await frontier.take(["skip_resurrection"], lease_ttl=0.05)
    assert len(taken) == 1

    # Skip the job
    await frontier.on_job_skipped("skip_resurrection")

    # Wait for lease to expire
    await asyncio.sleep(0.1)

    # Reap expired leases - should NOT resurrect the skipped job
    await frontier.reap_expired_soft_leases()

    # Job should NOT be in ready queue
    ready = await frontier.peek_ready(10)
    assert "skip_resurrection" not in [wi.id for wi in ready]


@pytest.mark.asyncio
async def test_on_job_retry_clears_lease_so_retry_delay_is_honored():
    """Retry delay should be honored exactly, not blocked by old lease."""
    # Use a frontier with a long default lease to make the bug obvious
    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=5.0)

    job = wi_factory("retry_test", priority=1)
    await add_ready_jobs(frontier, job)

    # Take the job with a long lease (5 seconds)
    taken = await frontier.take(["retry_test"], lease_ttl=5.0)
    assert len(taken) == 1

    # Verify job is now leased and not ready
    ready = await frontier.peek_ready(10)
    assert "retry_test" not in [wi.id for wi in ready]

    # Verify lease is set
    assert "retry_test" in frontier.leased_until

    # Create a work_item with short retry_delay for the retry call
    retry_work_item = NS(retry_delay=0.1)  # 100ms retry delay

    # Call on_job_retry - this should clear the lease
    await frontier.on_job_retry("retry_test", retry_work_item)

    # Lease should be cleared
    assert "retry_test" not in frontier.leased_until

    # Job should NOT be ready yet (start_after hasn't passed)
    ready = await frontier.peek_ready(10)
    assert "retry_test" not in [wi.id for wi in ready]

    # Wait for retry_delay to pass (plus small buffer)
    await asyncio.sleep(0.15)

    # Job should now be ready (if lease was still blocking, it would take 5s)
    ready = await frontier.peek_ready(10)
    assert "retry_test" in [wi.id for wi in ready]


@pytest.mark.asyncio
async def test_on_job_retry_without_clearing_lease_would_delay():
    """Demonstrates the bug scenario: if lease wasn't cleared, retry would be late."""
    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.5)

    job = wi_factory("retry_timing", priority=1)
    await add_ready_jobs(frontier, job)

    # Take with 0.5s lease
    await frontier.take(["retry_timing"], lease_ttl=0.5)

    # Retry with 0.1s delay - should be ready in 0.1s, not 0.5s
    retry_work_item = NS(retry_delay=0.1)
    await frontier.on_job_retry("retry_timing", retry_work_item)

    # After 0.15s, job should be ready (retry_delay passed)
    await asyncio.sleep(0.15)
    ready = await frontier.peek_ready(10)
    assert "retry_timing" in [wi.id for wi in ready], \
        "Job should be ready after retry_delay, not blocked by old lease"


@pytest.mark.asyncio
async def test_heap_ordering_matches_planner_priority_before_level(frontier: MemoryFrontier):
    """Heap must order by priority before level to match GlobalPriorityExecutionPlanner."""
    deep_low_pri = wi_factory("deep", job_level=10, priority=0)
    shallow_high_pri = wi_factory("shallow", job_level=1, priority=100)

    await add_ready_jobs(frontier, deep_low_pri, shallow_high_pri)

    ready = await frontier.peek_ready(2)
    ids = [wi.id for wi in ready]
    assert ids[0] == "shallow", "High-priority shallow job should come before low-priority deep job"


@pytest.mark.asyncio
async def test_heap_ordering_matches_planner_sla_before_level(frontier: MemoryFrontier):
    """Heap must order by SLA urgency before level to match GlobalPriorityExecutionPlanner."""
    now = datetime.now(timezone.utc)
    deep_no_sla = wi_factory("deep", job_level=10, priority=0)
    shallow_urgent = wi_factory(
        "shallow",
        job_level=1,
        priority=0,
        hard_sla=now - timedelta(hours=1),  # missed hard SLA
    )

    await add_ready_jobs(frontier, deep_no_sla, shallow_urgent)

    ready = await frontier.peek_ready(2)
    ids = [wi.id for wi in ready]
    assert ids[0] == "shallow", "Urgent shallow job should come before non-urgent deep job"


@pytest.mark.asyncio
async def test_heap_ordering_priority_then_sla_then_level(frontier: MemoryFrontier):
    """Full ordering: priority → SLA → level (matches GlobalPriorityExecutionPlanner)."""
    now = datetime.now(timezone.utc)

    # priority=0, no SLA, level=10
    deep_neutral = wi_factory("A", job_level=10, priority=0)
    # priority=0, hard SLA missed, level=1
    shallow_urgent = wi_factory("B", job_level=1, priority=0, hard_sla=now - timedelta(hours=1))
    # priority=50, no SLA, level=1
    shallow_high_pri = wi_factory("C", job_level=1, priority=50)
    # priority=50, hard SLA missed, level=5
    mid_urgent_high_pri = wi_factory("D", job_level=5, priority=50, hard_sla=now - timedelta(hours=2))

    await add_ready_jobs(frontier, deep_neutral, shallow_urgent, shallow_high_pri, mid_urgent_high_pri)

    ready = await frontier.peek_ready(4)
    ids = [wi.id for wi in ready]
    # D: priority=50, SLA=high → first
    # C: priority=50, SLA=0 → second (same priority, lower SLA)
    # B: priority=0, SLA=high → third
    # A: priority=0, SLA=0 → last
    assert ids == ["D", "C", "B", "A"]


@pytest.mark.asyncio
async def test_refresh_ready_ordering_rebuilds_heap_for_time_based_sla_changes(
    frontier: MemoryFrontier, monkeypatch
):
    buckets = {"deep": 0, "shallow": 0}

    def fake_bucket(_now, _soft_sla, hard_sla):
        return buckets[hard_sla]

    monkeypatch.setattr(
        memory_frontier_module,
        "compute_sla_priority_bucket",
        fake_bucket,
    )

    deep = wi_factory("deep", job_level=10, priority=0, hard_sla="deep")
    shallow = wi_factory("shallow", job_level=1, priority=0, hard_sla="shallow")
    await add_ready_jobs(frontier, deep, shallow)

    ready_before = await frontier.peek_ready(2)
    assert [wi.id for wi in ready_before] == ["deep", "shallow"]

    buckets["shallow"] = 1000
    await frontier.refresh_ready_ordering()

    ready_after = await frontier.peek_ready(2)
    assert [wi.id for wi in ready_after] == ["shallow", "deep"]


def generate_mixed_sla_jobs(count: int, seed: int = 42):
    """Generate jobs with mixed SLA configurations (25% each category)."""
    import random
    random.seed(seed)
    now = datetime.now(timezone.utc)
    jobs = []
    digits = len(str(count))
    for i in range(count):
        sla_type = i % 4
        if sla_type == 0:
            soft_sla, hard_sla = None, None
        elif sla_type == 1:
            soft_sla = now + timedelta(minutes=random.randint(5, 120))
            hard_sla = soft_sla + timedelta(hours=1)
        elif sla_type == 2:
            soft_sla = now - timedelta(minutes=random.randint(1, 60))
            hard_sla = now + timedelta(hours=random.randint(1, 4))
        else:
            soft_sla = now - timedelta(hours=random.randint(1, 5))
            hard_sla = now - timedelta(minutes=random.randint(1, 60))
        jobs.append(wi_factory(
            f"job_{i:0{digits}d}",
            job_level=random.randint(0, 10),
            priority=random.randint(0, 100),
            soft_sla=soft_sla,
            hard_sla=hard_sla,
        ))
    return jobs


async def run_frontier_stress_test(job_count: int, thresholds: dict):
    """Run stress test with given job count and performance thresholds."""
    import time

    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=60.0)
    jobs = generate_mixed_sla_jobs(job_count)

    start = time.monotonic()
    await add_ready_jobs(frontier, *jobs)
    add_time = time.monotonic() - start
    assert len(frontier.jobs_by_id) == job_count
    assert add_time < thresholds["add"], f"Adding {job_count} jobs took {add_time:.2f}s"

    start = time.monotonic()
    top_100 = await frontier.peek_ready(100)
    peek_time = time.monotonic() - start
    assert len(top_100) == 100
    assert peek_time < thresholds["peek"], f"Peeking 100 jobs took {peek_time:.2f}s"

    for i in range(len(top_100) - 1):
        assert top_100[i].priority >= top_100[i + 1].priority, f"Priority ordering violated at {i}"

    start = time.monotonic()
    await frontier.refresh_ready_ordering()
    refresh_time = time.monotonic() - start
    assert refresh_time < thresholds["refresh"], f"Refresh took {refresh_time:.2f}s"

    summary = frontier.summary(detail=True, top_n=10)
    sla = summary["sla"]
    assert sla["tracked"] == job_count
    assert sla["no_sla"] == job_count // 4
    assert sla["hard_missed"] == job_count // 4
    assert len(sla["top_urgent"]) == 10

    select_count = thresholds.get("select_count", 50)
    start = time.monotonic()
    selected = await frontier.select_ready(select_count, lease_ttl=1.0)
    select_time = time.monotonic() - start
    assert len(selected) == select_count
    assert select_time < thresholds["select"], f"Selecting {select_count} jobs took {select_time:.2f}s"

    for wi in selected:
        assert wi.id in frontier.leased_until

    return frontier


@pytest.mark.asyncio
async def test_large_frontier_stress_10k_jobs_with_mixed_slas():
    """Stress test: 10k jobs with mixed SLAs, priorities, and levels."""
    await run_frontier_stress_test(10_000, {
        "add": 5.0,
        "peek": 1.0,
        "refresh": 2.0,
        "select": 1.0,
    })


@pytest.mark.asyncio
async def test_large_frontier_stress_100k_jobs_with_mixed_slas():
    """Stress test: 100k jobs with mixed SLAs, priorities, and levels."""
    frontier = await run_frontier_stress_test(100_000, {
        "add": 30.0,
        "peek": 2.0,
        "refresh": 15.0,
        "select": 2.0,
        "select_count": 100,
    })

    import time
    start = time.monotonic()
    for _ in range(10):
        await frontier.peek_ready(50)
    repeated_peek_time = time.monotonic() - start
    assert repeated_peek_time < 5.0, f"10 repeated peeks took {repeated_peek_time:.2f}s"


# ── dag_remaining_counts tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_dag_remaining_counts_basic(frontier: MemoryFrontier):
    """Non-terminal jobs are counted per DAG."""
    a = wi_factory("A1", dag_id="D1")
    b = wi_factory("B1", dag_id="D1")
    c = wi_factory("C1", dag_id="D2")
    await add_ready_jobs(frontier, a, b, c)

    counts = frontier.dag_remaining_counts()
    assert counts == {"D1": 2, "D2": 1}


@pytest.mark.asyncio
async def test_dag_remaining_counts_all_terminal(frontier: MemoryFrontier):
    """All-terminal DAG should report 0 remaining."""
    a = wi_factory("A1", dag_id="D1")
    b = wi_factory("B1", dag_id="D1")
    await add_ready_jobs(frontier, a, b)

    await frontier.on_job_completed("A1")
    await frontier.on_job_completed("B1")

    counts = frontier.dag_remaining_counts()
    assert counts["D1"] == 0


@pytest.mark.asyncio
async def test_dag_remaining_counts_mixed_terminal(frontier: MemoryFrontier):
    """Mix of terminal and non-terminal states counted correctly."""
    a = wi_factory("A1", dag_id="D1")
    b = wi_factory("B1", dag_id="D1")
    c = wi_factory("C1", dag_id="D1")
    await add_ready_jobs(frontier, a, b, c)

    await frontier.on_job_completed("A1")
    await frontier.on_job_failed("B1")

    counts = frontier.dag_remaining_counts()
    assert counts["D1"] == 1
