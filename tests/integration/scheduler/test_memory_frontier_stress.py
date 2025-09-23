import os
import time

import pytest

from marie.scheduler.memory_frontier import MemoryFrontier

# ------------ Config ------------
RUN_STRESS = os.getenv("RUN_STRESS", "0") == "1"
TOTAL_ITEMS = int(os.getenv("STRESS_TOTAL_ITEMS", str(2_500_000)))  # 2.5M default
CHUNK = int(os.getenv("STRESS_CHUNK", "100000"))                    # add in 100k chunks
TOP_K = int(os.getenv("STRESS_TOP_K", "2048"))                      # top-k to fetch
SCAN_BUDGET = int(os.getenv("STRESS_SCAN_BUDGET", "4096"))          # for select_ready


# ------------ Lightweight WorkInfo stub ------------
class WI:
    __slots__ = ("id", "name", "dag_id", "job_level", "priority", "dependencies", "data")
    def __init__(self, jid: str, level: int, pri: int, exe: str):
        self.id = jid
        self.name = "job"
        self.dag_id = "D1"
        self.job_level = level
        self.priority = pri
        self.dependencies = []
        self.data = {"metadata": {"on": exe}}


def _mk_job(i: int) -> WI:
    # Mix levels and priorities a bit so ordering & tie-breaks are realistic.
    level = (i % 6)         # 0..5
    pri = 1 + (i % 5)       # 1..5
    # Simulate multiple executors, some “blocked”
    exe_name = "exe://ok" if (i % 7) else "exe://blocked"
    return WI(f"J{i}", level, pri, exe_name)


@pytest.fixture(scope="module")
def frontier_stress():
    if not RUN_STRESS:
        pytest.skip("Set RUN_STRESS=1 to run 2.5M-item stress tests")
    f = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)

    # Stream-load in chunks to avoid huge temporary lists
    remaining = TOTAL_ITEMS
    idx = 0
    while remaining > 0:
        n = min(CHUNK, remaining)
        batch = [_mk_job(idx + j) for j in range(n)]
        # Use public API; add roots by calling add_dag with no deps
        f.add_dag(None, batch)
        remaining -= n
        idx += n
    return f


@pytest.mark.asyncio
async def test_peek_ready_topk_timing(frontier_stress, record_property):
    """
    Measure peek_ready(TOP_K) on ~2.5M ready items.
    Asserts only correctness/shape; duration is recorded as a property.
    """
    start = time.perf_counter()
    out = await frontier_stress.peek_ready(TOP_K)
    dur = time.perf_counter() - start
    record_property("peek_ready_seconds", f"{dur:.4f}")

    # Sanity asserts (not performance thresholds)
    assert len(out) == TOP_K
    ids = [wi.id for wi in out]
    assert len(ids) == len(set(ids))  # unique
    # Ordered by (-level, -priority, age, seq) — we can't easily assert that here
    # without poking internals; correctness is covered in unit tests.


@pytest.mark.asyncio
async def test_select_ready_topk_timing_slot_filter(frontier_stress, record_property):
    """
    Measure select_ready(TOP_K) with a slot-aware filter and bounded scan_budget.
    Ensures we can skip past blocked heads and still get TOP_K runnable items.
    """
    def slot_filter(wi):
        ep = wi.data.get("metadata", {}).get("on", "")
        # reject "blocked" executor
        return not ep.endswith("blocked")

    start = time.perf_counter()
    picked = await frontier_stress.select_ready(
        TOP_K, filter_fn=slot_filter, lease_ttl=0.1, scan_budget=SCAN_BUDGET
    )
    dur = time.perf_counter() - start
    record_property("select_ready_seconds", f"{dur:.4f}")

    assert len(picked) == TOP_K
    pids = [wi.id for wi in picked]
    assert len(pids) == len(set(pids))
    # Ensure none of the picked are from the blocked executor
    for wi in picked:
        assert not wi.data["metadata"]["on"].endswith("blocked")


@pytest.mark.asyncio
async def test_compact_ready_heap_timing(frontier_stress, record_property):
    """
    Simulate some stale entries and measure compaction cost. We only check it runs
    and removes something; duration is informational via record_property.
    """
    # Drop ~1% from ready_set to simulate stales
    sample = 0
    target = max(1, TOTAL_ITEMS // 100)  # ~1%
    # NOTE: accessing internal state is OK in a stress test; unit tests verify contract.
    for jid in list(frontier_stress._ready_set):
        frontier_stress._remove_from_ready_set(jid)
        sample += 1
        if sample >= target:
            break

    start = time.perf_counter()
    removed = frontier_stress.compact_ready_heap(max_scan=200_000)  # bounded
    dur = time.perf_counter() - start
    record_property("compact_ready_heap_seconds", f"{dur:.4f}")
    assert removed >= 1

# To run the stress tests (takes a few minutes):
# RUN_STRESS=1 pytest -q tests/test_memory_frontier_stress.py

# Tune size if needed:
# RUN_STRESS=1 STRESS_TOTAL_ITEMS=1500000 STRESS_CHUNK=50000 STRESS_TOP_K=4096 \
#   pytest -q tests/test_memory_frontier_stress.py