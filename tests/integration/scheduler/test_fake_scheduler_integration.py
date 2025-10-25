from datetime import datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.memory_frontier import MemoryFrontier


def wi(
        jid: str,
        *,
        dag_id: str = "D1",
        name: str = "job",
        level: int = 0,
        pri: int = 1,
        deps=None,
        exe: str = "exe://ok",
):
    return NS(
        id=jid,
        name=name,
        dag_id=dag_id,
        job_level=level,
        priority=pri,
        dependencies=list(deps or []),
        data={"metadata": {"on": exe}},
    )


# ---------- Fakes for DB + Runner ----------
class FakeDB:
    def __init__(self, leased_subset: set[str] | None = None):
        # If None, lease all requested; else only lease the provided subset.
        self.leased_subset = leased_subset

    async def lease(self, job_name_to_ids: dict[str, list[str]]) -> set[str]:
        req = {jid for ids in job_name_to_ids.values() for jid in ids}
        if self.leased_subset is None:
            return req
        return req & set(self.leased_subset)

    async def release(self, ids: list[str]) -> None:
        return

    async def activate_from_lease(self, ids: list[str]) -> set[str]:
        # In “real” flow, this may return a subset; tests can patch this method if needed.
        return set(ids)


class FakeRunner:
    """
    dispatch_outcomes: dict[jid] -> bool  (True = accepted, False = failed)
    activation_ok: set[jid] that successfully activate (others fail activate)
    """

    def __init__(self, dispatch_outcomes: dict[str, bool] | None = None, activation_ok: set[str] | None = None):
        self.dispatch_outcomes = dispatch_outcomes or {}
        self.activation_ok = activation_ok or set()

    async def enqueue(self, wi) -> bool:
        return self.dispatch_outcomes.get(wi.id, True)

    async def activate(self, jid: str) -> bool:
        return jid in self.activation_ok if self.activation_ok else True


# ---------- FakeScheduler ----------
class FakeScheduler:
    def __init__(self, frontier: MemoryFrontier, db, runner, planner: GlobalPriorityExecutionPlanner):
        self.frontier = frontier
        self.db = db
        self.runner = runner
        self.planner = planner

    @staticmethod
    def _endpoint(wi):
        # Use the endpoint stored in metadata.on (e.g., "exe://A")
        return wi.data.get("metadata", {}).get("on", "") or ""

    @staticmethod
    def _member_of_endpoint(ep: str) -> str:
        # "exe://A" -> "A"
        return ep.split("://", 1)[1] if "://" in ep else ep

    def slot_filter(self, slots_by_executor: dict[str, int]):
        def _f(wi):
            ep = self._endpoint(wi)
            member = self._member_of_endpoint(ep)
            # Allow unknown/noop members unconditionally; otherwise require available member slots
            return True if not member or member == "noop" else (slots_by_executor.get(member, 0) > 0)

        return _f

    @staticmethod
    def _ensure_min_fields_for_planner(candidates: list):
        # Ensure minimal fields that some planners expect to exist
        now = datetime.now()
        for wi in candidates:
            if not hasattr(wi, "state"):
                wi.state = "created"
            if not hasattr(wi, "retry_limit"):
                wi.retry_limit = 0
            if not hasattr(wi, "retry_delay"):
                wi.retry_delay = 0
            if not hasattr(wi, "retry_backoff"):
                wi.retry_backoff = False
            if not hasattr(wi, "start_after"):
                wi.start_after = now
            if not hasattr(wi, "expire_in_seconds"):
                wi.expire_in_seconds = 3600
            if not hasattr(wi, "keep_until"):
                wi.keep_until = now + timedelta(days=1)
            if not hasattr(wi, "dependencies"):
                wi.dependencies = []

    async def cycle_peek_plan_take_backfill_dispatch(
            self,
            *,
            slots_by_executor: dict[str, int],
            batch_size: int = 64,
            lease_ttl: float = 0.5,
            backfill_scan_budget: int = 4096,
    ) -> dict:
        """
        One scheduling cycle:
          peek → plan → take (+ backfill) → cap-to-slots → DB lease → dispatch → activate
        Returns: {"planned": list[(ep, wi)], "leased": set[str], "dispatched": set[str], "activated": set[str]}
        """
        # PEAK RUNNABLE WINDOW (slot-aware)
        free = sum(max(v, 0) for v in slots_by_executor.values())
        window = max(batch_size, min(2000, 4 * free + 64))
        filt = self.slot_filter(slots_by_executor)

        candidates = await self.frontier.peek_ready(window, filter_fn=filt)
        if not candidates:
            return {"planned": [], "leased": set(), "dispatched": set(), "activated": set()}

        # Normalize fields for planner expectations
        self._ensure_min_fields_for_planner(candidates)

        # Prepare planner candidates as (endpoint, wi)
        planner_candidates = [(self._endpoint(wi), wi) for wi in candidates]

        # Planner: provide scheme-level capacity only (e.g., "exe"), per-member caps enforced later
        schemes_present = {ep.split("://", 1)[0] for ep, _ in planner_candidates if "://" in ep}
        sum_all = sum(max(v, 0) for v in slots_by_executor.values())
        planner_slots = {sch: sum_all for sch in schemes_present}

        planned = self.planner.plan(
            planner_candidates,
            planner_slots,
            active_dags=set(),
            exclude_blocked=True,
        )
        if not planned:
            return {"planned": [], "leased": set(), "dispatched": set(), "activated": set()}

        # TAKE CHOSEN (soft-lease)
        planned_ids = [wi.id for _, wi in planned]
        chosen_wi = await self.frontier.take(planned_ids, lease_ttl=lease_ttl)

        # BACKFILL MISSES (atomic)
        missing = len(planned_ids) - len(chosen_wi)
        if missing > 0:
            refill = await self.frontier.select_ready(
                missing, filter_fn=filt, lease_ttl=lease_ttl, scan_budget=backfill_scan_budget
            )
            for wi2 in refill:
                planned.append((self._endpoint(wi2), wi2))
                chosen_wi.append(wi2)

        if not chosen_wi:
            return {"planned": planned, "leased": set(), "dispatched": set(), "activated": set()}

        # ENFORCE MEMBER-LEVEL CAPS BEFORE DB LEASE
        remaining = {k: max(v, 0) for k, v in slots_by_executor.items()}
        eligible: list = []
        overflow: list = []
        chosen_ids_set = {w.id for w in chosen_wi}

        for ep, wi in planned:
            if wi.id not in chosen_ids_set:
                continue
            member = self._member_of_endpoint(ep)
            if not member or member == "noop":
                eligible.append(wi)
                continue
            if remaining.get(member, 0) > 0:
                eligible.append(wi)
                remaining[member] -= 1
            else:
                overflow.append(wi)

        # Over-capacity → release soft lease immediately
        for wi in overflow:
            await self.frontier.release_lease_local(wi.id)

        if not eligible:
            return {"planned": planned, "leased": set(), "dispatched": set(), "activated": set()}

        # DB LEASE (only for eligible set)
        by_name: dict[str, list[str]] = {}
        for wi in eligible:
            by_name.setdefault(wi.name, []).append(wi.id)

        leased = await self.db.lease(by_name)
        if not leased:
            for wi in eligible:
                await self.frontier.release_lease_local(wi.id)
            return {"planned": planned, "leased": set(), "dispatched": set(), "activated": set()}

        # Release soft lease for non-leased
        for wi in eligible:
            if wi.id not in leased:
                await self.frontier.release_lease_local(wi.id)

        # DISPATCH → ACTIVATE
        dispatched_ok: set[str] = set()
        activated: set[str] = set()

        for ep, wi in planned:
            if wi.id not in leased:
                continue
            # dispatch
            ok = await self.runner.enqueue(wi)
            if not ok:
                await self.db.release([wi.id])
                await self.frontier.release_lease_local(wi.id)
                continue
            dispatched_ok.add(wi.id)

            # activation (runner + DB)
            if await self.runner.activate(wi.id):
                activated.add(wi.id)
            else:
                await self.db.release([wi.id])
                await self.frontier.release_lease_local(wi.id)

        return {
            "planned": planned,
            "leased": leased,
            "dispatched": dispatched_ok,
            "activated": set(activated),
        }


@pytest.fixture
def frontier():
    return MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)


async def add_ready(frontier: MemoryFrontier, jobs):
    # Clear dependencies to make jobs immediately runnable
    for j in jobs:
        j.dependencies = []

    await frontier.add_dag(None, list(jobs))


# ---------- Tests ----------

@pytest.mark.asyncio
async def test_happy_path(frontier):
    jobs = [wi(f"A{i}", level=3, exe="exe://A") for i in range(4)] + [wi(f"B{i}", level=2, exe="exe://B") for i in
                                                                      range(3)]
    await add_ready(frontier, jobs)

    slots = {"A": 2, "B": 1}
    db = FakeDB()  # lease all
    runner = FakeRunner()  # all dispatch+activate OK
    sched = FakeScheduler(frontier, db, runner, GlobalPriorityExecutionPlanner())

    res = await sched.cycle_peek_plan_take_backfill_dispatch(slots_by_executor=slots, batch_size=64, lease_ttl=0.4)
    # Expect 3 scheduled (2 A + 1 B)
    assert len(res["dispatched"]) == 3
    assert res["dispatched"] == res["activated"]
    # Leased set should include exactly dispatched/activated ids
    assert res["leased"] == res["activated"]


@pytest.mark.asyncio
async def test_backfill_on_take_race(frontier):
    # Two runnable for A, and extras to backfill
    runnable = [wi("A0", level=5, exe="exe://A"), wi("A1", level=5, exe="exe://A")]
    extra = [wi(f"Ae{i}", level=4, exe="exe://A") for i in range(3)]
    await add_ready(frontier, runnable + extra)

    # Simulate race: remove A0 from ready_set before cycle
    frontier._remove_from_ready_set("A0")

    slots = {"A": 2}
    db = FakeDB()  # lease all
    runner = FakeRunner()
    sched = FakeScheduler(frontier, db, runner, GlobalPriorityExecutionPlanner())

    res = await sched.cycle_peek_plan_take_backfill_dispatch(slots_by_executor=slots, batch_size=16, lease_ttl=0.3)
    # Despite losing A0 at take-time, we should backfill and still dispatch 2
    assert len(res["dispatched"]) == 2
    assert res["dispatched"] == res["activated"]


@pytest.mark.asyncio
async def test_partial_db_lease_releases_non_leased(frontier):
    jobs = [wi(f"A{i}", level=3, exe="exe://A") for i in range(4)]
    await add_ready(frontier, jobs)

    slots = {"A": 3}
    # DB will only lease the first two IDs lexicographically
    lease_subset = {"A0", "A1"}
    db = FakeDB(leased_subset=lease_subset)
    runner = FakeRunner()
    sched = FakeScheduler(frontier, db, runner, GlobalPriorityExecutionPlanner())

    res = await sched.cycle_peek_plan_take_backfill_dispatch(slots_by_executor=slots, batch_size=32, lease_ttl=0.3)
    # Only leased subset can proceed
    assert res["leased"] == lease_subset
    assert res["dispatched"] <= res["leased"]
    # Non-leased must have had their soft leases released => show up in peek
    peek_ids = {wi.id for wi in await frontier.peek_ready(10)}
    assert ("A2" in peek_ids) or ("A3" in peek_ids)


@pytest.mark.asyncio
async def test_dispatch_and_activation_failures_release(frontier):
    jobs = [wi(f"A{i}", level=4, exe="exe://A") for i in range(3)]
    await add_ready(frontier, jobs)

    slots = {"A": 3}
    db = FakeDB()  # lease all
    # A0: dispatch fails; A1: dispatch ok but activation fails; A2: success
    dispatch_outcomes = {"A0": False, "A1": True, "A2": True}
    activation_ok = {"A2"}
    runner = FakeRunner(dispatch_outcomes=dispatch_outcomes, activation_ok=activation_ok)
    sched = FakeScheduler(frontier, db, runner, GlobalPriorityExecutionPlanner())

    res = await sched.cycle_peek_plan_take_backfill_dispatch(slots_by_executor=slots, batch_size=32, lease_ttl=0.5)
    # A2 activated; A0 and A1 should be released back to ready
    assert "A2" in res["activated"]
    assert "A2" in res["dispatched"]
    assert "A0" not in res["activated"] and "A1" not in res["activated"]

    peek_ids = {wi.id for wi in await frontier.peek_ready(10)}
    assert "A0" in peek_ids and "A1" in peek_ids
    assert "A2" not in peek_ids  # active job should not reappear
