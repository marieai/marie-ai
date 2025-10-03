import time
import uuid

import pytest

from marie.serve.discovery.etcd_client import EtcdClient
from marie.state.semaphore_store import SemaphoreHolder, SemaphoreStore


@pytest.fixture(scope="function")
def etcd_client():
    c = EtcdClient("localhost", 2379)
    yield c
    try:
        c.delete_prefix("")
    except Exception:
        pass


@pytest.fixture(scope="function")
def sema(etcd_client: EtcdClient):
    return SemaphoreStore(etcd_client, default_lease_ttl=5)


def _slot() -> str:
    return f"slot-{uuid.uuid4()}"


def _ticket() -> str:
    return f"t-{uuid.uuid4()}"


def test_capacity_set_get(sema: SemaphoreStore):
    slot = _slot()
    assert sema.get_capacity(slot) is None

    sema.set_capacity(slot, 3)
    assert sema.get_capacity(slot) == 3

    # update capacity
    sema.set_capacity(slot, 5)
    assert sema.get_capacity(slot) == 5


def test_available_count_and_read_count(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 2)

    assert sema.read_count(slot) == 0
    assert sema.available_slot_count(slot) == 2

    ok1 = sema.reserve(slot, _ticket(), job_id="j1", node="n1")
    ok2 = sema.reserve(slot, _ticket(), job_id="j2", node="n2")

    assert ok1 is True and ok2 is True
    assert sema.read_count(slot) == 2
    assert sema.available_slot_count(slot) == 0

    # third should fail due to capacity
    ok3 = sema.reserve(slot, _ticket(), job_id="j3", node="n3")
    assert ok3 is False


def test_reserve_success_and_release(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 1)

    ticket = _ticket()
    ok_r = sema.reserve(slot, ticket, job_id="job", node="node")
    assert ok_r is True
    assert sema.read_count(slot) == 1

    ok_rel = sema.release(slot, ticket)
    assert ok_rel is True
    assert sema.read_count(slot) == 0
    assert sema.available_slot_count(slot) == 1


def test_release_requires_existing_holder_and_count(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 1)

    # release without reserve -> False (no counter)
    assert sema.release(slot, _ticket()) is False

    # reserve once
    ticket = _ticket()
    assert sema.reserve(slot, ticket, job_id="j", node="n") is True

    # releasing unknown ticket -> False (holder missing)
    assert sema.release(slot, _ticket()) is False

    # correct ticket -> True
    assert sema.release(slot, ticket) is True


def test_list_holders(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 3)

    t1, t2 = _ticket(), _ticket()
    assert sema.reserve(slot, t1, job_id="j1", node="n1") is True
    assert sema.reserve(slot, t2, job_id="j2", node="n2") is True

    holders = sema.list_holders(slot)
    assert isinstance(holders, dict)
    assert t1 in holders and t2 in holders
    assert isinstance(holders[t1], SemaphoreHolder)
    assert holders[t1].job_id == "j1"
    assert holders[t2].node == "n2"


def test_reconcile_updates_counter(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 5)

    # reserve three
    tickets = [_ticket() for _ in range(3)]
    for i, t in enumerate(tickets):
        assert sema.reserve(slot, t, job_id=f"j{i}", node=f"n{i}") is True

    # manually skew the counter down by one using direct put via client to emulate drift
    cnt_key = f"semaphores/{slot}/count"
    sema.etcd.put(cnt_key, "1")

    # before reconcile, read_count sees 1
    assert sema.read_count(slot) == 1

    # reconcile should compute from holders (=3) and CAS to update
    new_count = sema.reconcile(slot)
    assert new_count == 3
    assert sema.read_count(slot) == 3

    # release one and reconcile again
    assert sema.release(slot, tickets[0]) is True
    # Intentionally skew count up
    sema.etcd.put(cnt_key, "10")
    assert sema.reconcile(slot) in (2, 10)  # if CAS lost due to concurrent ops it's okay
    # ensure final count is at least consistent after a deterministic reconcile
    fixed = sema.reconcile(slot)
    assert fixed == 2


def test_lease_ttl_does_not_block_basic_flow(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 2)

    t = _ticket()
    # use a shorter ttl to ensure lease mechanics don't raise in basic reserve path
    ok = sema.reserve(slot, t, job_id="job", node="node", ttl=2)
    assert ok is True

    # wait a bit (not necessarily beyond ttl to keep test quick)
    time.sleep(0.05)

    # release should still work if holder exists
    assert sema.release(slot, t) is True
# ... existing code ...

def test_list_slot_types_and_capacities_all(sema: SemaphoreStore):
    s1, s2, s3 = _slot(), _slot(), _slot()

    # set capacities for three slot types
    sema.set_capacity(s1, 3)
    sema.set_capacity(s2, 1)
    sema.set_capacity(s3, 0)  # still should appear in slots and capacities

    # create activity for s1 to ensure semaphores/ paths exist too
    t1 = _ticket()
    assert sema.reserve(s1, t1, job_id="j1", node="n1") is True

    slots = sema.list_slot_types()
    assert {s1, s2, s3}.issubset(slots)

    caps = sema.capacities_all()
    assert caps.get(s1) == 3
    assert caps.get(s2) == 1
    assert caps.get(s3) == 0


def test_read_count_all_and_holder_counts_and_list_holders_all(sema: SemaphoreStore):
    s1, s2, s3 = _slot(), _slot(), _slot()

    sema.set_capacity(s1, 3)
    sema.set_capacity(s2, 2)
    sema.set_capacity(s3, 5)

    # reservations
    t1a, t1b = _ticket(), _ticket()
    t2a = _ticket()

    assert sema.reserve(s1, t1a, job_id="j1a", node="n1") is True
    assert sema.reserve(s1, t1b, job_id="j1b", node="n1") is True
    assert sema.reserve(s2, t2a, job_id="j2a", node="n2") is True
    # s3 has no holders

    # read_count_all should reflect used counts
    used = sema.read_count_all()
    assert used.get(s1) == 2
    assert used.get(s2) == 1
    # even if no counter exists yet, slot must be present with 0
    assert used.get(s3, 0) == 0

    # holder_counts_all should match number of holders per slot
    hcnt = sema.holder_counts_all()
    assert hcnt.get(s1) == 2
    assert hcnt.get(s2) == 1
    assert hcnt.get(s3) == 0  # ensured present with zero

    # list_holders_all returns mapping per slot, including empty for s3
    all_holders = sema.list_holders_all()
    assert s1 in all_holders and s2 in all_holders and s3 in all_holders
    assert isinstance(all_holders[s1], dict) and isinstance(all_holders[s2], dict)
    assert isinstance(all_holders[s3], dict) and len(all_holders[s3]) == 0

    # specific tickets present and parsed as SemaphoreHolder
    assert t1a in all_holders[s1] and isinstance(all_holders[s1][t1a], SemaphoreHolder)
    assert all_holders[s1][t1a].job_id == "j1a"
    assert t2a in all_holders[s2]


def test_available_count_all(sema: SemaphoreStore):
    s1, s2 = _slot(), _slot()
    sema.set_capacity(s1, 2)
    sema.set_capacity(s2, 1)

    t1 = _ticket()
    assert sema.reserve(s1, t1, job_id="j1", node="n1") is True

    avail = sema.available_count_all()
    # s1: cap 2, used 1
    assert avail.get(s1) == 1
    # s2: cap 1, used 0
    assert avail.get(s2) == 1


def test_snapshot_all_basic_and_with_holders(sema: SemaphoreStore):
    s1, s2, s3 = _slot(), _slot(), _slot()
    sema.set_capacity(s1, 3)
    sema.set_capacity(s2, 1)
    sema.set_capacity(s3, 2)

    t1a, t1b, t2a = _ticket(), _ticket(), _ticket()
    assert sema.reserve(s1, t1a, job_id="j1a", node="n1") is True
    assert sema.reserve(s1, t1b, job_id="j1b", node="n1") is True
    assert sema.reserve(s2, t2a, job_id="j2a", node="n2") is True
    # s3 has no holders

    snap = sema.snapshot_all(include_holders=False)
    # Ensure all slots present
    assert set([s1, s2, s3]).issubset(snap.keys())

    # Validate core fields
    assert snap[s1]["capacity"] == 3
    assert snap[s1]["used"] == 2
    assert snap[s1]["available"] == 1
    assert snap[s1]["holder_count"] == 2
    assert "holders" not in snap[s1]

    assert snap[s2]["capacity"] == 1
    assert snap[s2]["used"] == 1
    assert snap[s2]["available"] == 0
    assert snap[s2]["holder_count"] == 1

    assert snap[s3]["capacity"] == 2
    assert snap[s3]["used"] == 0
    assert snap[s3]["available"] == 2
    assert snap[s3]["holder_count"] == 0

    # Now include holders
    snap_h = sema.snapshot_all(include_holders=True)
    assert "holders" in snap_h[s1]
    assert isinstance(snap_h[s1]["holders"], dict)
    assert t1a in snap_h[s1]["holders"]
    assert isinstance(snap_h[s1]["holders"][t1a], SemaphoreHolder)
    # s3 should include empty holders map
    assert isinstance(snap_h[s3]["holders"], dict) and len(snap_h[s3]["holders"]) == 0