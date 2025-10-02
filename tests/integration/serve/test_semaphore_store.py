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
