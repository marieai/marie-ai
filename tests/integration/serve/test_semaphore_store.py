import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pytest

from marie.serve.discovery.etcd_client import EtcdClient
from marie.state.semaphore_store import SemaphoreHolder, SemaphoreStore


class _TransactionProxy:
    def __init__(
        self,
        transaction: Any,
        commit: Callable[[Any], tuple[bool, Any]],
    ) -> None:
        self._transaction = transaction
        self._commit = commit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transaction, name)

    def commit(self) -> tuple[bool, Any]:
        return self._commit(self._transaction)


@pytest.fixture(scope="function")
def etcd_client():
    # Unique namespace per test: the teardown below range-deletes the client's
    # entire namespace, which on the default "marie" namespace wiped the live
    # keyspace (2026-07-09 outage). Never point this at "marie".
    c = EtcdClient("localhost", 2379, namespace=f"marie-test-{uuid.uuid4().hex[:8]}")
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

    ok1 = sema.reserve(slot, _ticket(),  node="n1")
    ok2 = sema.reserve(slot, _ticket(),  node="n2")

    assert ok1 is True and ok2 is True
    assert sema.read_count(slot) == 2
    assert sema.available_slot_count(slot) == 0

    # third should fail due to capacity
    ok3 = sema.reserve(slot, _ticket(), node="n3")
    assert ok3 is False


def test_reserve_success_and_release(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 1)

    ticket = _ticket()
    ok_r = sema.reserve(slot, ticket,  node="node")
    assert ok_r is True
    assert sema.read_count(slot) == 1

    ok_rel = sema.release(slot, ticket)
    assert ok_rel is True
    assert sema.read_count(slot) == 0
    assert sema.available_slot_count(slot) == 1


def test_attempt_bound_holder_rejects_stale_renew_and_release(
    sema: SemaphoreStore,
) -> None:
    slot = _slot()
    ticket = _ticket()
    sema.set_capacity(slot, 1)
    assert sema.reserve(
        slot,
        ticket,
        node="scheduler",
        owner=ticket,
        run_attempt_id="attempt-1",
    )

    holder = sema.get_holder(slot, ticket)
    assert holder is not None
    assert holder.run_attempt_id == "attempt-1"
    assert sema.validate_holder(
        slot,
        ticket,
        owner=ticket,
        run_attempt_id="attempt-1",
    )
    assert not sema.validate_holder(
        slot,
        ticket,
        owner=ticket,
        run_attempt_id="attempt-2",
    )
    assert not sema.renew(
        slot,
        ticket,
        owner=ticket,
        run_attempt_id="attempt-2",
    )
    stale_release = sema.release_owned_result(
        slot,
        ticket,
        owner=ticket,
        run_attempt_id="attempt-2",
    )
    assert not stale_release
    assert stale_release.reason == "attempt_mismatch"
    assert sema.release_owned(
        slot,
        ticket,
        owner=ticket,
        run_attempt_id="attempt-1",
    )


def test_renew_keeps_existing_holder_lease(sema: SemaphoreStore) -> None:
    slot = _slot()
    ticket = _ticket()
    sema.set_capacity(slot, 1)
    assert sema.reserve(
        slot,
        ticket,
        node="scheduler",
        owner=ticket,
        run_attempt_id="attempt-1",
    )
    _raw_before, meta_before = sema._get_holder_raw(slot, ticket)

    assert sema.renew(
        slot,
        ticket,
        owner=ticket,
        run_attempt_id="attempt-1",
    )

    _raw_after, meta_after = sema._get_holder_raw(slot, ticket)
    assert meta_before.lease_id == meta_after.lease_id


def test_concurrent_owned_releases_retry_counter_contention(
    sema: SemaphoreStore,
) -> None:
    slot = _slot()
    tickets = [_ticket(), _ticket()]
    sema.set_capacity(slot, 2)
    for ticket in tickets:
        assert sema.reserve(slot, ticket, node="scheduler", owner=ticket)

    real_txn = sema.etcd.txn
    commit_barrier = threading.Barrier(2)
    commit_lock = threading.Lock()
    commit_count = 0

    def contended_commit(transaction: Any) -> tuple[bool, Any]:
        nonlocal commit_count
        with commit_lock:
            commit_count += 1
            synchronize = commit_count <= 2
        if synchronize:
            commit_barrier.wait(timeout=2.0)
        return transaction.commit()

    def contended_txn() -> _TransactionProxy:
        return _TransactionProxy(real_txn(), contended_commit)

    sema.etcd.txn = contended_txn
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda ticket: sema.release_owned_result(
                        slot,
                        ticket,
                        owner=ticket,
                    ),
                    tickets,
                )
            )
    finally:
        sema.etcd.txn = real_txn

    assert all(result.success for result in results)
    assert max(result.attempts for result in results) >= 2
    assert sema.read_count(slot) == 0
    assert sema.list_holders(slot) == {}


def test_absent_holder_release_reconciles_stale_counter(
    sema: SemaphoreStore,
) -> None:
    slot = _slot()
    ticket = _ticket()
    sema.set_capacity(slot, 1)
    assert sema.reserve(slot, ticket, node="scheduler", owner=ticket)

    sema.etcd.delete(f"semaphores/{slot}/holders/{ticket}")
    assert sema.read_count(slot) == 1

    result = sema.release_owned_result(slot, ticket, owner=ticket)

    assert result.success
    assert result.reason == "already_absent"
    assert result.holder_absent
    assert result.counter_reconciled
    assert sema.read_count(slot) == 0


def test_reconcile_retries_counter_contention(sema: SemaphoreStore) -> None:
    slot = _slot()
    sema.set_capacity(slot, 2)
    assert sema.reserve(slot, "t1", node="scheduler")
    sema.etcd.put(f"semaphores/{slot}/count", "2")

    real_txn = sema.etcd.txn
    failed_commits = 0

    def contended_commit(transaction: Any) -> tuple[bool, Any]:
        nonlocal failed_commits
        failed_commits += 1
        if failed_commits <= 2:
            return False, []
        return transaction.commit()

    def contended_txn() -> _TransactionProxy:
        return _TransactionProxy(real_txn(), contended_commit)

    sema.etcd.txn = contended_txn
    try:
        assert sema.reconcile(slot) == 1
    finally:
        sema.etcd.txn = real_txn

    assert failed_commits == 3
    assert sema.read_count(slot) == 1


def test_reserve_many_reserves_available_tickets(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 3)

    tickets = [_ticket() for _ in range(3)]
    run_attempt_ids = {
        ticket: f"attempt-{index}" for index, ticket in enumerate(tickets)
    }
    reserved = sema.reserve_many(
        slot,
        tickets,
        node="scheduler",
        owner_by_ticket={ticket: ticket for ticket in tickets},
        run_attempt_id_by_ticket=run_attempt_ids,
    )

    assert reserved == set(tickets)
    assert sema.read_count(slot) == 3

    holders = sema.list_holders(slot)
    assert set(holders) == set(tickets)
    for ticket in tickets:
        assert holders[ticket].owner == ticket
        assert holders[ticket].run_attempt_id == run_attempt_ids[ticket]


def test_reserve_many_caps_at_available_capacity(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 2)

    tickets = [_ticket() for _ in range(4)]
    reserved = sema.reserve_many(slot, tickets, node="scheduler")

    assert reserved == set(tickets[:2])
    assert sema.read_count(slot) == 2
    assert sema.available_slot_count(slot) == 0


def test_reserve_many_falls_back_when_one_ticket_exists(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 3)

    existing = _ticket()
    first_new = _ticket()
    second_new = _ticket()
    assert sema.reserve(slot, existing, node="scheduler") is True

    reserved = sema.reserve_many(
        slot,
        [existing, first_new, second_new],
        node="scheduler",
    )

    assert reserved == {first_new, second_new}
    assert sema.read_count(slot) == 3


def test_release_requires_existing_holder_and_count(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 1)

    # release without reserve -> False (no counter)
    assert sema.release(slot, _ticket()) is False

    # reserve once
    ticket = _ticket()
    assert sema.reserve(slot, ticket, node="n") is True

    # releasing unknown ticket -> False (holder missing)
    assert sema.release(slot, _ticket()) is False

    # correct ticket -> True
    assert sema.release(slot, ticket) is True


def test_list_holders(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 3)

    t1, t2 = _ticket(), _ticket()
    assert sema.reserve(slot, t1,  node="n1") is True
    assert sema.reserve(slot, t2,  node="n2") is True

    holders = sema.list_holders(slot)
    assert isinstance(holders, dict)
    assert t1 in holders and t2 in holders
    assert isinstance(holders[t1], SemaphoreHolder)
    assert holders[t1].ticket_id == t1
    assert holders[t2].node == "n2"


def test_reconcile_updates_counter(sema: SemaphoreStore):
    slot = _slot()
    sema.set_capacity(slot, 5)

    # reserve three
    tickets = [_ticket() for _ in range(3)]
    for i, t in enumerate(tickets):
        assert sema.reserve(slot, t, node=f"n{i}") is True

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
    ok = sema.reserve(slot, t, node="node", ttl=2)
    assert ok is True

    # wait a bit (not necessarily beyond ttl to keep test quick)
    time.sleep(0.05)

    # release should still work if holder exists
    assert sema.release(slot, t) is True


def test_list_slot_types_and_capacities_all(sema: SemaphoreStore):
    s1, s2, s3 = _slot(), _slot(), _slot()

    # set capacities for three slot types
    sema.set_capacity(s1, 3)
    sema.set_capacity(s2, 1)
    sema.set_capacity(s3, 0)  # still should appear in slots and capacities

    # create activity for s1 to ensure semaphores/ paths exist too
    t1 = _ticket()
    assert sema.reserve(s1, t1, node="n1") is True

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

    assert sema.reserve(s1, t1a, node="n1") is True
    assert sema.reserve(s1, t1b, node="n1") is True
    assert sema.reserve(s2, t2a, node="n2") is True
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
    assert all_holders[s1][t1a].ticket_id == t1a
    assert t2a in all_holders[s2]


def test_available_count_all(sema: SemaphoreStore):
    s1, s2 = _slot(), _slot()
    sema.set_capacity(s1, 2)
    sema.set_capacity(s2, 1)

    t1 = _ticket()
    assert sema.reserve(s1, t1, node="n1") is True

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
    assert sema.reserve(s1, t1a, node="n1") is True
    assert sema.reserve(s1, t1b, node="n1") is True
    assert sema.reserve(s2, t2a, node="n2") is True
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

def test_release_with_missing_counter(sema: SemaphoreStore):
    """
    Bug #2 fix: Test that release() handles missing counter gracefully.
    Previously, if the counter was deleted but holder existed, release would fail
    and the holder would be stuck until lease expiration.
    """
    slot = _slot()
    sema.set_capacity(slot, 2)

    # Reserve a slot
    ticket = _ticket()
    assert sema.reserve(slot, ticket, node="node1") is True
    assert sema.read_count(slot) == 1

    # Simulate counter deletion (e.g., manual deletion, etcd issue, bug)
    cnt_key = f"semaphores/{slot}/count"
    sema.etcd.delete(cnt_key)

    # Verify counter is missing but holder still exists
    assert sema.read_count(slot) == 0  # Counter missing returns 0
    holders = sema.list_holders(slot)
    assert ticket in holders  # Holder still exists

    # The fix: release() should succeed even with missing counter
    ok_rel = sema.release(slot, ticket)
    assert ok_rel is True

    # After release:
    # - Holder should be deleted
    # - Counter should be initialized to 0
    assert sema.read_count(slot) == 0
    holders_after = sema.list_holders(slot)
    assert ticket not in holders_after


def test_release_owned_with_missing_counter(sema: SemaphoreStore):
    """
    Bug #2 fix: Test that release_owned() handles missing counter gracefully
    while still enforcing ownership checks.
    """
    slot = _slot()
    sema.set_capacity(slot, 2)

    # Reserve with specific owner
    ticket = _ticket()
    owner = "worker-123"
    assert sema.reserve(slot, ticket, node="node1", owner=owner) is True
    assert sema.read_count(slot) == 1

    # Verify holder has correct owner
    holder = sema.get_holder(slot, ticket)
    assert holder is not None
    assert holder.owner == owner

    # Simulate counter deletion
    cnt_key = f"semaphores/{slot}/count"
    sema.etcd.delete(cnt_key)

    # Verify counter is missing but holder still exists
    assert sema.read_count(slot) == 0
    holders = sema.list_holders(slot)
    assert ticket in holders

    # The fix: release_owned() should succeed with correct owner
    ok_rel = sema.release_owned(slot, ticket, owner=owner)
    assert ok_rel is True

    # After release:
    # - Holder should be deleted
    # - Counter should be initialized to 0
    assert sema.read_count(slot) == 0
    holders_after = sema.list_holders(slot)
    assert ticket not in holders_after


def test_release_owned_with_missing_counter_wrong_owner(sema: SemaphoreStore):
    """
    Bug #2 fix: Test that release_owned() still enforces ownership checks
    even when counter is missing.
    """
    slot = _slot()
    sema.set_capacity(slot, 2)

    # Reserve with specific owner
    ticket = _ticket()
    correct_owner = "worker-123"
    wrong_owner = "worker-456"
    assert sema.reserve(slot, ticket, node="node1", owner=correct_owner) is True

    # Simulate counter deletion
    cnt_key = f"semaphores/{slot}/count"
    sema.etcd.delete(cnt_key)

    # Attempt to release with wrong owner should fail
    ok_rel = sema.release_owned(slot, ticket, owner=wrong_owner)
    assert ok_rel is False

    # Holder should still exist (not released)
    holders = sema.list_holders(slot)
    assert ticket in holders

    # Now release with correct owner should succeed
    ok_rel_correct = sema.release_owned(slot, ticket, owner=correct_owner)
    assert ok_rel_correct is True

    # Holder should now be deleted
    holders_after = sema.list_holders(slot)
    assert ticket not in holders_after


def test_multiple_releases_with_missing_counter(sema: SemaphoreStore):
    """
    Bug #2 fix: Test that multiple holders can be released when counter is missing.
    """
    slot = _slot()
    sema.set_capacity(slot, 3)

    # Reserve multiple slots
    t1, t2, t3 = _ticket(), _ticket(), _ticket()
    assert sema.reserve(slot, t1, node="n1") is True
    assert sema.reserve(slot, t2, node="n2") is True
    assert sema.reserve(slot, t3, node="n3") is True
    assert sema.read_count(slot) == 3

    # Simulate counter deletion
    cnt_key = f"semaphores/{slot}/count"
    sema.etcd.delete(cnt_key)

    # Release all three - each should succeed
    assert sema.release(slot, t1) is True
    assert sema.release(slot, t2) is True
    assert sema.release(slot, t3) is True

    # All holders should be gone
    holders_after = sema.list_holders(slot)
    assert len(holders_after) == 0
    assert sema.read_count(slot) == 0


def test_set_capacity_safe_never_below_concurrent_usage(sema):
    slot = _slot()
    sema.set_capacity(slot, 2)
    assert sema.reserve(slot, "t1", node="n1")  # used = 1

    real_get_raw = sema._get_raw
    injected = {"done": False}

    def racing_get_raw(key):
        val = real_get_raw(key)
        if key.endswith("/count") and not injected["done"]:
            injected["done"] = True  # guard BEFORE reserving (reserve re-enters _get_raw)
            assert sema.reserve(slot, "t2", node="n2")  # commits after the count read
        return val

    sema._get_raw = racing_get_raw
    try:
        sema.set_capacity_safe(slot, 0)
    finally:
        sema._get_raw = real_get_raw

    assert sema.read_count(slot) == 2
    assert sema.get_capacity(slot) >= 2  # old code writes 1 → capacity < used


def test_set_capacity_safe_create_path_guards_count(sema, etcd_client):
    """Create path (no capacity key yet): the count must be txn-guarded too —
    a bare put_if_absent would commit a target computed from a stale count."""
    slot = _slot()
    # count exists without a capacity key (e.g. capacity key was deleted)
    etcd_client.put(f"semaphores/{slot}/count", "5")

    real_get_raw = sema._get_raw
    injected = {"done": False}

    def racing_get_raw(key):
        val = real_get_raw(key)
        if key.endswith("/count") and not injected["done"]:
            injected["done"] = True
            # count jumps AFTER set_capacity_safe read it, BEFORE it writes
            etcd_client.put(f"semaphores/{slot}/count", "8")
        return val

    sema._get_raw = racing_get_raw
    try:
        result = sema.set_capacity_safe(slot, 3)
    finally:
        sema._get_raw = real_get_raw

    assert result >= 8            # recomputed from the fresh count
    assert sema.get_capacity(slot) >= 8   # old create path writes 5 < used 8


def test_set_capacity_safe_refuses_malformed_count(sema, etcd_client):
    slot = _slot()
    etcd_client.put(f"semaphores/{slot}/count", "not-an-int")
    with pytest.raises(RuntimeError, match="malformed count"):
        sema.set_capacity_safe(slot, 3)


def test_reconcile_concurrent_reserve_does_not_undercount(sema):
    slot = _slot()
    sema.set_capacity(slot, 3)
    assert sema.reserve(slot, "t1", node="n1")  # count = 1

    real_get_prefix = sema.etcd.client.get_prefix
    injected = {"done": False}

    def racing_get_prefix(key, *args, **kwargs):
        results = list(real_get_prefix(key, *args, **kwargs))
        kb = key if isinstance(key, bytes) else str(key).encode()
        if b"/holders/" in kb and not injected["done"]:
            injected["done"] = True
            assert sema.reserve(slot, "t2", node="n2")  # commits mid-reconcile
        return iter(results)

    sema.etcd.client.get_prefix = racing_get_prefix
    try:
        sema.reconcile(slot)
    finally:
        sema.etcd.client.get_prefix = real_get_prefix

    # old order: reconcile reads count AFTER the racing reserve (sees 2),
    # scanned holders BEFORE it (saw 1) -> CAS passes -> count clobbered to 1.
    assert sema.read_count(slot) == 2
    assert sema.reserve(slot, "t3", node="n3") is True
    assert sema.read_count(slot) == 3


def test_malformed_count_reserve_and_release_self_repair(sema, etcd_client):
    slot = _slot()
    sema.set_capacity(slot, 2)
    etcd_client.put(f"semaphores/{slot}/count", "not-an-int")

    assert sema.read_count(slot) == 0          # already tolerant today
    assert sema.reserve(slot, "t1", node="n1") is True   # ValueError today
    assert sema.read_count(slot) == 1
    assert sema.release(slot, "t1") is True
    assert sema.read_count(slot) == 0


def test_reconcile_repairs_malformed_count(sema, etcd_client):
    slot = _slot()
    sema.set_capacity(slot, 2)
    assert sema.reserve(slot, "t1", node="n1")
    etcd_client.put(f"semaphores/{slot}/count", "not-an-int")

    assert sema.reconcile(slot) == 1           # ValueError today
    assert sema.read_count(slot) == 1


def test_fixture_cleanup_is_scoped_to_test_namespace(etcd_client):
    """Guard for the 2026-07-09 outage class: the fixture teardown
    (delete_prefix("")) must only ever range-delete the fixture's own
    namespace, and that namespace must never be the production "marie"."""
    live = EtcdClient("localhost", 2379, namespace="marie-wipe-guard")
    try:
        live.put("sentinel", "alive")
        etcd_client.put("victim", "x")

        # the exact teardown call the fixture performs
        etcd_client.delete_prefix("")

        assert etcd_client.get("victim") is None
        assert live.get("sentinel") is not None  # other namespaces untouched
        assert etcd_client.ns != "marie"
    finally:
        live.delete_prefix("")


def test_scan_raises_typed_error_when_channel_closed(sema):
    from marie.state.semaphore_store import EtcdStoreUnavailable

    def _boom(_prefix):
        raise ValueError("Cannot invoke RPC on closed channel!")

    sema.etcd.client.get_prefix = _boom
    try:
        with pytest.raises(EtcdStoreUnavailable):
            sema.available_count_all()
        with pytest.raises(EtcdStoreUnavailable):
            sema.list_slot_types()
    finally:
        del sema.etcd.client.get_prefix


def test_scan_raises_typed_error_on_connection_failed(sema):
    """etcd3's own _manage_grpc_errors translates a real outage (server down)
    into etcd3.exceptions.ConnectionFailedError before it ever reaches
    _scan_prefix as a grpc.RpcError — the helper must still classify it."""
    import etcd3.exceptions

    from marie.state.semaphore_store import EtcdStoreUnavailable

    def _boom(_prefix):
        raise etcd3.exceptions.ConnectionFailedError()

    sema.etcd.client.get_prefix = _boom
    try:
        with pytest.raises(EtcdStoreUnavailable):
            sema.available_count_all()
    finally:
        del sema.etcd.client.get_prefix


def test_scan_raises_typed_error_on_connection_timeout(sema):
    """Same as above for the paused-etcd/DEADLINE_EXCEEDED case, which etcd3
    translates into ConnectionTimeoutError before _scan_prefix sees it."""
    import etcd3.exceptions

    from marie.state.semaphore_store import EtcdStoreUnavailable

    def _boom(_prefix):
        raise etcd3.exceptions.ConnectionTimeoutError()

    sema.etcd.client.get_prefix = _boom
    try:
        with pytest.raises(EtcdStoreUnavailable):
            sema.available_count_all()
    finally:
        del sema.etcd.client.get_prefix


def test_scan_typed_error_covers_both_closed_channel_phrasings(sema, monkeypatch):
    from marie.state.semaphore_store import EtcdStoreUnavailable

    for msg in (
        "Cannot invoke RPC on closed channel!",
        "Cannot invoke RPC: Channel closed!",
    ):
        def _boom(_prefix, _msg=msg):
            raise ValueError(_msg)

        monkeypatch.setattr(sema.etcd.client, "get_prefix", _boom, raising=False)
        with pytest.raises(EtcdStoreUnavailable):
            sema.available_count_all()
        monkeypatch.undo()
        del sema.etcd.client.get_prefix
