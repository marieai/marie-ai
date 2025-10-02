import uuid

import pytest
from etcd3 import transactions as tx

from marie.serve.discovery.etcd_client import EtcdClient, _prefix_end_key


@pytest.fixture(scope="function")
def client():
    c = EtcdClient("localhost", 2379)
    yield c
    # best-effort cleanup for the test namespace
    try:
        c.delete_prefix("")
    except Exception:
        pass


def _B(v):
    return v if isinstance(v, (bytes, bytearray)) else str(v).encode("utf-8")


def _S(v):
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


def _flatten_nested_prefix_dict(d, base=""):
    leaves = {}
    for k, v in d.items():
        full = f"{base}/{k}" if base else k
        if isinstance(v, dict):
            leaves.update(_flatten_nested_prefix_dict(v, full))
        else:
            leaves[full] = v
    return leaves


def test_txn_if_missing_put_then_succeeds(client: EtcdClient):
    key = f"txn/if_missing/{uuid.uuid4()}"
    mk = client._mangle_key(key)
    val = b"alpha"

    cmp_list = [tx.Version(mk) == 0]
    ops_then = [tx.Put(mk, val)]
    ops_else = []

    ok, resp = client.transaction(compare=cmp_list, success=ops_then, failure=ops_else)
    assert bool(ok) is True

    got, meta = client.get(key, metadata=True, serializable=False)
    assert got == val
    assert meta is not None

    # Idempotency check: now Version(mk)==0 is false -> should not put again
    ok2, _ = client.transaction(compare=[tx.Version(mk) == 0], success=[tx.Put(mk, b"beta")], failure=[])
    assert bool(ok2) is False
    got2, _ = client.get(key, metadata=True, serializable=False)
    assert got2 == val


def test_txn_else_branch_on_exists(client: EtcdClient):
    prefix = f"txn/else/{uuid.uuid4()}"
    key = f"{prefix}/a"
    else_key = f"{prefix}/else"
    client.put(key, "exists")

    mk = client._mangle_key(key)
    mk_else = client._mangle_key(else_key)

    # if missing -> then Put(key,"X"), else -> Put(else_key,"ELSE")
    ok, _ = client.transaction(
        compare=[tx.Version(mk) == 0],
        success=[tx.Put(mk, b"X")],
        failure=[tx.Put(mk_else, b"ELSE")],
    )
    assert bool(ok) is False  # because condition false

    val, _ = client.get(key, metadata=True)
    assert _S(val) == "exists"
    else_val, _ = client.get(else_key, metadata=True)
    assert _S(else_val) == "ELSE"


def test_txn_value_cas_swap(client: EtcdClient):
    key = f"txn/value/{uuid.uuid4()}"
    client.put(key, "v1")

    mk = client._mangle_key(key)
    ok, _ = client.transaction(
        compare=[tx.Value(mk) == b"v1"],
        success=[tx.Put(mk, b"v2")],
        failure=[],
    )
    assert bool(ok) is True

    got, _ = client.get(key, metadata=True)
    assert _S(got) == "v2"

    # wrong expected -> shouldn't swap
    ok2, _ = client.transaction(
        compare=[tx.Value(mk) == b"v1"],  # stale expectation
        success=[tx.Put(mk, b"v3")],
        failure=[],
    )
    assert bool(ok2) is False
    got2, _ = client.get(key, metadata=True)
    assert _S(got2) == "v2"


def test_txn_create_revision_guard(client: EtcdClient):
    key = f"txn/crev/{uuid.uuid4()}"
    mk = client._mangle_key(key)

    # If key missing -> Version == 0, then create
    ok, _ = client.transaction(
        compare=[tx.Version(mk) == 0],
        success=[tx.Put(mk, b"init")],
        failure=[],
    )
    assert bool(ok) is True
    v, _ = client.get(key, metadata=True)
    assert _S(v) == "init"

    # Now it exists, Version==0 should fail
    ok2, _ = client.transaction(
        compare=[tx.Version(mk) == 0],
        success=[tx.Put(mk, b"other")],
        failure=[],
    )
    assert bool(ok2) is False
    v2, _ = client.get(key, metadata=True)
    assert _S(v2) == "init"


def test_txn_multi_success_ops(client: EtcdClient):
    base = f"txn/multi/{uuid.uuid4()}"
    k1, k2 = f"{base}/k1", f"{base}/k2"
    mk1, mk2 = client._mangle_key(k1), client._mangle_key(k2)

    # both ops in success
    ok, _ = client.transaction(
        compare=[tx.Version(mk1) == 0, tx.Version(mk2) == 0],
        success=[tx.Put(mk1, b"one"), tx.Put(mk2, b"two")],
        failure=[],
    )
    assert bool(ok) is True

    v1, _ = client.get(k1, metadata=True)
    v2, _ = client.get(k2, metadata=True)
    assert _S(v1) == "one"
    assert _S(v2) == "two"


def test_txn_delete_prefix_range(client: EtcdClient):
    pfx = f"txn/delprefix/{uuid.uuid4()}"
    keys = [f"{pfx}/a", f"{pfx}/b", f"{pfx}/c/d"]
    for k in keys:
        client.put(k, "x")

    # Guard: ensure keys exist (count leaves under prefix)
    nested = client.get_prefix_dict(pfx)
    flat = _flatten_nested_prefix_dict(nested)
    assert len(flat) == 3

    start = client._mangle_key(pfx)
    end = _prefix_end_key(start)

    # Transactionally delete the entire prefix
    ok, _ = client.transaction(
        compare=[tx.Version(start) > -1],  # always-true compare to execute success
        success=[tx.Delete(start, range_end=end)],
        failure=[],
    )
    assert bool(ok) is True

    nested_after = client.get_prefix_dict(pfx)
    flat_after = _flatten_nested_prefix_dict(nested_after)
    assert len(flat_after) == 0


def test_txn_noop_when_compare_false_and_no_else(client: EtcdClient):
    key = f"txn/noop/{uuid.uuid4()}"
    client.put(key, "keep")
    mk = client._mangle_key(key)

    ok, _ = client.transaction(
        compare=[tx.Value(mk) == b"nope"],
        success=[tx.Put(mk, b"mutate")],
        failure=[],  # no else
    )
    assert bool(ok) is False
    v, _ = client.get(key, metadata=True)
    assert _S(v) == "keep"


def test_txn_else_executes_when_compare_false(client: EtcdClient):
    base = f"txn/else2/{uuid.uuid4()}"
    key, else_key = f"{base}/k", f"{base}/else"
    client.put(key, "present")

    mk = client._mangle_key(key)
    mk_else = client._mangle_key(else_key)

    ok, _ = client.transaction(
        compare=[tx.Version(mk) == 0],  # false
        success=[tx.Put(mk, b"then")],
        failure=[tx.Put(mk_else, b"else")],
    )
    assert bool(ok) is False
    v, _ = client.get(key, metadata=True)
    v_else, _ = client.get(else_key, metadata=True)
    assert _S(v) == "present"
    assert _S(v_else) == "else"
