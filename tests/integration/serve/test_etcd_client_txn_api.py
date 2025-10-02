import uuid

import pytest
from etcd3 import transactions as tx

from marie.serve.discovery.etcd_client import EtcdClient


@pytest.fixture(scope="function")
def client():
    c = EtcdClient("localhost", 2379)
    yield c
    # best-effort cleanup for the test namespace
    try:
        c.delete_prefix("")
    except Exception:
        pass


def _S(v):
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


def test_fluent_if_missing_then_put(client: EtcdClient):
    key = f"fluent/if_missing/{uuid.uuid4()}"

    with client.txn() as t:
        t.if_missing(key).put(key, "alpha")
        ok, _ = t.commit()

    assert bool(ok) is True
    got, _ = client.get(key, metadata=True)
    assert _S(got) == "alpha"

    # second attempt should not overwrite
    with client.txn() as t2:
        t2.if_missing(key).put(key, "beta")
        ok2, _ = t2.commit()
    assert bool(ok2) is False
    got2, _ = client.get(key, metadata=True)
    assert _S(got2) == "alpha"


def test_fluent_if_value_cas(client: EtcdClient):
    key = f"fluent/if_value/{uuid.uuid4()}"
    client.put(key, "v1")

    # correct expected -> swap
    with client.txn() as t:
        t.if_value(key, "==", "v1").put(key, "v2")
        ok, _ = t.commit()
    assert bool(ok) is True
    got, _ = client.get(key, metadata=True)
    assert _S(got) == "v2"

    # wrong expected -> no-op
    with client.txn() as t2:
        t2.if_value(key, "==", "v1").put(key, "v3")
        ok2, _ = t2.commit()
    assert bool(ok2) is False
    got2, _ = client.get(key, metadata=True)
    assert _S(got2) == "v2"


def test_fluent_else_branch(client: EtcdClient):
    base = f"fluent/else/{uuid.uuid4()}"
    key = f"{base}/present"
    else_key = f"{base}/else"

    client.put(key, "exists")

    with client.txn() as t:
        # if key is missing THEN put "X", OTHERWISE put to else_key
        t.if_missing(key).put(key, "X").put(else_key, "ELSE", else_branch=True)
        ok, _ = t.commit()

    # condition false -> failure branch executed
    assert bool(ok) is False
    v, _ = client.get(key, metadata=True)
    v_else, _ = client.get(else_key, metadata=True)
    assert _S(v) == "exists"
    assert _S(v_else) == "ELSE"


def test_fluent_delete_and_delete_prefix(client: EtcdClient):
    base = f"fluent/del/{uuid.uuid4()}"
    k1 = f"{base}/a"
    k2 = f"{base}/b"
    k3 = f"{base}/nested/c"
    client.put(k1, "x")
    client.put(k2, "y")
    client.put(k3, "z")

    # delete one explicitly
    with client.txn() as t:
        t.if_exists(k1).delete(k1)
        ok1, _ = t.commit()
    assert bool(ok1) is True
    v1, _ = client.get(k1, metadata=True)
    assert v1 is None

    # delete remaining under prefix
    with client.txn() as t2:
        t2.delete_prefix(base)
        ok2, _ = t2.commit()
    assert bool(ok2) is True

    # verify nothing remains under base
    nested = client.get_prefix_dict(base)
    assert nested == {}


def test_fluent_then_otherwise_accessors(client: EtcdClient):
    base = f"fluent/accessors/{uuid.uuid4()}"
    key = f"{base}/k"
    else_key = f"{base}/else"

    with client.txn() as t:
        t.if_missing(key).put(key, "alpha").put(else_key, "beta", else_branch=True)

        # Ensure accessors expose non-empty ops on both branches
        then_ops = t.then()
        else_ops = t.otherwise()
        assert isinstance(then_ops, list) and len(then_ops) >= 1
        assert isinstance(else_ops, list) and len(else_ops) >= 1

        ok, _ = t.commit()
    # key missing initially -> then branch executed
    assert bool(ok) is True
    v, _ = client.get(key, metadata=True)
    v_else, _ = client.get(else_key, metadata=True)
    assert _S(v) == "alpha"
    # since 'then' executed, else branch hasn't been applied
    assert v_else is None


def test_fluent_commit_twice_raises(client: EtcdClient):
    key = f"fluent/commit_twice/{uuid.uuid4()}"

    with client.txn() as t:
        t.if_missing(key).put(key, "once")
        ok, _ = t.commit()
        assert bool(ok) is True
        with pytest.raises(RuntimeError):
            t.commit()


@pytest.mark.skipif(not all(hasattr(tx, a) for a in ("Mod",)), reason="Mod comparator not available in this etcd3 build")
def test_fluent_if_mod_revision(client: EtcdClient):
    key = f"fluent/mod/{uuid.uuid4()}"
    client.put(key, "first")
    _, meta = client.get(key, metadata=True, serializable=False)

    with client.txn() as t:
        t.if_mod_revision(key, "==", meta.mod_revision).put(key, "second")
        ok, _ = t.commit()
    assert bool(ok) is True
    v, _ = client.get(key, metadata=True)
    assert _S(v) == "second"

    # stale mod should fail
    with client.txn() as t2:
        t2.if_mod_revision(key, "==", meta.mod_revision).put(key, "third")
        ok2, _ = t2.commit()
    assert bool(ok2) is False
    v2, _ = client.get(key, metadata=True)
    assert _S(v2) == "second"


@pytest.mark.skipif(not all(hasattr(tx, a) for a in ("Create",)), reason="Create comparator not available in this etcd3 build")
def test_fluent_if_create_revision(client: EtcdClient):
    key = f"fluent/create/{uuid.uuid4()}"

    # key missing -> create branch should succeed
    with client.txn() as t:
        t.if_create_revision(key, "==", 0).put(key, "init")
        ok, _ = t.commit()
    assert bool(ok) is True
    v, _ = client.get(key, metadata=True)
    assert _S(v) == "init"

    # key exists now -> should fail compare (no overwrite)
    with client.txn() as t2:
        t2.if_create_revision(key, "==", 0).put(key, "other")
        ok2, _ = t2.commit()
    assert bool(ok2) is False
    v2, _ = client.get(key, metadata=True)
    assert _S(v2) == "init"

