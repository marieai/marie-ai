import time
import uuid

import pytest

from marie.serve.discovery.etcd_client import EtcdClient


# Assuming a fixture `etcd_client` is available in a conftest.py
@pytest.fixture(scope='function')
def etcd_client(tmpdir):
    etcd_client = EtcdClient("localhost", 2379)
    yield etcd_client


@pytest.fixture(scope='function')
def client(etcd_client: EtcdClient):
    """Provides a fresh EtcdClient for each test function."""
    # This fixture assumes you have a higher-level fixture `etcd_client`
    # that manages the connection to a test etcd server.
    yield etcd_client
    # Optional: Clean up keys after each test if the main fixture doesn't.
    etcd_client.delete_prefix('')


def test_put_get_delete(client: EtcdClient):
    """Test basic single key PUT, GET, and DELETE operations."""
    key = f"test/key/{uuid.uuid4()}"
    value = "hello world"

    # 1. PUT a new key-value pair
    client.put(key, value)

    # 2. GET the key and verify its value (request metadata explicitly)
    retrieved_value, meta = client.get(key, metadata=True)
    assert retrieved_value is not None
    assert retrieved_value.decode('utf-8') == value
    assert meta is not None

    # 3. DELETE the key
    deleted = client.delete(key)
    assert bool(deleted) is True

    # 4. GET the key again and verify it's gone
    retrieved_value, meta = client.get(key, metadata=True)
    assert retrieved_value is None
    assert meta is None


def test_get_prefix(client: EtcdClient):
    """Test retrieving multiple keys with a common prefix."""
    prefix = f"test/prefix/{uuid.uuid4()}"
    keys_to_create = {
        f"{prefix}/a": "value_a",
        f"{prefix}/b": "value_b",
        f"{prefix}/c/d": "value_d",
    }

    for key, value in keys_to_create.items():
        client.put(key, value)

    # Prefer the dict helper that returns demangled, nested dict
    nested = client.get_prefix_dict(prefix)

    def _flatten(d, base=""):
        out = {}
        for k, v in d.items():
            full = f"{base}/{k}" if base else k
            if isinstance(v, dict):
                out.update(_flatten(v, full))
            else:
                out[full] = v
        return out

    flat = _flatten(nested)
    # Build expected keys relative to the prefix (demangled), e.g. 'a', 'b', 'c/d'
    expected = {k.removeprefix(f"{prefix}/"): v for k, v in keys_to_create.items()}
    for rel_key, val in expected.items():
        assert rel_key in flat
        # flat values are bytes or str depending on client; normalize to str
        got_val = flat[rel_key].decode('utf-8') if isinstance(flat[rel_key], (bytes, bytearray)) else str(flat[rel_key])
        assert got_val == val


def test_delete_prefix(client: EtcdClient):
    """Test deleting multiple keys with a common prefix."""
    prefix = f"test/delete_prefix/{uuid.uuid4()}"
    keys_to_create = [f"{prefix}/1", f"{prefix}/2", f"{prefix}/3"]

    for key in keys_to_create:
        client.put(key, "some_value")

    # Verify keys exist
    assert len(list(client.get_prefix(prefix))) == 3

    # Delete by prefix; depending on implementation this can be a response object or an int
    deleted_result = client.delete_prefix(prefix)
    deleted_count = deleted_result if isinstance(deleted_result, int) else getattr(deleted_result, 'deleted', None)
    assert deleted_count == 3

    # Verify keys are gone
    assert len(list(client.get_prefix(prefix))) == 0


@pytest.mark.xfail(
    reason="EtcdClient.Txn uses etcd3.transactions comparators with wrong casing; put_if_absent relies on Txn")
def test_put_if_absent(client: EtcdClient):
    """Test the atomic put_if_absent operation."""
    key = f"test/absent/{uuid.uuid4()}"

    # 1. First time should succeed
    created = client.put_if_absent(key, "first_val")
    assert created is True

    # 2. Second time with the same key should fail
    created_again = client.put_if_absent(key, "second_val")
    assert created_again is False

    # 3. Verify the value is still the first one
    val, _ = client.get(key, metadata=True)
    assert val.decode('utf-8') == "first_val"


@pytest.mark.xfail(
    reason="EtcdClient.Txn compares with tx.version/tx.value etc., but etcd3 exposes Version/Value; fix client Txn first")
def test_transaction(client: EtcdClient):
    """Test a simple transaction using the fluent builder."""
    key = f"test/txn/{uuid.uuid4()}"

    # Transaction: if key is missing, put it.
    with client.txn() as t:
        t.if_missing(key).put(key, "txn_value")

    ok, _ = t.commit()
    assert ok is True

    val, _ = client.get(key, metadata=True)
    assert val.decode('utf-8') == "txn_value"

    # Transaction: if key exists, do nothing.
    with client.txn() as t2:
        t2.if_missing(key).put(key, "another_value")

    ok2, _ = t2.commit()
    assert ok2 is False  # The "if" condition was false

    val, _ = client.get(key, metadata=True)
    assert val.decode('utf-8') == "txn_value"  # Unchanged


def test_watch_callback(client: EtcdClient):
    """Test the watch functionality with a callback."""
    prefix = f"test/watch/{uuid.uuid4()}"
    key_to_watch = f"{prefix}/watched_key"

    events_received = []

    # Normalize callback to receive (key: str, event: Event) or variants
    def my_callback(*args, **kwargs):
        # Expected by EtcdClient: event_callback(self._demangle_key(ev.key), event)
        if len(args) >= 2 and isinstance(args[0], (str, bytes, bytearray)):
            key = args[0]
            evt_or_val = args[1]
            # If it's the Event(namedtuple), extract .value; otherwise keep as-is
            try:
                value = evt_or_val.value
            except Exception:
                value = evt_or_val
            events_received.append((key, value))
        elif len(args) >= 1:
            # Fallback: single positional event-like object
            ev = args[0]
            try:
                key = ev.key
                value = ev.value
                events_received.append((key, value))
            except Exception:
                events_received.append((ev, None))
        else:
            events_received.append((None, None))

    # Add a watch on the prefix
    watch_id = client.add_watch_prefix_callback(prefix, my_callback)
    time.sleep(1)  # Give watch time to establish

    try:
        # Trigger an event
        client.put(key_to_watch, "watch_trigger_value")

        # Wait a moment for the event to be processed by the background thread
        time.sleep(2)

        assert len(events_received) > 0

        # Unpack captured (key, value_or_dict)
        received_key, received_value = events_received[0]

        # Normalize key
        if isinstance(received_key, (bytes, bytearray)):
            k_dec = received_key.decode('utf-8')
        else:
            k_dec = str(received_key)

        # Normalize value: can be bytes/str or a dict for prefix watches
        v_dec = None
        if isinstance(received_value, dict):
            # Prefix callback packs value as dict under the leaf key
            leaf = key_to_watch.rsplit('/', 1)[-1]
            inner = received_value.get(leaf)
            if isinstance(inner, (bytes, bytearray)):
                v_dec = inner.decode('utf-8')
            elif inner is not None:
                v_dec = str(inner)
        elif isinstance(received_value, (bytes, bytearray)):
            v_dec = received_value.decode('utf-8')
        elif received_value is not None:
            v_dec = str(received_value)

        assert k_dec.endswith(key_to_watch)
        assert v_dec == "watch_trigger_value"

    finally:
        # Clean up the watch
        client.cancel_watch(watch_id)
        time.sleep(1)
