import os
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import DEFAULT, Mock

import pytest
from etcd3.exceptions import ConnectionFailedError, ConnectionTimeoutError

from marie.serve.discovery.etcd_client import EtcdClient
from marie.state.state_store import DesiredStore, StatusStore


@pytest.fixture
def client() -> Iterator[EtcdClient]:
    store = EtcdClient(
        os.environ.get("ETCD_TEST_HOST", "localhost"),
        int(os.environ.get("ETCD_TEST_PORT", "2379")),
        namespace=f"marie-test-{uuid.uuid4().hex}",
    )
    try:
        yield store
    finally:
        try:
            store.delete_prefix("")
        finally:
            store.close()


def test_prefix_keys_requests_only_keys_in_namespace(
    client: EtcdClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.get_prefix_keys("deployments/") == []
    key = "deployments/node%3A1/llm/desired"
    client.client.put(client._mangle_key(key), b"\xff")
    client.put("deployments-old/node/llm/desired", "unrelated")
    scan = Mock(wraps=client.client.get_prefix)
    monkeypatch.setattr(client.client, "get_prefix", scan)

    assert client.get_prefix_keys("deployments/") == [key]
    scan.assert_called_once_with(f"/{client.ns}/deployments/".encode(), keys_only=True)


def test_store_pairs_use_complete_keys_and_preserve_point_read_identity(
    client: EtcdClient,
) -> None:
    desired = DesiredStore(client)
    status = StatusStore(client)
    assert list(desired.list_pairs()) == list(status.list_pairs()) == []
    for node, depl in (
        ("worker:1", "llm"),
        ("worker%3A1", "llm"),
        ("worker", "rep%2F1"),
    ):
        desired.set(node, depl, {"id": node})
    assert status.claim("worker%3A1", "llm", worker_id="first", epoch=1)
    assert status.claim("other", "parser", worker_id="second", epoch=2)
    for key in (
        "deployments/worker/desired",
        "deployments/worker/llm/extra/desired",
        "deployments//llm/status",
        "deployments/worker//desired",
        "deployments/other/parser/desired/",
        "deployments/other/parser/metrics",
        "deployments-old/other/llm/desired",
    ):
        client.put(key, "ignore")

    pairs = sorted(desired.list_pairs())
    assert pairs == [
        ("worker", "rep%2F1"),
        ("worker%3A1", "llm"),
        ("worker:1", "llm"),
    ]
    for node, depl in pairs:
        assert desired.get(node, depl).params == {"id": node}
    assert {(n, d): status.read(n, d).owner for n, d in status.list_pairs()} == {
        ("other", "parser"): "second",
        ("worker%3A1", "llm"): "first",
    }


def test_prefix_keys_retries_partial_range_and_propagates_timeout(
    client: EtcdClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "deployments/worker/llm/desired"
    client.put(key, "value")

    def interrupted_range() -> Iterator[tuple[bytes, SimpleNamespace]]:
        yield (
            b"",
            SimpleNamespace(key=client._mangle_key("deployments/stale/llm/desired")),
        )
        raise ConnectionFailedError()

    scan = Mock(
        wraps=client.client.get_prefix,
        side_effect=[interrupted_range(), DEFAULT, ConnectionTimeoutError()],
    )
    monkeypatch.setattr(client.client, "get_prefix", scan)
    assert client.get_prefix_keys("deployments/") == [key]
    assert scan.call_count == 2
    with pytest.raises(ConnectionTimeoutError):
        list(DesiredStore(client).list_pairs())
