from typing import Any

import pytest
from marie.engine.llm_queue.queue_keys import QueueKeys
from marie.engine.llm_queue.store import RequestStore, StoreLimits, StoreUnavailable


class MetadataClient:
    def __init__(self, *, failed: str | None = None) -> None:
        self.failed = failed
        self.reads: list[tuple[str, Any]] = []
        self.records = {
            'bad': {
                'producer_id': 'producer',
                'schema_version': '1',
                'state': 'ready',
                'cost': 'PHI private content',
            },
            'good': {
                'producer_id': 'producer',
                'schema_version': '1',
                'state': 'ready',
                'cost': '2',
                'pool_id': 'pool',
            },
        }

    def read(self, method: str, detail: Any) -> None:
        self.reads.append((method, detail))
        if method == self.failed:
            raise ConnectionError('PHI private credential data:image')

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        self.read('lrange', key)
        return ['bad', 'good'][start : end + 1]

    def hmget(self, key: str, *names: str) -> list[str | None]:
        self.read('hmget', names)
        assert 'payload' not in names and 'result' not in names
        record = self.records.get(key.rsplit(':', 1)[-1], {})
        return [record.get(name) for name in names]

    def lindex(self, key: str, index: int) -> str:
        self.read('lindex', key)
        return 'good'

    def llen(self, key: str) -> int:
        self.read('llen', key)
        return 2


def metadata_store(client: MetadataClient) -> RequestStore:
    store = object.__new__(RequestStore)
    store.client = client
    store.keys = QueueKeys('fabric')
    store.limits = StoreLimits()
    store._client_error = ConnectionError
    return store


@pytest.mark.parametrize('method', ['lrange', 'hmget', 'lindex', 'llen'])
def test_durable_read_errors_are_typed_and_content_free(method: str) -> None:
    client = MetadataClient(failed=method)
    store = metadata_store(client)
    read = {
        'lrange': lambda: store.ready_metadata('pool', limit=2),
        'hmget': lambda: store.metadata('good'),
        'lindex': lambda: store.ready_head('pool'),
        'llen': lambda: store.ready_depth('pool'),
    }[method]
    with pytest.raises(StoreUnavailable) as caught:
        read()
    assert str(caught.value) == 'Queue store operation failed; state is unknown'
    assert caught.value.__cause__ is None
    assert [name for name, _ in client.reads] == [method]


def test_ready_metadata_preserves_valid_rows_beside_corruption_without_body_reads() -> (
    None
):
    client = MetadataClient()
    store = metadata_store(client)
    rows, malformed = store.ready_metadata('pool', limit=2)
    assert malformed == 1
    assert [row.attempt_id for row in rows] == ['good']
    assert rows[0].cost == 2
    assert [name for name, _ in client.reads] == ['lrange', 'hmget', 'hmget']


def test_transport_failure_is_not_counted_as_malformed_or_empty_metadata() -> None:
    client = MetadataClient(failed='hmget')
    store = metadata_store(client)
    with pytest.raises(StoreUnavailable):
        store.ready_metadata('pool', limit=2)
    assert [name for name, _ in client.reads] == ['lrange', 'hmget']


def test_depth_transport_failure_never_becomes_successful_zero() -> None:
    store = metadata_store(MetadataClient(failed='llen'))
    with pytest.raises(StoreUnavailable):
        store.ready_depth('pool')


def test_missing_record_is_accounted_without_touching_prompt_or_result() -> None:
    client = MetadataClient()
    client.records.pop('bad')
    store = metadata_store(client)
    rows, malformed = store.ready_metadata('pool', limit=2)
    assert malformed == 1
    assert [row.attempt_id for row in rows] == ['good']


@pytest.mark.asyncio
async def test_dispatch_head_read_failure_cannot_authorize_claim_or_charge() -> None:
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    client = MetadataClient(failed='lindex')
    store = metadata_store(client)
    store.server_time_ms = lambda: 1
    store.usage = lambda: {'reserved_items': 0}
    store.route_status = lambda pool: {
        'reserved_items': 0,
        'reserved_bytes': 0,
        'enabled': '1',
    }
    store.claim_and_charge = lambda *args, **kwargs: pytest.fail(
        'failed head read cannot authorize a claim'
    )
    runtime = RequestDispatcher(
        store=store,
        endpoints=[RegisteredEndpoint('endpoint', 'https://example.test')],
        lanes=[DispatchLane('pool', 'endpoint')],
    )
    try:
        with pytest.raises(StoreUnavailable):
            await runtime._dispatch()
        assert runtime.scheduler.inflight_count == 0
        assert runtime.scheduler.lane_metadata('pool')['committed_charge'] == 0
        assert runtime._counts['invalid_head_metadata'] == 0
    finally:
        for worker in (
            runtime._store_workers,
            runtime._lease_worker,
            runtime._refresh_worker,
        ):
            worker.shutdown()
