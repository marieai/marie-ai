from types import SimpleNamespace

import pytest
from marie.engine.completion_contract import QueuedCompletionEnvelope
from marie.engine.llm_queue.dispatcher import QueuedBatchDispatcher
from marie.engine.llm_queue.queue_io import (
    InMemoryListQueueClient,
    StoreListQueueClient,
)
from marie.engine.llm_queue.queue_keys import request_queue_key
from marie.engine.llm_queue.scheduler import DrrLaneConfig, DrrLaneScheduler
from test_queue_path import _call, _Logger, _queue_config


def test_drr_transport_peek_failure_never_pops_or_counts_malformed():
    from marie.engine.llm_queue.queue_io import QueueUnavailable

    client = SimpleNamespace(
        request_queue_depth=lambda pool: 1,
        peek_request=lambda pool: (_ for _ in ()).throw(QueueUnavailable()),
        try_pop_request=lambda pool: pytest.fail(
            'transport read must not authorize pop'
        ),
    )
    scheduler = DrrLaneScheduler(
        queue_client=client,
        lanes=[DrrLaneConfig(pool_id='default')],
        total_concurrent_dispatch=1,
    )
    with pytest.raises(QueueUnavailable):
        scheduler.select_next()
    assert scheduler.lane_snapshots()[0].malformed_requests_dropped == 0


def test_malformed_compare_discard_preserves_replaced_head():
    from marie.engine.llm_queue.queue_io import MalformedQueueRequest

    queue = InMemoryListQueueClient()
    key = request_queue_key('default')
    queue._lists[key].append('PHI invalid JSON')
    with pytest.raises(MalformedQueueRequest) as caught:
        queue.peek_request('default')
    assert 'PHI' not in str(caught.value)
    queue._lists[key].popleft()
    queue._lists[key].append('replacement')
    assert not queue.discard_malformed_head('default', caught.value.digest)
    assert queue._lists[key][0] == 'replacement'


def test_valkey_read_errors_are_typed_content_free():
    from marie.engine.llm_queue.queue_io import QueueUnavailable

    queue = object.__new__(StoreListQueueClient)
    queue._client = SimpleNamespace(
        lindex=lambda *args: (_ for _ in ()).throw(ConnectionError('PHI secret'))
    )
    with pytest.raises(QueueUnavailable) as caught:
        queue.peek_request('default')
    assert 'PHI' not in str(caught.value)


def test_fifo_fill_transport_failure_executes_already_popped_item():
    from marie.engine.llm_queue.queue_io import QueueUnavailable

    queue = InMemoryListQueueClient()
    queue.set_producer_alive('p', 'p', 30)
    queue.push_request(
        QueuedCompletionEnvelope(
            request_id='r',
            producer_id='p',
            pool_id='default',
            submitted_at=1,
            call=_call([]),
        )
    )
    queue.try_pop_request = lambda pool: (_ for _ in ()).throw(QueueUnavailable())

    class Adapter:
        async def execute(self, call, **kwargs):
            return {'choices': []}

    runtime = QueuedBatchDispatcher(
        queue_client=queue,
        execution_adapter=Adapter(),
        config=_queue_config(),
        logger=_Logger(),
    )
    assert runtime.run_once() == 1
    assert runtime.health()['transport_failures'] == 1
    assert runtime.health()['malformed_requests_dropped'] == 0
    assert runtime.run_once() == 0
    assert runtime.health()['last_error'] == 'queue_unavailable'


def test_drr_malformed_count_is_aggregated_once():
    from marie.engine.llm_queue import registry
    from marie.engine.llm_queue.dispatcher import DrrQueuedBatchDispatcher

    queue = InMemoryListQueueClient()
    queue._lists[request_queue_key('default')].append('invalid')
    runtime = DrrQueuedBatchDispatcher(
        queue_client=queue,
        execution_adapter=object(),
        config=_queue_config(),
        logger=_Logger(),
        lanes=[DrrLaneConfig(pool_id='default')],
        total_concurrent_dispatch=1,
    )
    assert runtime.run_once() == 0
    registry.register_dispatcher(runtime.dispatcher_id, runtime)
    try:
        snapshot = registry.dispatch_runtime_live_state()
        assert snapshot['runtime_summary']['malformed_requests_dropped'] == 1
        assert snapshot['pool_config'][0]['malformed_requests_dropped'] == 1
    finally:
        registry.unregister_dispatcher(runtime.dispatcher_id)


def test_drr_transport_counter_is_separate_from_malformed():
    from marie.engine.llm_queue.dispatcher import DrrQueuedBatchDispatcher
    from marie.engine.llm_queue.queue_io import QueueUnavailable

    queue = SimpleNamespace(
        request_queue_depth=lambda _: 1,
        peek_request=lambda _: (_ for _ in ()).throw(QueueUnavailable()),
    )
    runtime = DrrQueuedBatchDispatcher(
        queue_client=queue,
        execution_adapter=object(),
        config=_queue_config(),
        logger=_Logger(),
        lanes=[DrrLaneConfig(pool_id='default')],
        total_concurrent_dispatch=1,
    )
    with pytest.raises(QueueUnavailable):
        runtime.run_once()
    assert runtime.health()['transport_failures'] == 1
    assert runtime.health()['malformed_requests_dropped'] == 0


def test_postpop_drr_has_no_fallible_depth_reads():
    from marie.engine.llm_queue.queue_io import QueueUnavailable

    queue = InMemoryListQueueClient()
    from test_scheduler import _request

    request = _request('default', 'one')
    queue.push_request(request)
    original = queue.try_pop_request

    def pop(pool):
        popped = original(pool)
        queue.request_queue_depth = lambda pool: (_ for _ in ()).throw(
            QueueUnavailable()
        )
        return popped

    queue.try_pop_request = pop
    scheduler = DrrLaneScheduler(
        queue_client=queue,
        lanes=[DrrLaneConfig('default')],
        total_concurrent_dispatch=1,
    )
    assert scheduler.select_next().request.request_id == 'one'
    scheduler.release('default')
    assert scheduler.inflight_count == 0


@pytest.mark.parametrize('failure_at', [1, 2])
def test_drr_unknown_liveness_holds_request_and_executes_selected_siblings(failure_at):
    from marie.engine.llm_queue.dispatcher import DrrQueuedBatchDispatcher
    from marie.engine.llm_queue.queue_io import QueueUnavailable
    from test_scheduler import _request

    queue = InMemoryListQueueClient()
    for index in range(2):
        queue.push_request(_request('default', str(index)))
    attempts = []

    def liveness(producer):
        attempts.append(producer)
        if len(attempts) == failure_at:
            raise QueueUnavailable()
        return True

    queue.is_producer_alive = liveness
    dispatcher = DrrQueuedBatchDispatcher(
        queue_client=queue,
        execution_adapter=None,
        config=_queue_config(),
        logger=_Logger(),
        lanes=[DrrLaneConfig('default', quantum=2)],
        total_concurrent_dispatch=2,
    )
    executed = []
    dispatcher._run_popped_batch = lambda batch, **kw: executed.extend(
        req.request_id for req in batch
    ) or len(batch)
    dispatcher.run_once()
    assert len(dispatcher._held_requests) == 1
    assert dispatcher.scheduler.inflight_count == 1
    dispatcher.run_once()
    assert sorted(executed) == ['0', '1']
    assert dispatcher.scheduler.inflight_count == 0
    assert not dispatcher._held_requests


def test_drr_reply_liveness_failure_never_reexecutes_and_retention_expires():
    from marie.engine.llm_queue.dispatcher import DrrQueuedBatchDispatcher
    from marie.engine.llm_queue.queue_io import QueueUnavailable
    from marie.engine.llm_queue.result_types import BatchResult
    from test_scheduler import _request

    queue = InMemoryListQueueClient()
    queue.push_request(_request('default', 'one'))
    live = [True, QueueUnavailable(), True]

    def liveness(producer):
        value = live.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    queue.is_producer_alive = liveness
    dispatcher = DrrQueuedBatchDispatcher(
        queue_client=queue,
        execution_adapter=None,
        config=_queue_config(),
        logger=_Logger(),
        lanes=[DrrLaneConfig('default')],
        total_concurrent_dispatch=1,
    )
    sends = []
    dispatcher._execute_batch = lambda batch: sends.append(1) or [
        BatchResult('one', {'choices': [{'message': {'content': 'ok'}}]}, None)
    ]
    assert dispatcher.run_once() == 1
    assert len(dispatcher._pending_replies) == 1
    assert dispatcher.scheduler.inflight_count == 0
    dispatcher.run_once()
    assert sends == [1]
    assert not dispatcher._pending_replies
    queue.push_request(_request('default', 'two'))
    queue.is_producer_alive = lambda producer: (_ for _ in ()).throw(QueueUnavailable())
    dispatcher.run_once()
    dispatcher._retained_until['two'] = 0
    dispatcher._expire_retained()
    assert dispatcher.scheduler.inflight_count == 0
    assert not dispatcher._held_requests


@pytest.mark.parametrize('policy', ['fifo', 'drr'])
def test_expired_terminal_replies_clear_snapshots_across_cycles_and_shutdown(policy):
    from marie.engine.llm_queue.dispatcher import DrrQueuedBatchDispatcher
    from marie.engine.llm_queue.queue_io import QueueUnavailable
    from marie.engine.llm_queue.result_types import BatchResult
    from test_scheduler import _request

    queue = InMemoryListQueueClient()
    queue.is_producer_alive = lambda _: True
    queue.push_reply = lambda *args, **kwargs: (_ for _ in ()).throw(QueueUnavailable())
    options = dict(
        queue_client=queue,
        execution_adapter=None,
        config=_queue_config(pool_id='default', max_batch_items=1),
        logger=_Logger(),
    )
    if policy == 'drr':
        dispatcher = DrrQueuedBatchDispatcher(
            **options, lanes=[DrrLaneConfig('default')], total_concurrent_dispatch=1
        )
    else:
        dispatcher = QueuedBatchDispatcher(**options)
    sends = []

    def execute(batch):
        sends.extend(request.request_id for request in batch)
        return [
            BatchResult(request.request_id, {'choices': []}, None) for request in batch
        ]

    dispatcher._execute_batch = execute
    for index in range(3):
        request_id = str(index)
        queue.push_request(_request('default', request_id))
        assert dispatcher.run_once() == 1
        assert len(dispatcher._pending_replies) == 1
        dispatcher._retained_until[request_id] = 0
        dispatcher._expire_retained()
        assert not dispatcher._pending_replies
        assert not dispatcher._retained_until
    assert dispatcher.inflight_requests_snapshot() == []
    assert QueuedBatchDispatcher.health(dispatcher)['inflight_request_count'] == 0
    assert sends == ['0', '1', '2']

    queue.push_request(_request('default', 'shutdown'))
    assert dispatcher.run_once() == 1
    assert len(dispatcher.inflight_requests_snapshot()) == 1
    dispatcher.stop()
    assert dispatcher.inflight_requests_snapshot() == []
    assert QueuedBatchDispatcher.health(dispatcher)['inflight_request_count'] == 0
    assert not dispatcher._pending_replies
    assert not dispatcher._retained_until
    assert sends == ['0', '1', '2', 'shutdown']
    if policy == 'drr':
        assert dispatcher.scheduler.inflight_count == 0
