import asyncio
import threading
import time
from copy import deepcopy
from unittest import mock

import pytest

from marie.engine.batch_processor import BatchProcessor
from marie.engine.completion_contract import (
    COMPLETION_QUEUE_CONTRACT_VERSION,
    CompletionCallParams,
    CompletionReplyEnvelope,
    QueuedCompletionEnvelope,
    build_completion_call,
    completion_payload_to_text,
)
from marie.engine.llm_queue import dispatcher as dispatcher_module
from marie.engine.llm_queue import queue_io as queue_io_module
from marie.engine.llm_queue.adapters.openai_compatible import (
    OpenAICompatibleExecutionAdapter,
)
from marie.engine.llm_queue.config import (
    DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
    LlmQueueConfig,
)
from marie.engine.llm_queue.dispatcher import QueuedBatchDispatcher
from marie.engine.llm_queue.queue_io import (
    InMemoryListQueueClient,
    ValkeyListQueueClient,
)
from marie.engine.llm_queue.registry import (
    dispatch_runtime_live_state,
    dispatch_runtime_snapshot,
    register_dispatcher,
    unregister_dispatcher,
)
from marie.engine.llm_queue.result_types import BatchResult
from marie.engine.llm_queue.submitter import QueuedBatchExecutor, _reply_to_batch_result
from marie.engine.llm_queue.valkey_keys import (
    producer_alive_key,
    queue_namespace,
    reply_queue_key,
    request_queue_key,
)


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _queue_config(**overrides) -> LlmQueueConfig:
    values = dict(
        enabled=True,
        valkey_url=None,
        pool_id="default",
        producer_id="producer-A",
        producer_ttl_seconds=5,
        producer_refresh_interval_seconds=0.05,
        reply_queue_ttl_seconds=60,
        reply_pop_timeout_seconds=0.05,
        dispatch_pop_timeout_seconds=0.05,
        max_batch_items=8,
        max_batch_wait_ms=25,
        max_buffered_requests_per_pool=8,
        max_inline_payload_bytes=DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
    )
    values.update(overrides)
    return LlmQueueConfig(**values)


def _call(messages, **overrides) -> CompletionCallParams:
    return build_completion_call(
        model="test-model",
        messages=messages,
        completion_params=overrides or None,
        stream=False,
    )


def _completion_payload(text: str, *, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"content": text},
                "finish_reason": finish_reason,
            }
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _CircuitBreaker:
    def is_available(self, address):
        return True

    def get_state(self, address):
        return "closed"

    def record_success(self, address):
        pass

    def record_failure(self, address):
        pass


class _FakePipeline:
    def __init__(self, client):
        self._client = client
        self._ops = []

    def rpush(self, key, value):
        self._ops.append(("rpush", key, value))
        return self

    def expire(self, key, ttl_seconds):
        self._ops.append(("expire", key, ttl_seconds))
        return self

    def execute(self):
        for op, *args in self._ops:
            getattr(self._client, op)(*args)
        self._client.pipeline_exec_count += 1
        self._ops.clear()


class _FakeSyncQueueBackend:
    def __init__(self):
        self.lists = {}
        self.kv = {}
        self.expiry = {}
        self.blpop_calls = []
        self.pipeline_exec_count = 0
        self.closed = False
        self._lock = threading.Lock()

    def rpush(self, key, value):
        with self._lock:
            self.lists.setdefault(key, []).append(value)

    def lpush(self, key, value):
        with self._lock:
            self.lists.setdefault(key, []).insert(0, value)

    def lpop(self, key):
        with self._lock:
            values = self.lists.get(key, [])
            if not values:
                return None
            return values.pop(0)

    def lindex(self, key, index):
        with self._lock:
            values = self.lists.get(key, [])
            if not values:
                return None
            return values[index]

    def blpop(self, key, timeout):
        with self._lock:
            self.blpop_calls.append((key, timeout))
        value = self.lpop(key)
        if value is None:
            return None
        return (key, value)

    def pipeline(self):
        return _FakePipeline(self)

    def expire(self, key, ttl_seconds):
        with self._lock:
            self.expiry[key] = ttl_seconds

    def set(self, key, value, ex=None):
        with self._lock:
            self.kv[key] = value
            if ex is not None:
                self.expiry[key] = ex

    def exists(self, key):
        with self._lock:
            return key in self.kv

    def llen(self, key):
        with self._lock:
            return len(self.lists.get(key, []))

    def lrange(self, key, start, stop):
        with self._lock:
            values = list(self.lists.get(key, []))
        if stop < 0:
            end = None
        else:
            end = stop + 1
        return values[start:end]

    def delete(self, key):
        with self._lock:
            self.kv.pop(key, None)
            self.expiry.pop(key, None)

    def close(self):
        with self._lock:
            self.closed = True


def _build_processor(*, client, queue_enabled: bool, queue_client=None) -> BatchProcessor:
    processor = object.__new__(BatchProcessor)
    processor.client = client
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = 4
    processor.batch_timeout = 2.0
    processor.backend_address = "http://test-backend:8000"
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    processor._circuit_breaker = _CircuitBreaker()
    processor._gate_lock = None
    processor._gate_lock_loop = None
    processor._queue_client = queue_client
    processor._queued_executor = None
    processor._queue_config = _queue_config(enabled=queue_enabled)
    return processor


def test_queued_batch_executor_demultiplexes_replies_from_same_producer():
    queue_client = InMemoryListQueueClient()
    executor = QueuedBatchExecutor(
        queue_client=queue_client,
        config=_queue_config(),
        logger=_Logger(),
    )

    def worker():
        requests = []
        for _ in range(2):
            request = queue_client.pop_request("default", timeout=1.0)
            assert request is not None
            requests.append(request)

        for request in reversed(requests):
            queue_client.push_reply(
                CompletionReplyEnvelope(
                    request_id=request.request_id,
                    producer_id=request.producer_id,
                    pool_id=request.pool_id,
                    status="ok",
                    completion=_completion_payload(f"resp:{request.request_id}"),
                    completed_at=time.time(),
                ),
                ttl_seconds=60,
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    results = executor.execute(
        calls=[
            _call([{"role": "user", "content": "a"}]),
            _call([{"role": "user", "content": "b"}]),
        ],
        batch_request_id="batch-1",
        batch_timeout=2.0,
    )

    assert [result.task_id for result in results] == [
        "batch-1_task_0",
        "batch-1_task_1",
    ]
    assert [result.response for result in results] == [
        "resp:batch-1_task_0",
        "resp:batch-1_task_1",
    ]


def test_queued_batch_executor_skips_malformed_reply_and_keeps_waiting():
    queue_client = InMemoryListQueueClient()
    executor = QueuedBatchExecutor(
        queue_client=queue_client,
        config=_queue_config(),
        logger=_Logger(),
    )

    def worker():
        request = queue_client.pop_request("default", timeout=1.0)
        assert request is not None

        queue_client._lists[reply_queue_key("producer-A")].append(
            CompletionReplyEnvelope(
                request_id=request.request_id,
                producer_id=request.producer_id,
                pool_id=request.pool_id,
                status="ok",
                completion=_completion_payload("bad"),
                completed_at=time.time(),
            ).to_json().replace(
                f"\"contract_version\":\"{COMPLETION_QUEUE_CONTRACT_VERSION}\"",
                "\"contract_version\":\"v1\"",
            )
        )
        queue_client.push_reply(
            CompletionReplyEnvelope(
                request_id=request.request_id,
                producer_id=request.producer_id,
                pool_id=request.pool_id,
                status="ok",
                completion=_completion_payload("good"),
                completed_at=time.time(),
            ),
            ttl_seconds=60,
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    results = executor.execute(
        calls=[_call([{"role": "user", "content": "hello"}])],
        batch_request_id="batch-malformed-reply",
        batch_timeout=2.0,
    )

    assert [result.task_id for result in results] == ["batch-malformed-reply_task_0"]
    assert [result.response for result in results] == ["good"]


def test_queue_keyspace_is_versioned():
    assert queue_namespace() == f"llm:{COMPLETION_QUEUE_CONTRACT_VERSION}"
    assert request_queue_key("default") == "list:llm:v2:requests:default"
    assert reply_queue_key("producer-A") == "list:llm:v2:replies:producer-A"
    assert producer_alive_key("producer-A") == "key:llm:v2:producer:producer-A:alive"


def test_valkey_list_queue_client_round_trip_and_pipeline_ttl(monkeypatch):
    backend = _FakeSyncQueueBackend()
    monkeypatch.setattr(queue_io_module, "_build_sync_client", lambda url: backend)

    queue_client = ValkeyListQueueClient("valkey://unit-test")
    request = QueuedCompletionEnvelope(
        request_id="req-1",
        producer_id="producer-A",
        pool_id="default",
        submitted_at=time.time(),
        call=_call([{"role": "user", "content": "hello"}]),
    )
    reply = CompletionReplyEnvelope(
        request_id="req-1",
        producer_id="producer-A",
        pool_id="default",
        status="ok",
        completion=_completion_payload("done"),
        completed_at=time.time(),
    )

    queue_client.push_request(request)
    peeked_request = queue_client.peek_request("default")
    assert peeked_request is not None
    assert peeked_request.request_id == "req-1"
    assert queue_client.request_queue_depth("default") == 1

    sampled_requests = queue_client.sample_requests("default", limit=10)
    assert [item.request_id for item in sampled_requests] == ["req-1"]
    restored_request = queue_client.pop_request("default", timeout=0.1)
    assert restored_request is not None
    assert restored_request.request_id == "req-1"
    assert backend.blpop_calls[-1] == (request_queue_key("default"), 1)

    queue_client.push_request_front(request)
    assert queue_client.try_pop_request("default").request_id == "req-1"
    assert queue_client.request_queue_depth("default") == 0
    assert queue_client.sample_requests("default", limit=10) == []

    queue_client.push_reply(reply, ttl_seconds=60)
    assert backend.pipeline_exec_count == 1
    assert backend.expiry[reply_queue_key("producer-A")] == 60
    restored_reply = queue_client.pop_reply("producer-A", timeout=0.1)
    assert restored_reply is not None
    assert restored_reply.request_id == "req-1"

    queue_client.set_producer_alive("producer-A", "alive", 30)
    assert queue_client.is_producer_alive("producer-A") is True
    assert backend.expiry[producer_alive_key("producer-A")] == 30
    queue_client.clear_producer_alive("producer-A")
    assert queue_client.is_producer_alive("producer-A") is False

    queue_client.close()
    assert backend.closed is True


def test_valkey_queue_runtime_end_to_end_with_dispatcher_thread(monkeypatch):
    backend = _FakeSyncQueueBackend()
    monkeypatch.setattr(queue_io_module, "_build_sync_client", lambda url: backend)

    queue_client = ValkeyListQueueClient("valkey://unit-test")

    class _Completions:
        async def create(self, **kwargs):
            request_text = kwargs["messages"][-1]["content"]
            return type(
                "Completion",
                (),
                {
                    "model_dump": lambda self: _completion_payload(
                        f"done:{request_text}"
                    )
                },
            )()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=OpenAICompatibleExecutionAdapter(
            client=_Client(),
            logger=_Logger(),
            default_timeout=2.0,
        ),
        config=_queue_config(),
        logger=_Logger(),
    )
    executor = QueuedBatchExecutor(
        queue_client=queue_client,
        config=_queue_config(),
        logger=_Logger(),
    )

    callback_results = []
    dispatcher.start()
    try:
        results = executor.execute(
            calls=[
                _call([{"role": "user", "content": "alpha"}]),
                _call([{"role": "user", "content": "beta"}]),
            ],
            batch_request_id="batch-e2e",
            batch_timeout=2.0,
            on_result=lambda task_id, response: callback_results.append(
                (task_id, response)
            ),
        )
    finally:
        dispatcher.stop()
        executor.close()
        queue_client.close()

    assert [result.task_id for result in results] == [
        "batch-e2e_task_0",
        "batch-e2e_task_1",
    ]
    assert [result.response for result in results] == ["done:alpha", "done:beta"]
    assert sorted(callback_results) == [
        ("batch-e2e_task_0", "done:alpha"),
        ("batch-e2e_task_1", "done:beta"),
    ]
    assert dispatcher.health()["processed_items"] == 2
    assert backend.closed is True


def test_envelopes_reject_mismatched_contract_version():
    request_payload = QueuedCompletionEnvelope(
        request_id="req-1",
        producer_id="producer-A",
        pool_id="default",
        submitted_at=time.time(),
        call=_call([{"role": "user", "content": "hello"}]),
    ).to_json().replace(
        f"\"contract_version\":\"{COMPLETION_QUEUE_CONTRACT_VERSION}\"",
        "\"contract_version\":\"v1\"",
    )
    with pytest.raises(ValueError, match="Unsupported request contract version"):
        QueuedCompletionEnvelope.from_json(request_payload)

    reply_payload = CompletionReplyEnvelope(
        request_id="req-1",
        producer_id="producer-A",
        pool_id="default",
        status="ok",
        completion=_completion_payload("hello"),
        completed_at=time.time(),
    ).to_json().replace(
        f"\"contract_version\":\"{COMPLETION_QUEUE_CONTRACT_VERSION}\"",
        "\"contract_version\":\"v1\"",
    )
    with pytest.raises(ValueError, match="Unsupported reply contract version"):
        CompletionReplyEnvelope.from_json(reply_payload)


class _FakeAdapter:
    def __init__(self):
        self.requests = []

    async def execute(self, call, *, timeout_seconds=None):
        request_text = call.messages[-1]["content"]
        self.requests.append(request_text)
        return type(
            "Completion",
            (),
            {"model_dump": lambda self: _completion_payload(f"done:{request_text}")},
        )()


def test_dispatcher_drops_requests_for_dead_producers():
    queue_client = InMemoryListQueueClient()
    config = _queue_config(producer_id="producer-live")
    adapter = _FakeAdapter()
    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=adapter,
        config=config,
        logger=_Logger(),
    )

    queue_client.set_producer_alive("producer-live", "producer-live", 5)

    live_request = QueuedCompletionEnvelope(
        request_id="live-1",
        producer_id="producer-live",
        pool_id="default",
        submitted_at=time.time(),
        call=_call([{"role": "user", "content": "keep"}]),
    )
    dead_request = QueuedCompletionEnvelope(
        request_id="dead-1",
        producer_id="producer-dead",
        pool_id="default",
        submitted_at=time.time(),
        call=_call([{"role": "user", "content": "drop"}]),
    )

    queue_client.push_request(dead_request)
    queue_client.push_request(live_request)

    dispatched = dispatcher.run_once()
    assert dispatched == 1
    assert adapter.requests == ["keep"]
    assert dispatcher.health()["offline_producer_requests_dropped"] == 1

    reply = queue_client.pop_reply("producer-live", timeout=1.0)
    assert reply is not None
    assert reply.request_id == "live-1"
    assert completion_payload_to_text(reply.completion) == "done:keep"


def test_dispatcher_skips_malformed_request_and_processes_next_live_one():
    queue_client = InMemoryListQueueClient()
    config = _queue_config(producer_id="producer-live")
    adapter = _FakeAdapter()
    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=adapter,
        config=config,
        logger=_Logger(),
    )

    queue_client.set_producer_alive("producer-live", "producer-live", 5)

    malformed_payload = QueuedCompletionEnvelope(
        request_id="bad-1",
        producer_id="producer-live",
        pool_id="default",
        submitted_at=time.time(),
        call=_call([{"role": "user", "content": "stale"}]),
    ).to_json().replace(
        f"\"contract_version\":\"{COMPLETION_QUEUE_CONTRACT_VERSION}\"",
        "\"contract_version\":\"v1\"",
    )
    queue_client._lists[request_queue_key("default")].append(malformed_payload)

    queue_client.push_request(
        QueuedCompletionEnvelope(
            request_id="live-1",
            producer_id="producer-live",
            pool_id="default",
            submitted_at=time.time(),
            call=_call([{"role": "user", "content": "keep"}]),
        )
    )

    assert dispatcher.run_once() == 1
    assert adapter.requests == ["keep"]
    assert dispatcher.health()["malformed_requests_dropped"] == 1

    reply = queue_client.pop_reply("producer-live", timeout=1.0)
    assert reply is not None
    assert reply.request_id == "live-1"
    assert completion_payload_to_text(reply.completion) == "done:keep"


def test_dispatcher_records_openinference_input_and_output(monkeypatch):
    queue_client = InMemoryListQueueClient()
    config = _queue_config(producer_id="producer-live")
    adapter = _FakeAdapter()
    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=adapter,
        config=config,
        logger=_Logger(),
    )
    messages = [{"role": "user", "content": "keep"}]
    recorded_io = []

    def record_llm_io(span, *, input_messages=None, output_messages=None):
        recorded_io.append(
            {
                "input_messages": input_messages,
                "output_messages": output_messages,
            }
        )

    monkeypatch.setattr(dispatcher_module, "set_llm_io", record_llm_io)
    queue_client.set_producer_alive("producer-live", "producer-live", 5)
    queue_client.push_request(
        QueuedCompletionEnvelope(
            request_id="live-1",
            producer_id="producer-live",
            pool_id="default",
            submitted_at=time.time(),
            call=_call(messages),
        )
    )

    assert dispatcher.run_once() == 1
    assert recorded_io == [
        {"input_messages": messages, "output_messages": None},
        {"input_messages": None, "output_messages": "done:keep"},
    ]


def test_batch_processor_uses_queued_executor_when_enabled(monkeypatch):
    processor = object.__new__(BatchProcessor)
    processor.client = None
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = 20
    processor.batch_timeout = 5.0
    processor.backend_address = "http://test-backend:8000"
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    processor._circuit_breaker = None
    processor._gate_lock = None
    processor._gate_lock_loop = None
    processor._queue_client = None
    processor._queued_executor = None
    processor._queue_config = _queue_config()

    class _QueuedExecutor:
        def execute(self, **kwargs):
            return [
                BatchResult("req_task_0", "resp-0", None),
                BatchResult("req_task_1", "resp-1", None),
            ]

    monkeypatch.setattr(processor, "_get_queued_executor", lambda: _QueuedExecutor())

    responses = processor.batch_generate(
        messages_list=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        guided_json=None,
    )

    assert responses == ["resp-0", "resp-1"]


def test_batch_processor_builds_queue_dispatcher():
    processor = object.__new__(BatchProcessor)
    processor.client = None
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = 20
    processor.batch_timeout = 5.0
    processor.backend_address = "http://test-backend:8000"
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    processor._circuit_breaker = None
    processor._gate_lock = None
    processor._gate_lock_loop = None
    processor._queue_client = InMemoryListQueueClient()
    processor._queued_executor = None
    processor._queue_config = _queue_config()

    dispatcher = processor.build_queue_dispatcher()

    assert isinstance(dispatcher, QueuedBatchDispatcher)
    assert dispatcher.health()["pool_id"] == "default"


def test_queued_batch_executor_rejects_oversized_payload():
    queue_client = InMemoryListQueueClient()
    executor = QueuedBatchExecutor(
        queue_client=queue_client,
        config=_queue_config(max_inline_payload_bytes=64),
        logger=_Logger(),
    )

    with pytest.raises(ValueError, match="max inline payload"):
        executor.execute(
            calls=[_call([{"role": "user", "content": "x" * 1024}])],
            batch_request_id="batch-too-large",
            batch_timeout=2.0,
        )
    assert queue_client.try_pop_request("default") is None


def test_dispatcher_drops_reply_if_producer_goes_offline_after_execute():
    queue_client = InMemoryListQueueClient()
    config = _queue_config(producer_id="producer-live")

    class _Adapter:
        async def execute(self, call, *, timeout_seconds=None):
            queue_client.clear_producer_alive("producer-live")
            request_text = call.messages[-1]["content"]
            return type(
                "Completion",
                (),
                {"model_dump": lambda self: _completion_payload(f"done:{request_text}")},
            )()

    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=_Adapter(),
        config=config,
        logger=_Logger(),
    )

    queue_client.set_producer_alive("producer-live", "producer-live", 5)
    queue_client.push_request(
        QueuedCompletionEnvelope(
            request_id="req-1",
            producer_id="producer-live",
            pool_id="default",
            submitted_at=time.time(),
            call=_call([{"role": "user", "content": "keep"}]),
        ),
    )

    assert dispatcher.run_once() == 1
    assert dispatcher.health()["offline_producer_replies_dropped"] == 1
    assert queue_client.pop_reply("producer-live", timeout=0.1) is None


def test_dispatcher_lifecycle_health():
    queue_client = InMemoryListQueueClient()
    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=_FakeAdapter(),
        config=_queue_config(
            fabric_group_id="runtime-fabric-default",
            gateway_id="gateway-localhost",
        ),
        logger=_Logger(),
    )

    dispatcher.start()
    try:
        health = dispatcher.health()
        assert health["running"] is True
        assert health["pool_id"] == "default"
        assert health["fabric_group_id"] == "runtime-fabric-default"
        assert health["gateway_id"] == "gateway-localhost"
        assert health["request_queue_depth"] == 0
        assert health["malformed_requests_dropped"] == 0
        assert health["offline_producer_requests_dropped"] == 0
        assert health["offline_producer_replies_dropped"] == 0
        snapshot = dispatch_runtime_snapshot()
        assert snapshot["registered_dispatchers"] >= 1
        assert snapshot["running_dispatchers"] >= 1
        assert any(
            item["dispatcher_id"] == health["dispatcher_id"]
            for item in snapshot["dispatchers"]
        )
    finally:
        dispatcher.stop()

    assert dispatcher.health()["running"] is False
    assert dispatch_runtime_snapshot()["registered_dispatchers"] == 0


def test_dispatch_runtime_live_state_merges_pending_and_inflight_requests():
    queue_client = InMemoryListQueueClient()
    started = threading.Event()
    release = threading.Event()

    class _SlowAdapter:
        async def execute(self, call, *, timeout_seconds=None):
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return type(
                "Completion",
                (),
                {
                    "model_dump": lambda self: _completion_payload(
                        f"done:{call.messages[-1]['content']}"
                    )
                },
            )()

    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=_SlowAdapter(),
        config=_queue_config(max_batch_items=1, max_batch_wait_ms=1),
        logger=_Logger(),
    )

    queue_client.set_producer_alive("producer-A", "producer-A", 5)
    queue_client.push_request(
        QueuedCompletionEnvelope(
            request_id="req-inflight",
            producer_id="producer-A",
            pool_id="default",
            submitted_at=time.time(),
            call=_call([{"role": "user", "content": "first prompt"}]),
        )
    )
    queue_client.push_request(
        QueuedCompletionEnvelope(
            request_id="req-pending",
            producer_id="producer-A",
            pool_id="default",
            submitted_at=time.time(),
            call=_call([{"role": "user", "content": "second prompt"}]),
        )
    )

    dispatcher.start()
    try:
        assert started.wait(timeout=2.0) is True

        runtime_state = dispatch_runtime_live_state(limit_per_pool=10)
        summary = runtime_state["runtime_summary"]
        live_requests = runtime_state["live_requests"]
        pool_config = runtime_state["pool_config"]

        assert summary["pending_request_count"] == 1
        assert summary["pending_request_sample_count"] == 1
        assert summary["inflight_request_count"] == 1
        assert summary["live_request_sample_limit_per_pool"] == 10
        assert pool_config[0]["scheduler_policy"] == "fifo"
        assert pool_config[0]["pool_id"] == "default"
        assert pool_config[0]["request_queue_depth"] == 1

        by_id = {item["request_id"]: item for item in live_requests}
        assert by_id["req-pending"]["state_source"] == "valkey"
        assert by_id["req-pending"]["lifecycle_stage"] == "pending"
        assert by_id["req-pending"]["dispatcher_id"] is None
        assert by_id["req-pending"]["estimated_cost_units"] == 1
        assert by_id["req-pending"]["request_summary"] == "user: second prompt"

        assert by_id["req-inflight"]["state_source"] == "dispatcher"
        assert by_id["req-inflight"]["lifecycle_stage"] == "executing"
        assert by_id["req-inflight"]["dispatcher_id"] == dispatcher.dispatcher_id
        assert by_id["req-inflight"]["inflight_age_seconds"] is not None
    finally:
        release.set()
        dispatcher.stop()


def test_inflight_snapshot_recomputes_age_on_each_read(monkeypatch):
    queue_client = InMemoryListQueueClient()
    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=_FakeAdapter(),
        config=_queue_config(),
        logger=_Logger(),
    )
    request = QueuedCompletionEnvelope(
        request_id="req-inflight",
        producer_id="producer-A",
        pool_id="default",
        submitted_at=95.0,
        call=_call([{"role": "user", "content": "first prompt"}]),
    )
    current_time = {"value": 100.0}
    monkeypatch.setattr(
        dispatcher_module.time,
        "time",
        lambda: current_time["value"],
    )

    dispatcher._mark_request_popped(request, lifecycle_stage="executing")
    first_snapshot = dispatcher.inflight_requests_snapshot()[0]
    current_time["value"] = 107.0
    second_snapshot = dispatcher.inflight_requests_snapshot()[0]

    assert first_snapshot["queue_wait_age_seconds"] == 5.0
    assert second_snapshot["queue_wait_age_seconds"] == 5.0
    assert first_snapshot["inflight_age_seconds"] == 0.0
    assert second_snapshot["inflight_age_seconds"] == 7.0


def test_dispatch_runtime_live_state_reports_full_pending_depth_with_sample_cap():
    queue_client = InMemoryListQueueClient()

    class _NeverUsedAdapter:
        async def execute(self, call, *, timeout_seconds=None):
            raise AssertionError("execute() should not be called in this test")

    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=_NeverUsedAdapter(),
        config=_queue_config(max_batch_items=1, max_batch_wait_ms=1),
        logger=_Logger(),
    )

    queue_client.set_producer_alive("producer-A", "producer-A", 5)
    for index in range(3):
        queue_client.push_request(
            QueuedCompletionEnvelope(
                request_id=f"req-{index}",
                producer_id="producer-A",
                pool_id="default",
                submitted_at=time.time(),
                call=_call([{"role": "user", "content": f"prompt-{index}"}]),
            )
        )

    try:
        register_dispatcher(dispatcher.dispatcher_id, dispatcher)
        runtime_state = dispatch_runtime_live_state(limit_per_pool=1)
        summary = runtime_state["runtime_summary"]

        assert summary["pending_request_count"] == 3
        assert summary["pending_request_sample_count"] == 1
        assert summary["live_request_sample_limit_per_pool"] == 1
        assert len(runtime_state["live_requests"]) == 1
        assert runtime_state["live_requests"][0]["state_source"] == "valkey"
    finally:
        unregister_dispatcher(dispatcher.dispatcher_id)


def test_openai_compatible_execution_adapter_enforces_timeout():
    class _Completions:
        async def create(self, **kwargs):
            await asyncio.sleep(0.2)

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    adapter = OpenAICompatibleExecutionAdapter(
        client=_Client(),
        logger=_Logger(),
        default_timeout=600.0,
    )
    with pytest.raises(TimeoutError):
        asyncio.run(
            adapter.execute(
                _call([{"role": "user", "content": "slow"}]),
                timeout_seconds=0.05,
            )
        )


def test_openai_compatible_execution_adapter_adds_endpoint_context():
    class _Completions:
        async def create(self, **kwargs):
            raise ConnectionError("connection refused")

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    adapter = OpenAICompatibleExecutionAdapter(
        client=_Client(),
        logger=_Logger(),
        default_timeout=600.0,
        backend_address="http://llm-backend:8000/v1",
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            adapter.execute(
                _call([{"role": "user", "content": "hello"}]),
                timeout_seconds=0.05,
            )
        )

    assert "http://llm-backend:8000/v1" in str(exc_info.value)
    assert "ConnectionError: connection refused" in str(exc_info.value)


def test_openai_compatible_execution_adapter_factory_closes_owned_client():
    payload = _completion_payload("done")
    clients = []

    class _Completions:
        async def create(self, **kwargs):
            return deepcopy(payload)

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    def client_factory():
        client = _Client()
        clients.append(client)
        return client

    adapter = OpenAICompatibleExecutionAdapter(
        client_factory=client_factory,
        logger=_Logger(),
        default_timeout=600.0,
        backend_address="http://llm-backend:8000/v1",
    )

    completion = asyncio.run(
        adapter.execute(
            _call([{"role": "user", "content": "hello"}]),
            timeout_seconds=0.05,
        )
    )

    assert completion == payload
    assert len(clients) == 1
    assert clients[0].closed is True


def test_reply_to_batch_result_raises_max_tokens_for_length_finish_reason():
    reply = CompletionReplyEnvelope(
        request_id="req-1",
        producer_id="producer-A",
        pool_id="default",
        status="ok",
        completion=_completion_payload("partial", finish_reason="length"),
        completed_at=time.time(),
    )

    result = _reply_to_batch_result(reply)
    assert result.response is None
    assert str(result.error) == "LLM hit max_tokens"


def test_reply_to_batch_result_preserves_dispatcher_error_origin():
    reply = CompletionReplyEnvelope(
        request_id="req-1",
        producer_id="producer-A",
        pool_id="default",
        status="error",
        completed_at=time.time(),
        error_type="RuntimeError",
        error_message="OpenAI-compatible dispatch failed",
        error_source="llm_dispatcher",
        dispatcher_id="default:dispatcher-1",
        execution_backend_address="http://llm-backend:8000/v1",
    )

    result = _reply_to_batch_result(reply)

    assert result.response is None
    assert (
        str(result.error)
        == "RuntimeError from llm_dispatcher dispatcher=default:dispatcher-1 "
        "backend=http://llm-backend:8000/v1: OpenAI-compatible dispatch failed"
    )


def test_multimodal_queue_request_round_trip_preserves_content_shape():
    queue_client = InMemoryListQueueClient()
    request = QueuedCompletionEnvelope(
        request_id="req-mm-1",
        producer_id="producer-A",
        pool_id="default",
        submitted_at=time.time(),
        call=_call(
            [
                {"role": "system", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe the image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,ZmFrZQ==",
                            },
                            "min_pixels": 401408,
                            "max_pixels": 2007040,
                        },
                    ],
                },
            ],
            temperature=0.0,
        ),
    )

    queue_client.push_request(request)
    restored = queue_client.pop_request("default", timeout=0.1)

    assert restored is not None
    assert restored.call.messages == request.call.messages
    assert restored.call.temperature == request.call.temperature


def test_batch_processor_forwards_guided_json_to_queued_executor(monkeypatch):
    processor = object.__new__(BatchProcessor)
    processor.client = None
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = 20
    processor.batch_timeout = 5.0
    processor.backend_address = "http://test-backend:8000"
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    processor._circuit_breaker = None
    processor._gate_lock = None
    processor._gate_lock_loop = None
    processor._queue_client = None
    processor._queued_executor = None
    processor._queue_config = _queue_config()

    captured = {}

    class _QueuedExecutor:
        def execute(self, **kwargs):
            captured.update(kwargs)
            return [BatchResult("req_task_0", "resp-0", None)]

    monkeypatch.setattr(processor, "_get_queued_executor", lambda: _QueuedExecutor())

    processor.batch_generate(
        messages_list=[[{"role": "user", "content": "a"}]],
        guided_json={"type": "object"},
    )

    assert captured["calls"][0].extra_body["guided_json"] == {"type": "object"}


def test_direct_and_queued_paths_return_same_text_for_same_completion_payload():
    payload = _completion_payload("<think>reasoning</think>final answer")

    class _Completions:
        async def create(self, **kwargs):
            return deepcopy(payload)

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    calls = [_call([{"role": "user", "content": "hello"}])]

    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        direct_processor = _build_processor(client=_Client(), queue_enabled=False)
        direct_responses = direct_processor.batch_generate_calls(
            calls=calls,
            request_id="direct-batch",
        )

        queue_client = InMemoryListQueueClient()
        queued_processor = _build_processor(
            client=_Client(),
            queue_enabled=True,
            queue_client=queue_client,
        )
        dispatcher = queued_processor.build_queue_dispatcher()
        dispatcher.start()
        try:
            queued_responses = queued_processor.batch_generate_calls(
                calls=calls,
                request_id="queued-batch",
            )
        finally:
            dispatcher.stop()
            queued_processor.close()

    assert direct_responses == ["final answer"]
    assert queued_responses == direct_responses
