import asyncio
import threading
import time

import pytest

from marie.engine.batch_processor import BatchProcessor
from marie.engine.llm_queue.adapters.litellm import LiteLlmExecutionAdapter
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.dispatcher import QueuedBatchDispatcher
from marie.engine.llm_queue.models import QueueReply, QueueRequest
from marie.engine.llm_queue.queue_io import InMemoryListQueueClient
from marie.engine.llm_queue.result_types import BatchResult
from marie.engine.llm_queue.submitter import QueuedBatchExecutor


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
        max_inline_payload_bytes=128 * 1024,
    )
    values.update(overrides)
    return LlmQueueConfig(**values)


def test_queued_batch_executor_demultiplexes_replies_from_same_producer():
    queue_client = InMemoryListQueueClient()
    executor = QueuedBatchExecutor(
        queue_client=queue_client,
        config=_queue_config(),
        model_string="test-model",
        logger=_Logger(),
    )

    def worker():
        requests = []
        for _ in range(2):
            payload = queue_client.pop_request("default", timeout=1.0)
            assert payload is not None
            requests.append(QueueRequest.from_json(payload))

        for request in reversed(requests):
            queue_client.push_reply(
                request.producer_id,
                QueueReply(
                    request_id=request.request_id,
                    producer_id=request.producer_id,
                    pool_id=request.pool_id,
                    route_key=request.route_key,
                    status="ok",
                    response=f"resp:{request.request_id}",
                    completed_at=time.time(),
                ).to_json(),
                ttl_seconds=60,
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    results = executor.execute(
        messages_list=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
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


class _FakeAdapter:
    def __init__(self):
        self.requests = []

    def execute_requests(self, requests):
        self.requests.append([request.request_id for request in requests])
        return [
            BatchResult(
                task_id=request.request_id,
                response=f"done:{request.request_id}",
                error=None,
            )
            for request in requests
        ]


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

    live_request = QueueRequest(
        request_id="live-1",
        producer_id="producer-live",
        pool_id="default",
        route_key="test-model:abc",
        submitted_at=time.time(),
        messages=[{"role": "user", "content": "keep"}],
    )
    dead_request = QueueRequest(
        request_id="dead-1",
        producer_id="producer-dead",
        pool_id="default",
        route_key="test-model:abc",
        submitted_at=time.time(),
        messages=[{"role": "user", "content": "drop"}],
    )

    queue_client.push_request("default", dead_request.to_json())
    queue_client.push_request("default", live_request.to_json())

    dispatched = dispatcher.run_once()
    assert dispatched == 1
    assert adapter.requests == [["live-1"]]

    payload = queue_client.pop_reply("producer-live", timeout=1.0)
    assert payload is not None
    reply = QueueReply.from_json(payload)
    assert reply.request_id == "live-1"
    assert reply.response == "done:live-1"


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
        model_string="test-model",
        logger=_Logger(),
    )

    with pytest.raises(ValueError, match="max inline payload"):
        executor.execute(
            messages_list=[[{"role": "user", "content": "x" * 1024}]],
            batch_request_id="batch-too-large",
            batch_timeout=2.0,
        )
    assert queue_client.try_pop_request("default") is None


def test_dispatcher_drops_reply_if_producer_goes_offline_after_execute():
    queue_client = InMemoryListQueueClient()
    config = _queue_config(producer_id="producer-live")

    class _Adapter:
        def execute_requests(self, requests):
            queue_client.clear_producer_alive("producer-live")
            return [
                BatchResult(
                    task_id=request.request_id,
                    response=f"done:{request.request_id}",
                    error=None,
                )
                for request in requests
            ]

    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=_Adapter(),
        config=config,
        logger=_Logger(),
    )

    queue_client.set_producer_alive("producer-live", "producer-live", 5)
    queue_client.push_request(
        "default",
        QueueRequest(
            request_id="req-1",
            producer_id="producer-live",
            pool_id="default",
            route_key="test-model:abc",
            submitted_at=time.time(),
            messages=[{"role": "user", "content": "keep"}],
        ).to_json(),
    )

    assert dispatcher.run_once() == 1
    assert queue_client.pop_reply("producer-live", timeout=0.1) is None


def test_dispatcher_lifecycle_health():
    queue_client = InMemoryListQueueClient()
    dispatcher = QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=_FakeAdapter(),
        config=_queue_config(),
        logger=_Logger(),
    )

    dispatcher.start()
    try:
        health = dispatcher.health()
        assert health["running"] is True
        assert health["pool_id"] == "default"
    finally:
        dispatcher.stop()

    assert dispatcher.health()["running"] is False


def test_litellm_execution_adapter_enforces_timeout():
    class _BatchProcessor:
        batch_timeout = 600.0

        async def load_batched_request(self, **kwargs):
            await asyncio.sleep(0.2)

    adapter = LiteLlmExecutionAdapter(_BatchProcessor())
    request = QueueRequest(
        request_id="req-1",
        producer_id="producer-A",
        pool_id="default",
        route_key="test-model:abc",
        submitted_at=time.time(),
        messages=[{"role": "user", "content": "slow"}],
        timeout_seconds=0.05,
    )

    with pytest.raises(TimeoutError):
        adapter.execute_requests([request])
