import time

import pytest

from marie.engine.completion_contract import (
    COMPLETION_QUEUE_CONTRACT_VERSION,
    QueuedCompletionEnvelope,
    build_completion_call,
    completion_payload_to_text,
)
from marie.engine.llm_queue.config import (
    DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
    LlmQueueConfig,
)
from marie.engine.llm_queue.dispatcher import DrrQueuedBatchDispatcher
from marie.engine.llm_queue.queue_io import InMemoryListQueueClient
from marie.engine.llm_queue.registry import (
    dispatch_runtime_live_state,
    register_dispatcher,
    unregister_dispatcher,
)
from marie.engine.llm_queue.scheduler import (
    DrrLaneConfig,
    DrrLaneScheduler,
    request_cost_units,
)
from marie.engine.llm_queue.valkey_keys import request_queue_key


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _queue_config(**overrides) -> LlmQueueConfig:
    values = dict(
        enabled=True,
        valkey_url=None,
        pool_id="drr-domain",
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


def _request(
    pool_id: str,
    request_id: str,
    *,
    estimated_cost_units: int | None = None,
    metadata: dict | None = None,
    messages: list[dict] | None = None,
    submitted_at: float | None = None,
) -> QueuedCompletionEnvelope:
    return QueuedCompletionEnvelope(
        request_id=request_id,
        producer_id="producer-A",
        pool_id=pool_id,
        submitted_at=submitted_at if submitted_at is not None else time.time(),
        call=build_completion_call(
            model="test-model",
            messages=messages or [{"role": "user", "content": request_id}],
        ),
        estimated_cost_units=estimated_cost_units,
        metadata=metadata,
    )


def test_request_cost_units_uses_explicit_or_page_image_metadata():
    assert (
        request_cost_units(_request("default", "explicit", estimated_cost_units=99))
        == 16
    )
    assert (
        request_cost_units(
            _request(
                "default",
                "metadata",
                metadata={"image_count": "2", "chunk_page_count": "17"},
            )
        )
        == 8
    )
    assert (
        request_cost_units(
            _request(
                "default",
                "multimodal",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:a"}},
                            {"type": "image_url", "image_url": {"url": "data:b"}},
                        ],
                    }
                ],
            )
        )
        == 5
    )


def test_drr_quantum_shapes_dispatch_share_for_equal_cost_work():
    queue_client = InMemoryListQueueClient()
    for index in range(12):
        queue_client.push_request(_request("interactive", f"interactive-{index}"))
        queue_client.push_request(_request("backfill", f"backfill-{index}"))

    scheduler = DrrLaneScheduler(
        queue_client=queue_client,
        total_concurrent_dispatch=100,
        lanes=[
            DrrLaneConfig(pool_id="interactive", quantum=4),
            DrrLaneConfig(pool_id="backfill", quantum=1),
        ],
    )

    selected = []
    for _ in range(10):
        dispatch = scheduler.select_next()
        assert dispatch is not None
        selected.append(dispatch.pool_id)
        scheduler.release(dispatch.pool_id)

    assert selected == [
        "interactive",
        "interactive",
        "interactive",
        "interactive",
        "backfill",
        "interactive",
        "interactive",
        "interactive",
        "interactive",
        "backfill",
    ]


def test_lane_min_concurrent_protects_slots_from_noisy_pool():
    queue_client = InMemoryListQueueClient()
    for index in range(3):
        queue_client.push_request(_request("interactive", f"interactive-{index}"))
    queue_client.push_request(_request("document-extract", "document-0"))

    scheduler = DrrLaneScheduler(
        queue_client=queue_client,
        total_concurrent_dispatch=3,
        lanes=[
            DrrLaneConfig(
                pool_id="interactive",
                quantum=10,
                min_concurrent=1,
                max_concurrent=3,
            ),
            DrrLaneConfig(
                pool_id="document-extract",
                quantum=10,
                min_concurrent=1,
                max_concurrent=3,
            ),
        ],
    )

    first = scheduler.select_next()
    second = scheduler.select_next()
    third = scheduler.select_next()

    assert first is not None
    assert second is not None
    assert third is not None
    assert [first.pool_id, second.pool_id, third.pool_id] == [
        "interactive",
        "interactive",
        "document-extract",
    ]


def test_idle_protected_slots_are_borrowable_to_keep_pipe_full():
    queue_client = InMemoryListQueueClient()
    for index in range(3):
        queue_client.push_request(_request("interactive", f"interactive-{index}"))

    scheduler = DrrLaneScheduler(
        queue_client=queue_client,
        total_concurrent_dispatch=3,
        lanes=[
            DrrLaneConfig(
                pool_id="interactive",
                quantum=10,
                min_concurrent=1,
                max_concurrent=3,
            ),
            DrrLaneConfig(
                pool_id="document-extract",
                quantum=10,
                min_concurrent=1,
                max_concurrent=3,
            ),
        ],
    )

    selected = [scheduler.select_next() for _ in range(3)]

    assert [item.pool_id for item in selected if item is not None] == [
        "interactive",
        "interactive",
        "interactive",
    ]
    assert scheduler.inflight_count == 3


def test_lane_min_concurrent_sum_cannot_exceed_global_capacity():
    queue_client = InMemoryListQueueClient()

    with pytest.raises(ValueError, match="min_concurrent"):
        DrrLaneScheduler(
            queue_client=queue_client,
            total_concurrent_dispatch=1,
            lanes=[
                DrrLaneConfig(pool_id="interactive", min_concurrent=1),
                DrrLaneConfig(pool_id="backfill", min_concurrent=1),
            ],
        )


def test_lane_snapshot_reports_head_cost_and_oldest_pending_age():
    queue_client = InMemoryListQueueClient()
    queue_client.push_request(
        _request(
            "document-extract",
            "doc-0",
            metadata={"chunk_page_count": 17},
            submitted_at=time.time() - 12.0,
        )
    )
    scheduler = DrrLaneScheduler(
        queue_client=queue_client,
        total_concurrent_dispatch=2,
        lanes=[
            DrrLaneConfig(
                pool_id="document-extract",
                display_name="Document Extract",
            )
        ],
    )

    snapshot = scheduler.lane_snapshots()[0]

    assert snapshot.display_name == "Document Extract"
    assert snapshot.head_cost_units == 4
    assert snapshot.oldest_pending_age_seconds is not None
    assert snapshot.oldest_pending_age_seconds >= 10.0


def test_high_cost_request_dispatches_after_credit_accumulates():
    queue_client = InMemoryListQueueClient()
    queue_client.push_request(
        _request("document-extract", "large-chunk", estimated_cost_units=9)
    )
    scheduler = DrrLaneScheduler(
        queue_client=queue_client,
        total_concurrent_dispatch=1,
        lanes=[DrrLaneConfig(pool_id="document-extract", quantum=4)],
    )

    assert scheduler.select_next() is None
    assert scheduler.select_next() is None

    dispatch = scheduler.select_next()

    assert dispatch is not None
    assert dispatch.request.request_id == "large-chunk"
    assert dispatch.cost_units == 9
    assert dispatch.deficit_after_dispatch == 3


def test_malformed_head_request_is_dropped_without_wedging_lane():
    queue_client = InMemoryListQueueClient()
    malformed_payload = _request("document-extract", "bad").to_json().replace(
        f"\"contract_version\":\"{COMPLETION_QUEUE_CONTRACT_VERSION}\"",
        "\"contract_version\":\"v1\"",
    )
    queue_client._lists[request_queue_key("document-extract")].append(malformed_payload)
    queue_client.push_request(_request("document-extract", "good"))

    scheduler = DrrLaneScheduler(
        queue_client=queue_client,
        total_concurrent_dispatch=1,
        lanes=[DrrLaneConfig(pool_id="document-extract", quantum=1)],
    )

    assert scheduler.select_next() is None
    snapshot = scheduler.lane_snapshots()[0]
    assert snapshot.malformed_requests_dropped == 1
    assert snapshot.skip_counts["malformed_head"] == 1

    dispatch = scheduler.select_next()
    assert dispatch is not None
    assert dispatch.request.request_id == "good"


def test_per_lane_max_concurrent_is_enforced():
    queue_client = InMemoryListQueueClient()
    queue_client.push_request(_request("backfill", "backfill-0"))
    queue_client.push_request(_request("backfill", "backfill-1"))
    scheduler = DrrLaneScheduler(
        queue_client=queue_client,
        total_concurrent_dispatch=10,
        lanes=[DrrLaneConfig(pool_id="backfill", quantum=10, max_concurrent=1)],
    )

    first = scheduler.select_next()
    second = scheduler.select_next()

    assert first is not None
    assert first.pool_id == "backfill"
    assert second is None
    assert scheduler.lane_snapshots()[0].skip_counts["lane_capacity"] == 1


def test_drr_dispatcher_runs_selected_lanes_as_one_concurrent_batch():
    queue_client = InMemoryListQueueClient()

    class _Adapter:
        def __init__(self):
            self.requests = []

        async def execute(self, call, *, timeout_seconds=None):
            request_text = call.messages[-1]["content"]
            self.requests.append(request_text)
            return {
                "choices": [
                    {
                        "message": {"content": f"done:{request_text}"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "test-model",
            }

    adapter = _Adapter()
    dispatcher = DrrQueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=adapter,
        config=_queue_config(),
        logger=_Logger(),
        total_concurrent_dispatch=3,
        lanes=[
            DrrLaneConfig(pool_id="interactive", quantum=2),
            DrrLaneConfig(pool_id="backfill", quantum=1),
        ],
    )

    queue_client.set_producer_alive("producer-A", "producer-A", 5)
    for index in range(3):
        queue_client.push_request(_request("interactive", f"interactive-{index}"))
    for index in range(2):
        queue_client.push_request(_request("backfill", f"backfill-{index}"))

    assert dispatcher.run_once() == 3
    assert adapter.requests == ["interactive-0", "interactive-1", "backfill-0"]

    replies = [
        queue_client.pop_reply("producer-A", timeout=0.1),
        queue_client.pop_reply("producer-A", timeout=0.1),
        queue_client.pop_reply("producer-A", timeout=0.1),
    ]
    assert [reply.request_id for reply in replies if reply is not None] == [
        "interactive-0",
        "interactive-1",
        "backfill-0",
    ]
    assert [
        completion_payload_to_text(reply.completion)
        for reply in replies
        if reply is not None
    ] == ["done:interactive-0", "done:interactive-1", "done:backfill-0"]

    health = dispatcher.health()
    assert health["scheduler_policy"] == "drr"
    assert health["total_concurrent_dispatch"] == 3
    assert health["request_queue_depth"] == 2
    assert {lane["pool_id"]: lane["inflight"] for lane in health["lanes"]} == {
        "backfill": 0,
        "interactive": 0,
    }


def test_drr_dispatcher_does_not_fifo_fill_after_single_selection():
    queue_client = InMemoryListQueueClient()

    class _Adapter:
        def __init__(self):
            self.requests = []

        async def execute(self, call, *, timeout_seconds=None):
            request_text = call.messages[-1]["content"]
            self.requests.append(request_text)
            return {
                "choices": [
                    {
                        "message": {"content": f"done:{request_text}"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "test-model",
            }

    adapter = _Adapter()
    dispatcher = DrrQueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=adapter,
        config=_queue_config(max_batch_items=8, max_buffered_requests_per_pool=8),
        logger=_Logger(),
        total_concurrent_dispatch=1,
        lanes=[DrrLaneConfig(pool_id="interactive", quantum=4)],
    )

    queue_client.set_producer_alive("producer-A", "producer-A", 5)
    queue_client.push_request(_request("interactive", "interactive-0"))
    queue_client.push_request(_request("interactive", "interactive-1"))

    assert dispatcher.run_once() == 1
    assert adapter.requests == ["interactive-0"]
    assert queue_client.request_queue_depth("interactive") == 1


def test_drr_dispatcher_uses_lane_execution_adapters():
    queue_client = InMemoryListQueueClient()

    class _Adapter:
        def __init__(self, backend_address):
            self.backend_address = backend_address
            self.requests = []

        async def execute(self, call, *, timeout_seconds=None):
            request_text = call.messages[-1]["content"]
            self.requests.append(request_text)
            return {
                "choices": [
                    {
                        "message": {"content": f"done:{request_text}"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "test-model",
            }

    default_adapter = _Adapter("http://default:4000/v1")
    interactive_adapter = _Adapter("http://interactive:4000/v1")
    backfill_adapter = _Adapter("http://backfill:4000/v1")
    dispatcher = DrrQueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=default_adapter,
        config=_queue_config(),
        logger=_Logger(),
        total_concurrent_dispatch=2,
        lanes=[
            DrrLaneConfig(pool_id="interactive", quantum=1),
            DrrLaneConfig(pool_id="backfill", quantum=1),
        ],
        execution_adapters_by_pool={
            "interactive": interactive_adapter,
            "backfill": backfill_adapter,
        },
    )

    queue_client.set_producer_alive("producer-A", "producer-A", 5)
    queue_client.push_request(_request("interactive", "interactive-0"))
    queue_client.push_request(_request("backfill", "backfill-0"))

    assert dispatcher.run_once() == 2
    assert interactive_adapter.requests == ["interactive-0"]
    assert backfill_adapter.requests == ["backfill-0"]
    assert default_adapter.requests == []

    replies = [
        queue_client.pop_reply("producer-A", timeout=0.1),
        queue_client.pop_reply("producer-A", timeout=0.1),
    ]
    backend_by_request = {
        reply.request_id: reply.execution_backend_address
        for reply in replies
        if reply is not None
    }
    assert backend_by_request == {
        "interactive-0": "http://interactive:4000/v1",
        "backfill-0": "http://backfill:4000/v1",
    }


def test_drr_pending_samples_apply_limit_per_lane():
    queue_client = InMemoryListQueueClient()
    dispatcher = DrrQueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=object(),
        config=_queue_config(),
        logger=_Logger(),
        total_concurrent_dispatch=2,
        lanes=[
            DrrLaneConfig(pool_id="interactive"),
            DrrLaneConfig(pool_id="backfill"),
        ],
    )

    queue_client.push_request(_request("interactive", "interactive-0"))
    queue_client.push_request(_request("interactive", "interactive-1"))
    queue_client.push_request(_request("backfill", "backfill-0"))
    queue_client.push_request(_request("backfill", "backfill-1"))

    samples = dispatcher.sample_pending_requests(limit=1)

    assert [sample["request_id"] for sample in samples] == [
        "interactive-0",
        "backfill-0",
    ]


def test_drr_live_state_exposes_pool_config_rows():
    queue_client = InMemoryListQueueClient()

    class _Adapter:
        backend_address = "http://interactive:4000/v1"

    dispatcher = DrrQueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=object(),
        config=_queue_config(fabric_group_id="fabric-a", gateway_id="gateway-a"),
        logger=_Logger(),
        total_concurrent_dispatch=2,
        lanes=[
            DrrLaneConfig(
                pool_id="interactive",
                display_name="Interactive",
                quantum=8,
                min_concurrent=1,
                max_concurrent=2,
            ),
            DrrLaneConfig(pool_id="backfill", display_name="Backfill", quantum=1),
        ],
        execution_adapters_by_pool={"interactive": _Adapter()},
    )
    queue_client.push_request(_request("interactive", "interactive-0"))
    queue_client.push_request(_request("backfill", "backfill-0"))

    register_dispatcher(dispatcher.dispatcher_id, dispatcher)
    try:
        live_state = dispatch_runtime_live_state(limit_per_pool=1)
    finally:
        unregister_dispatcher(dispatcher.dispatcher_id)

    assert live_state["runtime_summary"]["pending_request_count"] == 2
    assert {
        request["request_id"] for request in live_state["live_requests"]
    } == {"backfill-0", "interactive-0"}

    rows = {row["pool_id"]: row for row in live_state["pool_config"]}
    assert rows["interactive"]["scheduler_policy"] == "drr"
    assert rows["interactive"]["fabric_group_id"] == "fabric-a"
    assert rows["interactive"]["gateway_id"] == "gateway-a"
    assert rows["interactive"]["display_name"] == "Interactive"
    assert rows["interactive"]["quantum"] == 8
    assert rows["interactive"]["min_concurrent"] == 1
    assert rows["interactive"]["max_concurrent"] == 2
    assert rows["interactive"]["request_queue_depth"] == 1
    assert rows["interactive"]["endpoint_url"] == "http://interactive:4000/v1"
