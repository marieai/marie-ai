import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from marie.serve.discovery import JsonAddress
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (  # noqa: F401
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.servers.marie_gateway import (
    SERVICE_SNAPSHOT_COMPLETE,
    EventKind,
    MarieServerGateway,
    ServiceEvent,
)


def _gateway(worker_count: int = 2) -> MarieServerGateway:
    gateway = object.__new__(MarieServerGateway)
    gateway.args = {"discovery_event_worker_count": worker_count}
    gateway.service_events_queue = asyncio.Queue()
    gateway.ready_event = asyncio.Event()
    gateway.deployment_nodes = {}
    gateway.logger = Mock()
    gateway._rebuild_task = None
    gateway._rebuild_requested = False
    gateway._streamer_update_lock = asyncio.Lock()
    gateway._service_retry_tasks = {}
    gateway._service_retry_attempts = {}
    gateway._service_readiness = {}
    gateway._debounce_s = 0
    gateway._rebuild_deployments_projection = Mock()
    gateway.update_gateway_streamer = AsyncMock()
    gateway._publish_capacity_event = AsyncMock()
    gateway.gateway_server_offline = AsyncMock(return_value=False)
    return gateway


def _event(key: str, name: str) -> ServiceEvent:
    address = key.rsplit("/", 1)[-1]
    return ServiceEvent(
        kind=EventKind.SERVICE,
        service="gateway/marie",
        ev_type="put",
        value={
            key: JsonAddress(
                address,
                metadata=f'{{"{name}":["{address}"]}}',
            ).add_value()
        },
        key=key,
    )


def _event_executor(value: dict) -> str:
    metadata = JsonAddress.from_value(value)._metadata
    return next(iter(json.loads(metadata)))


def _snapshot_complete() -> ServiceEvent:
    return ServiceEvent(
        kind=EventKind.SERVICE,
        service="gateway/marie",
        ev_type=SERVICE_SNAPSHOT_COMPLETE,
        value=None,
        key="gateway/marie",
    )


def _keys_for_different_workers(worker_count: int) -> tuple[str, str]:
    keys_by_worker = {}
    for index in range(100):
        key = f"gateway/marie/node-{index}"
        keys_by_worker.setdefault(hash(key) % worker_count, key)
        if len(keys_by_worker) == 2:
            return tuple(keys_by_worker.values())
    raise AssertionError("could not find keys for different workers")


@pytest.mark.asyncio
async def test_snapshot_marker_is_enqueued_after_initial_events():
    gateway = _gateway()
    gateway._loop = asyncio.get_running_loop()
    first = SimpleNamespace(
        event="put",
        value={"name": "first"},
        key="gateway/marie/node-1",
    )
    second = SimpleNamespace(
        event="put",
        value={"name": "second"},
        key="gateway/marie/node-2",
    )

    gateway._on_service_event("gateway/marie", first)
    gateway._on_service_event("gateway/marie", second)
    gateway._on_service_snapshot_complete("gateway/marie", 2)

    for _ in range(10):
        if gateway.service_events_queue.qsize() == 3:
            break
        await asyncio.sleep(0)

    events = [gateway.service_events_queue.get_nowait() for _ in range(3)]
    assert [event.ev_type for event in events] == [
        "put",
        "put",
        SERVICE_SNAPSHOT_COMPLETE,
    ]


@pytest.mark.asyncio
async def test_initial_snapshot_does_not_wait_for_registration_probes():
    gateway = _gateway()
    slow_key, fast_key = _keys_for_different_workers(2)
    slow_release = asyncio.Event()
    fast_processed = asyncio.Event()

    async def gateway_server_online(_service, value):
        if _event_executor(value) == "slow":
            await slow_release.wait()
        else:
            fast_processed.set()
        return True

    gateway.gateway_server_online = gateway_server_online
    processor = asyncio.create_task(gateway.process_service_events())

    await gateway.service_events_queue.put(_event(slow_key, "slow"))
    await gateway.service_events_queue.put(_event(fast_key, "fast"))
    await gateway.service_events_queue.put(_snapshot_complete())

    await asyncio.wait_for(gateway.ready_event.wait(), timeout=1)

    assert not slow_release.is_set()
    gateway.update_gateway_streamer.assert_awaited_once()
    assert gateway._publish_capacity_event.await_count >= 1

    await asyncio.wait_for(fast_processed.wait(), timeout=1)
    slow_release.set()

    processor.cancel()
    await asyncio.gather(processor, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_initial_snapshot_marks_gateway_ready():
    gateway = _gateway()
    gateway.gateway_server_online = AsyncMock(return_value=False)
    processor = asyncio.create_task(gateway.process_service_events())

    await gateway.service_events_queue.put(_snapshot_complete())
    await asyncio.wait_for(gateway.ready_event.wait(), timeout=1)

    gateway.update_gateway_streamer.assert_awaited_once()

    processor.cancel()
    await asyncio.gather(processor, return_exceptions=True)


@pytest.mark.asyncio
async def test_unready_registration_retries_without_blocking_snapshot():
    gateway = _gateway()
    gateway.gateway_server_online = AsyncMock(return_value=None)
    gateway._schedule_service_retry = Mock()
    processor = asyncio.create_task(gateway.process_service_events())
    event = _event("gateway/marie/node-1", "starting")

    await gateway.service_events_queue.put(event)
    await gateway.service_events_queue.put(_snapshot_complete())
    await asyncio.wait_for(gateway.ready_event.wait(), timeout=1)

    for _ in range(10):
        if gateway._schedule_service_retry.called:
            break
        await asyncio.sleep(0)
    gateway._schedule_service_retry.assert_called_once_with(event)

    processor.cancel()
    await asyncio.gather(processor, return_exceptions=True)


@pytest.mark.asyncio
async def test_unready_registration_is_exposed_with_retry_details():
    gateway = _gateway()
    gateway.ready_event.set()
    gateway.gateway_server_online = AsyncMock(return_value=None)
    processor = asyncio.create_task(gateway.process_service_events())
    event = _event(
        "gateway/marie/172.20.10.67:62177",
        "corr_indexing_executor",
    )

    await gateway.service_events_queue.put(event)
    for _ in range(20):
        entry = gateway._service_readiness.get(event.key)
        if entry and entry["state"] == "retrying":
            break
        await asyncio.sleep(0)

    snapshot = gateway._discovery_readiness_snapshot("unready")

    assert snapshot["readiness"] == "degraded"
    assert snapshot["summary"] == {
        "registered": 1,
        "ready": 0,
        "unready": 1,
        "checking": 0,
        "retrying": 1,
        "error": 0,
    }
    entry = snapshot["gateways"][0]
    assert {
        key: entry[key]
        for key in (
            "key",
            "address",
            "host",
            "port",
            "registered",
            "ready",
            "state",
            "retry_attempt",
            "probe_attempt_limit",
            "last_error",
            "executors",
        )
    } == {
        "key": event.key,
        "address": "172.20.10.67:62177",
        "host": "172.20.10.67",
        "port": 62177,
        "registered": True,
        "ready": False,
        "state": "retrying",
        "retry_attempt": 1,
        "probe_attempt_limit": 3,
        "last_error": "gRPC health check did not return SERVING",
        "executors": ["corr_indexing_executor"],
    }
    assert entry["first_seen_at"] is not None
    assert entry["last_checked_at"] is not None
    assert entry["next_retry_at"] is not None

    processor.cancel()
    await asyncio.gather(processor, return_exceptions=True)


@pytest.mark.asyncio
async def test_ready_registration_is_removed_after_discovery_delete():
    gateway = _gateway()
    gateway.ready_event.set()
    gateway.gateway_server_online = AsyncMock(return_value=True)
    processor = asyncio.create_task(gateway.process_service_events())
    event = _event("gateway/marie/172.20.10.67:62177", "corr_indexing_executor")

    await gateway.service_events_queue.put(event)
    for _ in range(20):
        entry = gateway._service_readiness.get(event.key)
        if entry and entry["state"] == "ready":
            break
        await asyncio.sleep(0)

    ready_snapshot = gateway._discovery_readiness_snapshot("ready")
    assert ready_snapshot["readiness"] == "ready"
    assert ready_snapshot["summary"]["ready"] == 1
    assert ready_snapshot["gateways"][0]["last_ready_at"] is not None

    await gateway.service_events_queue.put(
        ServiceEvent(
            kind=EventKind.SERVICE,
            service=event.service,
            ev_type="delete",
            value=None,
            key=event.key,
        )
    )
    for _ in range(20):
        if event.key not in gateway._service_readiness:
            break
        await asyncio.sleep(0)

    removed_snapshot = gateway._discovery_readiness_snapshot()
    assert removed_snapshot["readiness"] == "ready"
    assert removed_snapshot["summary"]["registered"] == 0
    assert removed_snapshot["gateways"] == []

    processor.cancel()
    await asyncio.gather(processor, return_exceptions=True)


@pytest.mark.asyncio
async def test_events_for_same_registration_remain_ordered():
    gateway = _gateway()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    order = []

    async def gateway_server_online(_service, value):
        name = _event_executor(value)
        order.append(name)
        if name == "first":
            await first_release.wait()
        else:
            second_started.set()
        return True

    gateway.gateway_server_online = gateway_server_online
    processor = asyncio.create_task(gateway.process_service_events())
    key = "gateway/marie/node-1"

    await gateway.service_events_queue.put(_event(key, "first"))
    await gateway.service_events_queue.put(_event(key, "second"))
    await gateway.service_events_queue.put(_snapshot_complete())
    await asyncio.sleep(0.02)

    assert not second_started.is_set()

    first_release.set()
    await asyncio.wait_for(second_started.wait(), timeout=1)

    assert order == ["first", "second"]

    processor.cancel()
    await asyncio.gather(processor, return_exceptions=True)


@pytest.mark.asyncio
async def test_scheduler_does_not_start_before_discovery_is_ready(monkeypatch):
    gateway = _gateway()
    gateway.job_scheduler = Mock(start=AsyncMock())
    gateway._start_gateway_background_runtimes = AsyncMock()
    gateway.args.update(
        {
            "sensors": {},
            "kv_store_kwargs": {},
        }
    )
    monkeypatch.setattr(
        "marie.serve.runtimes.servers.marie_gateway.setup_sensor_worker",
        Mock(),
    )
    monkeypatch.setattr(
        "marie.serve.runtimes.servers.marie_gateway.attach_sensor_worker_scheduler",
        Mock(return_value=False),
    )

    starter = asyncio.create_task(gateway.wait_and_start_scheduler(timeout=0.01))
    await asyncio.sleep(0.03)

    gateway.job_scheduler.start.assert_not_awaited()

    gateway.ready_event.set()
    await asyncio.wait_for(starter, timeout=1)

    gateway.job_scheduler.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_incremental_update_skips_unchanged_executor_addresses():
    gateway = _gateway()
    gateway.streamer = Mock(
        update_executor_addresses=AsyncMock(),
        remove_connection=AsyncMock(),
    )
    gateway._last_deployments_addresses = {
        "stable": ["node-1:5000"],
        "changed": ["node-2:5000"],
    }

    await gateway._apply_incremental_updates(
        {
            "stable": ["node-1:5000"],
            "changed": ["node-3:5000"],
        }
    )

    gateway.streamer.update_executor_addresses.assert_awaited_once_with(
        "changed", ["node-3:5000"]
    )


@pytest.mark.asyncio
async def test_full_generation_swaps_before_old_streamer_drains(monkeypatch):
    gateway = _gateway()
    new_streamer = Mock()

    async def close_old_streamer():
        assert gateway.streamer is new_streamer
        assert gateway.distributor.streamer is new_streamer

    old_streamer = Mock(close=AsyncMock(side_effect=close_old_streamer))
    gateway.streamer = old_streamer
    gateway.distributor = SimpleNamespace(streamer=old_streamer)
    gateway.runtime_args = SimpleNamespace(grpc_channel_options=None)
    monkeypatch.setattr(
        "marie.serve.runtimes.servers.marie_gateway.GatewayStreamer",
        Mock(return_value=new_streamer),
    )

    await gateway._create_new_gateway_streamer(
        {"start-gateway": []},
        {},
        {},
    )

    old_streamer.close.assert_awaited_once()
