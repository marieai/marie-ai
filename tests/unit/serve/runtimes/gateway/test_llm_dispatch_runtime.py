import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest
from marie.engine.llm_queue.config import (
    DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
    INTERNAL_LEGACY_POOL_ID,
    LlmQueueRuntimeConfig,
)
from marie.engine.llm_queue.scheduler_config import (
    DatabaseSchedulerConfigSource,
    LlmQueueSchedulerConfig,
    scheduler_config_from_mapping,
)

from marie.excepts import RuntimeFailToStart
from marie.job.common import JobStatus
from marie.job.event_publisher import EventPublisher
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
    _scheduler_fabric_group_id,
    _scheduler_repository_config,
)
from marie.serve.runtimes.servers.marie_gateway import (
    LLM_DISPATCH_RUNTIME_EVENT,
    LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S,
    LLM_DISPATCH_RUNTIME_MARKER,
    LLM_DISPATCH_RUNTIME_SOURCE,
    MarieServerGateway,
    _llm_dispatch_runtime_event_message,
    _llm_queue_runtime_config,
    _should_publish_llm_dispatch_runtime_event,
)


def _format_log_message(args):
    if not args:
        return ""
    message = str(args[0])
    if len(args) > 1:
        try:
            return message % args[1:]
        except TypeError:
            return message
    return message


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []
        self.exception_messages = []

    def info(self, *args, **kwargs):
        self.info_messages.append(_format_log_message(args))

    def warning(self, *args, **kwargs):
        self.warning_messages.append(_format_log_message(args))

    def error(self, *args, **kwargs):
        self.error_messages.append(_format_log_message(args))

    def exception(self, *args, **kwargs):
        self.exception_messages.append(_format_log_message(args))


def _queue_config(**overrides) -> LlmQueueRuntimeConfig:
    values = dict(
        enabled=True,
        queue_url="redis://valkey:6379/0",
        pool_id="default",
        producer_id="producer-A",
        producer_ttl_seconds=30,
        producer_refresh_interval_seconds=10.0,
        reply_queue_ttl_seconds=300,
        reply_pop_timeout_seconds=1.0,
        dispatch_pop_timeout_seconds=1.0,
        max_batch_items=8,
        max_batch_wait_ms=100,
        max_buffered_requests_per_pool=32,
        max_inline_payload_bytes=DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
    )
    values.update(overrides)
    return LlmQueueRuntimeConfig(**values)


class _FakeSchedulerConfigRepository:
    def __init__(self, payload):
        self.payload = payload
        self.fabric_group_ids = []

    def load_scheduler_config(self, fabric_group_id: str):
        self.fabric_group_ids.append(fabric_group_id)
        return self.payload


@pytest.mark.asyncio
async def test_gateway_llm_dispatch_runtime_is_noop_when_disabled():
    runtime = GatewayLlmDispatchRuntime(
        logger=_Logger(),
        queue_config=_queue_config(enabled=False, queue_url=None),
    )

    await runtime.start()

    health = runtime.health()
    assert health["enabled"] is False
    assert runtime.mode == "direct-batch"
    assert health["running"] is False

    await runtime.stop()


def test_llm_queue_config_reads_runtime_fabric_identity_from_env():
    with mock.patch.dict(
        "os.environ",
        {
            "LLM_QUEUE_ENABLED": "true",
            "LLM_QUEUE_VALKEY_URL": "redis://valkey:6379/0",
            "LLM_QUEUE_FABRIC_GROUP_ID": "default",
            "LLM_QUEUE_GATEWAY_ID": "gateway-localhost",
        },
    ):
        config = LlmQueueRuntimeConfig.from_env()

    assert config.fabric_group_id == "default"
    assert config.gateway_id == "gateway-localhost"


def test_llm_dispatch_runtime_event_uses_control_plane_event_shape():
    snapshot = {
        "mode": "queued-dispatch",
        "runtime_summary": {
            "registered_dispatchers": 1,
            "running_dispatchers": 1,
            "pending_request_count": 2,
        },
        "live_requests": [],
        "dispatchers": [],
    }

    event = _llm_dispatch_runtime_event_message(
        snapshot=snapshot,
        queue_config=_queue_config(
            fabric_group_id="default",
            gateway_id="gateway-localhost",
        ),
    )

    assert event.event == "engine.event"
    assert event.source == LLM_DISPATCH_RUNTIME_SOURCE
    assert event.source == "gateway://control-plane"
    assert event.api_key == "system:gateway"
    assert event.jobid == "gateway"
    assert event.jobtag == LLM_DISPATCH_RUNTIME_MARKER
    assert event.status == "INFO"

    assert event.payload["message"] == "LLM dispatch runtime snapshot updated"
    assert event.payload["marker_start"] == LLM_DISPATCH_RUNTIME_MARKER
    assert event.payload["component"] == "llm_dispatch_runtime"
    assert event.payload["event_type"] == LLM_DISPATCH_RUNTIME_EVENT
    assert event.payload["fabric_group_id"] == "default"
    assert event.payload["gateway_id"] == "gateway-localhost"
    assert event.payload["pool_id"] == "default"
    assert event.payload["result"] == snapshot
    assert event.payload["metadata"]["llm_dispatch_runtime"].value == snapshot


@pytest.mark.asyncio
async def test_llm_dispatch_runtime_event_joins_database_drain_references(
    monkeypatch,
):
    from marie.serve.runtimes.servers import marie_gateway as module

    snapshot = {
        'fabric_group_id': 'default',
        'runtime_summary': {},
        'live_requests': [],
        'dispatchers': [],
        'pools': [
            {
                'pool_id': 'document-small',
                'drain_references': {
                    'postgres': 2,
                    'valkey': 1,
                    'total': 3,
                },
            }
        ],
    }
    reads = []

    async def read_observation(**kwargs):
        reads.append(kwargs)
        return snapshot

    published = []

    async def notify(*args):
        published.append(args)

    monkeypatch.setattr(module, 'read_operator_runtime_snapshot', read_observation)
    monkeypatch.setattr(module.Toast, 'notify', notify)
    repository = object()
    gateway = SimpleNamespace(
        llm_dispatch_runtime=SimpleNamespace(
            config=_queue_config(
                fabric_group_id='default',
                gateway_id='gateway-localhost',
            ),
            _scheduler_config_source=SimpleNamespace(repository=repository),
        ),
        _last_llm_dispatch_event_fingerprint=None,
        _last_llm_dispatch_event_monotonic=0.0,
    )

    await MarieServerGateway._publish_llm_dispatch_runtime_event(gateway)

    assert reads == [
        {
            'fabric_group_id': 'default',
            'limit': 50,
            'policy_repository': repository,
        }
    ]
    assert len(published) == 1
    assert published[0][1].payload['result']['pools'][0]['drain_references'] == {
        'postgres': 2,
        'valkey': 1,
        'total': 3,
    }


def test_llm_dispatch_runtime_event_publish_policy_repeats_idle_snapshots():
    assert _should_publish_llm_dispatch_runtime_event(
        fingerprint="snapshot-a",
        last_fingerprint=None,
        last_published_at=0.0,
        now=1.0,
        unchanged_interval_s=LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S,
    )
    assert not _should_publish_llm_dispatch_runtime_event(
        fingerprint="snapshot-a",
        last_fingerprint="snapshot-a",
        last_published_at=10.0,
        now=10.5,
        unchanged_interval_s=LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S,
    )
    assert _should_publish_llm_dispatch_runtime_event(
        fingerprint="snapshot-a",
        last_fingerprint="snapshot-a",
        last_published_at=10.0,
        now=10.0 + LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S,
        unchanged_interval_s=LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S,
    )
    assert _should_publish_llm_dispatch_runtime_event(
        fingerprint="snapshot-b",
        last_fingerprint="snapshot-a",
        last_published_at=10.0,
        now=10.5,
        unchanged_interval_s=LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S,
    )


def test_database_scheduler_config_source_reads_repository_mapping():
    repository = _FakeSchedulerConfigRepository(
        {
            "policy": "drr",
            "total_concurrent_dispatch": 60,
            "lanes": [
                {
                    "pool_id": "interactive",
                    "quantum": 8,
                    "min_concurrent": 10,
                },
                {
                    "pool_id": "backfill",
                    "quantum": 1,
                    "max_burst_per_visit": 1,
                },
            ],
        }
    )

    config = DatabaseSchedulerConfigSource(
        repository=repository,
        fabric_group_id="fabric-a",
    ).load()

    assert config.is_drr is True
    assert config.total_concurrent_dispatch == 60
    assert repository.fabric_group_ids == ["fabric-a"]
    assert [lane.pool_id for lane in config.lanes] == [
        "interactive",
        "backfill",
        INTERNAL_LEGACY_POOL_ID,
    ]
    assert config.lanes[0].min_concurrent == 10
    assert config.lanes[1].max_burst_per_visit == 1


def test_scheduler_config_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unsupported LLM queue scheduler policy"):
        scheduler_config_from_mapping({"policy": "weighted"})


def test_runtime_scheduler_repository_config_uses_llm_queue_block():
    config = {
        "fabric_group_id": "default",
        "scheduler": {
            "storage": {
                "psql": {
                    "provider": "postgresql",
                    "hostname": "postgres",
                    "port": 5432,
                    "username": "marie",
                    "password": "test",
                    "database": "marie",
                    "schema": "marie_scheduler",
                }
            }
        },
    }

    repository_config = _scheduler_repository_config(config)

    assert repository_config is not None
    assert repository_config["hostname"] == "postgres"
    assert repository_config["schema"] == "marie_scheduler"
    assert _scheduler_fabric_group_id(_queue_config(), config) == "default"


def test_llm_queue_runtime_config_inherits_job_scheduler_postgres_config():
    config = _llm_queue_runtime_config(
        {
            "llm_queue": {"fabric_group_id": "default"},
            "job_scheduler_kwargs": {
                "provider": "postgresql",
                "hostname": "postgres",
                "port": 5432,
                "username": "marie",
                "password": "test",
                "database": "postgres",
                "schema": "marie_scheduler",
            },
        }
    )

    repository_config = _scheduler_repository_config(config)

    assert repository_config is not None
    assert repository_config["hostname"] == "postgres"
    assert repository_config["schema"] == "marie_scheduler"
    assert config["fabric_group_id"] == "default"


def test_llm_queue_runtime_config_preserves_explicit_scheduler_postgres_config():
    config = _llm_queue_runtime_config(
        {
            "llm_queue": {
                "scheduler": {
                    "psql": {
                        "provider": "postgresql",
                        "hostname": "llm-postgres",
                        "schema": "llm_scheduler",
                    }
                }
            },
            "job_scheduler_kwargs": {
                "provider": "postgresql",
                "hostname": "job-postgres",
                "schema": "marie_scheduler",
            },
        }
    )

    repository_config = _scheduler_repository_config(config)

    assert repository_config is not None
    assert repository_config["hostname"] == "llm-postgres"
    assert repository_config["schema"] == "llm_scheduler"


@pytest.mark.asyncio
async def test_gateway_llm_dispatch_runtime_requires_store_when_enabled():
    runtime = GatewayLlmDispatchRuntime(
        logger=_Logger(),
        queue_config=_queue_config(queue_url=None, fabric_group_id="default"),
    )

    assert runtime.health()["queue_configured"] is False
    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with pytest.raises(RuntimeFailToStart, match="store is unavailable"):
            await runtime.start()


def test_gateway_llm_dispatch_runtime_reports_canonical_queue_diagnostic():
    with mock.patch.dict(
        "os.environ",
        {
            "LLM_QUEUE_ENABLED": "true",
            "LLM_QUEUE_URL": "redis://queue:6379/0",
        },
        clear=True,
    ):
        health = GatewayLlmDispatchRuntime(logger=_Logger()).health()

    assert health["queue_configured"] is True
    assert "valkey_configured" not in health


@pytest.mark.asyncio
async def test_gateway_background_runtime_start_calls_llm_dispatch_runtime():
    started = 0

    class _Runtime:
        mode = "queued-dispatch"
        enabled = True
        _dispatcher = None
        _queue_client = None

        async def start(self):
            nonlocal started
            started += 1

    gateway = object.__new__(MarieServerGateway)
    gateway.logger = _Logger()
    gateway.llm_dispatch_runtime = _Runtime()

    await gateway._start_gateway_background_runtimes()

    assert started == 1
    assert "LLM execution mode: queued-dispatch" in gateway.logger.info_messages


@pytest.mark.asyncio
async def test_gateway_run_server_tracks_control_plane_tasks():
    server_stopped = asyncio.Event()

    class _Server:
        async def run_server(self) -> None:
            await server_stopped.wait()

    async def wait_forever(*_args, **_kwargs) -> None:
        await asyncio.Event().wait()

    gateway = object.__new__(MarieServerGateway)
    gateway.servers = [_Server()]
    gateway.grpc_broker = None
    gateway._background_services_shutdown = False
    gateway._control_plane_tasks = set()
    gateway.process_service_events = wait_forever
    gateway.process_state_events = wait_forever
    gateway.wait_and_start_scheduler = wait_forever
    gateway._reconcile_loop = wait_forever
    gateway._capacity_broadcast_loop = wait_forever

    run_task = asyncio.create_task(gateway.run_server())
    await asyncio.sleep(0)

    assert {task.get_name() for task in gateway._control_plane_tasks} == {
        "gateway-service-events",
        "gateway-state-events",
        "gateway-scheduler-start",
        "gateway-reconcile",
        "gateway-capacity-broadcast",
    }

    gateway._background_services_shutdown = True
    gateway._rebuild_task = None
    await gateway._stop_control_plane_tasks()
    assert not run_task.done()

    server_stopped.set()
    await run_task


@pytest.mark.asyncio
async def test_gateway_background_shutdown_is_idempotent():
    class _AsyncStopper:
        def __init__(self, name: str, calls: list[str]) -> None:
            self.name = name
            self.order = calls
            self.calls = 0

        async def stop(self) -> None:
            self.calls += 1
            self.order.append(self.name)

        async def shutdown(self) -> None:
            self.calls += 1
            self.order.append(self.name)

    order: list[str] = []

    class _Resolver:
        def stop(self) -> None:
            order.append("resolver")

    stop = asyncio.Event()

    async def control_plane_task(name: str) -> None:
        try:
            await stop.wait()
        finally:
            order.append(name)

    gateway = object.__new__(MarieServerGateway)
    gateway.logger = _Logger()
    gateway.resolver = _Resolver()
    gateway.etcd_client = mock.MagicMock()
    gateway.etcd_client.close.side_effect = lambda: order.append("etcd")
    gateway.job_scheduler = _AsyncStopper("scheduler", order)
    gateway.job_manager = _AsyncStopper("job-manager", order)
    gateway.grpc_broker = _AsyncStopper("broker", order)
    gateway.llm_dispatch_runtime = _AsyncStopper("llm", order)
    gateway._background_services_shutdown = False
    gateway._background_services_lock = asyncio.Lock()
    gateway._control_plane_tasks = {
        asyncio.create_task(control_plane_task("control-plane"))
    }
    gateway._rebuild_task = asyncio.create_task(control_plane_task("rebuild"))

    await asyncio.sleep(0)

    await gateway._shutdown_background_services()
    await gateway._shutdown_background_services()

    assert gateway.job_scheduler.calls == 1
    assert gateway.job_manager.calls == 1
    assert gateway.grpc_broker.calls == 1
    assert gateway.llm_dispatch_runtime.calls == 1
    assert order[0] == "resolver"
    assert set(order[1:3]) == {"control-plane", "rebuild"}
    assert order[3:] == ["scheduler", "job-manager", "broker", "llm", "etcd"]
    assert not gateway._control_plane_tasks
    assert gateway._rebuild_task is None


@pytest.mark.asyncio
async def test_gateway_shutdown_drains_pending_job_events_before_unsubscribe():
    release = asyncio.Event()
    received: list[int] = []
    publisher = EventPublisher(
        max_queue_size=4,
        worker_count=1,
        publish_blocking=True,
        subscriber_timeout_s=0,
    )

    async def lifecycle_handler(_event_type: str, message: dict) -> None:
        await release.wait()
        received.append(message["sequence"])

    publisher.subscribe(JobStatus.SUCCEEDED, lifecycle_handler)

    class _JobManager:
        async def shutdown(self) -> None:
            await publisher.stop()

    class _Scheduler:
        async def stop(self) -> None:
            await publisher.join()
            publisher.unsubscribe(JobStatus.SUCCEEDED, lifecycle_handler)

    class _Stopper:
        async def stop(self) -> None:
            return None

    gateway = object.__new__(MarieServerGateway)
    gateway.logger = _Logger()
    gateway.job_manager = _JobManager()
    gateway.job_scheduler = _Scheduler()
    gateway.grpc_broker = None
    gateway.resolver = None
    gateway.etcd_client = mock.MagicMock()
    gateway.llm_dispatch_runtime = _Stopper()
    gateway._background_services_shutdown = False
    gateway._background_services_lock = asyncio.Lock()
    gateway._control_plane_tasks = set()
    gateway._rebuild_task = None

    await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1", "sequence": 1})
    await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1", "sequence": 2})
    shutdown = asyncio.create_task(gateway._shutdown_background_services())
    await asyncio.sleep(0)

    assert not shutdown.done()

    release.set()
    await asyncio.wait_for(shutdown, timeout=1)

    assert received == [1, 2]
