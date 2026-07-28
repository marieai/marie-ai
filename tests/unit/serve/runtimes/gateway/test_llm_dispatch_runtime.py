import asyncio
from unittest import mock

import pytest
from marie.engine.llm_queue.config import (
    DEFAULT_LLM_QUEUE_POOL_ID,
    DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
    LlmQueueConfig,
)
from marie.engine.llm_queue.dispatcher import DrrQueuedBatchDispatcher
from marie.engine.llm_queue.scheduler import DrrLaneConfig
from marie.engine.llm_queue.scheduler_config import (
    DatabaseSchedulerConfigSource,
    LlmQueueSchedulerConfig,
    StaticSchedulerConfigSource,
    scheduler_config_from_mapping,
)

from marie.excepts import RuntimeFailToStart
from marie.job.common import JobStatus
from marie.job.event_publisher import EventPublisher
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
    _build_dispatcher,
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


def _queue_config(**overrides) -> LlmQueueConfig:
    values = dict(
        enabled=True,
        valkey_url="redis://valkey:6379/0",
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
    return LlmQueueConfig(**values)


class _FakeQueueClient:
    def __init__(self, url: str):
        self.url = url
        self.closed = False
        self.depth_calls = []

    def request_queue_depth(self, pool_id: str) -> int:
        self.depth_calls.append(pool_id)
        return 0

    def close(self) -> None:
        self.closed = True


class _FakeDispatcher:
    def __init__(
        self, *, queue_client, client, client_factory, config, scheduler_config, logger
    ):
        self.queue_client = queue_client
        self.client = client
        self.client_factory = client_factory
        self.config = config
        self.scheduler_config = scheduler_config
        self.logger = logger
        self.start_calls = 0
        self.stop_calls = 0
        self.running = False

    def start(self) -> None:
        self.start_calls += 1
        self.running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def health(self) -> dict[str, object]:
        health = {
            "enabled": self.config.enabled,
            "pool_id": self.config.pool_id,
            "running": self.running,
            "request_queue_depth": 0,
        }
        if self.scheduler_config.is_drr:
            health["scheduler_policy"] = "drr"
            health["lanes"] = [
                {"pool_id": lane.pool_id}
                for lane in self.scheduler_config.lanes
                if lane.enabled
            ]
        return health


class _FakeSchedulerConfigRepository:
    def __init__(self, payload):
        self.payload = payload
        self.fabric_group_ids = []

    def load_scheduler_config(self, fabric_group_id: str):
        self.fabric_group_ids.append(fabric_group_id)
        return self.payload


class _FakeOpenAIClient:
    def __init__(self):
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_gateway_llm_dispatch_runtime_is_noop_when_disabled():
    runtime = GatewayLlmDispatchRuntime(
        logger=_Logger(),
        queue_config=_queue_config(enabled=False, valkey_url=None),
    )

    await runtime.start()

    health = runtime.health()
    assert health["enabled"] is False
    assert health["running"] is False

    await runtime.stop()


@pytest.mark.asyncio
async def test_gateway_llm_dispatch_runtime_starts_and_stops_cleanly():
    queue_clients = []
    openai_clients = []
    dispatchers = []

    def queue_client_factory(url: str):
        client = _FakeQueueClient(url)
        queue_clients.append(client)
        return client

    def openai_client_factory(api_key: str, base_url: str | None):
        assert api_key == "test-key"
        assert base_url == "http://queue-backend:4000/v1"
        client = _FakeOpenAIClient()
        openai_clients.append(client)
        return client

    def dispatcher_factory(
        *,
        queue_client,
        client,
        client_factory,
        client_factory_for_base_url,
        config,
        scheduler_config,
        logger,
        backend_address,
    ):
        assert backend_address == "http://queue-backend:4000/v1"
        assert client is None
        assert callable(client_factory)
        assert callable(client_factory_for_base_url)
        dispatcher = _FakeDispatcher(
            queue_client=queue_client,
            client=client,
            client_factory=client_factory,
            config=config,
            scheduler_config=scheduler_config,
            logger=logger,
        )
        dispatchers.append(dispatcher)
        return dispatcher

    runtime = GatewayLlmDispatchRuntime(
        logger=_Logger(),
        queue_config=_queue_config(),
        queue_client_factory=queue_client_factory,
        openai_client_factory=openai_client_factory,
        dispatcher_factory=dispatcher_factory,
    )

    with mock.patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE": "http://queue-backend:4000/v1",
        },
    ):
        await runtime.start()
        assert dispatchers[0].start_calls == 1
        assert runtime.health()["running"] is True

        await runtime.stop()

    assert dispatchers[0].stop_calls == 1
    assert queue_clients[0].closed is True
    assert openai_clients == []
    assert runtime.health()["running"] is False


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
        config = LlmQueueConfig.from_env()

    assert config.fabric_group_id == "default"
    assert config.gateway_id == "gateway-localhost"


def test_llm_dispatch_runtime_event_uses_control_plane_event_shape():
    snapshot = {
        "contract_version": "v2",
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
        DEFAULT_LLM_QUEUE_POOL_ID,
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


def test_build_dispatcher_uses_drr_dispatcher_for_drr_policy():
    scheduler_config = LlmQueueSchedulerConfig(
        policy="drr",
        total_concurrent_dispatch=3,
        lanes=(
            DrrLaneConfig(pool_id="interactive", quantum=2),
            DrrLaneConfig(pool_id="backfill", quantum=1),
        ),
    )

    dispatcher = _build_dispatcher(
        queue_client=_FakeQueueClient("redis://valkey:6379/0"),
        client=object(),
        config=_queue_config(),
        scheduler_config=scheduler_config,
        logger=_Logger(),
        backend_address="http://queue-backend:4000/v1",
    )

    assert isinstance(dispatcher, DrrQueuedBatchDispatcher)
    assert dispatcher.scheduler.total_concurrent_dispatch == 3


def test_build_dispatcher_builds_lane_endpoint_adapters():
    scheduler_config = LlmQueueSchedulerConfig(
        policy="drr",
        total_concurrent_dispatch=1,
        lanes=(
            DrrLaneConfig(
                pool_id="interactive",
                endpoint_url="http://user:secret@interactive:4000/v1",
            ),
        ),
    )

    def client_factory_for_base_url(base_url):
        assert base_url == "http://user:secret@interactive:4000/v1"
        return lambda: object()

    dispatcher = _build_dispatcher(
        queue_client=_FakeQueueClient("redis://valkey:6379/0"),
        client=None,
        client_factory=lambda: object(),
        client_factory_for_base_url=client_factory_for_base_url,
        config=_queue_config(),
        scheduler_config=scheduler_config,
        logger=_Logger(),
        backend_address="http://queue-backend:4000/v1",
    )

    lane = next(
        lane
        for lane in dispatcher.health()["lanes"]
        if lane["pool_id"] == "interactive"
    )
    assert lane["pool_id"] == "interactive"
    assert lane["endpoint_url"] == "http://interactive:4000/v1"


@pytest.mark.asyncio
async def test_gateway_runtime_uses_injected_scheduler_config_source():
    queue_clients = []
    dispatchers = []
    logger = _Logger()
    scheduler_config = LlmQueueSchedulerConfig(
        policy="drr",
        total_concurrent_dispatch=2,
        lanes=(
            DrrLaneConfig(
                pool_id="interactive",
                endpoint_url="http://user:secret@interactive:4000/v1",
            ),
            DrrLaneConfig(pool_id="backfill"),
        ),
    )

    def queue_client_factory(url: str):
        client = _FakeQueueClient(url)
        queue_clients.append(client)
        return client

    def dispatcher_factory(
        *,
        queue_client,
        client,
        client_factory,
        client_factory_for_base_url,
        config,
        scheduler_config,
        logger,
        backend_address,
    ):
        dispatcher = _FakeDispatcher(
            queue_client=queue_client,
            client=client,
            client_factory=client_factory,
            config=config,
            scheduler_config=scheduler_config,
            logger=logger,
        )
        dispatchers.append(dispatcher)
        return dispatcher

    runtime = GatewayLlmDispatchRuntime(
        logger=logger,
        queue_config=_queue_config(),
        queue_client_factory=queue_client_factory,
        dispatcher_factory=dispatcher_factory,
        scheduler_config_source=StaticSchedulerConfigSource(scheduler_config),
    )

    with mock.patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE": "http://queue-backend:4000/v1",
        },
    ):
        await runtime.start()
        health = runtime.health()
        await runtime.stop()

    assert queue_clients[0].depth_calls == [
        "interactive",
        "backfill",
        DEFAULT_LLM_QUEUE_POOL_ID,
    ]
    assert health["pool_ids"] == [
        "interactive",
        "backfill",
        DEFAULT_LLM_QUEUE_POOL_ID,
    ]
    assert health["pool_count"] == 3
    started_message = next(
        message
        for message in logger.info_messages
        if message.startswith("Started LLM DRR dispatch runtime")
    )
    assert "  pools: 3" in started_message
    assert (
        "interactive -> http://interactive:4000/v1 "
        "(explicit; quantum=1, protected=0, max=unbounded, burst=default)"
        in started_message
    )
    assert (
        "backfill -> http://queue-backend:4000/v1 "
        "(runtime default; quantum=1, protected=0, max=unbounded, burst=default)"
        in started_message
    )
    assert (
        "default -> http://queue-backend:4000/v1 "
        "(runtime default; quantum=1, protected=0, max=unbounded, burst=default)"
        in started_message
    )
    assert (
        "Stopped LLM DRR dispatch runtime for 3 pools: interactive, backfill, default"
        in logger.info_messages
    )
    assert [lane.pool_id for lane in dispatchers[0].scheduler_config.lanes] == [
        "interactive",
        "backfill",
        DEFAULT_LLM_QUEUE_POOL_ID,
    ]


@pytest.mark.asyncio
async def test_gateway_llm_dispatch_runtime_requires_valkey_when_enabled():
    runtime = GatewayLlmDispatchRuntime(
        logger=_Logger(),
        queue_config=_queue_config(valkey_url=None),
    )

    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with pytest.raises(RuntimeFailToStart, match="LLM_QUEUE_VALKEY_URL"):
            await runtime.start()


@pytest.mark.asyncio
async def test_gateway_background_runtime_start_calls_llm_dispatch_runtime():
    started = 0

    class _Runtime:
        async def start(self):
            nonlocal started
            started += 1

    gateway = object.__new__(MarieServerGateway)
    gateway.logger = _Logger()
    gateway.llm_dispatch_runtime = _Runtime()

    await gateway._start_gateway_background_runtimes()

    assert started == 1


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

    await gateway._shutdown_background_services()
    await gateway._shutdown_background_services()

    assert gateway.job_scheduler.calls == 1
    assert gateway.job_manager.calls == 1
    assert gateway.grpc_broker.calls == 1
    assert gateway.llm_dispatch_runtime.calls == 1
    assert order == ["resolver", "scheduler", "job-manager", "broker", "llm", "etcd"]


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

    await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1", "sequence": 1})
    await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1", "sequence": 2})
    shutdown = asyncio.create_task(gateway._shutdown_background_services())
    await asyncio.sleep(0)

    assert not shutdown.done()

    release.set()
    await asyncio.wait_for(shutdown, timeout=1)

    assert received == [1, 2]
