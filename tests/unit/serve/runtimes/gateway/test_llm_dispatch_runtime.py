import asyncio
from unittest import mock

import pytest

from marie.engine.llm_queue.config import (
    DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
    LlmQueueConfig,
)
from marie.excepts import RuntimeFailToStart
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.servers.marie_gateway import MarieServerGateway


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


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
    def __init__(self, *, queue_client, client, client_factory, config, logger):
        self.queue_client = queue_client
        self.client = client
        self.client_factory = client_factory
        self.config = config
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
        return {
            "enabled": self.config.enabled,
            "pool_id": self.config.pool_id,
            "running": self.running,
            "request_queue_depth": 0,
        }


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
        *, queue_client, client, client_factory, config, logger, backend_address
    ):
        assert backend_address == "http://queue-backend:4000/v1"
        assert client is None
        assert callable(client_factory)
        dispatcher = _FakeDispatcher(
            queue_client=queue_client,
            client=client,
            client_factory=client_factory,
            config=config,
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
        def __init__(self):
            self.calls = 0

        async def stop(self):
            self.calls += 1

    gateway = object.__new__(MarieServerGateway)
    gateway.logger = _Logger()
    gateway.job_scheduler = _AsyncStopper()
    gateway.grpc_broker = _AsyncStopper()
    gateway.llm_dispatch_runtime = _AsyncStopper()
    gateway._background_services_shutdown = False
    gateway._background_services_lock = asyncio.Lock()

    await gateway._shutdown_background_services()
    await gateway._shutdown_background_services()

    assert gateway.job_scheduler.calls == 1
    assert gateway.grpc_broker.calls == 1
    assert gateway.llm_dispatch_runtime.calls == 1
