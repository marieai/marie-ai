import logging

import pytest

# Import first to break a pre-existing circular import between
# marie.serve.runtimes.gateway.marie.__init__ and marie_gateway.
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (  # noqa: F401
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.servers import marie_gateway as marie_gateway_module
from marie.serve.runtimes.servers.composite import CompositeServer
from marie.serve.runtimes.servers.marie_gateway import MarieServerGateway


def _patch_setup_server_collaborators(monkeypatch, sensor_calls):
    """Stub out everything setup_server() touches besides the sensor wiring."""

    monkeypatch.setattr(marie_gateway_module, "setup_toast_events", lambda cfg: None)
    monkeypatch.setattr(marie_gateway_module, "setup_storage", lambda cfg: None)
    monkeypatch.setattr(marie_gateway_module, "setup_auth", lambda cfg: None)
    monkeypatch.setattr(
        marie_gateway_module, "setup_llm_tracking", lambda *args, **kwargs: None
    )

    def fake_setup_sensor_worker(sensor_config, db_config):
        sensor_calls.append((sensor_config, db_config))

    monkeypatch.setattr(
        marie_gateway_module, "setup_sensor_worker", fake_setup_sensor_worker
    )

    async def fake_super_setup_server(self):
        return None

    monkeypatch.setattr(CompositeServer, "setup_server", fake_super_setup_server)

    async def fake_setup_service_discovery(self, **kwargs):
        return None

    monkeypatch.setattr(
        MarieServerGateway, "setup_service_discovery", fake_setup_service_discovery
    )


def _gateway_for_setup_server(args):
    gateway = object.__new__(MarieServerGateway)
    gateway.logger = logging.getLogger("test-setup-server")
    gateway.servers = []
    gateway.args = args
    return gateway


@pytest.mark.asyncio
async def test_setup_server_starts_sensor_worker_when_configured(monkeypatch):
    sensor_calls = []
    _patch_setup_server_collaborators(monkeypatch, sensor_calls)

    sensor_config = {"enabled": True, "daemon_interval_seconds": 5}
    kv_store_kwargs = {"provider": "postgresql", "hostname": "localhost"}
    gateway = _gateway_for_setup_server(
        {
            "sensors": sensor_config,
            "kv_store_kwargs": kv_store_kwargs,
            "discovery_host": "0.0.0.0",
            "discovery_port": 2379,
            "discovery_service_name": "gateway/marie",
        }
    )

    await gateway.setup_server()

    assert sensor_calls == [(sensor_config, kv_store_kwargs)]


@pytest.mark.asyncio
async def test_setup_server_sensor_worker_noop_when_sensors_config_absent(
    monkeypatch,
):
    sensor_calls = []
    _patch_setup_server_collaborators(monkeypatch, sensor_calls)

    # No "sensors" key at all -- must not raise, and setup_sensor_worker
    # itself no-ops on an empty config.
    gateway = _gateway_for_setup_server(
        {
            "kv_store_kwargs": {"provider": "postgresql", "hostname": "localhost"},
            "discovery_host": "0.0.0.0",
            "discovery_port": 2379,
            "discovery_service_name": "gateway/marie",
        }
    )

    await gateway.setup_server()

    assert sensor_calls == [({}, {"provider": "postgresql", "hostname": "localhost"})]
