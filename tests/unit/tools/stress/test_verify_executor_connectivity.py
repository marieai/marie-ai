from __future__ import annotations

import argparse
import asyncio
import json

import grpc
import pytest
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from marie.proto import jina_pb2, jina_pb2_grpc
from marie.serve.discovery.address import JsonAddress
from tools.stress.verify_executor_connectivity import (
    MARIE_DATA_SERVICE,
    ProbeResult,
    Target,
    _extract_deployment_addresses,
    _normalize_target,
    _parse_args,
    _run_probes,
    _TargetProbe,
)


class _DiscoveryServicer(jina_pb2_grpc.JinaDiscoverEndpointsRPCServicer):
    async def endpoint_discovery(self, request, context):
        return jina_pb2.EndpointsProto(endpoints=["/document/index", "/dry_run"])


def test_normalize_target_supports_plain_and_tls_addresses() -> None:
    plain = _normalize_target("172.20.10.49:53267")
    secure = _normalize_target("grpcs://executor.internal:53268")

    assert plain.address == "172.20.10.49:53267"
    assert plain.host == "172.20.10.49"
    assert plain.port == 53267
    assert plain.tls is False
    assert secure.address == "executor.internal:53268"
    assert secure.tls is True


def test_extract_deployment_addresses_decodes_registry_metadata() -> None:
    metadata = {
        "corr_indexing_executor": ["172.20.10.49:53267"],
        "other_executor": ["172.20.10.50:51000"],
    }
    encoded = JsonAddress("172.20.10.49:53267", json.dumps(metadata)).add_value()
    unrelated = JsonAddress(
        "172.20.10.51:53269",
        json.dumps({"other_executor": ["172.20.10.51:53269"]}),
    ).add_value()
    registrations = [
        JsonAddress.from_value({"172.20.10.49:53267": encoded}),
        JsonAddress.from_value({"172.20.10.51:53269": unrelated}),
    ]

    addresses, controls = _extract_deployment_addresses(
        registrations, "corr_indexing_executor"
    )

    assert addresses == {"172.20.10.49:53267"}
    assert controls == {"172.20.10.49:53267"}


@pytest.mark.asyncio
async def test_probe_verifies_health_and_endpoint_discovery() -> None:
    server = grpc.aio.server()
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    jina_pb2_grpc.add_JinaDiscoverEndpointsRPCServicer_to_server(
        _DiscoveryServicer(), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    await health_servicer.set(
        MARIE_DATA_SERVICE, health_pb2.HealthCheckResponse.SERVING
    )

    probe = _TargetProbe(_normalize_target(f"127.0.0.1:{port}"), timeout=1.0)
    try:
        result = await probe.probe()
    finally:
        await probe.close()
        await server.stop(grace=None)

    assert result.tcp_ok is True
    assert result.grpc_ok is True
    assert result.grpc_code == "OK"
    assert result.health_status == "SERVING"
    assert result.endpoints == ("/document/index", "/dry_run")


@pytest.mark.asyncio
async def test_probe_reports_named_service_not_serving() -> None:
    server = grpc.aio.server()
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    await health_servicer.set(
        MARIE_DATA_SERVICE, health_pb2.HealthCheckResponse.NOT_SERVING
    )

    probe = _TargetProbe(_normalize_target(f"127.0.0.1:{port}"), timeout=1.0)
    try:
        result = await probe.probe()
    finally:
        await probe.close()
        await server.stop(grace=None)

    assert result.tcp_ok is True
    assert result.grpc_ok is False
    assert result.grpc_phase == "health"
    assert result.grpc_code == "NOT_SERVING"
    assert result.health_status == "NOT_SERVING"


@pytest.mark.asyncio
async def test_run_probes_sends_bounded_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConcurrentProbe:
        active = 0
        max_active = 0
        closed = False

        def __init__(self, target: Target, _timeout: float) -> None:
            self.target = target

        async def probe(self) -> ProbeResult:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            await asyncio.sleep(0)
            type(self).active -= 1
            return ProbeResult(
                timestamp="2026-07-26T00:00:00+00:00",
                address=self.target.address,
                tcp_ok=True,
                tcp_latency_ms=1.0,
                tcp_error=None,
                grpc_ok=True,
                grpc_phase="endpoint_discovery",
                grpc_latency_ms=1.0,
                grpc_code="OK",
                grpc_details=None,
                health_status="SERVING",
                endpoints=("/dry_run",),
            )

        async def close(self) -> None:
            type(self).closed = True

    monkeypatch.setattr(
        "tools.stress.verify_executor_connectivity._TargetProbe", ConcurrentProbe
    )
    args = argparse.Namespace(
        timeout=1.0,
        count=5,
        concurrency=3,
        interval=0,
        jsonl=None,
    )

    results = await _run_probes([_normalize_target("127.0.0.1:5000")], args)

    assert len(results) == 5
    assert ConcurrentProbe.max_active == 3
    assert ConcurrentProbe.closed is True


def test_parse_args_rejects_invalid_concurrency() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--address", "127.0.0.1:5000", "--concurrency", "0"])
