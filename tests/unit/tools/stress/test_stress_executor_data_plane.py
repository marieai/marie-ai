from __future__ import annotations

import argparse
import asyncio

import grpc
import pytest
from grpc_reflection.v1alpha import reflection

from marie.proto import jina_pb2, jina_pb2_grpc
from tools.stress.stress_executor_data_plane import (
    PING_ENDPOINT,
    StressResult,
    _build_request,
    _DataPlaneProbe,
    _normalize_target,
    _run_requests,
)


class _ResetServicer(jina_pb2_grpc.JinaSingleDataRequestRPCServicer):
    async def process_single_data(self, request, context):
        await context.abort(
            grpc.StatusCode.UNAVAILABLE,
            "recvmsg:Connection reset by peer",
        )


class _EchoServicer(jina_pb2_grpc.JinaSingleDataRequestRPCServicer):
    def __init__(self) -> None:
        self.endpoints: list[str] = []

    async def process_single_data(self, request, context):
        metadata = dict(context.invocation_metadata())
        self.endpoints.append(metadata["endpoint"])
        return request


def test_build_request_matches_data_and_ping_shapes() -> None:
    data_request, data_request_id, data_endpoint = _build_request(
        kind="data",
        sequence=4,
        deployment="corr_indexing_executor",
        endpoint="/document/index",
        asset_key="s3://marie/stress/sample.tif",
        pages=[0, 2],
        parameter_template={"payload": {"mode": "sparse"}},
    )
    ping_request, ping_request_id, ping_endpoint = _build_request(
        kind="ping",
        sequence=5,
        deployment="corr_indexing_executor",
        endpoint="/document/index",
        asset_key=None,
        pages=[],
        parameter_template={},
    )

    assert data_endpoint == "/document/index"
    assert data_request.header.target_executor == "corr_indexing_executor"
    assert data_request.parameters["job_id"] == data_request_id
    assert data_request.parameters["payload"] == {"mode": "sparse"}
    assert data_request.data.docs[0].asset_key == "s3://marie/stress/sample.tif"
    assert data_request.data.docs[0].pages == [0, 2]
    assert ping_endpoint == PING_ENDPOINT
    assert ping_request.header.target_executor == "corr_indexing_executor"
    assert ping_request.parameters == {"job_id": ping_request_id}
    assert ping_request.data.docs[0].text.startswith("ping : corr_indexing_executor")


@pytest.mark.asyncio
async def test_probe_sends_single_data_rpc_with_endpoint_metadata() -> None:
    servicer = _EchoServicer()
    server = grpc.aio.server()
    jina_pb2_grpc.add_JinaSingleDataRequestRPCServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    probe = _DataPlaneProbe(
        target=_normalize_target(f"127.0.0.1:{port}"),
        deployment="corr_indexing_executor",
        endpoint="/document/index",
        asset_key="s3://marie/stress/sample.tif",
        pages=[],
        parameters={},
        timeout=1.0,
        ping_channel="shared",
    )

    try:
        result = await probe.send("data", 0)
    finally:
        await probe.close()
        await server.stop(grace=None)

    assert result.ok is True
    assert result.grpc_code == "OK"
    assert result.response_code == "SUCCESS"
    assert servicer.endpoints == ["/document/index"]


@pytest.mark.asyncio
async def test_probe_captures_connection_reset_details() -> None:
    server = grpc.aio.server()
    jina_pb2_grpc.add_JinaSingleDataRequestRPCServicer_to_server(
        _ResetServicer(), server
    )
    reflection.enable_server_reflection(
        (
            jina_pb2.DESCRIPTOR.services_by_name[
                "JinaSingleDataRequestRPC"
            ].full_name,
            reflection.SERVICE_NAME,
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    probe = _DataPlaneProbe(
        target=_normalize_target(f"127.0.0.1:{port}"),
        deployment="corr_indexing_executor",
        endpoint="/document/index",
        asset_key=None,
        pages=[],
        parameters={},
        timeout=1.0,
        ping_channel="fresh",
    )

    try:
        result = await probe.send("ping", 0)
    finally:
        await probe.close()
        await server.stop(grace=None)

    assert result.ok is False
    assert result.request_kind == "ping"
    assert result.channel_mode == "fresh"
    assert result.phase == "process_single_data"
    assert result.grpc_code == "UNAVAILABLE"
    assert result.grpc_details == "recvmsg:Connection reset by peer"


@pytest.mark.asyncio
async def test_run_requests_bounds_combined_data_and_ping_concurrency() -> None:
    class ConcurrentProbe:
        active = 0
        max_active = 0

        async def send(self, kind: str, sequence: int) -> StressResult:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            await asyncio.sleep(0)
            type(self).active -= 1
            return StressResult(
                timestamp="2026-07-26T00:00:00+00:00",
                sequence=sequence,
                request_id=f"request-{sequence}-{kind}",
                request_kind=kind,
                channel_mode="fresh" if kind == "ping" else "shared",
                phase="process_single_data",
                address="127.0.0.1:5000",
                endpoint=PING_ENDPOINT if kind == "ping" else "/document/index",
                ok=True,
                latency_ms=1.0,
                grpc_code="OK",
                grpc_details=None,
                response_code="SUCCESS",
                response_description=None,
            )

        async def close(self) -> None:
            return None

    args = argparse.Namespace(
        count=3,
        mode="both",
        concurrency=4,
        interval=0,
        jsonl=None,
    )
    probe = ConcurrentProbe()

    results = await _run_requests(probe, args)  # type: ignore[arg-type]

    assert len(results) == 6
    assert sum(result.request_kind == "data" for result in results) == 3
    assert sum(result.request_kind == "ping" for result in results) == 3
    assert ConcurrentProbe.max_active == 4
