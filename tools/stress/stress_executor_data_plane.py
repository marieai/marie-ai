#!/usr/bin/env python3
"""Stress executor data and dry-run RPCs with controlled channel reuse."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import grpc
from docarray import DocList
from docarray.documents import TextDoc

from marie.api.docs import AssetKeyDoc
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.serve.networking.connection_stub import _ConnectionStubs
from marie.serve.networking.instrumentation import _NetworkingHistograms
from marie.serve.networking.utils import get_grpc_channel
from marie.types_core.request.data import DataRequest

if __package__:
    from .verify_executor_connectivity import (
        Target,
        _normalize_target,
        _one_line,
        _percentile,
    )
else:
    from verify_executor_connectivity import (
        Target,
        _normalize_target,
        _one_line,
        _percentile,
    )

RequestKind = Literal["data", "ping"]
PingChannel = Literal["fresh", "shared"]
PING_ENDPOINT = "_jina_dry_run_"


@dataclass(frozen=True)
class StressResult:
    timestamp: str
    sequence: int
    request_id: str
    request_kind: RequestKind
    channel_mode: str
    phase: str
    address: str
    endpoint: str
    ok: bool
    latency_ms: float
    grpc_code: str
    grpc_details: str | None
    response_code: str | None
    response_description: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_request(
    kind: RequestKind,
    sequence: int,
    deployment: str | None,
    endpoint: str,
    asset_key: str | None,
    pages: Sequence[int],
    parameter_template: dict[str, Any],
) -> tuple[DataRequest, str, str]:
    request_id = f"grpc-stress-{sequence}-{uuid.uuid4()}"
    request = DataRequest()
    request.header.request_id = request_id
    request.header.exec_endpoint = PING_ENDPOINT if kind == "ping" else endpoint
    if deployment:
        request.header.target_executor = deployment

    if kind == "ping":
        docs = DocList[TextDoc](
            [TextDoc(text=f"ping : {deployment or 'executor'}@{PING_ENDPOINT}")]
        )
        request.parameters = {"job_id": request_id}
    else:
        if asset_key is None:
            raise ValueError("--asset-key is required for data requests")
        docs = DocList[AssetKeyDoc](
            [AssetKeyDoc(asset_key=asset_key, pages=list(pages) or None)]
        )
        parameters = copy.deepcopy(parameter_template)
        parameters["job_id"] = request_id
        parameters.setdefault("ref_id", "grpc-stress")
        parameters.setdefault("ref_type", "stress")
        parameters.setdefault("queue_id", "grpc-stress")
        parameters.setdefault("payload", {})
        request.parameters = parameters

    request.document_array_cls = docs.__class__
    request.data.docs = docs
    return request, request_id, request.header.exec_endpoint


class _DataPlaneProbe:
    def __init__(
        self,
        target: Target,
        deployment: str | None,
        endpoint: str,
        asset_key: str | None,
        pages: Sequence[int],
        parameters: dict[str, Any],
        timeout: float,
        ping_channel: PingChannel,
    ) -> None:
        self.target = target
        self.deployment = deployment
        self.endpoint = endpoint
        self.asset_key = asset_key
        self.pages = pages
        self.parameters = parameters
        self.timeout = timeout
        self.ping_channel = ping_channel
        self.channel = get_grpc_channel(
            target.address,
            asyncio=True,
            tls=target.tls,
        )
        self.stub = jina_pb2_grpc.JinaSingleDataRequestRPCStub(self.channel)

    async def close(self) -> None:
        await self.channel.close()

    async def send(self, kind: RequestKind, sequence: int) -> StressResult:
        request, request_id, endpoint = _build_request(
            kind=kind,
            sequence=sequence,
            deployment=self.deployment,
            endpoint=self.endpoint,
            asset_key=self.asset_key,
            pages=self.pages,
            parameter_template=self.parameters,
        )
        started = time.perf_counter()
        channel_mode = self.ping_channel if kind == "ping" else "shared"
        phase = "process_single_data"
        try:
            if kind == "ping" and self.ping_channel == "fresh":
                async with get_grpc_channel(
                    self.target.address,
                    asyncio=True,
                    tls=self.target.tls,
                ) as channel:
                    connection = _ConnectionStubs(
                        address=self.target.address,
                        channel=channel,
                        deployment_name=self.deployment or "executor",
                        histograms=_NetworkingHistograms(),
                    )
                    phase = "service_discovery"
                    await connection._init_stubs()
                    phase = "process_single_data"
                    response, _ = await connection.send_requests(
                        requests=[request],
                        metadata={},
                        compression=False,
                        timeout=self.timeout,
                    )
            else:
                metadata = () if kind == "ping" else (("endpoint", endpoint),)
                response = await self.stub.process_single_data(
                    request,
                    metadata=metadata,
                    timeout=self.timeout,
                )
            latency_ms = (time.perf_counter() - started) * 1000
            response_code = jina_pb2.StatusProto.StatusCode.Name(
                response.header.status.code
            )
            return StressResult(
                timestamp=_utc_now(),
                sequence=sequence,
                request_id=request_id,
                request_kind=kind,
                channel_mode=channel_mode,
                phase=phase,
                address=self.target.address,
                endpoint=endpoint,
                ok=response.header.status.code == jina_pb2.StatusProto.SUCCESS,
                latency_ms=latency_ms,
                grpc_code="OK",
                grpc_details=None,
                response_code=response_code,
                response_description=response.header.status.description or None,
            )
        except grpc.aio.AioRpcError as exc:
            return StressResult(
                timestamp=_utc_now(),
                sequence=sequence,
                request_id=request_id,
                request_kind=kind,
                channel_mode=channel_mode,
                phase=phase,
                address=self.target.address,
                endpoint=endpoint,
                ok=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                grpc_code=exc.code().name,
                grpc_details=exc.details(),
                response_code=None,
                response_description=None,
            )


def _print_result(result: StressResult) -> None:
    outcome = "ok" if result.ok else "FAIL"
    detail = result.grpc_details or result.response_description
    print(
        f"{result.timestamp} seq={result.sequence} kind={result.request_kind} "
        f"channel={result.channel_mode} phase={result.phase} "
        f"request_id={result.request_id} address={result.address} "
        f"endpoint={result.endpoint} {outcome} {result.latency_ms:.1f}ms "
        f"grpc={result.grpc_code} response={result.response_code or '-'} "
        f"details={_one_line(detail)}"
    )


def _request_kinds(mode: str) -> tuple[RequestKind, ...]:
    if mode == "both":
        return "data", "ping"
    return (mode,)  # type: ignore[return-value]


async def _run_requests(
    probe: _DataPlaneProbe, args: argparse.Namespace
) -> list[StressResult]:
    work = [
        (kind, sequence)
        for sequence in range(args.count)
        for kind in _request_kinds(args.mode)
    ]
    results: list[StressResult] = []
    output = Path(args.jsonl).open("a", encoding="utf-8") if args.jsonl else None
    try:
        for batch_start in range(0, len(work), args.concurrency):
            batch = work[batch_start : batch_start + args.concurrency]
            tasks = [
                asyncio.create_task(probe.send(kind, sequence))
                for kind, sequence in batch
            ]
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                _print_result(result)
                if output is not None:
                    output.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                    output.flush()
            if batch_start + len(batch) < len(work) and args.interval > 0:
                await asyncio.sleep(args.interval)
    finally:
        if output is not None:
            output.close()
        await probe.close()
    return results


def _print_summary(results: Sequence[StressResult]) -> bool:
    failed = False
    print("\nExecutor data-plane stress summary")
    groups = sorted({(result.request_kind, result.channel_mode) for result in results})
    for kind, channel_mode in groups:
        kind_results = [
            result
            for result in results
            if result.request_kind == kind and result.channel_mode == channel_mode
        ]
        successes = sum(result.ok for result in kind_results)
        latencies = [result.latency_ms for result in kind_results]
        failures = Counter(
            (
                f"grpc:{result.grpc_code}"
                if result.grpc_code != "OK"
                else f"response:{result.response_code}"
            )
            for result in kind_results
            if not result.ok
        )
        failed = failed or successes != len(kind_results)
        print(
            f"{kind}/{channel_mode}: requests={len(kind_results)} "
            f"ok={successes}/{len(kind_results)} "
            f"p50={_percentile(latencies, 0.50):.1f}ms "
            f"p95={_percentile(latencies, 0.95):.1f}ms "
            f"max={max(latencies, default=0.0):.1f}ms"
        )
        if failures:
            print(
                "  failures: "
                + ", ".join(f"{reason}={count}" for reason, count in failures.items())
            )
    return failed


def _load_parameters(args: argparse.Namespace) -> dict[str, Any]:
    if args.parameters_file:
        raw = Path(args.parameters_file).read_text(encoding="utf-8")
    else:
        raw = args.parameters_json or "{}"
    parameters = json.loads(raw)
    if not isinstance(parameters, dict):
        raise ValueError("executor parameters must be a JSON object")
    return parameters


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send concurrent Marie process_single_data and _jina_dry_run_ requests "
            "directly to an executor with shared or fresh ping channels."
        )
    )
    parser.add_argument("--address", required=True, help="Executor host:port")
    parser.add_argument(
        "--deployment",
        help="Executor deployment name placed in target_executor",
    )
    parser.add_argument("--endpoint", default="/document/index")
    parser.add_argument(
        "--mode",
        choices=("data", "ping", "both"),
        default="both",
        help="Request type to send; count applies to each selected type",
    )
    parser.add_argument(
        "--asset-key",
        help="Asset URI placed in AssetKeyDoc for data requests",
    )
    parser.add_argument("--page", action="append", type=int, default=[])
    parameter_group = parser.add_mutually_exclusive_group()
    parameter_group.add_argument(
        "--parameters-json",
        help="Executor parameters as a JSON object; job_id is replaced per request",
    )
    parameter_group.add_argument(
        "--parameters-file",
        help="Path to executor parameters JSON; job_id is replaced per request",
    )
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--ping-channel",
        choices=("fresh", "shared"),
        default="fresh",
        help="Use a fresh reflected channel like JobSupervisor.ping, or the shared data channel",
    )
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--jsonl", help="Append every result to this JSONL file")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mode in {"data", "both"} and not args.asset_key:
        parser.error("--asset-key is required when --mode includes data")
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.endpoint and not args.endpoint.startswith("/"):
        args.endpoint = f"/{args.endpoint}"
    return args


async def _execute(
    target: Target, args: argparse.Namespace, parameters: dict[str, Any]
) -> list[StressResult]:
    probe = _DataPlaneProbe(
        target=target,
        deployment=args.deployment,
        endpoint=args.endpoint,
        asset_key=args.asset_key,
        pages=args.page,
        parameters=parameters,
        timeout=args.timeout,
        ping_channel=args.ping_channel,
    )
    return await _run_requests(probe, args)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        target = _normalize_target(args.address, args.tls)
        parameters = _load_parameters(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid stress configuration: {exc}", file=sys.stderr)
        return 2

    kinds = _request_kinds(args.mode)
    total = args.count * len(kinds)
    print(
        f"Stressing {target.address} kinds={','.join(kinds)} "
        f"requests={total} concurrency={args.concurrency} "
        f"ping-channel={args.ping_channel} interval={args.interval:.3f}s "
        f"timeout={args.timeout:.3f}s"
    )
    if "data" in kinds:
        print(
            f"WARNING: data requests execute {args.endpoint} against {args.asset_key}"
        )

    results = asyncio.run(_execute(target, args, parameters))
    return 1 if _print_summary(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
