#!/usr/bin/env python3
"""Verify an executor's advertised TCP and Marie gRPC data-plane endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import grpc
from google.protobuf import empty_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc

from marie.proto import jina_pb2_grpc

MARIE_DATA_SERVICE = "jina.JinaSingleDataRequestRPC"


@dataclass(frozen=True)
class Target:
    address: str
    host: str
    port: int
    tls: bool


@dataclass(frozen=True)
class ProbeResult:
    timestamp: str
    address: str
    tcp_ok: bool
    tcp_latency_ms: float
    tcp_error: str | None
    grpc_ok: bool
    grpc_phase: str
    grpc_latency_ms: float
    grpc_code: str
    grpc_details: str | None
    health_status: str | None
    endpoints: tuple[str, ...]


def _normalize_target(raw_address: str, force_tls: bool = False) -> Target:
    raw_address = raw_address.strip()
    if not raw_address:
        raise ValueError("executor address must not be empty")

    parsed = urlparse(raw_address if "://" in raw_address else f"//{raw_address}")
    if parsed.scheme and parsed.scheme not in {"grpc", "grpcs"}:
        raise ValueError(
            f"unsupported address scheme {parsed.scheme!r}; use grpc or grpcs"
        )
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"invalid executor address {raw_address!r}; expected host:port"
        )

    host = parsed.hostname
    display_host = f"[{host}]" if ":" in host else host
    return Target(
        address=f"{display_host}:{parsed.port}",
        host=host,
        port=parsed.port,
        tls=force_tls or parsed.scheme == "grpcs",
    )


def _decode_metadata(raw_metadata: object) -> dict[str, Any]:
    metadata = raw_metadata
    for _ in range(2):
        if not isinstance(metadata, str):
            break
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        raise ValueError("discovery metadata is not a JSON object")
    return metadata


def _extract_deployment_addresses(
    registrations: Iterable[object], deployment: str
) -> tuple[set[str], set[str]]:
    addresses: set[str] = set()
    control_addresses: set[str] = set()
    for registration in registrations:
        control_address = getattr(registration, "_addr", None)
        raw_metadata = getattr(registration, "_metadata", None)
        metadata = _decode_metadata(raw_metadata)
        if deployment not in metadata:
            continue
        if control_address:
            control_addresses.add(str(control_address))
        deployment_addresses = metadata.get(deployment, [])
        if isinstance(deployment_addresses, str):
            deployment_addresses = [deployment_addresses]
        if not isinstance(deployment_addresses, list):
            raise ValueError(f"discovery addresses for {deployment!r} are not a list")
        addresses.update(str(value) for value in deployment_addresses)
    return addresses, control_addresses


def _resolve_from_etcd(args: argparse.Namespace) -> tuple[set[str], set[str]]:
    from marie.serve.discovery.address import JsonAddress
    from marie.serve.discovery.etcd_client import EtcdClient

    client = EtcdClient(
        etcd_host=args.discovery_host,
        etcd_port=args.discovery_port,
        namespace=args.discovery_namespace,
        retry_times=1,
        timeout=args.timeout,
    )
    try:
        values = client.get_prefix(args.discovery_service_name)
        registrations = [
            JsonAddress.from_value({control_address: value})
            for control_address, value in values.items()
        ]
        return _extract_deployment_addresses(registrations, args.deployment)
    finally:
        client.close()


async def _probe_tcp(target: Target, timeout: float) -> tuple[bool, float, str | None]:
    started = time.perf_counter()
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(target.host, target.port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True, (time.perf_counter() - started) * 1000, None
    except (TimeoutError, OSError) as exc:
        return False, (time.perf_counter() - started) * 1000, str(exc)


class _TargetProbe:
    def __init__(self, target: Target, timeout: float) -> None:
        self.target = target
        self.timeout = timeout
        if target.tls:
            self.channel = grpc.aio.secure_channel(
                target.address, grpc.ssl_channel_credentials()
            )
        else:
            self.channel = grpc.aio.insecure_channel(target.address)
        self.health_stub = health_pb2_grpc.HealthStub(self.channel)
        self.endpoint_stub = jina_pb2_grpc.JinaDiscoverEndpointsRPCStub(self.channel)

    async def close(self) -> None:
        await self.channel.close()

    async def probe(self) -> ProbeResult:
        tcp_ok, tcp_latency_ms, tcp_error = await _probe_tcp(self.target, self.timeout)
        grpc_started = time.perf_counter()
        phase = "health"
        try:
            response = await self.health_stub.Check(
                health_pb2.HealthCheckRequest(service=MARIE_DATA_SERVICE),
                timeout=self.timeout,
            )
            health_status = health_pb2.HealthCheckResponse.ServingStatus.Name(
                response.status
            )
            if response.status != health_pb2.HealthCheckResponse.SERVING:
                return ProbeResult(
                    timestamp=_utc_now(),
                    address=self.target.address,
                    tcp_ok=tcp_ok,
                    tcp_latency_ms=tcp_latency_ms,
                    tcp_error=tcp_error,
                    grpc_ok=False,
                    grpc_phase=phase,
                    grpc_latency_ms=(time.perf_counter() - grpc_started) * 1000,
                    grpc_code="NOT_SERVING",
                    grpc_details=f"health status is {health_status}",
                    health_status=health_status,
                    endpoints=(),
                )

            phase = "endpoint_discovery"
            endpoint_response = await self.endpoint_stub.endpoint_discovery(
                empty_pb2.Empty(), timeout=self.timeout
            )
            return ProbeResult(
                timestamp=_utc_now(),
                address=self.target.address,
                tcp_ok=tcp_ok,
                tcp_latency_ms=tcp_latency_ms,
                tcp_error=tcp_error,
                grpc_ok=True,
                grpc_phase=phase,
                grpc_latency_ms=(time.perf_counter() - grpc_started) * 1000,
                grpc_code="OK",
                grpc_details=None,
                health_status=health_status,
                endpoints=tuple(endpoint_response.endpoints),
            )
        except grpc.aio.AioRpcError as exc:
            return ProbeResult(
                timestamp=_utc_now(),
                address=self.target.address,
                tcp_ok=tcp_ok,
                tcp_latency_ms=tcp_latency_ms,
                tcp_error=tcp_error,
                grpc_ok=False,
                grpc_phase=phase,
                grpc_latency_ms=(time.perf_counter() - grpc_started) * 1000,
                grpc_code=exc.code().name,
                grpc_details=exc.details(),
                health_status=None,
                endpoints=(),
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _one_line(value: str | None, limit: int = 240) -> str:
    if value is None:
        return "-"
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 3]}..."


def _print_result(result: ProbeResult) -> None:
    tcp = f"ok {result.tcp_latency_ms:.1f}ms"
    if not result.tcp_ok:
        tcp = f"FAIL {result.tcp_latency_ms:.1f}ms {_one_line(result.tcp_error)}"

    grpc_result = (
        f"ok {result.grpc_latency_ms:.1f}ms "
        f"health={result.health_status} endpoints={len(result.endpoints)}"
    )
    if not result.grpc_ok:
        grpc_result = (
            f"FAIL {result.grpc_latency_ms:.1f}ms phase={result.grpc_phase} "
            f"code={result.grpc_code} details={_one_line(result.grpc_details)}"
        )
    print(f"{result.timestamp} {result.address} tcp=[{tcp}] grpc=[{grpc_result}]")


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _print_summary(results: Sequence[ProbeResult]) -> bool:
    failed = False
    print("\nExecutor connectivity summary")
    for address in sorted({result.address for result in results}):
        target_results = [result for result in results if result.address == address]
        tcp_successes = sum(result.tcp_ok for result in target_results)
        grpc_successes = sum(result.grpc_ok for result in target_results)
        failures = Counter(
            f"{result.grpc_phase}:{result.grpc_code}"
            for result in target_results
            if not result.grpc_ok
        )
        grpc_latencies = [result.grpc_latency_ms for result in target_results]
        failed = failed or tcp_successes != len(target_results)
        failed = failed or grpc_successes != len(target_results)
        print(
            f"{address}: samples={len(target_results)} "
            f"tcp_ok={tcp_successes}/{len(target_results)} "
            f"grpc_ok={grpc_successes}/{len(target_results)} "
            f"grpc_p50={_percentile(grpc_latencies, 0.50):.1f}ms "
            f"grpc_p95={_percentile(grpc_latencies, 0.95):.1f}ms "
            f"grpc_max={max(grpc_latencies, default=0.0):.1f}ms"
        )
        if failures:
            print(
                "  failures: "
                + ", ".join(f"{reason}={count}" for reason, count in failures.items())
            )
    return failed


async def _run_probes(
    targets: Sequence[Target], args: argparse.Namespace
) -> list[ProbeResult]:
    probes = [_TargetProbe(target, args.timeout) for target in targets]
    results: list[ProbeResult] = []
    output = Path(args.jsonl).open("a", encoding="utf-8") if args.jsonl else None
    try:
        for batch_start in range(0, args.count, args.concurrency):
            batch_size = min(args.concurrency, args.count - batch_start)
            sample_results = await asyncio.gather(
                *(probe.probe() for probe in probes for _ in range(batch_size))
            )
            for result in sample_results:
                results.append(result)
                _print_result(result)
                if output is not None:
                    output.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                    output.flush()
            if batch_start + batch_size < args.count and args.interval > 0:
                await asyncio.sleep(args.interval)
    finally:
        if output is not None:
            output.close()
        await asyncio.gather(*(probe.close() for probe in probes))
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly verify raw TCP, Marie gRPC health, and endpoint discovery "
            "for executor data-plane addresses. Run it in the gateway container or "
            "the same network namespace as the gateway."
        )
    )
    parser.add_argument(
        "--address",
        action="append",
        default=[],
        help="Executor host:port or grpc[s]://host:port; may be repeated",
    )
    parser.add_argument(
        "--deployment",
        help="Deployment name to resolve from Marie's etcd service registration",
    )
    parser.add_argument("--discovery-host", help="Etcd host used by the gateway")
    parser.add_argument("--discovery-port", type=int, default=2379)
    parser.add_argument("--discovery-namespace", default="marie")
    parser.add_argument("--discovery-service-name", default="gateway/marie")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent probes per endpoint (default: 1)",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--tls", action="store_true", help="Use TLS for all supplied addresses"
    )
    parser.add_argument("--jsonl", help="Append every probe result to this JSONL file")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.address and not (args.deployment and args.discovery_host):
        parser.error(
            "provide --address, or provide --deployment together with --discovery-host"
        )
    if args.discovery_host and not args.deployment:
        parser.error("--discovery-host requires --deployment")
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    supplied_addresses = set(args.address)
    initial_addresses: set[str] = set()
    initial_controls: set[str] = set()

    if args.discovery_host:
        try:
            initial_addresses, initial_controls = _resolve_from_etcd(args)
        except Exception as exc:
            print(
                f"Discovery lookup failed: {type(exc).__name__}: {exc}", file=sys.stderr
            )
            return 2
        print(
            f"Discovery {args.discovery_namespace}/{args.discovery_service_name} "
            f"deployment={args.deployment}: "
            f"addresses={sorted(initial_addresses)} controls={sorted(initial_controls)}"
        )

    raw_addresses = supplied_addresses | initial_addresses
    if not raw_addresses:
        print(
            f"No addresses registered for deployment {args.deployment!r}",
            file=sys.stderr,
        )
        return 2

    try:
        targets = sorted(
            {_normalize_target(value, args.tls) for value in raw_addresses},
            key=lambda target: target.address,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    unregistered = {
        _normalize_target(value, args.tls).address for value in supplied_addresses
    } - {_normalize_target(value, args.tls).address for value in initial_addresses}
    discovery_mismatch = bool(args.discovery_host and unregistered)
    if discovery_mismatch:
        print(
            "WARNING: supplied address(es) are not currently registered for "
            f"{args.deployment}: {sorted(unregistered)}"
        )

    print(
        f"Probing {len(targets)} endpoint(s), samples-per-endpoint={args.count}, "
        f"concurrency={args.concurrency}, interval={args.interval:.3f}s, "
        f"timeout={args.timeout:.3f}s"
    )
    results = asyncio.run(_run_probes(targets, args))
    failed = _print_summary(results)

    if args.discovery_host:
        try:
            final_addresses, final_controls = _resolve_from_etcd(args)
        except Exception as exc:
            print(
                f"Final discovery lookup failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
        if (initial_addresses, initial_controls) != (
            final_addresses,
            final_controls,
        ):
            discovery_mismatch = True
            print(
                "WARNING: discovery registration changed during the probe: "
                f"addresses {sorted(initial_addresses)} -> {sorted(final_addresses)}, "
                f"controls {sorted(initial_controls)} -> {sorted(final_controls)}"
            )

    return 1 if failed or discovery_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
