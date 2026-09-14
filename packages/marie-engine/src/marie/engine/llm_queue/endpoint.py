"""Registered queue endpoints and content-free terminal HTTP transport outcomes."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import ipaddress
import json
import logging
import math
import socket
import time
import traceback
from contextvars import ContextVar
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from httpcore import AsyncNetworkBackend
from httpcore._backends.anyio import AnyIOBackend
from marie.engine.completion_contract import (
    CompletionCallParams,
    require_terminal_completion,
)
from marie.engine.llm_queue.queue_keys import validate_identifier

_queue_transport = ContextVar('llm_queue_transport', default=False)


class _QueueTransportLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _queue_transport.get()


# Suppress only this queue call's raw transport events; direct calls retain their logging.
for _logger_name in (
    'httpx',
    'httpcore.connection',
    'httpcore.http11',
    'httpcore.http2',
    'httpcore.proxy',
    'httpcore.socks',
):
    logging.getLogger(_logger_name).addFilter(_QueueTransportLogFilter())


class EndpointPolicyError(ValueError):
    """A registered address failed the connection-time operator policy."""


@dataclass(frozen=True, slots=True)
class RegisteredEndpoint:
    endpoint_id: str
    base_url: str = field(repr=False)
    api_key: str | None = field(default=None, repr=False)
    allow_private: bool = False
    allow_loopback: bool = False
    execution_limit: int = 8
    execution_bytes: int = 64 * 1024 * 1024
    call_timeout_seconds: float = 60.0
    max_response_bytes: int = 1024 * 1024
    retry_429: bool = False
    enabled: bool = True

    def target_fingerprint(self) -> str:
        parsed = urlsplit(self.base_url)
        target = [
            parsed.scheme.lower(),
            parsed.hostname.lower(),
            parsed.port or (443 if parsed.scheme == 'https' else 80),
            parsed.path.rstrip('/'),
        ]
        return hashlib.sha256(
            json.dumps(target, separators=(',', ':')).encode()
        ).hexdigest()

    def transport_fingerprint(self) -> str:
        """Opaque immutable binding; never include this value in diagnostics."""
        value = json.dumps(
            [
                self.base_url.rstrip('/'),
                self.api_key,
                self.allow_private,
                self.allow_loopback,
                self.call_timeout_seconds,
                self.max_response_bytes,
                self.retry_429,
            ],
            separators=(',', ':'),
        )
        return hashlib.sha256(value.encode()).hexdigest()

    def __post_init__(self) -> None:
        validate_identifier(self.endpoint_id)
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {'http', 'https'}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or '?' in self.base_url
            or '#' in self.base_url
        ):
            raise ValueError('Invalid registered endpoint address')
        try:
            port = parsed.port
        except ValueError:
            raise ValueError('Invalid registered endpoint port') from None
        if port is not None and not 0 < port < 65536:
            raise ValueError('Invalid registered endpoint port')
        if any(
            type(value) is not bool
            for value in (
                self.enabled,
                self.allow_private,
                self.allow_loopback,
                self.retry_429,
            )
        ):
            raise ValueError('Endpoint policy flags must be booleans')
        if any(
            type(value) is not int
            for value in (
                self.execution_limit,
                self.execution_bytes,
                self.max_response_bytes,
            )
        ):
            raise ValueError('Endpoint capacity limits must be integers')
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.execution_limit,
                self.execution_bytes,
                self.call_timeout_seconds,
                self.max_response_bytes,
            )
        ):
            raise ValueError('Endpoint limits must be positive')
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            return
        self.check_address(address)

    def check_address(
        self, address: ipaddress.IPv4Address | ipaddress.IPv6Address
    ) -> None:
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if (
            str(address) in {'100.100.100.200', '168.63.129.16'}
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            raise EndpointPolicyError('Endpoint address is disallowed')
        if address.is_loopback:
            if self.allow_loopback:
                return
            raise EndpointPolicyError('Loopback endpoint requires operator policy')
        if not address.is_global and not self.allow_private:
            raise EndpointPolicyError('Private endpoint requires operator policy')


class _PinnedNetworkBackend(AsyncNetworkBackend):
    def __init__(self, endpoint: RegisteredEndpoint) -> None:
        self.endpoint = endpoint
        self.backend = AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        # Resolve on each new connection and connect the checked numeric address.
        # httpcore still owns the original origin and TLS server_hostname/SNI.
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM),
            timeout=timeout,
        )
        for item in addresses:
            self.endpoint.check_address(ipaddress.ip_address(item[4][0]))
        return await self.backend.connect_tcp(
            addresses[0][4][0], port, timeout, local_address, socket_options
        )

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(slots=True)
class ExecutionOutcome:
    response: dict[str, Any] | None = None
    category: str = ''
    retryable: bool = False
    remote_settled: bool = False
    availability_success: bool = False
    availability_failure: bool = False
    retry_after_ms: int = 0


def _transport_outcome(error: BaseException) -> ExecutionOutcome:
    if isinstance(error, httpx.PoolTimeout):
        return ExecutionOutcome(
            category='pool_pressure', retryable=True, remote_settled=True
        )
    current: BaseException | None = error
    for _ in range(12):
        if current is None:
            break
        if (
            isinstance(current, OSError)
            and current.errno == errno.ECONNREFUSED
            and isinstance(error, httpx.ConnectError)
        ):
            return ExecutionOutcome(
                category='connect_refused',
                retryable=True,
                remote_settled=True,
                availability_failure=True,
            )
        current = current.__cause__ or current.__context__
    categories = {
        httpx.ReadTimeout: 'read_timeout',
        httpx.WriteTimeout: 'write_timeout',
        httpx.ConnectTimeout: 'connect_timeout',
        httpx.ReadError: 'read_error',
        httpx.WriteError: 'write_error',
        httpx.ConnectError: 'connect_error',
        httpx.RemoteProtocolError: 'protocol_error',
        TimeoutError: 'call_timeout',
    }
    return ExecutionOutcome(
        category=categories.get(type(error), 'transport_unknown'),
        availability_failure=True,
    )


def _clear_transport_tracebacks(error: BaseException) -> None:
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(
            linked
            for linked in (current.__cause__, current.__context__)
            if linked is not None
        )
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = current.__context__ = None


def _finite_json_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite response number')
    return number


class EndpointClient:
    """One reusable client per registered endpoint, owned by the dispatch loop."""

    def __init__(self, endpoint: RegisteredEndpoint) -> None:
        self.endpoint = endpoint
        transport = httpx.AsyncHTTPTransport(
            retries=0,
            limits=httpx.Limits(
                max_connections=endpoint.execution_limit,
                max_keepalive_connections=endpoint.execution_limit,
            ),
        )
        transport._pool._network_backend = _PinnedNetworkBackend(endpoint)
        self.client = httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            trust_env=False,
            timeout=endpoint.call_timeout_seconds,
        )

    async def execute(
        self, call: CompletionCallParams, *, timeout_seconds: float
    ) -> ExecutionOutcome:
        token = _queue_transport.set(True)
        try:
            return await self._execute(call, timeout_seconds=timeout_seconds)
        finally:
            call = None
            _queue_transport.reset(token)

    async def _execute(
        self, call: CompletionCallParams, *, timeout_seconds: float
    ) -> ExecutionOutcome:
        require_terminal_completion(call)
        body = call.to_create_kwargs()
        extra_body = body.pop('extra_body', None)
        # SDK request options cannot override this queue-owned HTTP boundary.
        for option in ('timeout', 'extra_headers', 'extra_query', 'max_retries'):
            if option in body:
                return ExecutionOutcome(category='invalid_request', remote_settled=True)
        if extra_body:
            body.update(extra_body)
        headers = (
            {'Authorization': f'Bearer {self.endpoint.api_key}'}
            if self.endpoint.api_key
            else {}
        )
        try:
            async with asyncio.timeout(timeout_seconds):
                async with self.client.stream(
                    'POST',
                    self.endpoint.base_url.rstrip('/') + '/chat/completions',
                    json=body,
                    headers=headers,
                ) as response:
                    status = response.status_code
                    if status == 429 and self.endpoint.retry_429:
                        retry_after = response.headers.get('retry-after', '')
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            try:
                                delay = max(
                                    0,
                                    parsedate_to_datetime(retry_after).timestamp()
                                    - time.time(),
                                )
                            except (ValueError, TypeError, OverflowError):
                                delay = -1
                        if 0 <= delay <= 30:
                            return ExecutionOutcome(
                                category='rate_limited',
                                retryable=True,
                                remote_settled=True,
                                retry_after_ms=int(delay * 1000),
                            )
                    if 400 <= status < 500 or 300 <= status < 400:
                        category = (
                            'authentication'
                            if status in {401, 403}
                            else (
                                'route_or_model'
                                if status == 404
                                else (
                                    'invalid_request'
                                    if status in {400, 422}
                                    else 'provider_rejected'
                                )
                            )
                        )
                        return ExecutionOutcome(category=category, remote_settled=True)
                    if status >= 500:
                        return ExecutionOutcome(
                            category='provider_5xx', availability_failure=True
                        )
                    data = bytearray()
                    async for part in response.aiter_bytes():
                        if len(data) + len(part) > self.endpoint.max_response_bytes:
                            return ExecutionOutcome(category='response_too_large')
                        data.extend(part)
                    parsed = json.loads(
                        data,
                        parse_float=_finite_json_number,
                        parse_constant=_finite_json_number,
                    )
                    if not isinstance(parsed, dict):
                        return ExecutionOutcome(
                            category='invalid_response', remote_settled=True
                        )
                    return ExecutionOutcome(
                        response=parsed, remote_settled=True, availability_success=True
                    )
        except (httpx.HTTPError, TimeoutError) as error:
            outcome = _transport_outcome(error)
            _clear_transport_tracebacks(error)
        except OSError:
            outcome = ExecutionOutcome(
                category='connect_unknown', availability_failure=True
            )
        except EndpointPolicyError:
            outcome = ExecutionOutcome(category='endpoint_policy', remote_settled=True)
        except (ValueError, RecursionError):
            outcome = ExecutionOutcome(category='invalid_response', remote_settled=True)
        finally:
            # Transport exception cycles can otherwise retain this completed frame.
            body = call = extra_body = None
        # No exception/traceback or HTTP request escapes into commit/retry containers.
        return outcome

    async def close(self) -> None:
        await self.client.aclose()
