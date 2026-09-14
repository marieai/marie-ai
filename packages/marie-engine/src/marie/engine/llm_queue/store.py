"""Opt-in V3 authoritative request storage; no provider or runtime integration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any, Callable, Literal
from uuid import uuid4

from marie.engine.completion_contract import (
    QueuedCompletionEnvelopeV3,
    require_terminal_completion,
)
from marie.engine.llm_queue.queue_io import _build_sync_client
from marie.engine.llm_queue.queue_keys import QueueKeys, validate_identifier


class StoreUnavailable(RuntimeError):
    """Unknown store state: pause starts and reconcile, never infer producer death."""


class AdmissionConflict(ValueError):
    """The attempt ID already belongs to another immutable admission."""


class StaleOwner(RuntimeError):
    """The dispatcher no longer holds the authoritative owner lease."""


class ProducerDead(RuntimeError):
    """The authoritative producer lease has ended."""


class TransitionRejected(RuntimeError):
    """The store rejected a payload fetch or session creation."""


@dataclass(frozen=True, slots=True)
class StoreLimits:
    max_active_items: int = 1000
    max_payload_bytes: int = 256 * 1024 * 1024
    max_inline_payload_bytes: int = 16 * 1024 * 1024
    max_records: int = 2000
    max_storage_bytes: int = 128 * 1024 * 1024
    max_ready_ids: int = 2000
    max_execution_items: int = 16
    max_execution_bytes: int = 64 * 1024 * 1024
    max_producers: int = 1000
    max_routes: int = 1000
    max_endpoints: int = 1000
    result_allowance: int = 32 * 1024
    metadata_bytes: int = 4096
    delivery_grace_ms: int = 30_000
    claim_lease_ms: int = 10_000
    remote_uncertainty_ms: int = 120_000
    max_deadline_ms: int = 900_000
    max_lease_ms: int = 60_000
    max_attempts: int = 5
    cleanup_page_size: int = 100

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or not 0 < value <= 2**40:
                raise ValueError(f'{name} must be a bounded positive integer')
        if self.result_allowance < 64 or self.metadata_bytes < 4096:
            raise ValueError(
                'Result allowance must be >=64 and metadata allowance >=4096'
            )
        if self.cleanup_page_size > 1000:
            raise ValueError('cleanup_page_size must be <=1000')


@dataclass(frozen=True, slots=True)
class OwnerToken:
    identity: str
    generation: int

    @property
    def value(self) -> str:
        return f'{self.identity}:{self.generation}'


@dataclass(frozen=True, slots=True)
class StoreReply:
    disposition: str
    state: str = ''
    execution_seq: int = 0
    expires_at_ms: int = 0
    cost: int = 0
    payload_bytes: int = 0
    attempt_id: str = ''


@dataclass(frozen=True, slots=True)
class RequestMetadata:
    attempt_id: str
    producer_id: str
    pool_id: str
    endpoint_id: str
    config_revision: str | None
    state: str
    execution_seq: int
    expires_at_ms: int
    payload_bytes: int
    cost: int
    claim_id: str | None
    owner_id: str | None
    next_eligible: int
    uncertainty_until: int
    retain_until: int
    last_error: str | None = None
    model: str | None = None
    admitted_at_ms: int | None = None


class RequestStore:
    """Synchronous store. All times are queue-server epoch milliseconds.

    Mutation replies may be lost. Reconcile using the same attempt and claim IDs;
    an uncertain start authorization never permits a speculative second send.
    """

    def __init__(
        self,
        url: str,
        *,
        fabric_id: str,
        version: str,
        limits: StoreLimits | None = None,
        _join_existing: bool = False,
        io_timeout_seconds: float = 2,
    ) -> None:
        if version != 'v3':
            raise ValueError('RequestStore requires explicit version v3')
        self.keys = QueueKeys(fabric_id)
        self.limits = limits or StoreLimits()
        self.client = _build_sync_client(
            url,
            socket_connect_timeout=io_timeout_seconds,
            socket_timeout=io_timeout_seconds,
            retry_on_timeout=False,
        )
        source = (
            files('marie.engine.llm_queue')
            .joinpath('lua/request_store.lua')
            .read_text()
        )
        self._script = self.client.register_script(
            'local cleanup_only = false\n' + source
        )
        self._cleanup = self.client.register_script(
            '#!lua flags=allow-oom\nlocal cleanup_only = true\n' + source
        )
        # The client may be valkey-py or the existing redis-py fallback.
        module = __import__(
            type(self.client).__module__.split('.')[0] + '.exceptions',
            fromlist=['exceptions'],
        )
        self._client_error = getattr(
            module, 'ValkeyError', getattr(module, 'RedisError', RuntimeError)
        )
        try:
            current_limits = self._read('get', self.keys.prefix + 'limits')
            if _join_existing:
                if current_limits is None:
                    raise StoreUnavailable('Queue fabric policy is not initialized')
                try:
                    policy = json.loads(current_limits)
                    if not isinstance(policy, dict) or set(policy) != set(
                        asdict(StoreLimits())
                    ):
                        raise ValueError
                    self.limits = StoreLimits(**policy)
                except (ValueError, TypeError):
                    raise ValueError(
                        'Invalid authoritative queue fabric policy'
                    ) from None
            self._limits_json = json.dumps(
                asdict(self.limits), sort_keys=True, separators=(',', ':')
            )
            if current_limits is None:
                self._change('initialize')
            elif current_limits != self._limits_json:
                raise ValueError(
                    'Store limits differ from the authoritative fabric limits'
                )
        except BaseException:
            self.close()
            raise

    @classmethod
    def for_producer(
        cls, url: str, *, fabric_id: str, io_timeout_seconds: float = 0.25
    ) -> RequestStore:
        """Join an initialized fabric without choosing or writing its policy."""
        return cls(
            url,
            fabric_id=fabric_id,
            version='v3',
            _join_existing=True,
            io_timeout_seconds=io_timeout_seconds,
        )

    def _read(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except self._client_error:
            raise StoreUnavailable(
                'Queue store operation failed; state is unknown'
            ) from None

    def _invoke(
        self,
        op: str,
        *,
        attempt_id: str | None = None,
        producer_id: str | None = None,
        pool_id: str | None = None,
        owner: OwnerToken | None = None,
        cleanup: bool = False,
        **args: Any,
    ) -> dict[str, Any]:
        endpoint_id = args.get('endpoint_id')
        if attempt_id is not None and op != 'admit':
            identity = self._read(
                'hmget',
                self.keys.request(attempt_id),
                'producer_id',
                'pool_id',
                'endpoint_id',
            )
            if identity[0]:
                if producer_id is not None and producer_id != identity[0]:
                    raise AdmissionConflict('Request belongs to another producer')
                if pool_id is not None and pool_id != identity[1]:
                    raise AdmissionConflict('Request belongs to another pool')
                producer_id, pool_id, endpoint_id = identity
        keys = [
            (
                self.keys.request(attempt_id)
                if attempt_id is not None
                else self.keys.prefix + 'control:request'
            ),
            (
                self.keys.alive(producer_id)
                if producer_id is not None
                else self.keys.prefix + 'control:alive'
            ),
            (
                self.keys.members(producer_id)
                if producer_id is not None
                else self.keys.prefix + 'control:members'
            ),
            (
                self.keys.ready(pool_id)
                if pool_id is not None
                else self.keys.prefix + 'control:ready'
            ),
            (
                self.keys.route(pool_id)
                if pool_id is not None
                else self.keys.prefix + 'control:route'
            ),
            self.keys.prefix + 'usage',
            self.keys.owner,
            self.keys.prefix + 'owner-generation',
            self.keys.prefix + 'delayed',
            self.keys.prefix + 'processing',
            self.keys.prefix + 'deadlines',
            self.keys.prefix + 'retention',
            self.keys.prefix + 'producers',
            self.keys.prefix + 'limits',
            self.keys.prefix + 'routes',
            (
                self.keys.endpoint(endpoint_id)
                if endpoint_id is not None
                else self.keys.prefix + 'control:endpoint'
            ),
            self.keys.prefix + 'endpoints',
        ]
        keys.extend(
            self.keys.endpoint(identity) for identity in args.get('registry_ids', [])
        )
        payload = dict(
            op=op,
            fabric_id=self.keys.fabric_id,
            id=attempt_id or '_',
            producer_id=producer_id or '_',
            pool_id=pool_id or '_',
            limits=asdict(self.limits),
            limits_json=self._limits_json,
            **args,
        )
        if owner is not None:
            validate_identifier(owner.identity)
            payload.update(owner=owner.value, generation=owner.generation)
        try:
            result = json.loads(
                (self._cleanup if cleanup else self._script)(
                    keys=keys,
                    args=[json.dumps(payload, separators=(',', ':'), allow_nan=False)],
                )
            )
        except self._client_error:
            raise StoreUnavailable(
                'Queue store operation failed; state is unknown'
            ) from None
        disposition = result['disposition']
        if disposition == 'invalid_limits':
            raise ValueError('Store limits differ from the authoritative fabric limits')
        if disposition == 'stale_owner':
            raise StaleOwner('Dispatcher owner lease is no longer current')
        if disposition == 'conflict':
            raise AdmissionConflict('Attempt ID conflicts with its original admission')
        return result

    def _change(self, op: str, **kwargs: Any) -> StoreReply:
        return StoreReply(**self._invoke(op, **kwargs))

    def _lease(self, lease_ms: int) -> None:
        if type(lease_ms) is not int or not 0 < lease_ms <= self.limits.max_lease_ms:
            raise ValueError('lease_ms is outside the configured lease bounds')

    def server_time_ms(self) -> int:
        """Read the authoritative clock before allocating a fixed request deadline."""
        seconds, microseconds = self._read('time')
        return seconds * 1000 + microseconds // 1000

    def create_producer(self, *, lease_ms: int) -> str:
        """Establish a fresh process session synchronously; never reuse an old ID."""
        self._lease(lease_ms)
        producer_id = uuid4().hex
        result = self._change(
            'producer_create', producer_id=producer_id, lease_ms=lease_ms
        )
        if result.disposition != 'created':
            raise TransitionRejected(result.disposition)
        return producer_id

    def renew_producer(self, producer_id: str, *, lease_ms: int) -> bool:
        """Renew only an existing live session; errors mean unknown liveness."""
        self._lease(lease_ms)
        return (
            self._change(
                'producer_renew', producer_id=producer_id, lease_ms=lease_ms
            ).disposition
            == 'renewed'
        )

    def close_producer(self, producer_id: str) -> None:
        """Immediately prohibit further work; expire_producer removes owned content."""
        self._change('producer_close', producer_id=producer_id, cleanup=True)

    def acquire_owner(self, identity: str, *, lease_ms: int) -> OwnerToken | None:
        """Acquire a new fenced generation, or return None while another lease lives."""
        validate_identifier(identity)
        self._lease(lease_ms)
        result = self._invoke('owner_acquire', identity=identity, lease_ms=lease_ms)
        return (
            OwnerToken(identity, result['generation'])
            if result['disposition'] == 'acquired'
            else None
        )

    def renew_owner(self, owner: OwnerToken, *, lease_ms: int) -> None:
        """Extend the current generation without reacquiring it."""
        self._lease(lease_ms)
        self._change('owner_renew', owner=owner, lease_ms=lease_ms)

    def release_owner(self, owner: OwnerToken) -> StoreReply:
        """Release only this still-current owner after local workers have stopped."""
        return self._change('owner_release', owner=owner)

    def endpoint_status(self, endpoint_id: str) -> dict[str, Any]:
        """Read bounded circuit metadata without credentials or addresses."""
        names = [
            'circuit',
            'failures',
            'next_probe',
            'probe_claim',
            'probe_successes',
            'category',
            'reserved_items',
            'reserved_bytes',
        ]
        values = self._read('hmget', self.keys.endpoint(endpoint_id), *names)
        result = dict(zip(names, values))
        result['waiting_reason'] = (
            'probe_unresolved'
            if result['probe_claim']
            else 'circuit_cooldown'
            if result['circuit'] == 'open'
            else None
        )
        return result

    def scan_routes(
        self, *, cursor: int = 0, limit: int = 100
    ) -> tuple[int, list[str]]:
        """Discover authoritative route IDs with a bounded SSCAN work hint."""
        self._page(limit)
        return self._read(
            'sscan', self.keys.prefix + 'routes', cursor=cursor, count=limit
        )

    def disable_route(self, owner: OwnerToken, pool_id: str) -> StoreReply:
        """Stop new admission without changing old requests or endpoint reservations."""
        return self._change('route_disable', owner=owner, pool_id=pool_id)

    def route_status(self, pool_id: str) -> dict[str, Any]:
        names = [
            'endpoint_id',
            'revision',
            'enabled',
            'reserved_items',
            'reserved_bytes',
        ]
        values = self._read('hmget', self.keys.route(pool_id), *names)
        result = dict(zip(names, values))
        for name in ('reserved_items', 'reserved_bytes'):
            result[name] = int(result[name] or 0)
        return result

    def resolve_route(self, pool_id: str) -> dict[str, str] | None:
        """Return the authoritative enabled admission binding, without credentials."""
        values = self._read(
            'hmget', self.keys.route(pool_id), 'endpoint_id', 'revision', 'enabled'
        )
        if values[2] != '1':
            return None
        return dict(pool_id=pool_id, endpoint_id=values[0], config_revision=values[1])

    def processing_ids(self, *, offset: int = 0, limit: int = 100) -> list[str]:
        """Read a bounded page for owner handoff even before lease/uncertainty expiry."""
        self._page(limit)
        return self._read(
            'zrange', self.keys.prefix + 'processing', offset, offset + limit - 1
        )

    def record_endpoint_outcome(
        self,
        owner: OwnerToken,
        attempt_id: str,
        *,
        claim_id: str,
        execution_seq: int,
        outcome: str,
        category: str = 'none',
        open_ms: int = 30_000,
    ) -> StoreReply:
        """Apply circuit feedback once per authoritative provider execution."""
        return self._change(
            'circuit_feedback',
            owner=owner,
            attempt_id=attempt_id,
            claim_id=claim_id,
            execution_seq=execution_seq,
            outcome=outcome,
            reason=category,
            open_ms=open_ms,
        )

    def reject_claim(
        self, owner: OwnerToken, attempt_id: str, *, claim_id: str, category: str
    ) -> StoreReply:
        """Terminate an unsent claimed request rejected by queue-only validation."""
        return self._change(
            'reject_claim',
            owner=owner,
            attempt_id=attempt_id,
            claim_id=claim_id,
            reason=category,
        )

    def mark_unknown(
        self,
        owner: OwnerToken,
        attempt_id: str,
        *,
        claim_id: str,
        execution_seq: int,
        category: str,
    ) -> StoreReply:
        """Keep a sent execution reserved without making it runnable again."""
        return self._change(
            'mark_unknown',
            owner=owner,
            attempt_id=attempt_id,
            claim_id=claim_id,
            execution_seq=execution_seq,
            reason=category,
        )

    def configure_route(
        self,
        owner: OwnerToken,
        pool_id: str,
        endpoint_id: str,
        *,
        revision: str,
        execution_limit: int,
        execution_bytes: int,
        enabled: bool = True,
        gate_open: bool = True,
    ) -> StoreReply:
        """Set the lane's registered endpoint and minimal execution gate."""
        validate_identifier(pool_id)
        validate_identifier(endpoint_id)
        validate_identifier(revision)
        if (
            type(execution_limit) is not int
            or type(execution_bytes) is not int
            or not (
                0 < execution_limit <= self.limits.max_execution_items
                and 0 < execution_bytes <= self.limits.max_execution_bytes
            )
        ):
            raise ValueError('Route execution limits exceed fabric bounds')
        return self._change(
            'route',
            owner=owner,
            pool_id=pool_id,
            endpoint_id=endpoint_id,
            revision=revision,
            execution_limit=execution_limit,
            execution_bytes=execution_bytes,
            enabled='1' if enabled else '0',
            gate='open' if gate_open else 'closed',
        )

    def configure_endpoint(
        self,
        owner: OwnerToken,
        endpoint_id: str,
        *,
        execution_limit: int,
        execution_bytes: int,
        gate_open: bool = True,
        transport_fingerprint: str | None = None,
        target_fingerprint: str | None = None,
    ) -> StoreReply:
        """Set a physical endpoint cap and gate shared by all of its pools."""
        validate_identifier(endpoint_id)
        if transport_fingerprint is not None and (
            len(transport_fingerprint) != 64
            or any(c not in '0123456789abcdef' for c in transport_fingerprint)
        ):
            raise ValueError('Invalid endpoint binding')
        if (
            type(execution_limit) is not int
            or type(execution_bytes) is not int
            or not (
                0 < execution_limit <= self.limits.max_execution_items
                and 0 < execution_bytes <= self.limits.max_execution_bytes
            )
        ):
            raise ValueError('Endpoint execution limits exceed fabric bounds')
        registry_ids = []
        if target_fingerprint is not None:
            if len(target_fingerprint) != 64 or any(
                c not in '0123456789abcdef' for c in target_fingerprint
            ):
                raise ValueError('Invalid endpoint target')
            cursor = 0
            registered = set()
            while True:
                cursor, identities = self._read(
                    'sscan',
                    self.keys.prefix + 'endpoints',
                    cursor=cursor,
                    count=self.limits.cleanup_page_size,
                )
                registered.update(identities)
                if len(registered) > self.limits.max_endpoints:
                    raise StoreUnavailable('Endpoint registry exceeds bounds')
                if cursor == 0:
                    break
            registry_ids = sorted(registered)
        return self._change(
            'endpoint',
            owner=owner,
            endpoint_id=endpoint_id,
            execution_limit=execution_limit,
            execution_bytes=execution_bytes,
            gate='open' if gate_open else 'closed',
            transport_fingerprint=transport_fingerprint,
            target_fingerprint=target_fingerprint,
            registry_ids=registry_ids,
        )

    def admit(self, request: QueuedCompletionEnvelopeV3) -> StoreReply:
        """Admit one canonical call or reconcile its prior immutable admission."""
        require_terminal_completion(request.call)
        if (
            request.contract_version != 'v3'
            or QueueKeys(request.fabric_group_id) != self.keys
        ):
            raise ValueError('Request version and fabric must match this V3 store')
        for value in (
            request.producer_id,
            request.attempt_id,
            request.pool_id,
            request.endpoint_id,
            request.config_revision,
            request.logical_batch_id,
            request.logical_task_id,
        ):
            validate_identifier(value)
        if (
            type(request.item_index) is not int
            or not 0 <= request.item_index <= 2**31 - 1
        ):
            raise ValueError('item_index must be a nonnegative bounded integer')
        if (
            type(request.estimated_cost_units) is not int
            or not 1 <= request.estimated_cost_units <= 1_000_000
        ):
            raise ValueError(
                'estimated_cost_units must be a trusted bounded positive integer'
            )
        if (
            type(request.expires_at_ms) is not int
            or not 0 < request.expires_at_ms < 2**53
        ):
            raise ValueError(
                'expires_at_ms must be an absolute server time in milliseconds'
            )
        prepared = request.call.to_create_kwargs()
        if prepared.get('extra_body'):
            prepared.update(prepared['extra_body'])
        model = prepared.get('model')
        if not isinstance(model, str) or not re.fullmatch(
            r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}', model
        ):
            raise ValueError('Invalid diagnostic model alias')
        del prepared
        canonical = asdict(request)
        canonical['fabric_group_id'] = self.keys.fabric_id
        payload = json.dumps(
            canonical, separators=(',', ':'), sort_keys=True, allow_nan=False
        )
        payload_bytes = len(payload.encode())
        if payload_bytes > self.limits.max_inline_payload_bytes:
            raise ValueError('Inline request exceeds configured payload limit')
        # A retry cannot extend the stored deadline; it is not part of the content digest.
        canonical.pop('expires_at_ms')
        digest = hashlib.sha256(
            json.dumps(
                canonical, separators=(',', ':'), sort_keys=True, allow_nan=False
            ).encode()
        ).hexdigest()
        return self._change(
            'admit',
            attempt_id=request.attempt_id,
            producer_id=request.producer_id,
            pool_id=request.pool_id,
            endpoint_id=request.endpoint_id,
            config_revision=request.config_revision,
            logical_batch_id=request.logical_batch_id,
            logical_task_id=request.logical_task_id,
            item_index=request.item_index,
            expires_at_ms=request.expires_at_ms,
            payload=payload,
            payload_bytes=payload_bytes,
            digest=digest,
            cost=request.estimated_cost_units,
            model=model,
        )

    def metadata(self, attempt_id: str) -> RequestMetadata | None:
        """Read selected fields only; never load input bodies or terminal results."""
        names = [
            field
            for field in RequestMetadata.__dataclass_fields__
            if field != 'attempt_id'
        ]
        values = self._read('hmget', self.keys.request(attempt_id), *names)
        if values[0] is None:
            return None
        data = dict(zip(names, values))
        for name in (
            'execution_seq',
            'expires_at_ms',
            'payload_bytes',
            'cost',
            'next_eligible',
            'uncertainty_until',
            'retain_until',
        ):
            data[name] = int(data[name] or 0)
        if data['admitted_at_ms'] is not None:
            data['admitted_at_ms'] = int(data['admitted_at_ms'])
        return RequestMetadata(attempt_id=attempt_id, **data)

    def read_result(self, producer_id: str, attempt_id: str) -> Any | None:
        """Return a terminal result only for its original still-live producer."""
        result = self._invoke('result', producer_id=producer_id, attempt_id=attempt_id)
        if result['disposition'] == 'producer_dead':
            raise ProducerDead('Original producer lease has ended')
        return (
            json.loads(result['result']) if result['disposition'] == 'result' else None
        )

    def claim(
        self,
        owner: OwnerToken,
        attempt_id: str,
        *,
        pool_id: str,
        claim_id: str,
        expected_cost: int | None = None,
    ) -> StoreReply:
        """Claim the expected ready head once, reserving execution capacity."""
        validate_identifier(pool_id)
        validate_identifier(claim_id)
        if expected_cost is not None and (
            type(expected_cost) is not int or not 1 <= expected_cost <= 1_000_000
        ):
            raise ValueError('Invalid expected cost')
        return self._change(
            'claim',
            owner=owner,
            attempt_id=attempt_id,
            pool_id=pool_id,
            claim_id=claim_id,
            expected_cost=expected_cost,
        )

    def fetch_payload(
        self, owner: OwnerToken, attempt_id: str, *, claim_id: str
    ) -> dict[str, Any]:
        """Fetch only a live reserved claim; authorize_start must precede HTTP send."""
        result = self._invoke(
            'payload', owner=owner, attempt_id=attempt_id, claim_id=claim_id
        )
        if result['disposition'] != 'payload':
            raise TransitionRejected(result['disposition'])
        payload = json.loads(result['payload'])
        QueuedCompletionEnvelopeV3.from_json(result['payload'])
        return payload

    def authorize_start(
        self, owner: OwnerToken, attempt_id: str, *, claim_id: str
    ) -> StoreReply:
        """Fence a provider start. already_executing never authorizes another send."""
        return self._change(
            'start', owner=owner, attempt_id=attempt_id, claim_id=claim_id
        )

    def defer(
        self,
        owner: OwnerToken,
        attempt_id: str,
        *,
        claim_id: str,
        execution_seq: int,
        delay_ms: int,
        reason: str,
        remote_settled: bool = False,
    ) -> StoreReply:
        """Release a settled call or unsent claim and schedule a bounded retry."""
        validate_identifier(reason)
        if (
            type(delay_ms) is not int
            or not 0 <= delay_ms <= self.limits.max_deadline_ms
        ):
            raise ValueError('delay_ms is outside the request budget bounds')
        return self._change(
            'defer',
            owner=owner,
            attempt_id=attempt_id,
            claim_id=claim_id,
            execution_seq=execution_seq,
            delay_ms=delay_ms,
            reason=reason,
            remote_settled=remote_settled,
        )

    def promote_due(
        self,
        owner: OwnerToken,
        *,
        limit: int = 100,
        offset: int = 0,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[StoreReply]:
        """Discover bounded candidates; stop between separately fenced transitions."""
        if should_stop and should_stop():
            return []
        results = []
        for attempt in self.due_ids('delayed', limit=limit, offset=offset):
            if should_stop and should_stop():
                break
            results.append(self._change('promote', owner=owner, attempt_id=attempt))
        return results

    def finish(
        self,
        owner: OwnerToken,
        attempt_id: str,
        *,
        claim_id: str,
        execution_seq: int,
        result: Any,
        success: bool = True,
    ) -> StoreReply:
        """Commit a settled remote outcome and atomically remove its input payload."""
        body = json.dumps(
            result, separators=(',', ':'), sort_keys=True, allow_nan=False
        )
        oversized = len(body.encode()) > self.limits.result_allowance
        if oversized:
            body = '{"error":"result_too_large"}'
        return self._change(
            'finish',
            owner=owner,
            attempt_id=attempt_id,
            claim_id=claim_id,
            execution_seq=execution_seq,
            result=body,
            success=success,
            oversized=oversized,
        )

    def cancel_or_expire(
        self, producer_id: str, attempt_id: str, *, expire: bool = False
    ) -> StoreReply:
        """End local work; unresolved remote execution keeps its durable reservation."""
        return self._change(
            'cancel', producer_id=producer_id, attempt_id=attempt_id, expire=expire
        )

    def recover_claim(self, owner: OwnerToken, attempt_id: str) -> StoreReply:
        """Recover unsent live work or adopt unknown executions without resending."""
        return self._change('recover', owner=owner, attempt_id=attempt_id)

    def settle_remote(
        self,
        owner: OwnerToken,
        attempt_id: str,
        *,
        claim_id: str,
        execution_seq: int,
        evidence: Literal['remote_completed', 'remote_cancelled'],
    ) -> StoreReply:
        """Release remote capacity only on affirmative remote completion or cancellation."""
        if evidence not in {
            'remote_completed',
            'remote_cancelled',
        }:
            raise ValueError(
                'Local task cancellation is not remote settlement evidence'
            )
        return self._change(
            'settle',
            owner=owner,
            attempt_id=attempt_id,
            claim_id=claim_id,
            execution_seq=execution_seq,
            evidence=evidence,
            cleanup=True,
        )

    def expire_producer(
        self,
        producer_id: str,
        *,
        limit: int = 100,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[StoreReply]:
        """Discard a bounded page after confirmed death, stopping between store units."""
        self._page(limit)
        if should_stop and should_stop():
            return []
        if (
            self._change(
                'producer_expire', producer_id=producer_id, cleanup=True
            ).disposition
            == 'producer_live'
        ):
            return []
        if should_stop and should_stop():
            return []
        ids = self._read('srandmember', self.keys.members(producer_id), limit)
        results = []
        for attempt in ids:
            if should_stop and should_stop():
                return results
            results.append(
                self._change(
                    'discard', attempt_id=attempt, producer_id=producer_id, cleanup=True
                )
            )
        if not should_stop or not should_stop():
            self._change('producer_expire', producer_id=producer_id, cleanup=True)
        return results

    def purge_terminal(self, owner: OwnerToken, attempt_id: str) -> StoreReply:
        """Purge after the original deadline and delivery grace, or producer death."""
        return self._change('purge', owner=owner, attempt_id=attempt_id, cleanup=True)

    def _page(self, limit: int) -> None:
        if type(limit) is not int or not 1 <= limit <= self.limits.cleanup_page_size:
            raise ValueError('limit exceeds the configured bounded cleanup page')

    def due_ids(
        self,
        index: Literal['delayed', 'processing', 'deadlines', 'retention', 'producers'],
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """Discover a bounded page; a candidate alone never authorizes a mutation."""
        if index not in {
            'delayed',
            'processing',
            'deadlines',
            'retention',
            'producers',
        }:
            raise ValueError('Unsupported V3 index')
        self._page(limit)
        if type(offset) is not int or not 0 <= offset <= self.limits.max_records:
            raise ValueError('Invalid bounded index offset')
        return self._read(
            'zrangebyscore',
            self.keys.prefix + index,
            '-inf',
            self.server_time_ms(),
            start=offset,
            num=limit,
        )

    def ready_metadata(
        self,
        pool_id: str,
        *,
        limit: int,
        before_read: Callable[[], None] | None = None,
        before_candidate: Callable[[], None] | None = None,
    ) -> tuple[list[RequestMetadata], int]:
        """Sample bounded IDs and metadata, preserving valid rows beside corruption."""
        self._page(limit)
        rows, malformed = [], 0
        if before_read:
            before_read()
        for attempt in self._read('lrange', self.keys.ready(pool_id), 0, limit - 1):
            if before_read:
                before_read()
            if before_candidate:
                before_candidate()
            try:
                row = self.metadata(attempt)
            except (ValueError, TypeError):
                malformed += 1
                continue
            if row is not None:
                rows.append(row)
            else:
                malformed += 1
        return rows, malformed

    def ready_depth(self, pool_id: str) -> int:
        return self._read('llen', self.keys.ready(pool_id))

    def ready_head(self, pool_id: str) -> str | None:
        """Read an expected head ID for a subsequent atomic claim."""
        return self._read('lindex', self.keys.ready(pool_id), 0)

    def prune_ready(
        self,
        owner: OwnerToken,
        pool_id: str,
        *,
        limit: int = 100,
        should_stop: Callable[[], bool] | None = None,
    ) -> int:
        """Remove bounded terminal/missing head IDs regardless of endpoint gating."""
        self._page(limit)
        count = 0
        for _ in range(limit):
            if should_stop and should_stop():
                break
            attempt = self.ready_head(pool_id)
            if attempt is None or (should_stop and should_stop()):
                break
            result = self._change(
                'prune', owner=owner, pool_id=pool_id, attempt_id=attempt, cleanup=True
            )
            if result.disposition != 'tombstone':
                break
            count += 1
        return count

    def usage(self) -> dict[str, int]:
        """Read only bounded fabric accounting counters."""
        names = [
            'active_items',
            'payload_bytes',
            'records',
            'storage_bytes',
            'ready_ids',
            'reserved_items',
            'reserved_bytes',
        ]
        return {
            key: int(value or 0)
            for key, value in zip(
                names, self._read('hmget', self.keys.prefix + 'usage', *names)
            )
        }

    def close(self) -> None:
        """Close this client's sockets without changing ownership or requests."""
        self.client.close()
