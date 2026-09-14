"""Fenced terminal dispatch. Durable request state belongs exclusively to RequestStore."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable
from uuid import uuid4

from marie.engine.completion_contract import (
    CompletionCallParams,
    UnsupportedQueueStreaming,
)
from marie.engine.llm_queue.endpoint import (
    EndpointClient,
    ExecutionOutcome,
    RegisteredEndpoint,
)
from marie.engine.llm_queue.queue_keys import validate_identifier
from marie.engine.llm_queue.registry import (
    SnapshotReadBudget,
    register_dispatcher,
    unregister_dispatcher,
)
from marie.engine.llm_queue.scheduler import DrrLaneConfig, DrrLaneScheduler
from marie.engine.llm_queue.store import (
    OwnerToken,
    RequestStore,
    StaleOwner,
    StoreUnavailable,
    TransitionRejected,
)
from marie.instrumentation import get_tracer
from opentelemetry.trace import StatusCode

_tracer = get_tracer('marie.engine.llm_queue.request_dispatcher')


class _DispatchStopped(Exception):
    pass


@dataclass(frozen=True, slots=True)
class DispatchLane:
    pool_id: str
    endpoint_id: str
    revision: str = 'r1'
    enabled: bool = True
    execution_limit: int = 8
    execution_bytes: int = 64 * 1024 * 1024
    quantum: int = 1
    min_concurrent: int = 0
    max_burst_per_visit: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.execution_limit) is not int
            or type(self.execution_bytes) is not int
        ):
            raise ValueError('Invalid lane policy types')
        DrrLaneConfig(
            pool_id=self.pool_id,
            quantum=self.quantum,
            min_concurrent=self.min_concurrent,
            max_concurrent=self.execution_limit,
            max_burst_per_visit=self.max_burst_per_visit,
            enabled=self.enabled,
        )
        for value in (self.pool_id, self.endpoint_id, self.revision):
            validate_identifier(value)


class RequestDispatcher:
    """One persistent async loop, with finite synchronous store I/O in worker threads."""

    def __init__(
        self,
        *,
        store: RequestStore,
        endpoints: list[RegisteredEndpoint],
        lanes: list[DispatchLane],
        owner_lease_ms: int = 10_000,
        poll_seconds: float = 0.1,
        drain_seconds: float = 10.0,
        retry_min_ms: int = 4000,
        retry_max_ms: int = 30_000,
        circuit_open_ms: int = 30_000,
        client_factory: Callable[[RegisteredEndpoint], EndpointClient] = EndpointClient,
        total_concurrent_dispatch: int | None = None,
        policy: str = 'drr',
        policy_loader: Callable[[], dict[str, Any]] | None = None,
        refresh_seconds: float = 5.0,
        refresh_timeout_seconds: float = 2.0,
    ) -> None:
        self.store = store
        self.endpoints = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
        self.lanes = tuple(lanes)
        if len(self.endpoints) != len(endpoints) or len(
            {lane.pool_id for lane in lanes}
        ) != len(lanes):
            raise ValueError('Duplicate endpoint or pool identity')
        if any(lane.endpoint_id not in self.endpoints for lane in lanes):
            raise ValueError('Lane must reference a registered endpoint')
        if not lanes or not endpoints:
            raise ValueError('V3 requires registered endpoints and explicit lanes')
        if (
            len(lanes) > store.limits.max_routes
            or len(endpoints) > store.limits.max_endpoints
        ):
            raise ValueError('Registered routes exceed fabric bounds')
        for bound in (*lanes, *endpoints):
            if not (
                0 < bound.execution_limit <= store.limits.max_execution_items
                and 0 < bound.execution_bytes <= store.limits.max_execution_bytes
            ):
                raise ValueError('Execution policy exceeds fabric bounds')
        if (
            not 0 < retry_min_ms <= retry_max_ms <= store.limits.max_deadline_ms
            or not 0 < circuit_open_ms <= store.limits.max_deadline_ms
        ):
            raise ValueError('Invalid retry policy bounds')
        if (
            not 0 < owner_lease_ms <= store.limits.max_lease_ms
            or poll_seconds <= 0
            or drain_seconds < 0
        ):
            raise ValueError('Invalid dispatcher lifecycle bounds')
        if (
            policy not in {'drr', 'fifo'}
            or refresh_seconds <= 0
            or refresh_timeout_seconds <= 0
        ):
            raise ValueError('Invalid scheduling refresh policy')
        total = (
            store.limits.max_execution_items
            if total_concurrent_dispatch is None
            else total_concurrent_dispatch
        )
        if type(total) is not int or not 1 <= total <= store.limits.max_execution_items:
            raise ValueError('Invalid dispatch capacity')
        self.policy = policy
        self.policy_loader = policy_loader
        self.refresh_seconds = refresh_seconds
        self.refresh_timeout_seconds = refresh_timeout_seconds
        self._refresh_future = None
        self._refresh_started = 0.0
        self._next_refresh = 0.0
        self._refresh_failures = 0
        self._policy_paused = False
        self._routes_disabled = False
        self._config_revision = 1
        self._refresh_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='llm-v3-policy'
        )
        self.scheduler = DrrLaneScheduler(
            queue_client=None,
            lanes=[
                DrrLaneConfig(
                    pool_id=lane.pool_id,
                    quantum=lane.quantum if policy == 'drr' else 1_000_000,
                    min_concurrent=lane.min_concurrent,
                    max_concurrent=lane.execution_limit,
                    max_burst_per_visit=(
                        lane.max_burst_per_visit if policy == 'drr' else 1
                    ),
                    enabled=lane.enabled,
                )
                for lane in lanes
            ],
            total_concurrent_dispatch=total,
        )
        self.owner_lease_ms = owner_lease_ms
        self.poll_seconds = poll_seconds
        self.drain_seconds = drain_seconds
        self.retry_min_ms = retry_min_ms
        self.retry_max_ms = retry_max_ms
        self.circuit_open_ms = circuit_open_ms
        self.client_factory = client_factory
        self.dispatcher_id = uuid4().hex
        self.owner: OwnerToken | None = None
        self._owner_ready = False
        self._next_config_retry = 0.0
        self._owner_until = 0.0
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._calling: set[str] = set()
        self._clients: dict[str, EndpointClient] = {}
        self._stop = asyncio.Event()
        self._stop_workers = threading.Event()
        self._route_cursor = 0
        self._runner: asyncio.Task[None] | None = None
        self._heartbeat: asyncio.Task[None] | None = None
        self._draining = False
        self._counts: Counter[str] = Counter()
        self._category: str | None = None
        self._last_error_at_ms: int | None = None
        self.metadata_unavailable = 0
        self._processing_offset = 0
        self._delayed_offset = 0
        self._store_workers = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix='llm-v3-store'
        )
        self._lease_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='llm-v3-lease'
        )
        self._store_futures: set[asyncio.Future[Any]] = set()

    async def _io(self, method: str, *args: Any, **kwargs: Any) -> Any:
        executor = (
            self._lease_worker if method == 'renew_owner' else self._store_workers
        )
        future = asyncio.get_running_loop().run_in_executor(
            executor, partial(getattr(self.store, method), *args, **kwargs)
        )
        self._store_futures.add(future)

        def completed(future: asyncio.Future[Any]) -> None:
            self._store_futures.discard(future)
            if not future.cancelled():
                future.exception()

        future.add_done_callback(completed)
        return await asyncio.shield(future)

    async def _maintenance_io(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._stop.is_set():
            raise _DispatchStopped
        if method in {'prune_ready', 'expire_producer', 'promote_due'}:
            kwargs['should_stop'] = self._stop_workers.is_set
        return await self._io(method, *args, **kwargs)

    def health(
        self, *, read_budget: SnapshotReadBudget | None = None
    ) -> dict[str, Any]:
        from marie.engine.llm_queue.registry import (
            MAX_SNAPSHOT_ROWS,
            SnapshotUnavailable,
        )

        if len(self.lanes) + len(self.endpoints) > MAX_SNAPSHOT_ROWS:
            raise SnapshotUnavailable('snapshot_budget_exceeded')
        read_budget = read_budget or SnapshotReadBudget(0, time.monotonic() + 1.0)
        read_budget.before_read()
        usage = self.store.usage()
        endpoint_states = {}
        for endpoint_id in dict.fromkeys(lane.endpoint_id for lane in self.lanes):
            read_budget.before_read()
            endpoint_states[endpoint_id] = self.store.endpoint_status(endpoint_id)
        lane_rows = []
        for lane in self.lanes:
            head, _ = self.store.ready_metadata(
                lane.pool_id, limit=1, before_read=read_budget.before_read
            )
            read_budget.before_read()
            depth = self.store.ready_depth(lane.pool_id)
            read_budget.before_read()
            route = self.store.route_status(lane.pool_id)
            fairness = self.scheduler.lane_metadata(lane.pool_id)
            fairness.update(
                quantum=lane.quantum,
                min_concurrent=lane.min_concurrent,
                max_burst_per_visit=lane.max_burst_per_visit,
                inflight=route['reserved_items'],
            )
            endpoint = endpoint_states[lane.endpoint_id]
            waiting = (
                'disabled'
                if not lane.enabled or not self.endpoints[lane.endpoint_id].enabled
                else endpoint.get('waiting_reason')
                or (
                    'empty'
                    if not head
                    else (
                        'lane_capacity'
                        if route['reserved_items'] >= lane.execution_limit
                        else (
                            'global_capacity'
                            if usage['reserved_items']
                            >= self.scheduler.total_concurrent_dispatch
                            else (
                                'insufficient_credit'
                                if self.owner is not None
                                and head[0].cost > fairness.get('deficit', 0)
                                else None
                            )
                        )
                    )
                )
            )
            lane_rows.append(
                dict(
                    pool_id=lane.pool_id,
                    endpoint_id=lane.endpoint_id,
                    enabled=lane.enabled and self.endpoints[lane.endpoint_id].enabled,
                    max_concurrent=lane.execution_limit,
                    **fairness,
                    reserved_items=route['reserved_items'],
                    reserved_bytes=route['reserved_bytes'],
                    protected_slots=min(lane.min_concurrent, route['reserved_items']),
                    borrowed_slots=max(
                        0, route['reserved_items'] - lane.min_concurrent
                    ),
                    waiting_reason=waiting,
                    request_queue_depth=depth,
                    head_cost_units=head[0].cost if head else None,
                    oldest_pending_admitted_at_ms=(
                        head[0].admitted_at_ms if head else None
                    ),
                )
            )
        read_budget.check()
        return dict(
            contract_version='v3',
            metadata_unavailable=self.metadata_unavailable,
            fabric_group_id=self.store.keys.fabric_id,
            dispatcher_id=self.dispatcher_id,
            running=self._runner is not None and not self._runner.done(),
            draining=self._draining,
            owner_generation=self.owner.generation if self.owner else None,
            role='owner' if self.owner else 'standby',
            pool_ids=[lane.pool_id for lane in self.lanes],
            endpoint_ids=list(self.endpoints),
            inflight_request_count=len(self._tasks),
            counters=dict(self._counts),
            execution_failures=self._counts['execution_errors'],
            usage=usage,
            endpoints=endpoint_states,
            lanes=lane_rows,
            last_error=self._category,
            last_error_at_ms=self._last_error_at_ms,
            enabled=True,
            scheduler_policy=self.policy,
            total_concurrent_dispatch=self.scheduler.total_concurrent_dispatch,
            config_generation=self._config_revision,
            waiting_reason=(
                'policy_refresh_unavailable' if self._policy_paused else None
            ),
        )

    def sample_pending_requests(
        self, limit: int, *, read_budget: SnapshotReadBudget | None = None
    ) -> list[dict[str, Any]]:
        read_budget = read_budget or SnapshotReadBudget(limit, time.monotonic() + 1.0)
        rows = []
        malformed = 0
        for lane in self.lanes:
            remaining = min(
                limit - len(rows) - malformed,
                read_budget.candidates_left,
                self.store.limits.cleanup_page_size,
            )
            if remaining <= 0:
                break
            records, invalid = self.store.ready_metadata(
                lane.pool_id,
                limit=remaining,
                before_read=read_budget.before_read,
                before_candidate=read_budget.before_candidate,
            )
            malformed += invalid
            rows.extend(
                dict(
                    request_id=record.attempt_id,
                    attempt_id=record.attempt_id,
                    pool_id=record.pool_id,
                    endpoint_id=record.endpoint_id,
                    fabric_group_id=self.store.keys.fabric_id,
                    contract_version='v3',
                    config_revision=record.config_revision,
                    lifecycle_stage=record.state,
                    state_source='store',
                    payload_bytes=record.payload_bytes,
                    cost=record.cost,
                    expires_at_ms=record.expires_at_ms,
                    execution_seq=record.execution_seq,
                    next_eligible=record.next_eligible,
                    last_error=record.last_error,
                    model=record.model,
                    admitted_at_ms=record.admitted_at_ms,
                )
                for record in records
            )
        self.metadata_unavailable = malformed
        return rows

    def inflight_requests_snapshot(self) -> list[dict[str, Any]]:
        return [
            dict(
                request_id=attempt,
                fabric_group_id=self.store.keys.fabric_id,
                contract_version='v3',
                lifecycle_stage='dispatching',
            )
            for attempt in self._tasks
        ]

    async def start(self) -> None:
        if self._runner is not None:
            return
        self._clients = {
            identity: self.client_factory(endpoint)
            for identity, endpoint in self.endpoints.items()
            if endpoint.enabled
            and any(
                lane.enabled and lane.endpoint_id == identity for lane in self.lanes
            )
        }
        self._runner = asyncio.create_task(self._run())
        register_dispatcher(self.dispatcher_id, self)

    async def stop(self) -> None:
        if self._runner is None:
            return
        self._draining = True
        self._stop_workers.set()
        self._stop.set()
        await asyncio.shield(self._runner)

    async def _pause(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _renew(self) -> None:
        while not self._draining or self._tasks:
            await asyncio.sleep(self.owner_lease_ms / 3000)
            owner = self.owner
            if owner is None:
                return
            try:
                began = time.monotonic()
                await self._io('renew_owner', owner, lease_ms=self.owner_lease_ms)
                self._owner_until = began + self.owner_lease_ms / 1000
            except (StaleOwner, StoreUnavailable):
                self._category = 'owner_uncertain'
                self._owner_until = 0
                for task in self._tasks.values():
                    task.cancel()
                return

    async def _configure(self) -> None:
        cursor = 0
        while True:
            cursor, pools = await self._maintenance_io(
                'scan_routes', cursor=cursor, limit=self.store.limits.cleanup_page_size
            )
            for pool_id in pools:
                await self._maintenance_io('disable_route', self.owner, pool_id)
            if cursor == 0:
                break
        for endpoint in self.endpoints.values():
            result = await self._maintenance_io(
                'configure_endpoint',
                self.owner,
                endpoint.endpoint_id,
                execution_limit=endpoint.execution_limit,
                execution_bytes=endpoint.execution_bytes,
                gate_open=endpoint.enabled,
                transport_fingerprint=endpoint.transport_fingerprint(),
                target_fingerprint=endpoint.target_fingerprint(),
            )
            if result.disposition != 'configured':
                raise TransitionRejected(result.disposition)
        for lane in self.lanes:
            result = await self._maintenance_io(
                'configure_route',
                self.owner,
                lane.pool_id,
                lane.endpoint_id,
                revision=lane.revision,
                execution_limit=lane.execution_limit,
                execution_bytes=lane.execution_bytes,
                enabled=lane.enabled and self.endpoints[lane.endpoint_id].enabled,
            )
            if result.disposition != 'configured':
                raise TransitionRejected(result.disposition)
        for endpoint in self.endpoints.values():
            result = await self._maintenance_io(
                'configure_endpoint',
                self.owner,
                endpoint.endpoint_id,
                execution_limit=endpoint.execution_limit,
                execution_bytes=endpoint.execution_bytes,
                gate_open=endpoint.enabled,
                transport_fingerprint=endpoint.transport_fingerprint(),
                target_fingerprint=endpoint.target_fingerprint(),
            )
            if result.disposition != 'configured':
                raise TransitionRejected(result.disposition)

    async def _refresh_policy(self) -> None:
        if self.policy_loader is None:
            return
        now = time.monotonic()
        if self._refresh_future is None and now >= self._next_refresh:
            self._refresh_started = now
            self._refresh_future = self._refresh_worker.submit(self.policy_loader)
        future = self._refresh_future
        if future is not None and future.done():
            self._refresh_future = None
            try:
                policy = future.result()
                if policy['limits'] != self.store.limits:
                    raise ValueError('Immutable fabric limits changed')
                endpoints = {entry.endpoint_id: entry for entry in policy['endpoints']}
                lanes = tuple(policy['lanes'])
                for identity, endpoint in endpoints.items():
                    previous = self.endpoints.get(identity)
                    if (
                        previous
                        and previous.transport_fingerprint()
                        != endpoint.transport_fingerprint()
                    ):
                        raise ValueError('Endpoint identity changed')
                scheduler = DrrLaneScheduler(
                    queue_client=None,
                    total_concurrent_dispatch=policy['total_concurrent_dispatch'],
                    lanes=[
                        DrrLaneConfig(
                            pool_id=lane.pool_id,
                            quantum=(
                                lane.quantum if policy['policy'] == 'drr' else 1_000_000
                            ),
                            min_concurrent=lane.min_concurrent,
                            max_concurrent=lane.execution_limit,
                            max_burst_per_visit=(
                                lane.max_burst_per_visit
                                if policy['policy'] == 'drr'
                                else 1
                            ),
                            enabled=lane.enabled,
                        )
                        for lane in lanes
                    ],
                )
                changed = (
                    lanes != self.lanes
                    or endpoints != self.endpoints
                    or policy['policy'] != self.policy
                    or scheduler.total_concurrent_dispatch
                    != self.scheduler.total_concurrent_dispatch
                )
                if changed or self._policy_paused:
                    old_lanes, old_endpoints = self.lanes, self.endpoints
                    self.lanes, self.endpoints = lanes, endpoints
                    self._routes_disabled = False
                    try:
                        await self._configure()
                    except BaseException:
                        self.lanes, self.endpoints = old_lanes, old_endpoints
                        raise
                    for identity, endpoint in endpoints.items():
                        if endpoint.enabled and identity not in self._clients:
                            self._clients[identity] = self.client_factory(endpoint)
                    self.scheduler = scheduler
                    self.policy = policy['policy']
                    self._config_revision += 1
                    self._counts['config_refreshes'] += 1
                self._policy_paused = False
                self._routes_disabled = False
                self._refresh_failures = 0
                self._next_refresh = now + self.refresh_seconds
            except Exception:
                self._policy_paused = True
                self._refresh_failures += 1
                self._counts['config_refresh_errors'] += 1
                self._category = 'policy_refresh_unavailable'
                self._next_refresh = now + min(
                    30.0, self.refresh_seconds * 2 ** min(self._refresh_failures, 3)
                )
        elif (
            future is not None
            and now - self._refresh_started >= self.refresh_timeout_seconds
        ):
            self._policy_paused = True
            self._category = 'policy_refresh_unavailable'
        if self._policy_paused and not self._routes_disabled:
            cursor = 0
            while True:
                cursor, pools = await self._maintenance_io(
                    'scan_routes',
                    cursor=cursor,
                    limit=self.store.limits.cleanup_page_size,
                )
                for pool in pools:
                    await self._maintenance_io('disable_route', self.owner, pool)
                if cursor == 0:
                    break
            self._routes_disabled = True

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    if self.owner is None:
                        began = time.monotonic()
                        self.owner = await self._maintenance_io(
                            'acquire_owner',
                            self.dispatcher_id,
                            lease_ms=self.owner_lease_ms,
                        )
                        if self.owner is None:
                            await self._pause(self.poll_seconds)
                            continue
                        self._owner_until = began + self.owner_lease_ms / 1000
                        self._heartbeat = asyncio.create_task(self._renew())
                        self._owner_ready = False
                    if not self._owner_ready:
                        if time.monotonic() < self._next_config_retry:
                            await self._maintenance()
                            await self._pause(self.poll_seconds)
                            continue
                        try:
                            await self._configure()
                        except TransitionRejected as exc:
                            self._category = str(exc)
                            self._counts['config_refresh_errors'] += 1
                            self._next_config_retry = (
                                time.monotonic() + self.refresh_seconds
                            )
                            await self._maintenance()
                            await self._pause(self.poll_seconds)
                            continue
                        # Snapshot bounded IDs before recovery removes unsent claims from the index.
                        previous: list[str] = []
                        while not self._stop.is_set():
                            ids = await self._maintenance_io(
                                'processing_ids',
                                offset=len(previous),
                                limit=self.store.limits.cleanup_page_size,
                            )
                            if not ids:
                                break
                            previous.extend(ids)
                        for attempt in previous:
                            if self._stop.is_set():
                                break
                            recovered = await self._maintenance_io(
                                'recover_claim', self.owner, attempt
                            )
                            if recovered.disposition == 'requeued':
                                self._counts['live_claims_recovered'] += 1
                        self._owner_ready = True
                    if time.monotonic() >= self._owner_until:
                        raise StaleOwner('owner expired locally')
                    await self._refresh_policy()
                    await self._maintenance()
                    if not self._policy_paused:
                        await self._dispatch()
                    await self._pause(self.poll_seconds)
                except _DispatchStopped:
                    break
                except StoreUnavailable:
                    self._counts['store_errors'] += 1
                    self._category = 'store_unavailable'
                    await self._pause(random.uniform(0.1, 0.5))
                except (StaleOwner, TransitionRejected):
                    self._category = 'owner_lost'
                    for task in self._tasks.values():
                        task.cancel()
                    if self._tasks:
                        await asyncio.gather(
                            *tuple(self._tasks.values()), return_exceptions=True
                        )
                    if self._heartbeat:
                        self._heartbeat.cancel()
                        await asyncio.gather(self._heartbeat, return_exceptions=True)
                    self.owner = None
                    self._owner_ready = False
                    await self._pause(self.owner_lease_ms / 1000)
                except Exception:
                    self._category = 'dispatch_error'
                    self._counts['dispatch_errors'] += 1
                    await self._pause(0.5)
        finally:
            self._draining = True
            if self._tasks:
                _, pending = await asyncio.wait(
                    tuple(self._tasks.values()), timeout=self.drain_seconds
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            if self._heartbeat:
                self._heartbeat.cancel()
                await asyncio.gather(self._heartbeat, return_exceptions=True)
            if self._store_futures:
                await asyncio.gather(
                    *tuple(self._store_futures), return_exceptions=True
                )
            for client in self._clients.values():
                await client.close()
            self._clients.clear()
            if self.owner:
                try:
                    await self._io('release_owner', self.owner)
                except (StaleOwner, StoreUnavailable):
                    pass
            self.owner = None
            unregister_dispatcher(self.dispatcher_id)
            await self._io('close')
            self._store_workers.shutdown(wait=True)
            self._lease_worker.shutdown(wait=True)
            self._refresh_worker.shutdown(wait=False, cancel_futures=True)

    async def _maintenance(self) -> None:
        page = self.store.limits.cleanup_page_size
        for attempt, task in tuple(self._tasks.items()):
            metadata = await self._maintenance_io('metadata', attempt)
            if attempt in self._calling and (
                metadata is None
                or metadata.state in {'cancelled', 'expired', 'abandoned'}
            ):
                task.cancel()
        for producer in await self._maintenance_io('due_ids', 'producers', limit=page):
            expired = await self._maintenance_io(
                'expire_producer', producer, limit=page
            )
            self._counts['producer_expired_requests'] += sum(
                reply.disposition == 'discarded' for reply in expired
            )
        for attempt in await self._maintenance_io('due_ids', 'deadlines', limit=page):
            metadata = await self._maintenance_io('metadata', attempt)
            if metadata:
                await self._maintenance_io(
                    'cancel_or_expire', metadata.producer_id, attempt, expire=True
                )
        for attempt in await self._maintenance_io('due_ids', 'retention', limit=page):
            await self._maintenance_io('purge_terminal', self.owner, attempt)
        ids = await self._maintenance_io(
            'processing_ids', offset=self._processing_offset, limit=page
        )
        self._processing_offset = (
            self._processing_offset + len(ids) if len(ids) == page else 0
        )
        for attempt in ids:
            if attempt not in self._tasks:
                recovered = await self._maintenance_io(
                    'recover_claim', self.owner, attempt
                )
                if recovered.disposition == 'requeued':
                    self._counts['live_claims_recovered'] += 1
        promoted = await self._maintenance_io(
            'promote_due', self.owner, limit=page, offset=self._delayed_offset
        )
        self._delayed_offset = (
            self._delayed_offset + len(promoted) if len(promoted) == page else 0
        )
        self._route_cursor, pools = await self._maintenance_io(
            'scan_routes', cursor=self._route_cursor, limit=page
        )
        for pool_id in pools:
            await self._maintenance_io('prune_ready', self.owner, pool_id, limit=page)

    async def _dispatch(self) -> None:
        heads = {}
        reserved = {}
        now = await self._maintenance_io('server_time_ms')
        usage = await self._maintenance_io('usage')
        for lane in self.lanes:
            route = await self._maintenance_io('route_status', lane.pool_id)
            reserved[lane.pool_id] = route['reserved_items']
            if not lane.enabled or not self.endpoints[lane.endpoint_id].enabled:
                continue
            endpoint = await self._maintenance_io('endpoint_status', lane.endpoint_id)
            if endpoint['probe_claim'] or (
                endpoint['circuit'] not in (None, 'closed')
                and int(endpoint['next_probe'] or 0) > now
            ):
                continue
            attempt = await self._maintenance_io('ready_head', lane.pool_id)
            if attempt:
                try:
                    head = await self._maintenance_io('metadata', attempt)
                except (ValueError, TypeError):
                    self._counts['invalid_head_metadata'] += 1
                    continue
                if (
                    head is not None
                    and head.state == 'ready'
                    and head.endpoint_id == lane.endpoint_id
                    and head.config_revision == lane.revision
                    and route['enabled'] == '1'
                    and 1 <= head.cost <= 1_000_000
                    and int(endpoint['reserved_items'] or 0)
                    < self.endpoints[lane.endpoint_id].execution_limit
                    and int(endpoint['reserved_bytes'] or 0) + head.payload_bytes
                    <= self.endpoints[lane.endpoint_id].execution_bytes
                    and route['reserved_bytes'] + head.payload_bytes
                    <= lane.execution_bytes
                ):
                    heads[lane.pool_id] = head
        self.scheduler.sync_reservations(reserved, usage['reserved_items'], set(heads))
        for _ in range(self.store.limits.max_execution_items):
            if (
                self._stop.is_set()
                or len(self._tasks) >= self.store.limits.max_execution_items
            ):
                return
            pool = self.scheduler.select_metadata(
                {pool: head.cost for pool, head in heads.items()}
            )
            if pool is None:
                return
            head = heads[pool]
            claim_id = uuid4().hex
            reply = await self._maintenance_io(
                'claim',
                self.owner,
                head.attempt_id,
                pool_id=pool,
                claim_id=claim_id,
                expected_cost=head.cost,
            )
            if reply.disposition != 'claimed':
                self.scheduler.rejected(pool, reply.disposition)
                heads.pop(pool)
                continue
            self.scheduler.claimed(pool, reply.cost)
            self._counts['claims'] += 1
            # Track the selected claim before any subsequent fallible store inspection.
            task = asyncio.create_task(
                self._execute(reply.attempt_id, claim_id, head.endpoint_id)
            )
            self._tasks[reply.attempt_id] = task
            task.add_done_callback(partial(self._completed_task, reply.attempt_id))
            attempt = await self._maintenance_io('ready_head', pool)
            try:
                next_head = (
                    await self._maintenance_io('metadata', attempt) if attempt else None
                )
            except (ValueError, TypeError):
                self._counts['invalid_head_metadata'] += 1
                next_head = None
            if (
                next_head is not None
                and next_head.state == 'ready'
                and next_head.endpoint_id == head.endpoint_id
                and next_head.config_revision == head.config_revision
                and 1 <= next_head.cost <= 1_000_000
            ):
                heads[pool] = next_head
            else:
                heads.pop(pool, None)

    def _completed_task(self, attempt: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(attempt, None)
        if not task.cancelled() and task.exception() is not None:
            self._category = 'dispatch_error'
            self._counts['dispatch_errors'] += 1

    async def _provider(
        self, attempt: str, claim_id: str, endpoint_id: str
    ) -> tuple[int, ExecutionOutcome]:
        if self._stop.is_set():
            raise TransitionRejected('start_stopped')
        payload = call = None
        self._calling.add(attempt)
        try:
            metadata = await self._io('metadata', attempt)
            payload = await self._io(
                'fetch_payload', self.owner, attempt, claim_id=claim_id
            )
            call = CompletionCallParams.from_dict(payload['call'])
            clock_read_at = time.monotonic()
            now = await self._io('server_time_ms')
            timeout = min(
                (payload['expires_at_ms'] - now) / 1000,
                self.endpoints[endpoint_id].call_timeout_seconds,
            )
            from marie.engine.completion_contract import require_terminal_completion

            require_terminal_completion(call)
            if (
                self._stop.is_set()
                or time.monotonic() >= self._owner_until
                or timeout <= 0
            ):
                raise TransitionRejected('start_stopped')
            endpoint_state = await self._io('endpoint_status', endpoint_id)
            started = await self._io(
                'authorize_start', self.owner, attempt, claim_id=claim_id
            )
            if started.disposition != 'started':
                raise TransitionRejected('start_unconfirmed')
            timeout -= time.monotonic() - clock_read_at
            if (
                timeout <= 0
                or self._stop.is_set()
                or time.monotonic() >= self._owner_until
            ):
                raise TransitionRejected('start_stopped')
            self._counts['provider_starts'] += 1
            if endpoint_state.get('probe_claim') == claim_id:
                self._counts['probe_starts'] += 1
            attributes = {
                'marie.llm_dispatch.request_id': attempt,
                'marie.llm_dispatch.claim_id': claim_id,
                'marie.llm_dispatch.execution_seq': started.execution_seq,
                'marie.llm_dispatch.fabric_group_id': self.store.keys.fabric_id,
                'marie.llm_dispatch.pool_id': metadata.pool_id,
                'marie.llm_dispatch.endpoint_id': endpoint_id,
                'marie.llm_dispatch.dispatcher_id': self.dispatcher_id,
                'marie.llm_dispatch.contract_version': 'v3',
                'marie.llm_dispatch.model': (
                    metadata.model
                    if metadata.model
                    and not metadata.model.lower().startswith(
                        ('http:', 'https:', 'data:')
                    )
                    else ''
                ),
                'marie.llm_dispatch.message_count': len(call.messages),
                'marie.llm_dispatch.queue_wait_ms': max(
                    0, now - metadata.admitted_at_ms
                ),
            }
            began_ns = time.time_ns()
            began = time.monotonic()
            outcome = await self._clients[endpoint_id].execute(
                call, timeout_seconds=timeout
            )
            payload = call = None
            _emit_execution_history(attributes, outcome, began_ns, began)
            return started.execution_seq, outcome
        finally:
            payload = call = None
            self._calling.discard(attempt)

    async def _commit(self, method: str, *args: Any, **kwargs: Any) -> Any:
        while True:
            try:
                return await self._io(method, self.owner, *args, **kwargs)
            except StoreUnavailable:
                self._counts['store_errors'] += 1
            if time.monotonic() >= self._owner_until:
                raise StaleOwner('owner uncertain')
            await asyncio.sleep(random.uniform(0.1, 0.5))

    async def _execute(self, attempt: str, claim_id: str, endpoint_id: str) -> None:
        outcome = None
        try:
            try:
                sequence, outcome = await self._provider(attempt, claim_id, endpoint_id)
            except UnsupportedQueueStreaming:
                await self._commit(
                    'reject_claim',
                    attempt,
                    claim_id=claim_id,
                    category='unsupported_streaming',
                )
                return
            except (StoreUnavailable, TransitionRejected):
                self._category = 'start_unconfirmed'
                return
            args = dict(claim_id=claim_id, execution_seq=sequence)
            if outcome.category:
                self._category = outcome.category
                self._last_error_at_ms = int(time.time() * 1000)
                self._counts['execution_errors'] += 1
            feedback = (
                'success'
                if outcome.availability_success
                else 'unavailable'
                if outcome.availability_failure
                else 'neutral'
            )
            await self._commit(
                'record_endpoint_outcome',
                attempt,
                **args,
                outcome=feedback,
                category=outcome.category or 'none',
                open_ms=self.circuit_open_ms,
            )
            if not outcome.remote_settled:
                await self._commit(
                    'mark_unknown', attempt, **args, category=outcome.category
                )
                self._counts['outcome_unknown'] += 1
                return
            if outcome.retryable and sequence < self.store.limits.max_attempts:
                delay = max(
                    outcome.retry_after_ms,
                    random.randint(
                        self.retry_min_ms,
                        min(self.retry_max_ms, self.retry_min_ms * 2 ** (sequence - 1)),
                    ),
                )
                reply = await self._commit(
                    'defer',
                    attempt,
                    **args,
                    delay_ms=delay,
                    reason=outcome.category,
                    remote_settled=True,
                )
                if reply.disposition == 'deferred':
                    self._counts['retries'] += 1
                    return
            result = (
                outcome.response
                if outcome.response is not None
                else {'error': outcome.category}
            )
            try:
                reply = await self._commit(
                    'finish',
                    attempt,
                    **args,
                    result=result,
                    success=outcome.response is not None,
                )
            except (ValueError, TypeError, OverflowError, RecursionError):
                outcome.response = None
                outcome.category = self._category = 'invalid_response'
                result = {'error': 'invalid_response'}
                reply = None
            # Leave the exception/serialization frame before retrying the bounded result.
            if reply is None:
                reply = await self._commit(
                    'finish', attempt, **args, result=result, success=False
                )
            # Cancellation/deadline/producer death can win while HTTP completes.
            if reply.disposition in {
                'existing',
                'producer_dead',
                'expired',
                'terminal',
            }:
                await self._commit(
                    'settle_remote', attempt, **args, evidence='remote_completed'
                )
            self._counts['completed'] += 1
        except StaleOwner:
            self._owner_until = 0
        finally:
            outcome = None


def _emit_execution_history(
    attributes: dict[str, Any], outcome: ExecutionOutcome, began_ns: int, began: float
) -> None:
    """Emit one physical attempt after transport returns, before result-commit retries."""
    elapsed = max(0.0, (time.monotonic() - began) * 1000)
    prefix = 'marie.llm_dispatch.'
    attributes[prefix + 'execution_ms'] = elapsed
    attributes[prefix + 'total_latency_ms'] = (
        attributes[prefix + 'queue_wait_ms'] + elapsed
    )
    attributes[prefix + 'status'] = 'ok' if outcome.response is not None else 'error'
    if outcome.category:
        attributes[prefix + 'error_type'] = outcome.category
    if outcome.response is not None:
        usage = outcome.response.get('usage')
        if isinstance(usage, dict):
            for source, target in (
                ('prompt_tokens', 'prompt'),
                ('completion_tokens', 'completion'),
                ('total_tokens', 'total'),
            ):
                count = usage.get(source)
                if type(count) is int and 0 <= count <= 2**53 - 1:
                    attributes['llm.token_count.' + target] = count
    # No active span context encloses payloads, exceptions, or commit retries.
    span = _tracer.start_span(
        'LLMDispatch.completion', start_time=began_ns, attributes=attributes
    )
    span.set_status(StatusCode.OK if outcome.response is not None else StatusCode.ERROR)
    span.end()
