"""Fenced terminal dispatch. Durable request state belongs exclusively to RequestStore."""

from __future__ import annotations

import asyncio
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable
from urllib.parse import urlsplit
from uuid import uuid4

from marie.engine.completion_contract import (
    CompletionCallParams,
    UnsupportedQueueStreaming,
    apply_repetition_recovery,
    completion_finish_reason,
    extract_completion_text,
    has_terminal_repetition,
    recover_complete_json,
)
from marie.engine.completion_observer import (
    CompletionAttempt,
    _raw_completion_text,
    _response_value,
    _token_count,
)
from marie.engine.llm_queue.endpoint import (
    EndpointClient,
    ExecutionOutcome,
    RegisteredEndpoint,
    RegisteredEndpointGroup,
    RegisteredReplica,
)
from marie.engine.llm_queue.metrics import dispatch_metrics
from marie.engine.llm_queue.queue_keys import validate_identifier
from marie.engine.llm_queue.registry import (
    SnapshotReadBudget,
    register_dispatcher,
    unregister_dispatcher,
)
from marie.engine.llm_queue.scheduler import DrrLaneConfig, DrrLaneScheduler
from marie.engine.llm_queue.store import (
    ClaimRecord,
    OwnerToken,
    RequestMetadata,
    RequestStore,
    StaleOwner,
    StoreUnavailable,
    TransitionRejected,
)
from marie.instrumentation import get_tracer, set_llm_io
from marie.instrumentation.context import (
    extract_trace_context,
    start_trace_span,
    use_trace_context,
)
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, StatusCode

_tracer = get_tracer("marie.engine.llm_queue.request_dispatcher")


# Records that still reserve capacity although no local work will settle them.
_HELD_STATES = frozenset({"abandoned", "failed", "cancelled", "expired"})


def _endpoint_port(base_url: str) -> int:
    parsed = urlsplit(base_url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


class _DispatchStopped(Exception):
    pass


class _NoReplicaAvailable(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _InflightClaim:
    claim: ClaimRecord
    metadata: RequestMetadata
    popped_at: float


@dataclass(frozen=True, slots=True)
class DispatchLane:
    pool_id: str
    endpoint_id: str | None = None
    revision: str = "r1"
    enabled: bool = True
    execution_limit: int = 8
    execution_bytes: int = 64 * 1024 * 1024
    quantum: int = 1
    min_concurrent: int = 0
    max_burst_per_visit: int | None = None
    endpoint_group_id: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.execution_limit) is not int
            or type(self.execution_bytes) is not int
        ):
            raise ValueError("Invalid lane policy types")
        DrrLaneConfig(
            pool_id=self.pool_id,
            quantum=self.quantum,
            min_concurrent=self.min_concurrent,
            max_concurrent=self.execution_limit,
            max_burst_per_visit=self.max_burst_per_visit,
            enabled=self.enabled,
        )
        if (
            self.endpoint_id is not None
            and self.endpoint_group_id is not None
            and self.endpoint_id != self.endpoint_group_id
        ):
            raise ValueError("Lane endpoint aliases disagree")
        endpoint_group_id = self.endpoint_group_id or self.endpoint_id
        if endpoint_group_id is None:
            raise ValueError("Lane endpoint group is required")
        object.__setattr__(self, "endpoint_id", endpoint_group_id)
        object.__setattr__(self, "endpoint_group_id", endpoint_group_id)
        for value in (self.pool_id, endpoint_group_id, self.revision):
            validate_identifier(value)


class RequestDispatcher:
    """One persistent async loop, with finite synchronous store I/O in worker threads."""

    def __init__(
        self,
        *,
        store: RequestStore,
        endpoints: list[RegisteredEndpoint] | None = None,
        endpoint_groups: list[RegisteredEndpointGroup] | None = None,
        lanes: list[DispatchLane],
        owner_lease_ms: int = 10_000,
        poll_seconds: float = 0.1,
        drain_seconds: float = 10.0,
        retry_min_ms: int = 4000,
        retry_max_ms: int = 30_000,
        circuit_open_ms: int = 30_000,
        client_factory: Callable[[RegisteredEndpoint], EndpointClient] = EndpointClient,
        total_concurrent_dispatch: int | None = None,
        policy: str = "drr",
        policy_loader: Callable[[], dict[str, Any]] | None = None,
        refresh_seconds: float = 5.0,
        refresh_timeout_seconds: float = 2.0,
        policy_generation: int | None = None,
        policy_digest: str | None = None,
    ) -> None:
        self.store = store
        if endpoints and endpoint_groups:
            raise ValueError("Use endpoint groups or legacy endpoints, not both")
        if endpoint_groups is None:
            endpoint_groups = [
                RegisteredEndpointGroup(
                    group_id=endpoint.endpoint_id,
                    revision="r1",
                    replicas=(
                        RegisteredReplica(
                            replica_id=endpoint.endpoint_id,
                            base_url=endpoint.base_url,
                            api_key=endpoint.api_key,
                            allow_private=endpoint.allow_private,
                            allow_loopback=endpoint.allow_loopback,
                            execution_limit=endpoint.execution_limit,
                            execution_bytes=endpoint.execution_bytes,
                            call_timeout_seconds=endpoint.call_timeout_seconds,
                            max_response_bytes=endpoint.max_response_bytes,
                            retry_429=endpoint.retry_429,
                            enabled=endpoint.enabled,
                        ),
                    ),
                )
                for endpoint in endpoints or []
            ]
        self.endpoint_groups = {group.group_id: group for group in endpoint_groups}
        replicas = [replica for group in endpoint_groups for replica in group.replicas]
        self.replicas = {replica.replica_id: replica for replica in replicas}
        self.endpoints = {
            replica.replica_id: replica.as_endpoint() for replica in replicas
        }
        self.lanes = tuple(lanes)
        if (
            len(self.endpoint_groups) != len(endpoint_groups)
            or len(self.replicas) != len(replicas)
            or len({lane.pool_id for lane in lanes}) != len(lanes)
        ):
            raise ValueError("Duplicate endpoint or pool identity")
        if any(lane.endpoint_group_id not in self.endpoint_groups for lane in lanes):
            raise ValueError("Lane must reference a registered endpoint group")
        if not lanes or not endpoint_groups:
            raise ValueError(
                "Queued dispatch requires registered endpoint groups and explicit lanes"
            )
        if (
            len(lanes) > store.limits.max_routes
            or len(replicas) > store.limits.max_endpoints
        ):
            raise ValueError("Registered routes exceed fabric bounds")
        for bound in (*lanes, *replicas):
            if not (
                0 < bound.execution_limit <= store.limits.max_execution_items
                and 0 < bound.execution_bytes <= store.limits.max_execution_bytes
            ):
                raise ValueError("Execution policy exceeds fabric bounds")
        if (
            not 0 < retry_min_ms <= retry_max_ms <= store.limits.max_deadline_ms
            or not 0 < circuit_open_ms <= store.limits.max_deadline_ms
        ):
            raise ValueError("Invalid retry policy bounds")
        if (
            not 0 < owner_lease_ms <= store.limits.max_lease_ms
            or poll_seconds <= 0
            or drain_seconds < 0
        ):
            raise ValueError("Invalid dispatcher lifecycle bounds")
        if (
            policy not in {"drr", "fifo"}
            or refresh_seconds <= 0
            or refresh_timeout_seconds <= 0
        ):
            raise ValueError("Invalid scheduling refresh policy")
        if (policy_generation is None) != (policy_digest is None) or (
            policy_generation is not None
            and (
                type(policy_generation) is not int
                or not 1 <= policy_generation <= 2**53 - 1
                or not isinstance(policy_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", policy_digest) is None
            )
        ):
            raise ValueError("Invalid observed policy identity")
        total = (
            store.limits.max_execution_items
            if total_concurrent_dispatch is None
            else total_concurrent_dispatch
        )
        if type(total) is not int or not 1 <= total <= store.limits.max_execution_items:
            raise ValueError("Invalid dispatch capacity")
        self.policy = policy
        self.policy_loader = policy_loader
        self.refresh_seconds = refresh_seconds
        self.refresh_timeout_seconds = refresh_timeout_seconds
        self._refresh_future = None
        self._refresh_started = 0.0
        self._next_refresh = 0.0
        self._refresh_failures = 0
        self._refresh_timed_out = False
        self._policy_paused = False
        self._routes_disabled = False
        self._config_revision = 1
        self.policy_generation = policy_generation
        self.policy_digest = policy_digest
        self._refresh_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm-dispatch-policy"
        )
        self.scheduler = DrrLaneScheduler(
            lanes=[
                DrrLaneConfig(
                    pool_id=lane.pool_id,
                    quantum=lane.quantum if policy == "drr" else 1_000_000,
                    min_concurrent=lane.min_concurrent,
                    max_concurrent=lane.execution_limit,
                    max_burst_per_visit=(
                        lane.max_burst_per_visit if policy == "drr" else 1
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
        self._inflight_claims: dict[str, _InflightClaim] = {}
        self._calling: set[str] = set()
        self._clients: dict[str, EndpointClient] = {}
        self._client_endpoints: dict[str, RegisteredEndpoint] = {}
        self._client_users: Counter[str] = Counter()
        self._replica_cursors: Counter[str] = Counter()
        self._stop = asyncio.Event()
        self._dispatch_wakeup: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
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
            max_workers=4, thread_name_prefix="llm-dispatch-store"
        )
        self._lease_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm-dispatch-lease"
        )
        self._store_futures: set[asyncio.Future[Any]] = set()

    async def select_replica(
        self, claim: ClaimRecord, *, excluded: set[str] | None = None
    ) -> RegisteredReplica:
        """Select one currently eligible physical replica within the durable group."""
        group = self.endpoint_groups.get(claim.endpoint_group_id)
        if group is None:
            raise TransitionRejected("endpoint_group_unavailable")
        metadata = await self._maintenance_io("metadata", claim.attempt_id)
        if metadata is None:
            raise TransitionRejected("request_missing")
        now = await self._maintenance_io("server_time_ms")
        excluded = excluded or set()
        replicas = list(group.replicas)
        start = self._replica_cursors[group.group_id] % len(replicas)
        replicas = replicas[start:] + replicas[:start]
        candidates = []
        for position, replica in enumerate(replicas):
            if not replica.enabled or replica.replica_id in excluded:
                continue
            status = await self._maintenance_io("endpoint_status", replica.replica_id)
            if not self._replica_is_eligible(
                replica, status, payload_bytes=metadata.payload_bytes, now=now
            ):
                continue
            candidates.append(
                (
                    (
                        int((status.get("circuit") or "closed") == "closed"),
                        replica.execution_limit
                        - int(status.get("reserved_items") or 0),
                        replica.execution_bytes
                        - int(status.get("reserved_bytes") or 0),
                        -position,
                    ),
                    replica,
                )
            )
        if candidates:
            return max(candidates, key=lambda candidate: candidate[0])[1]
        raise TransitionRejected("replica_unavailable")

    @staticmethod
    def _replica_is_eligible(
        replica: RegisteredReplica,
        status: dict[str, Any],
        *,
        payload_bytes: int,
        now: int,
    ) -> bool:
        circuit = status.get("circuit") or "closed"
        return not (
            not replica.enabled
            or status.get("gate") == "closed"
            or status.get("probe_claim")
            or (circuit != "closed" and int(status.get("next_probe") or 0) > now)
            or int(status.get("reserved_items") or 0) >= replica.execution_limit
            or int(status.get("reserved_bytes") or 0) + payload_bytes
            > replica.execution_bytes
        )

    async def _io(self, method: str, *args: Any, **kwargs: Any) -> Any:
        executor = (
            self._lease_worker if method == "renew_owner" else self._store_workers
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

    def release_hold(self, attempt_id: str) -> str:
        """Release one held reservation on an operator's word that the call stopped."""
        owner = self.owner
        if owner is None:
            return "not_owner"
        if attempt_id in self._tasks:
            return "in_flight"
        metadata = self.store.metadata(attempt_id)
        if metadata is None:
            return "missing"
        if metadata.reserved != 1 or metadata.state not in _HELD_STATES:
            return "not_held"
        disposition = self.store.settle_remote(
            owner,
            attempt_id,
            claim_id=metadata.claim_id,
            execution_seq=metadata.execution_seq,
            evidence="remote_cancelled",
        ).disposition
        if disposition == "settled":
            self._counts["operator_holds_released"] += 1
        return disposition

    async def _maintenance_io(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._stop.is_set():
            raise _DispatchStopped
        if method in {"prune_ready", "expire_producer", "promote_due"}:
            kwargs["should_stop"] = self._stop_workers.is_set
        return await self._io(method, *args, **kwargs)

    def health(
        self, *, read_budget: SnapshotReadBudget | None = None
    ) -> dict[str, Any]:
        from marie.engine.llm_queue.registry import MAX_SNAPSHOT_ROWS

        read_budget = read_budget or SnapshotReadBudget(0, time.monotonic() + 1.0)
        visible_lanes = []
        visible_group_ids: set[str] = set()
        visible_replica_ids: set[str] = set()
        rows_used = 1
        detail_row_limit = max(
            1,
            min(
                MAX_SNAPSHOT_ROWS - read_budget.candidates_left,
                read_budget.detail_rows_left - max(0, read_budget.dispatchers_left - 1),
            ),
        )
        for lane in self.lanes:
            group = self.endpoint_groups[lane.endpoint_group_id]
            new_group = group.group_id not in visible_group_ids
            new_replicas = {
                replica.replica_id for replica in group.replicas
            } - visible_replica_ids
            row_cost = 2 + (1 if new_group else 0) + 2 * len(new_replicas)
            if rows_used + row_cost > detail_row_limit:
                break
            visible_lanes.append(lane)
            visible_group_ids.add(group.group_id)
            visible_replica_ids.update(new_replicas)
            rows_used += row_cost
        details_truncated = (
            len(visible_lanes) < len(self.lanes)
            or len(visible_group_ids) < len(self.endpoint_groups)
            or len(visible_replica_ids) < len(self.replicas)
        )
        read_budget.before_read()
        usage = self.store.usage()
        read_budget.before_read()
        now_ms = self.store.server_time_ms()
        endpoint_states = {}
        for endpoint_id in sorted(visible_replica_ids):
            read_budget.before_read()
            endpoint_states[endpoint_id] = {
                **self.store.endpoint_status(endpoint_id),
                "port": _endpoint_port(self.endpoints[endpoint_id].base_url),
            }
        processing_limit = min(
            self.store.limits.cleanup_page_size,
            max(1, detail_row_limit - rows_used),
        )
        read_budget.before_read()
        processing_ids = self.store.processing_ids(limit=processing_limit)
        processing_rows = []
        for attempt_id in processing_ids:
            read_budget.before_read()
            metadata = self.store.metadata(attempt_id)
            if metadata is not None:
                processing_rows.append(metadata)
        processing_truncated = usage["reserved_items"] > len(processing_rows)
        lane_rows = []
        for lane in visible_lanes:
            head, _ = self.store.ready_metadata(
                lane.pool_id, limit=1, before_read=read_budget.before_read
            )
            read_budget.before_read()
            depth = self.store.ready_depth(lane.pool_id)
            read_budget.before_read()
            route = self.store.route_status(lane.pool_id)
            read_budget.before_read()
            route_totals = self.store.route_totals(lane.pool_id)
            lane_processing = [
                row for row in processing_rows if row.pool_id == lane.pool_id
            ]
            state_counts = Counter(row.state for row in lane_processing)
            fairness = self.scheduler.lane_metadata(lane.pool_id)
            fairness.update(
                quantum=lane.quantum,
                min_concurrent=lane.min_concurrent,
                max_burst_per_visit=lane.max_burst_per_visit,
                inflight=route["reserved_items"],
            )
            group = self.endpoint_groups[lane.endpoint_group_id]
            group_states = [
                endpoint_states[replica.replica_id] for replica in group.replicas
            ]
            replica_waiting_reasons = [
                (
                    "replica_unavailable"
                    if not replica.enabled or state.get("gate") == "closed"
                    else state.get("waiting_reason")
                    or (
                        "replica_capacity"
                        if int(state.get("reserved_items") or 0)
                        >= replica.execution_limit
                        or int(state.get("reserved_bytes") or 0)
                        + (head[0].payload_bytes if head else 0)
                        > replica.execution_bytes
                        else None
                    )
                )
                for replica, state in zip(group.replicas, group_states, strict=True)
            ]
            group_waiting_reason = None
            if all(replica_waiting_reasons):
                group_waiting_reason = next(
                    (
                        reason
                        for reason in (
                            "probe_unresolved",
                            "circuit_cooldown",
                            "replica_capacity",
                            "replica_unavailable",
                        )
                        if reason in replica_waiting_reasons
                    ),
                    "replica_unavailable",
                )
            waiting = (
                "disabled"
                if not lane.enabled
                or not any(replica.enabled for replica in group.replicas)
                else group_waiting_reason
                or (
                    "empty"
                    if not head
                    else (
                        "lane_capacity"
                        if route["reserved_items"] >= lane.execution_limit
                        else (
                            "global_capacity"
                            if usage["reserved_items"]
                            >= self.scheduler.total_concurrent_dispatch
                            else (
                                "insufficient_credit"
                                if self.owner is not None
                                and head[0].cost > fairness.get("deficit", 0)
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
                    endpoint_group_id=lane.endpoint_group_id,
                    enabled=lane.enabled
                    and any(replica.enabled for replica in group.replicas),
                    max_concurrent=lane.execution_limit,
                    **fairness,
                    reserved_items=route["reserved_items"],
                    reserved_bytes=route["reserved_bytes"],
                    protected_slots=min(lane.min_concurrent, route["reserved_items"]),
                    borrowed_slots=max(
                        0, route["reserved_items"] - lane.min_concurrent
                    ),
                    waiting_reason=waiting,
                    request_queue_depth=depth,
                    head_cost_units=head[0].cost if head else None,
                    oldest_pending_admitted_at_ms=(
                        head[0].admitted_at_ms if head else None
                    ),
                    oldest_pending_age_seconds=(
                        max(
                            0.0,
                            (now_ms - head[0].admitted_at_ms) / 1000,
                        )
                        if head and head[0].admitted_at_ms is not None
                        else None
                    ),
                    charged_cost=route_totals["charged"],
                    refunded_cost=route_totals["refunded"],
                    accepted=route_totals["accepted"],
                    completed=route_totals["completed"],
                    state_counts={
                        "ready": depth,
                        "claimed": state_counts["claimed"],
                        "executing": state_counts["executing"],
                        "outcome_unknown": state_counts["outcome_unknown"],
                        "abandoned": state_counts["abandoned"],
                        "failed": state_counts["failed"],
                        "cancelled": state_counts["cancelled"],
                        "expired": state_counts["expired"],
                    },
                    state_counts_truncated=processing_truncated,
                    drain_references=depth + len(lane_processing),
                )
            )
        group_rows = []
        for group in self.endpoint_groups.values():
            if group.group_id not in visible_group_ids:
                continue
            group_lanes = [
                lane
                for lane in visible_lanes
                if lane.endpoint_group_id == group.group_id
            ]
            active = [
                row for row in processing_rows if row.endpoint_id == group.group_id
            ]
            selected = next(
                (row.replica_id for row in active if row.replica_id is not None), None
            )
            group_rows.append(
                {
                    "group_id": group.group_id,
                    "revision": group.revision,
                    "selected_replica_id": selected,
                    "drain_references": sum(
                        row["drain_references"]
                        for row in lane_rows
                        if row["pool_id"] in {lane.pool_id for lane in group_lanes}
                    ),
                    "replicas": [
                        {
                            "replica_id": replica.replica_id,
                            "enabled": replica.enabled,
                            "execution_limit": replica.execution_limit,
                            "execution_bytes": replica.execution_bytes,
                            **endpoint_states[replica.replica_id],
                        }
                        for replica in group.replicas
                    ],
                }
            )
        read_budget.check()
        return dict(
            mode="queued-dispatch",
            sampling_available=True,
            metadata_unavailable=self.metadata_unavailable,
            fabric_group_id=self.store.keys.fabric_id,
            dispatcher_id=self.dispatcher_id,
            running=self._runner is not None and not self._runner.done(),
            draining=self._draining,
            owner_generation=self.owner.generation if self.owner else None,
            role="owner" if self.owner else "standby",
            configuration_ready=self.owner is not None
            and self._owner_ready
            and not self._policy_paused,
            pool_ids=[lane.pool_id for lane in visible_lanes],
            endpoint_ids=sorted(visible_replica_ids),
            inflight_request_count=len(self._tasks),
            counters={
                **self._counts,
                "abandoned_executions": usage.get("abandoned_executions", 0),
            },
            held_reservations_sampled=sum(
                row.reserved == 1 and row.state in _HELD_STATES
                for row in processing_rows
            ),
            held_requests=[
                dict(
                    attempt_id=row.attempt_id,
                    pool_id=row.pool_id,
                    replica_id=row.replica_id,
                    state=row.state,
                    execution_seq=row.execution_seq,
                    last_error=row.last_error,
                    uncertainty_until=row.uncertainty_until,
                    release_after_ms=row.uncertainty_until
                    + self.store.limits.max_deadline_ms,
                )
                for row in processing_rows
                if row.reserved == 1
                and row.state in _HELD_STATES
                and row.attempt_id not in self._tasks
            ],
            observed_at_ms=int(time.time() * 1000),
            policy_generation=self.policy_generation,
            policy_digest=self.policy_digest,
            pool_count=len(self.lanes),
            endpoint_count=len(self.endpoints),
            endpoint_group_count=len(self.endpoint_groups),
            details_truncated=details_truncated,
            execution_failures=self._counts["execution_errors"],
            usage=usage,
            endpoints=endpoint_states,
            endpoint_groups=group_rows,
            lanes=lane_rows,
            last_error=self._category,
            last_error_at_ms=self._last_error_at_ms,
            enabled=True,
            scheduler_policy=self.policy,
            total_concurrent_dispatch=self.scheduler.total_concurrent_dispatch,
            config_generation=self._config_revision,
            waiting_reason=(
                "policy_refresh_unavailable" if self._policy_paused else None
            ),
            processing_truncated=processing_truncated,
            policy_refresh_inflight=self._refresh_future is not None,
            policy_refresh_timed_out=self._refresh_timed_out,
        )

    def routing_resource_references(
        self,
        *,
        resource_type: str,
        resource_id: str,
        revision: str | None,
        limit: int,
    ) -> dict[str, Any]:
        """Return bounded queue references for deletion preflight."""
        page_limit = min(limit, self.store.limits.cleanup_page_size)
        usage = self.store.usage()
        processing_ids = self.store.processing_ids(limit=page_limit)
        rows = [
            metadata
            for attempt_id in processing_ids
            if (metadata := self.store.metadata(attempt_id)) is not None
        ]
        truncated = usage["reserved_items"] > len(rows)
        if resource_type == "pool":
            matching_lanes = [
                lane for lane in self.lanes if lane.pool_id == resource_id
            ]
        elif resource_type == "endpoint_group":
            matching_lanes = [
                lane
                for lane in self.lanes
                if lane.endpoint_group_id == resource_id
                and (revision is None or lane.revision == revision)
            ]
        elif resource_type == "replica":
            group_ids = {
                group.group_id
                for group in self.endpoint_groups.values()
                if any(replica.replica_id == resource_id for replica in group.replicas)
                and (revision is None or group.revision == revision)
            }
            matching_lanes = [
                lane for lane in self.lanes if lane.endpoint_group_id in group_ids
            ]
        elif resource_type == "policy_generation":
            generation = int(resource_id)
            ready_rows = []
            ready_total = 0
            remaining = page_limit
            for lane in self.lanes:
                depth = self.store.ready_depth(lane.pool_id)
                ready_total += depth
                if remaining <= 0 or depth <= 0:
                    continue
                records, invalid = self.store.ready_metadata(
                    lane.pool_id, limit=min(remaining, depth)
                )
                ready_rows.extend(records)
                remaining -= len(records) + invalid
            matching_ready = sum(
                row.policy_generation == generation for row in ready_rows
            )
            matching_active = sum(row.policy_generation == generation for row in rows)
            unknown_generation = any(
                row.policy_generation == 0 for row in (*ready_rows, *rows)
            )
            return {
                "valkey": matching_ready + matching_active,
                "ready": matching_ready,
                "active": matching_active,
                "truncated": truncated
                or ready_total > len(ready_rows)
                or unknown_generation,
            }
        else:
            raise ValueError("Invalid routing resource type")
        pools = {lane.pool_id for lane in matching_lanes}
        ready = sum(self.store.ready_depth(pool_id) for pool_id in pools)
        active = sum(row.pool_id in pools for row in rows)
        return {
            "valkey": ready + active,
            "ready": ready,
            "active": active,
            "truncated": truncated,
        }

    def sample_pending_requests(
        self, limit: int, *, read_budget: SnapshotReadBudget | None = None
    ) -> list[dict[str, Any]]:
        read_budget = read_budget or SnapshotReadBudget(limit, time.monotonic() + 1.0)
        observed_at = time.time()
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
                    mode="queued-dispatch",
                    config_revision=record.config_revision,
                    lifecycle_stage=record.state,
                    state_source="store",
                    payload_bytes=record.payload_bytes,
                    cost=record.cost,
                    expires_at_ms=record.expires_at_ms,
                    execution_seq=record.execution_seq,
                    next_eligible=record.next_eligible,
                    last_error=record.last_error,
                    model=record.model,
                    admitted_at_ms=record.admitted_at_ms,
                    submitted_at=(
                        record.admitted_at_ms / 1000
                        if record.admitted_at_ms is not None
                        else None
                    ),
                    popped_at=None,
                    state_updated_at=(
                        record.admitted_at_ms / 1000
                        if record.admitted_at_ms is not None
                        else observed_at
                    ),
                    queue_wait_age_seconds=(
                        max(0.0, observed_at - record.admitted_at_ms / 1000)
                        if record.admitted_at_ms is not None
                        else None
                    ),
                    inflight_age_seconds=None,
                    endpoint_group_id=record.endpoint_id,
                    replica_id=getattr(record, "replica_id", None),
                    policy_generation=getattr(record, "policy_generation", 0),
                    charged_cost=getattr(record, "charged_cost", 0),
                    charge_sequence=getattr(record, "charge_sequence", 0),
                    refund_state=getattr(record, "refund_state", None),
                )
                for record in records
            )
        self.metadata_unavailable = malformed
        return rows

    def inflight_requests_snapshot(self) -> list[dict[str, Any]]:
        observed_at = time.time()
        return [
            dict(
                request_id=inflight.claim.attempt_id,
                attempt_id=inflight.claim.attempt_id,
                fabric_group_id=self.store.keys.fabric_id,
                mode="queued-dispatch",
                lifecycle_stage="dispatching",
                state_source="dispatcher",
                dispatcher_id=self.dispatcher_id,
                pool_id=inflight.claim.pool_id,
                endpoint_id=inflight.claim.endpoint_group_id,
                endpoint_group_id=inflight.claim.endpoint_group_id,
                config_revision=inflight.metadata.config_revision,
                model=inflight.metadata.model,
                payload_bytes=inflight.metadata.payload_bytes,
                cost=inflight.metadata.cost,
                admitted_at_ms=inflight.metadata.admitted_at_ms,
                submitted_at=(
                    inflight.metadata.admitted_at_ms / 1000
                    if inflight.metadata.admitted_at_ms is not None
                    else None
                ),
                expires_at_ms=inflight.metadata.expires_at_ms,
                popped_at=inflight.popped_at,
                state_updated_at=inflight.popped_at,
                queue_wait_age_seconds=(
                    max(
                        0.0,
                        inflight.popped_at - inflight.metadata.admitted_at_ms / 1000,
                    )
                    if inflight.metadata.admitted_at_ms is not None
                    else None
                ),
                inflight_age_seconds=max(0.0, observed_at - inflight.popped_at),
                execution_seq=inflight.claim.execution_sequence,
                claim_id=inflight.claim.claim_id,
                owner_generation=inflight.claim.owner_generation,
            )
            for inflight in self._inflight_claims.values()
        ]

    async def start(self, *, owner: OwnerToken | None = None) -> None:
        if self._runner is not None:
            return
        if owner is not None:
            began = time.monotonic()
            await self._io('renew_owner', owner, lease_ms=self.owner_lease_ms)
            self.owner = owner
            self._owner_until = began + self.owner_lease_ms / 1000
        self._clients = {
            identity: self.client_factory(endpoint)
            for identity, endpoint in self.endpoints.items()
            if endpoint.enabled
            and any(
                lane.enabled
                and any(
                    replica.replica_id == identity
                    for replica in self.endpoint_groups[lane.endpoint_group_id].replicas
                )
                for lane in self.lanes
            )
        }
        if owner is not None:
            self._heartbeat = asyncio.create_task(self._renew())
        self._runner = asyncio.create_task(self._run())
        self._client_endpoints = {
            identity: self.endpoints[identity] for identity in self._clients
        }
        register_dispatcher(self.dispatcher_id, self)

    async def stop(self) -> None:
        if self._runner is None:
            return
        self._draining = True
        self._stop_workers.set()
        self._stop.set()
        self._wake_dispatcher()
        await asyncio.shield(self._runner)

    def _wake_dispatcher(self) -> None:
        if not self._dispatch_wakeup.full():
            self._dispatch_wakeup.put_nowait(None)

    async def _wait_for_dispatch(self) -> None:
        try:
            await asyncio.wait_for(
                self._dispatch_wakeup.get(), timeout=self.poll_seconds
            )
        except TimeoutError:
            pass

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
                await self._io("renew_owner", owner, lease_ms=self.owner_lease_ms)
                self._owner_until = began + self.owner_lease_ms / 1000
            except (StaleOwner, StoreUnavailable):
                self._category = "owner_uncertain"
                self._owner_until = 0
                for task in self._tasks.values():
                    task.cancel()
                return

    async def _configure(self) -> None:
        cursor = 0
        while True:
            cursor, pools = await self._maintenance_io(
                "scan_routes", cursor=cursor, limit=self.store.limits.cleanup_page_size
            )
            for pool_id in pools:
                await self._maintenance_io("disable_route", self.owner, pool_id)
            if cursor == 0:
                break
        for endpoint in self.endpoints.values():
            result = await self._maintenance_io(
                "configure_endpoint",
                self.owner,
                endpoint.endpoint_id,
                execution_limit=endpoint.execution_limit,
                execution_bytes=endpoint.execution_bytes,
                gate_open=endpoint.enabled,
                transport_fingerprint=endpoint.transport_fingerprint(),
                target_fingerprint=endpoint.target_fingerprint(),
            )
            if result.disposition != "configured":
                raise TransitionRejected(result.disposition)
        for lane in self.lanes:
            group = self.endpoint_groups[lane.endpoint_group_id]
            legacy_endpoint = (
                len(group.replicas) == 1
                and group.replicas[0].replica_id == group.group_id
            )
            result = await self._maintenance_io(
                "configure_route",
                self.owner,
                lane.pool_id,
                lane.endpoint_group_id,
                revision=lane.revision,
                execution_limit=lane.execution_limit,
                execution_bytes=lane.execution_bytes,
                enabled=lane.enabled
                and any(replica.enabled for replica in group.replicas),
                legacy_endpoint=legacy_endpoint,
            )
            if result.disposition != "configured":
                raise TransitionRejected(result.disposition)
        for endpoint in self.endpoints.values():
            result = await self._maintenance_io(
                "configure_endpoint",
                self.owner,
                endpoint.endpoint_id,
                execution_limit=endpoint.execution_limit,
                execution_bytes=endpoint.execution_bytes,
                gate_open=endpoint.enabled,
                transport_fingerprint=endpoint.transport_fingerprint(),
                target_fingerprint=endpoint.target_fingerprint(),
            )
            if result.disposition != "configured":
                raise TransitionRejected(result.disposition)

    async def _refresh_policy(self) -> None:
        if self.policy_loader is None:
            return
        now = time.monotonic()
        if self._refresh_future is None and now >= self._next_refresh:
            self._refresh_started = now
            self._refresh_timed_out = False
            self._refresh_future = self._refresh_worker.submit(self.policy_loader)
        future = self._refresh_future
        if future is not None and future.done():
            self._refresh_future = None
            try:
                policy = future.result()
                generation = policy.get("policy_generation")
                digest = policy.get("policy_digest")
                if (generation is None) != (digest is None) or (
                    generation is not None
                    and (
                        type(generation) is not int
                        or not 1 <= generation <= 2**53 - 1
                        or not isinstance(digest, str)
                        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    )
                ):
                    raise ValueError("Invalid observed policy identity")
                if policy["limits"] != self.store.limits:
                    raise ValueError("Immutable fabric limits changed")
                endpoint_groups = list(policy.get("endpoint_groups") or [])
                if not endpoint_groups:
                    endpoint_groups = [
                        RegisteredEndpointGroup(
                            group_id=endpoint.endpoint_id,
                            revision="r1",
                            replicas=(
                                RegisteredReplica(
                                    replica_id=endpoint.endpoint_id,
                                    base_url=endpoint.base_url,
                                    api_key=endpoint.api_key,
                                    allow_private=endpoint.allow_private,
                                    allow_loopback=endpoint.allow_loopback,
                                    execution_limit=endpoint.execution_limit,
                                    execution_bytes=endpoint.execution_bytes,
                                    call_timeout_seconds=endpoint.call_timeout_seconds,
                                    max_response_bytes=endpoint.max_response_bytes,
                                    retry_429=endpoint.retry_429,
                                    enabled=endpoint.enabled,
                                ),
                            ),
                        )
                        for endpoint in policy["endpoints"]
                    ]
                groups = {group.group_id: group for group in endpoint_groups}
                for group_id, group in groups.items():
                    previous = self.endpoint_groups.get(group_id)
                    if (
                        previous is not None
                        and previous.revision == group.revision
                        and tuple(replica.replica_id for replica in previous.replicas)
                        != tuple(replica.replica_id for replica in group.replicas)
                    ):
                        raise ValueError("Endpoint group membership changed in place")
                replicas = {
                    replica.replica_id: replica
                    for group in endpoint_groups
                    for replica in group.replicas
                }
                endpoints = {
                    identity: replica.as_endpoint()
                    for identity, replica in replicas.items()
                }
                lanes = tuple(policy["lanes"])
                for identity, endpoint in endpoints.items():
                    previous = self.endpoints.get(
                        identity
                    ) or self._client_endpoints.get(identity)
                    if (
                        previous
                        and previous.transport_fingerprint()
                        != endpoint.transport_fingerprint()
                    ):
                        raise ValueError("Endpoint identity changed")
                scheduler = DrrLaneScheduler(
                    total_concurrent_dispatch=policy["total_concurrent_dispatch"],
                    lanes=[
                        DrrLaneConfig(
                            pool_id=lane.pool_id,
                            quantum=(
                                lane.quantum if policy["policy"] == "drr" else 1_000_000
                            ),
                            min_concurrent=lane.min_concurrent,
                            max_concurrent=lane.execution_limit,
                            max_burst_per_visit=(
                                lane.max_burst_per_visit
                                if policy["policy"] == "drr"
                                else 1
                            ),
                            enabled=lane.enabled,
                        )
                        for lane in lanes
                    ],
                )
                changed = (
                    lanes != self.lanes
                    or groups != self.endpoint_groups
                    or endpoints != self.endpoints
                    or policy["policy"] != self.policy
                    or scheduler.total_concurrent_dispatch
                    != self.scheduler.total_concurrent_dispatch
                )
                if changed or self._policy_paused:
                    old_state = (
                        self.lanes,
                        self.endpoint_groups,
                        self.replicas,
                        self.endpoints,
                    )
                    self.lanes = lanes
                    self.endpoint_groups = groups
                    self.replicas = replicas
                    self.endpoints = endpoints
                    self._routes_disabled = False
                    try:
                        await self._configure()
                    except BaseException:
                        (
                            self.lanes,
                            self.endpoint_groups,
                            self.replicas,
                            self.endpoints,
                        ) = old_state
                        raise
                    for identity, endpoint in endpoints.items():
                        if endpoint.enabled and identity not in self._clients:
                            self._clients[identity] = self.client_factory(endpoint)
                            self._client_endpoints[identity] = endpoint
                    await self._retire_clients()
                    self.scheduler = scheduler
                    self.policy = policy["policy"]
                    self._config_revision += 1
                    self._counts["config_refreshes"] += 1
                if self._policy_paused:
                    self._counts["config_refresh_recoveries"] += 1
                self._policy_paused = False
                self._refresh_timed_out = False
                self._routes_disabled = False
                self._refresh_failures = 0
                self._next_refresh = now + self.refresh_seconds
                self.policy_generation = policy.get("policy_generation")
                self.policy_digest = policy.get("policy_digest")
            except Exception:
                self._policy_paused = True
                self._refresh_failures += 1
                self._counts["config_refresh_errors"] += 1
                self._category = "policy_refresh_unavailable"
                self._next_refresh = now + min(
                    30.0, self.refresh_seconds * 2 ** min(self._refresh_failures, 3)
                )
        elif (
            future is not None
            and now - self._refresh_started >= self.refresh_timeout_seconds
        ):
            if not self._refresh_timed_out:
                self._counts["config_refresh_timeouts"] += 1
                self._refresh_timed_out = True
            self._policy_paused = True
            self._category = "policy_refresh_unavailable"
        if self._policy_paused and not self._routes_disabled:
            cursor = 0
            while True:
                cursor, pools = await self._maintenance_io(
                    "scan_routes",
                    cursor=cursor,
                    limit=self.store.limits.cleanup_page_size,
                )
                for pool in pools:
                    await self._maintenance_io("disable_route", self.owner, pool)
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
                            "acquire_owner",
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
                            self._counts["config_refresh_errors"] += 1
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
                                "processing_ids",
                                offset=len(previous),
                                limit=self.store.limits.cleanup_page_size,
                            )
                            if not ids:
                                break
                            previous.extend(ids)
                        charges: list[ClaimRecord] = []
                        page = self.store.limits.cleanup_page_size
                        for offset in range(0, len(previous), page):
                            charges.extend(
                                await self._maintenance_io(
                                    "reconcile_charges",
                                    self.owner,
                                    attempt_ids=previous[offset : offset + page],
                                )
                            )
                        self.scheduler.reconcile_charges(charges)
                        charges_by_attempt = {
                            claim.attempt_id: claim for claim in charges
                        }
                        for attempt in previous:
                            if self._stop.is_set():
                                break
                            recovered = await self._maintenance_io(
                                "recover_claim", self.owner, attempt
                            )
                            if recovered.disposition == "requeued":
                                self._counts["live_claims_recovered"] += 1
                                claim = charges_by_attempt.get(attempt)
                                if claim is not None:
                                    self.scheduler.refunded(
                                        claim.pool_id,
                                        claim.charged_cost,
                                        charge_sequence=claim.charge_sequence,
                                    )
                            elif recovered.disposition == "finished":
                                self._counts["remote_outcomes_recovered"] += 1
                            elif recovered.disposition == "settled":
                                self._counts["terminal_reservations_recovered"] += 1
                            elif recovered.disposition == "hold_expired":
                                self._counts["held_reservations_expired"] += 1
                        self._owner_ready = True
                    if time.monotonic() >= self._owner_until:
                        raise StaleOwner("owner expired locally")
                    await self._refresh_policy()
                    await self._maintenance()
                    if not self._policy_paused:
                        await self._dispatch()
                    await self._wait_for_dispatch()
                except _DispatchStopped:
                    break
                except StoreUnavailable:
                    self._counts["store_errors"] += 1
                    self._category = "store_unavailable"
                    await self._pause(random.uniform(0.1, 0.5))
                except (StaleOwner, TransitionRejected):
                    self._category = "owner_lost"
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
                    self._category = "dispatch_error"
                    self._counts["dispatch_errors"] += 1
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
            for identity in tuple(self._clients):
                await self._close_client(identity)
            if self.owner:
                try:
                    await self._io("release_owner", self.owner)
                except (StaleOwner, StoreUnavailable):
                    pass
            self.owner = None
            unregister_dispatcher(self.dispatcher_id)
            await self._io("close")
            self._store_workers.shutdown(wait=True)
            self._lease_worker.shutdown(wait=True)
            self._refresh_worker.shutdown(wait=False, cancel_futures=True)

    async def _close_client(self, identity: str) -> None:
        client = self._clients.pop(identity, None)
        if client is None:
            return
        self._client_endpoints.pop(identity, None)
        self._client_users.pop(identity, None)
        try:
            await client.close()
        except Exception:
            self._counts["client_close_errors"] += 1
            self._category = "client_close_failed"

    async def _retire_clients(self) -> None:
        for identity in tuple(self._clients):
            if identity not in self.endpoints and not self._client_users[identity]:
                await self._close_client(identity)

    async def _maintenance(self) -> None:
        page = self.store.limits.cleanup_page_size
        for attempt, task in tuple(self._tasks.items()):
            metadata = await self._maintenance_io("metadata", attempt)
            # Keep the bounded HTTP observer to obtain actual settlement evidence.
            if attempt in self._calling and metadata is None:
                task.cancel()
        for producer in await self._maintenance_io("due_ids", "producers", limit=page):
            expired = await self._maintenance_io(
                "expire_producer", producer, limit=page
            )
            self._counts["producer_expired_requests"] += sum(
                reply.disposition == "discarded" for reply in expired
            )
        for attempt in await self._maintenance_io("due_ids", "deadlines", limit=page):
            metadata = await self._maintenance_io("metadata", attempt)
            if metadata:
                await self._maintenance_io(
                    "cancel_or_expire", metadata.producer_id, attempt, expire=True
                )
        for attempt in await self._maintenance_io("due_ids", "retention", limit=page):
            await self._maintenance_io("purge_terminal", self.owner, attempt)
        ids = await self._maintenance_io(
            "processing_ids", offset=self._processing_offset, limit=page
        )
        self._processing_offset = (
            self._processing_offset + len(ids) if len(ids) == page else 0
        )
        for attempt in ids:
            if attempt not in self._tasks:
                recovered = await self._maintenance_io(
                    "recover_claim", self.owner, attempt
                )
                if recovered.disposition == "requeued":
                    self._counts["live_claims_recovered"] += 1
                elif recovered.disposition == "finished":
                    self._counts["remote_outcomes_recovered"] += 1
                elif recovered.disposition == "settled":
                    self._counts["terminal_reservations_recovered"] += 1
                elif recovered.disposition == "hold_expired":
                    self._counts["held_reservations_expired"] += 1
        promoted = await self._maintenance_io(
            "promote_due", self.owner, limit=page, offset=self._delayed_offset
        )
        self._delayed_offset = (
            self._delayed_offset + len(promoted) if len(promoted) == page else 0
        )
        self._route_cursor, pools = await self._maintenance_io(
            "scan_routes", cursor=self._route_cursor, limit=page
        )
        for pool_id in pools:
            await self._maintenance_io("prune_ready", self.owner, pool_id, limit=page)

    async def _dispatch(self) -> None:
        heads = {}
        reserved = {}
        now = await self._maintenance_io("server_time_ms")
        replica_statuses: dict[str, dict[str, Any]] = {}
        usage = await self._maintenance_io("usage")
        for lane in self.lanes:
            route = await self._maintenance_io("route_status", lane.pool_id)
            reserved[lane.pool_id] = route["reserved_items"]
            group = self.endpoint_groups[lane.endpoint_group_id]
            if not lane.enabled or not any(
                replica.enabled for replica in group.replicas
            ):
                continue
            attempt = await self._maintenance_io("ready_head", lane.pool_id)
            if attempt:
                try:
                    head = await self._maintenance_io("metadata", attempt)
                except (ValueError, TypeError):
                    self._counts["invalid_head_metadata"] += 1
                    continue
                if (
                    head is not None
                    and head.state == "ready"
                    and head.endpoint_id == lane.endpoint_id
                    and head.config_revision == lane.revision
                    and route["enabled"] == "1"
                    and 1 <= head.cost <= 1_000_000
                    and route["reserved_bytes"] + head.payload_bytes
                    <= lane.execution_bytes
                ):
                    available = False
                    for replica in group.replicas:
                        status = replica_statuses.get(replica.replica_id)
                        if status is None:
                            status = await self._maintenance_io(
                                "endpoint_status", replica.replica_id
                            )
                            replica_statuses[replica.replica_id] = status
                        if self._replica_is_eligible(
                            replica,
                            status,
                            payload_bytes=head.payload_bytes,
                            now=now,
                        ):
                            available = True
                            break
                    if available:
                        heads[lane.pool_id] = head
        self.scheduler.sync_reservations(reserved, usage["reserved_items"], set(heads))
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
                "claim_and_charge",
                self.owner,
                pool,
                expected_attempt=head.attempt_id,
                claim_id=claim_id,
                expected_cost=head.cost,
            )
            if not isinstance(reply, ClaimRecord):
                self.scheduler.rejected(pool, reply.disposition)
                heads.pop(pool)
                continue
            self.scheduler.claimed(
                pool,
                reply.charged_cost,
                charge_sequence=reply.charge_sequence,
            )
            self._counts["claims"] += 1
            # Track the selected claim before any subsequent fallible store inspection.
            self._inflight_claims[reply.attempt_id] = _InflightClaim(
                claim=reply,
                metadata=head,
                popped_at=time.time(),
            )
            task = asyncio.create_task(self._execute(reply))
            self._tasks[reply.attempt_id] = task
            task.add_done_callback(partial(self._completed_task, reply.attempt_id))
            attempt = await self._maintenance_io("ready_head", pool)
            try:
                next_head = (
                    await self._maintenance_io("metadata", attempt) if attempt else None
                )
            except (ValueError, TypeError):
                self._counts["invalid_head_metadata"] += 1
                next_head = None
            if (
                next_head is not None
                and next_head.state == "ready"
                and next_head.endpoint_id == head.endpoint_id
                and next_head.config_revision == head.config_revision
                and 1 <= next_head.cost <= 1_000_000
            ):
                heads[pool] = next_head
            else:
                heads.pop(pool, None)

    def _completed_task(self, attempt: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(attempt, None)
        self._inflight_claims.pop(attempt, None)
        self._wake_dispatcher()
        if not task.cancelled() and task.exception() is not None:
            self._category = "dispatch_error"
            self._counts["dispatch_errors"] += 1

    async def _reserve_replica(
        self, claim: ClaimRecord, excluded: set[str]
    ) -> RegisteredReplica:
        group = self.endpoint_groups.get(claim.endpoint_group_id)
        if group is None:
            raise _NoReplicaAvailable
        while len(excluded) < len(group.replicas):
            try:
                replica = await self.select_replica(claim, excluded=excluded)
            except TransitionRejected as exc:
                if str(exc) == "replica_unavailable":
                    raise _NoReplicaAvailable from None
                raise
            reply = await self._io(
                "reserve_replica", self.owner, claim, replica.replica_id
            )
            if reply.disposition in {"reserved", "existing"}:
                self._replica_cursors[group.group_id] += 1
                return replica
            if reply.disposition not in {"capacity", "gated", "replica_unavailable"}:
                raise TransitionRejected(reply.disposition)
            excluded.add(replica.replica_id)
        raise _NoReplicaAvailable

    async def _return_untransmitted(self, claim: ClaimRecord) -> None:
        while True:
            reply = await self._commit("return_untransmitted_and_refund", claim)
            if reply.disposition in {"returned", "existing"}:
                if reply.refund_state == "refunded":
                    self.scheduler.refunded(
                        claim.pool_id,
                        claim.charged_cost,
                        charge_sequence=claim.charge_sequence,
                    )
                return
            if reply.disposition != "backpressure":
                raise TransitionRejected(reply.disposition)
            await asyncio.sleep(self.poll_seconds)

    async def _record_physical_attempt(
        self,
        claim: ClaimRecord,
        *,
        task_id: str,
        sequence: int,
        physical_attempt: int,
        began: float,
        outcome: ExecutionOutcome,
    ) -> str | None:
        try:
            response = outcome.response
            usage = _response_value(response, "usage")
            raw_response = _raw_completion_text(response)
            error = outcome.category or None
            if response is not None and raw_response is None:
                error = error or "invalid_response"
            evidence = CompletionAttempt(
                task_id=task_id,
                attempt_number=physical_attempt,
                raw_response=raw_response,
                finish_reason=(
                    completion_finish_reason(response)
                    if raw_response is not None
                    else None
                ),
                response_model=_response_value(response, "model"),
                input_tokens=_token_count(usage, "prompt_tokens", "input_tokens"),
                output_tokens=_token_count(usage, "completion_tokens", "output_tokens"),
                elapsed_ms=max(0.0, (time.monotonic() - began) * 1000),
                error=error,
                execution_sequence=sequence,
                physical_attempt=physical_attempt,
            )
            recorded = await self._commit(
                "record_completion_attempt",
                claim.attempt_id,
                claim_id=claim.claim_id,
                execution_seq=sequence,
                physical_attempt=physical_attempt,
                attempt=evidence,
            )
            failure = (
                None
                if recorded.disposition in {"recorded", "existing"}
                else "attempt_evidence_unavailable"
            )
            if recorded.disposition in {"evidence_too_large", "evidence_conflict"}:
                failure = recorded.disposition
        except (OSError, ValueError, TypeError, OverflowError, RecursionError):
            failure = "evidence_write_failed"
        return failure

    async def _provider(self, claim: ClaimRecord) -> tuple[int, ExecutionOutcome, str]:
        attempt = claim.attempt_id
        claim_id = claim.claim_id
        if self._stop.is_set():
            raise TransitionRejected("start_stopped")
        payload = call = None
        used_clients: set[str] = set()
        self._calling.add(attempt)
        try:
            excluded: set[str] = set()
            try:
                replica = await self._reserve_replica(claim, excluded)
            except _NoReplicaAvailable:
                await self._return_untransmitted(claim)
                raise
            client = self._clients.get(replica.replica_id)
            if client is None:
                await self._commit(
                    "release_replica_for_failover", claim, replica.replica_id
                )
                await self._return_untransmitted(claim)
                raise _NoReplicaAvailable
            used_clients.add(replica.replica_id)
            self._client_users[replica.replica_id] += 1
            metadata = await self._io("metadata", attempt)
            payload = await self._io(
                "fetch_payload", self.owner, attempt, claim_id=claim_id
            )
            call = CompletionCallParams.from_dict(payload["call"])
            clock_read_at = time.monotonic()
            now = await self._io("server_time_ms")
            queue_clock_read_at = time.monotonic()
            request_deadline = clock_read_at + (payload["expires_at_ms"] - now) / 1000
            from marie.engine.completion_contract import require_terminal_completion

            require_terminal_completion(call)
            if (
                self._stop.is_set()
                or time.monotonic() >= self._owner_until
                or request_deadline <= clock_read_at
            ):
                raise TransitionRejected("start_stopped")
            started = await self._io(
                "authorize_start", self.owner, attempt, claim_id=claim_id
            )
            if started.disposition != "started":
                raise TransitionRejected("start_unconfirmed")
            repetition_retry_used = False
            physical_attempt = 0
            capture_attempts = payload.get("capture_attempts", False)
            transmitted = False
            while True:
                timeout = min(
                    request_deadline - time.monotonic(),
                    replica.call_timeout_seconds,
                )
                if timeout <= 0:
                    if transmitted:
                        return (
                            started.execution_seq,
                            ExecutionOutcome(
                                category="call_timeout", remote_settled=True
                            ),
                            replica.replica_id,
                        )
                    await self._commit(
                        "release_replica_for_failover",
                        claim,
                        replica.replica_id,
                    )
                    await self._return_untransmitted(claim)
                    raise _NoReplicaAvailable
                if self._stop.is_set() or time.monotonic() >= self._owner_until:
                    raise TransitionRejected("start_stopped")
                endpoint_state = await self._io("endpoint_status", replica.replica_id)
                self._counts["provider_starts"] += 1
                if endpoint_state.get("probe_claim") == claim_id:
                    self._counts["probe_starts"] += 1
                attributes = {
                    "marie.llm_dispatch.request_id": attempt,
                    "marie.llm_dispatch.claim_id": claim_id,
                    "marie.llm_dispatch.execution_seq": started.execution_seq,
                    "marie.llm_dispatch.fabric_group_id": self.store.keys.fabric_id,
                    "marie.llm_dispatch.pool_id": metadata.pool_id,
                    "marie.llm_dispatch.endpoint_group_id": claim.endpoint_group_id,
                    "marie.llm_dispatch.replica_id": replica.replica_id,
                    "marie.llm_dispatch.dispatcher_id": self.dispatcher_id,
                    "marie.llm_dispatch.mode": "queued-dispatch",
                    "marie.llm_dispatch.model": (
                        metadata.model
                        if metadata.model
                        and not metadata.model.lower().startswith(
                            ("http:", "https:", "data:")
                        )
                        else ""
                    ),
                    "marie.llm_dispatch.message_count": len(call.messages),
                    "marie.llm_dispatch.queue_wait_ms": max(
                        0,
                        now
                        - metadata.admitted_at_ms
                        + (time.monotonic() - queue_clock_read_at) * 1000,
                    ),
                }
                parent_context = trace.set_span_in_context(trace.get_current_span())
                if not trace.get_current_span().get_span_context().is_valid:
                    parent_context = extract_trace_context(
                        metadata.traceparent, metadata.tracestate
                    )
                began_ns = time.time_ns()
                began = time.monotonic()
                span = start_trace_span(
                    _tracer,
                    "LLMDispatch.completion",
                    context=parent_context,
                    start_time=began_ns,
                    kind=trace.SpanKind.CLIENT,
                    attributes={
                        **attributes,
                        "openinference.span.kind": "LLM",
                        "llm.model_name": attributes["marie.llm_dispatch.model"],
                    },
                )
                if span.is_recording():
                    set_llm_io(span, input_messages=call.messages, context=call.context)
                physical_attempt += 1
                outcome = None
                interrupted = None
                evidence_failure = None
                try:
                    with use_trace_context(trace.set_span_in_context(span)):
                        outcome = await client.execute(call, timeout_seconds=timeout)
                except BaseException as error:
                    interrupted = error
                    raise
                finally:
                    try:
                        if capture_attempts:
                            evidence_failure = await self._record_physical_attempt(
                                claim,
                                task_id=payload["logical_task_id"],
                                sequence=started.execution_seq,
                                physical_attempt=physical_attempt,
                                began=began,
                                outcome=outcome
                                or ExecutionOutcome(category="execution_interrupted"),
                            )
                    except (StaleOwner, StoreUnavailable, TransitionRejected):
                        if interrupted is None:
                            raise
                    finally:
                        if outcome is None:
                            _emit_execution_history(
                                attributes,
                                ExecutionOutcome(category="execution_interrupted"),
                                began_ns,
                                began,
                                span=span,
                            )
                if evidence_failure is not None:
                    outcome.retryable = False
                    outcome.response = None
                    outcome.category = evidence_failure
                    _emit_execution_history(
                        attributes, outcome, began_ns, began, span=span
                    )
                    return started.execution_seq, outcome, replica.replica_id
                if not outcome.request_started and transmitted:
                    outcome.retryable = False
                    _emit_execution_history(
                        attributes, outcome, began_ns, began, span=span
                    )
                    return started.execution_seq, outcome, replica.replica_id
                transmitted = transmitted or outcome.request_started
                recovered, repetitive = _normalize_length_response(outcome.response)
                if recovered:
                    self._counts["missing_eos_recovered"] += 1
                    _emit_execution_history(
                        attributes, outcome, began_ns, began, span=span
                    )
                elif repetitive:
                    outcome.category = "repetition"
                    _emit_execution_history(
                        attributes, outcome, began_ns, began, span=span
                    )
                    recovered_call = apply_repetition_recovery(call)
                    if not repetition_retry_used and recovered_call is not None:
                        permission = await self._commit(
                            "can_continue_execution",
                            attempt,
                            claim_id=claim_id,
                            execution_seq=started.execution_seq,
                        )
                        if permission.disposition == "allowed":
                            call = recovered_call
                            repetition_retry_used = True
                            self._counts["repetition_retries"] += 1
                            continue
                    outcome = ExecutionOutcome(
                        category="repetition",
                        remote_settled=True,
                        availability_success=True,
                    )
                else:
                    _emit_execution_history(
                        attributes, outcome, began_ns, began, span=span
                    )
                if outcome.request_started:
                    return started.execution_seq, outcome, replica.replica_id
                await self._commit(
                    "record_endpoint_outcome",
                    attempt,
                    claim_id=claim_id,
                    execution_seq=started.execution_seq,
                    replica_id=replica.replica_id,
                    outcome=(
                        "unavailable" if outcome.availability_failure else "neutral"
                    ),
                    category=outcome.category or "none",
                    open_ms=self.circuit_open_ms,
                )
                released = await self._commit(
                    "release_replica_for_failover",
                    claim,
                    replica.replica_id,
                )
                if released.disposition not in {"released", "existing"}:
                    await self._return_untransmitted(claim)
                    raise _NoReplicaAvailable
                excluded.add(replica.replica_id)
                try:
                    replica = await self._reserve_replica(claim, excluded)
                    client = self._clients.get(replica.replica_id)
                    if client is None:
                        await self._commit(
                            "release_replica_for_failover", claim, replica.replica_id
                        )
                        raise _NoReplicaAvailable
                    if replica.replica_id not in used_clients:
                        used_clients.add(replica.replica_id)
                        self._client_users[replica.replica_id] += 1
                except _NoReplicaAvailable:
                    await self._return_untransmitted(claim)
                    raise
        finally:
            payload = call = None
            self._calling.discard(attempt)
            for identity in used_clients:
                self._client_users[identity] -= 1
            await self._retire_clients()

    async def _commit(self, method: str, *args: Any, **kwargs: Any) -> Any:
        while True:
            try:
                return await self._io(method, self.owner, *args, **kwargs)
            except StoreUnavailable:
                self._counts["store_errors"] += 1
            if time.monotonic() >= self._owner_until:
                raise StaleOwner("owner uncertain")
            await asyncio.sleep(random.uniform(0.1, 0.5))

    async def _execute(self, claim: ClaimRecord) -> None:
        span = None
        try:
            metadata = await self._io("metadata", claim.attempt_id)
            parent = extract_trace_context(
                metadata.traceparent if metadata else None,
                metadata.tracestate if metadata else None,
            )
            span = start_trace_span(
                _tracer,
                "LLMDispatch.dispatch",
                context=parent,
                attributes={
                    "openinference.span.kind": "CHAIN",
                    "marie.llm_dispatch.request_id": claim.attempt_id,
                    "marie.llm_dispatch.claim_id": claim.claim_id,
                    "marie.llm_dispatch.fabric_group_id": self.store.keys.fabric_id,
                    "marie.llm_dispatch.pool_id": claim.pool_id,
                    "marie.llm_dispatch.dispatcher_id": self.dispatcher_id,
                    "marie.llm_dispatch.mode": "queued-dispatch",
                },
            )
            with use_trace_context(trace.set_span_in_context(span)):
                await self._execute_claim(claim, span)
        except StaleOwner:
            self._owner_until = 0
            if span is not None:
                self._observe_outcome(
                    span, claim, "interrupted", category="owner_uncertain"
                )
        except (StoreUnavailable, TransitionRejected):
            self._category = "start_unconfirmed"
            if span is not None:
                self._observe_outcome(
                    span, claim, "interrupted", category="start_unconfirmed"
                )
        except asyncio.CancelledError:
            if span is not None:
                self._observe_outcome(
                    span, claim, "interrupted", category="dispatch_cancelled"
                )
            raise
        except Exception:
            if span is not None:
                self._observe_outcome(
                    span, claim, "interrupted", category="dispatch_error"
                )
            raise
        finally:
            self.scheduler.retire_charge(claim.charge_sequence)
            if span is not None:
                span.end()

    def _observe_outcome(
        self,
        span: Span,
        claim: ClaimRecord,
        disposition: str,
        *,
        category: str = "none",
        **attributes: Any,
    ) -> None:
        span.add_event(disposition, attributes=attributes)
        span.set_attribute("marie.llm_dispatch.disposition", disposition)
        span.set_attribute("marie.llm_dispatch.error_type", category)
        if category != "none":
            span.set_status(StatusCode.ERROR)
        elif disposition == "result_committed":
            span.set_status(StatusCode.OK)
        dispatch_metrics.record_request_outcome(
            pool_id=claim.pool_id,
            dispatcher_id=self.dispatcher_id,
            fabric_group_id=self.store.keys.fabric_id,
            disposition=disposition,
            category=category,
        )

    async def _execute_claim(self, claim: ClaimRecord, span: Span) -> None:
        attempt = claim.attempt_id
        claim_id = claim.claim_id
        outcome = None
        try:
            try:
                sequence, outcome, replica_id = await self._provider(claim)
            except UnsupportedQueueStreaming:
                reply = await self._commit(
                    "reject_claim",
                    attempt,
                    claim_id=claim_id,
                    category="unsupported_streaming",
                )
                if reply.refund_state == "refunded":
                    self.scheduler.refunded(
                        claim.pool_id,
                        claim.charged_cost,
                        charge_sequence=claim.charge_sequence,
                    )
                self._observe_outcome(
                    span, claim, "rejected", category="unsupported_streaming"
                )
                return
            except _NoReplicaAvailable:
                self._category = "replica_unavailable"
                self._observe_outcome(
                    span, claim, "returned", category="replica_unavailable"
                )
                return
            except (StoreUnavailable, TransitionRejected):
                self._category = "start_unconfirmed"
                self._observe_outcome(
                    span, claim, "interrupted", category="start_unconfirmed"
                )
                return
            args = dict(claim_id=claim_id, execution_seq=sequence)
            if outcome.category:
                self._category = outcome.category
                self._last_error_at_ms = int(time.time() * 1000)
                self._counts["execution_errors"] += 1
            feedback = (
                "success"
                if outcome.availability_success
                else "unavailable"
                if outcome.availability_failure
                else "neutral"
            )
            await self._commit(
                "record_endpoint_outcome",
                attempt,
                **args,
                replica_id=replica_id,
                outcome=feedback,
                category=outcome.category or "none",
                open_ms=self.circuit_open_ms,
            )
            if not outcome.remote_settled:
                reply = await self._commit(
                    "mark_unknown", attempt, **args, category=outcome.category
                )
                if reply.disposition == "finished":
                    self._counts["remote_outcomes_failed"] += 1
                self._observe_outcome(
                    span,
                    claim,
                    "remote_unknown",
                    category=outcome.category or "remote_outcome_unknown",
                    remote_settled=False,
                )
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
                    "defer",
                    attempt,
                    **args,
                    delay_ms=delay,
                    reason=outcome.category,
                    remote_settled=True,
                )
                if reply.disposition == "deferred":
                    self._counts["retries"] += 1
                    self._observe_outcome(
                        span,
                        claim,
                        "deferred",
                        category=outcome.category or "none",
                        delay_ms=delay,
                    )
                    return
            result = (
                outcome.response
                if outcome.response is not None
                else {"error": outcome.category}
            )
            try:
                reply = await self._commit(
                    "finish",
                    attempt,
                    **args,
                    result=result,
                    success=outcome.response is not None,
                )
            except (ValueError, TypeError, OverflowError, RecursionError):
                outcome.response = None
                outcome.category = self._category = "invalid_response"
                result = {"error": "invalid_response"}
                reply = None
            # Leave the exception/serialization frame before retrying the bounded result.
            if reply is None:
                reply = await self._commit(
                    "finish", attempt, **args, result=result, success=False
                )
            # Cancellation/deadline/producer death can win while HTTP completes.
            if reply.disposition in {
                "existing",
                "producer_dead",
                "expired",
                "terminal",
                "held",
            }:
                settlement = await self._commit(
                    "settle_remote", attempt, **args, evidence="remote_completed"
                )
                self._observe_outcome(
                    span,
                    claim,
                    "remote_settled",
                    evidence="remote_completed",
                    settlement=settlement.disposition,
                )
            self._observe_outcome(
                span,
                claim,
                "result_committed",
                category=outcome.category
                or (
                    "none"
                    if reply.state == "succeeded"
                    else reply.state or reply.disposition
                ),
                state=reply.state,
                remote_settled=outcome.remote_settled,
            )
            self._counts["completed"] += 1
        except StaleOwner:
            self._owner_until = 0
            self._observe_outcome(
                span, claim, "interrupted", category="owner_uncertain"
            )
        finally:
            outcome = None


def _emit_execution_history(
    attributes: dict[str, Any],
    outcome: ExecutionOutcome,
    began_ns: int,
    began: float,
    *,
    parent_context: Context | None = None,
    span: Span | None = None,
) -> None:
    """Emit one physical attempt after transport returns, before result-commit retries."""
    elapsed = max(0.0, (time.monotonic() - began) * 1000)
    prefix = "marie.llm_dispatch."
    attributes[prefix + "execution_ms"] = elapsed
    attributes[prefix + "remote_settled"] = outcome.remote_settled
    attributes[prefix + "request_started"] = outcome.request_started
    attributes[prefix + "retryable"] = outcome.retryable
    attributes["openinference.span.kind"] = "LLM"
    attributes["llm.model_name"] = attributes.get(prefix + "model", "")
    attributes[prefix + "total_latency_ms"] = (
        attributes[prefix + "queue_wait_ms"] + elapsed
    )
    failed = outcome.response is None or bool(outcome.category)
    attributes[prefix + "status"] = "error" if failed else "ok"
    if outcome.category:
        attributes[prefix + "error_type"] = outcome.category
    if outcome.response is not None:
        usage = outcome.response.get("usage")
        if isinstance(usage, dict):
            for source, target in (
                ("prompt_tokens", "prompt"),
                ("completion_tokens", "completion"),
                ("total_tokens", "total"),
            ):
                count = usage.get(source)
                if type(count) is int and 0 <= count <= 2**53 - 1:
                    attributes["llm.token_count." + target] = count
    if span is None:
        span = start_trace_span(
            _tracer,
            "LLMDispatch.completion",
            context=parent_context,
            start_time=began_ns,
            attributes=attributes,
        )
    else:
        span.set_attributes(attributes)
    if not failed and span.is_recording():
        output = _completion_output_text(outcome.response)
        if output is not None:
            set_llm_io(span, output_messages=output)
    span.set_status(StatusCode.ERROR if failed else StatusCode.OK)
    span.end()
    dispatch_metrics.record_request_execution(
        pool_id=attributes.get(prefix + "pool_id", ""),
        dispatcher_id=attributes.get(prefix + "dispatcher_id", ""),
        fabric_group_id=attributes.get(prefix + "fabric_group_id", ""),
        mode="queued-dispatch",
        duration_seconds=elapsed / 1000,
        ok=not failed,
        category=outcome.category or "none",
        remote_settled=outcome.remote_settled,
        queue_wait_seconds=attributes[prefix + "queue_wait_ms"] / 1000,
    )


def _completion_output_text(response: dict[str, Any] | None) -> str | None:
    """Return deliverable assistant text, or None for truncated or malformed shapes."""
    try:
        if completion_finish_reason(response) == "length":
            return None
        _, text = extract_completion_text(response)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return None
    return text


def _normalize_length_response(
    response: dict[str, Any] | None,
) -> tuple[bool, bool]:
    if response is None:
        return False, False
    try:
        if completion_finish_reason(response) != "length":
            return False, False
        _, text = extract_completion_text(response)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return False, False
    recovered = recover_complete_json(text or "")
    if recovered is None:
        return False, has_terminal_repetition(text or "")

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False, False
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return False, False
    message["content"] = recovered
    choices[0]["finish_reason"] = "stop"
    response["marie_recovery"] = {
        "method": "complete_json",
        "reason": "missing_eos",
    }
    return True, False
