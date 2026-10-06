from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from marie.engine.completion_contract import COMPLETION_QUEUE_CONTRACT_VERSION


class DispatchRuntime(Protocol):
    def health(
        self, *, read_budget: SnapshotReadBudget | None = None
    ) -> dict[str, Any]: ...
    def sample_pending_requests(
        self, limit: int, *, read_budget: SnapshotReadBudget | None = None
    ) -> list[dict[str, Any]]: ...
    def inflight_requests_snapshot(self) -> list[dict[str, Any]]: ...


class SnapshotUnavailable(RuntimeError):
    pass


_REGISTRY_LOCK = threading.Lock()
_DISPATCHERS: dict[str, DispatchRuntime] = {}
_READ_LOCK = threading.Lock()
_READ_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-observe")
MAX_SNAPSHOT_BYTES = 262144
MAX_SNAPSHOT_ROWS = 250
MAX_SNAPSHOT_READ_UNITS = 1000
OBSERVATION_STALE_AFTER_MS = 15_000


@dataclass
class SnapshotReadBudget:
    candidates_left: int
    deadline: float
    cancel: threading.Event | None = None
    units_left: int = MAX_SNAPSHOT_READ_UNITS
    detail_rows_left: int = MAX_SNAPSHOT_ROWS
    dispatchers_left: int = 1

    def check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise SnapshotUnavailable("runtime_read_cancelled")
        if time.monotonic() >= self.deadline:
            raise SnapshotUnavailable("runtime_read_timeout")

    def before_read(self) -> None:
        self.check()
        if self.units_left <= 0:
            raise SnapshotUnavailable("snapshot_budget_exceeded")
        self.units_left -= 1

    def before_candidate(self) -> None:
        self.check()
        if self.candidates_left <= 0:
            raise SnapshotUnavailable("snapshot_budget_exceeded")
        self.candidates_left -= 1


_FIELDS = frozenset(
    """contract_version fabric_group_id pool_id request_id attempt_id
endpoint_id config_revision dispatcher_id gateway_id model lifecycle_stage state_source
submitted_at admitted_at_ms oldest_pending_admitted_at_ms popped_at expires_at_ms payload_bytes cost execution_seq next_eligible
state_updated_at queue_wait_age_seconds inflight_age_seconds
running draining role owner_generation enabled scheduler_policy quantum deficit inflight
min_concurrent max_concurrent max_burst_per_visit request_queue_depth oldest_pending_at
head_cost_units processed_batches processed_items last_processed_at last_batch_size
execution_failures malformed_requests_dropped offline_producer_requests_dropped
offline_producer_replies_dropped inflight_request_count pool_count queue_configured
total_concurrent_dispatch sampling_available metadata_unavailable
reserved_items reserved_bytes failures circuit next_probe probe_successes category open_until probe claim_id execution_limit
execution_bytes protected_slots borrowed_slots config_generation waiting_reason last_error last_error_at_ms request_queue_depth_error transport_failures
observed_at_ms policy_generation policy_digest endpoint_group_id revision replica_id group_id selected_replica_id
charged_cost refunded_cost committed_charge accepted completed drain_references oldest_pending_age_seconds gate processing_truncated""".split()
    + """ charge_sequence refund_state endpoint_count endpoint_group_count details_truncated port""".split()
)
_ERRORS = frozenset(
    """connect_refused connect_timeout timeout outcome_unknown
store_unavailable invalid_response response_too_large endpoint_policy request_rejected
rate_limited authentication call_timeout connect_error invalid_request protocol_error provider_rejected read_error read_timeout route_or_model transport_unknown write_error write_timeout connect_unknown pool_pressure provider_unavailable provider_5xx replica_unavailable unsupported_streaming v3_start_failed owner_lost cancelled expired queue_unavailable dispatch_error owner_uncertain start_unconfirmed policy_refresh_unavailable binding_conflict binding_migration_required""".split()
)

_NUMBER_FIELDS = frozenset(
    """submitted_at admitted_at_ms oldest_pending_admitted_at_ms popped_at
state_updated_at queue_wait_age_seconds inflight_age_seconds
expires_at_ms payload_bytes cost execution_seq next_eligible owner_generation quantum deficit inflight
min_concurrent max_concurrent max_burst_per_visit request_queue_depth oldest_pending_at head_cost_units
processed_batches processed_items last_processed_at last_batch_size execution_failures transport_failures
malformed_requests_dropped offline_producer_requests_dropped offline_producer_replies_dropped
inflight_request_count pool_count total_concurrent_dispatch metadata_unavailable reserved_items
reserved_bytes protected_slots borrowed_slots config_generation failures next_probe probe_successes open_until execution_limit execution_bytes last_error_at_ms""".split()
    + """ observed_at_ms policy_generation charged_cost refunded_cost committed_charge accepted completed charge_sequence drain_references oldest_pending_age_seconds""".split()
    + """ endpoint_count endpoint_group_count port""".split()
)

_STATE_NAMES = frozenset({"ready", "claimed", "executing", "outcome_unknown"})


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in _FIELDS & row.keys():
        value = row[key]
        if key in _NUMBER_FIELDS and isinstance(value, str):
            result[key] = int(value) if value.isdecimal() and len(value) <= 16 else None
        elif key == "policy_digest":
            result[key] = (
                value
                if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                else None
            )
        elif key == "model" and isinstance(value, str):
            result[key] = (
                value
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value)
                and not value.lower().startswith(("data:", "http:", "https:"))
                else None
            )
        elif key in {"last_error", "request_queue_depth_error", "category"}:
            result[key] = (
                value if value in _ERRORS or value is None else "runtime_error"
            )
        elif value is None or isinstance(value, (bool, int, float)):
            result[key] = value
        elif isinstance(value, str) and len(value) <= 128:
            result[key] = (
                None
                if value.lower().startswith(("data:", "http:", "https:"))
                else value
            )
    for key in ("counters", "usage", "skip_counts"):
        if isinstance(row.get(key), dict):
            result[key] = {
                name: value
                for name, value in list(row[key].items())[:64]
                if isinstance(name, str)
                and name.replace("_", "").isalnum()
                and len(name) <= 64
                and type(value) is int
            }
    return result


def _clean_state_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result = {
        {"executing": "running", "outcome_unknown": "unknown"}.get(name, name): count
        for name, count in value.items()
        if name in _STATE_NAMES and type(count) is int and count >= 0
    }
    return {
        name: result.get(name, 0) for name in ("ready", "claimed", "running", "unknown")
    }


def _clean_replica(row: dict[str, Any]) -> dict[str, Any]:
    clean = _clean(row)
    reserved_items = clean.get("reserved_items")
    execution_limit = clean.get("execution_limit")
    reserved_bytes = clean.get("reserved_bytes")
    execution_bytes = clean.get("execution_bytes")
    if isinstance(reserved_items, int) and isinstance(execution_limit, int):
        clean["available_items"] = max(0, execution_limit - reserved_items)
    if isinstance(reserved_bytes, int) and isinstance(execution_bytes, int):
        clean["available_bytes"] = max(0, execution_bytes - reserved_bytes)
    return clean


def _clean_group(row: dict[str, Any], limit: int) -> dict[str, Any]:
    clean = _clean(row)
    replicas = row.get("replicas")
    if isinstance(replicas, list):
        clean["replicas"] = [
            _clean_replica(replica)
            for replica in replicas[:limit]
            if isinstance(replica, dict)
        ]
        clean["replica_count"] = len(replicas)
        clean["replicas_truncated"] = len(replicas) > limit
    return clean


def _fabric(dispatcher: DispatchRuntime) -> str:
    store = getattr(dispatcher, "store", None)
    if store is not None:
        return store.keys.fabric_id
    return str(
        getattr(getattr(dispatcher, "config", None), "fabric_group_id", "") or ""
    )


def register_dispatcher(dispatcher_id: str, dispatcher: DispatchRuntime) -> None:
    with _REGISTRY_LOCK:
        _DISPATCHERS[dispatcher_id] = dispatcher


def unregister_dispatcher(dispatcher_id: str) -> None:
    with _REGISTRY_LOCK:
        _DISPATCHERS.pop(dispatcher_id, None)


def dispatch_runtime_live_state(
    limit_per_pool: int = 50,
    *,
    fabric_group_id: str | None = None,
    _cancel: threading.Event | None = None,
    _deadline: float | None = None,
) -> dict[str, Any]:
    limit = max(1, min(limit_per_pool, MAX_SNAPSHOT_ROWS))
    with _REGISTRY_LOCK:
        items = [
            (identity, runtime)
            for identity, runtime in _DISPATCHERS.items()
            if fabric_group_id is None or _fabric(runtime) == fabric_group_id
        ]
    if len(items) > MAX_SNAPSHOT_ROWS:
        raise SnapshotUnavailable("snapshot_budget_exceeded")
    read_budget = SnapshotReadBudget(
        min(limit, max(0, MAX_SNAPSHOT_ROWS - max(1, len(items)))),
        _deadline if _deadline is not None else time.monotonic() + 1.0,
        _cancel,
    )
    rows_left, bytes_left = MAX_SNAPSHOT_ROWS, MAX_SNAPSHOT_BYTES

    def bounded(row: dict[str, Any], count: int = 1) -> dict[str, Any]:
        nonlocal rows_left, bytes_left
        rows_left -= count
        bytes_left -= len(json.dumps(row, allow_nan=False).encode())
        if rows_left < 0 or bytes_left < 0:
            raise SnapshotUnavailable("snapshot_budget_exceeded")
        return row

    dispatchers, pools, requests, endpoint_groups = [], [], [], []
    counted, sampled = set(), set()
    pending = 0
    available = True
    for index, (identity, runtime) in enumerate(items):
        read_budget.check()
        read_budget.detail_rows_left = max(0, rows_left - read_budget.candidates_left)
        read_budget.dispatchers_left = len(items) - index
        try:
            raw = runtime.health(read_budget=read_budget)
        except Exception:
            raise SnapshotUnavailable("runtime_read_failed") from None
        if raw.get("request_queue_depth_error"):
            raise SnapshotUnavailable("runtime_read_failed")
        health = _clean(raw)
        health["dispatcher_id"] = identity
        fabric = _fabric(runtime)
        health["fabric_group_id"] = fabric
        raw_lanes = raw.get("lanes")
        lanes = raw_lanes if isinstance(raw_lanes, list) else [raw]
        clean_lanes = []
        for lane in lanes:
            read_budget.check()
            if len(pools) >= MAX_SNAPSHOT_ROWS:
                raise SnapshotUnavailable("snapshot_budget_exceeded")
            if "request_queue_depth" in lane and lane["request_queue_depth"] is None:
                raise SnapshotUnavailable("runtime_read_failed")
            clean = _clean(lane)
            state_counts = _clean_state_counts(lane.get("state_counts"))
            if state_counts:
                clean["state_counts"] = state_counts
            if isinstance(clean.get("oldest_pending_age_seconds"), (int, float)):
                clean["oldest_ready_age_seconds"] = clean.pop(
                    "oldest_pending_age_seconds"
                )
            clean["fabric_group_id"] = fabric
            clean["scheduler_policy"] = health.get("scheduler_policy", "fifo")
            clean["gateway_id"] = health.get("gateway_id")
            key = (fabric, clean.get("pool_id"))
            if key not in counted:
                counted.add(key)
                pools.append(bounded(clean))
                depth = clean.get("request_queue_depth")
                if isinstance(depth, int):
                    pending += depth
                else:
                    available = False
            clean_lanes.append(clean)
        if isinstance(raw_lanes, list):
            health["lanes"] = clean_lanes
        health["endpoints"] = {
            name: _clean(state)
            for name, state in list((raw.get("endpoints") or {}).items())[
                :MAX_SNAPSHOT_ROWS
            ]
        }
        for group in (raw.get("endpoint_groups") or [])[:MAX_SNAPSHOT_ROWS]:
            if not isinstance(group, dict):
                continue
            cleaned_group = _clean_group(group, limit)
            cleaned_group["fabric_group_id"] = fabric
            endpoint_groups.append(
                bounded(cleaned_group, 1 + len(cleaned_group.get("replicas", [])))
            )
        dispatchers.append(
            bounded(health, 1 + len(health.get("lanes", [])) + len(health["endpoints"]))
        )
        sample_key = (fabric, tuple(lane.get("pool_id") for lane in lanes))
        try:
            read_budget.check()
            if (
                read_budget.candidates_left > 0
                and len(requests) < limit
                and sample_key not in sampled
            ):
                sampled.add(sample_key)
                requests.extend(
                    bounded(_clean(row))
                    for row in runtime.sample_pending_requests(
                        min(read_budget.candidates_left, limit - len(requests)),
                        read_budget=read_budget,
                    )[: limit - len(requests)]
                )
            read_budget.check()
            if read_budget.candidates_left > 0 and len(requests) < limit:
                for row in runtime.inflight_requests_snapshot()[
                    : min(read_budget.candidates_left, limit - len(requests))
                ]:
                    read_budget.before_candidate()
                    requests.append(bounded(_clean(row)))
        except Exception:
            raise SnapshotUnavailable("runtime_read_failed") from None
        if hasattr(runtime, "metadata_unavailable"):
            health["metadata_unavailable"] = runtime.metadata_unavailable
    versions = {
        row.get("contract_version", COMPLETION_QUEUE_CONTRACT_VERSION)
        for row in dispatchers
    }
    observed_generations = {
        row.get("policy_generation")
        for row in dispatchers
        if isinstance(row.get("policy_generation"), int)
    }
    complete_generation_observation = sum(
        isinstance(row.get("policy_generation"), int) for row in dispatchers
    ) == len(dispatchers)
    observed_digests = {
        row.get("policy_digest")
        for row in dispatchers
        if isinstance(row.get("policy_digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", row["policy_digest"])
    }
    complete_digest_observation = sum(
        isinstance(row.get("policy_digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", row["policy_digest"]) is not None
        for row in dispatchers
    ) == len(dispatchers)
    observed_times = [
        int(row["observed_at_ms"])
        for row in dispatchers
        if isinstance(row.get("observed_at_ms"), (int, float))
    ]
    observed_at_ms = min(observed_times) if observed_times else int(time.time() * 1000)
    drr = {
        field: sum(
            int(pool.get(field) or 0)
            for pool in pools
            if isinstance(pool.get(field), int)
        )
        for field in ("charged_cost", "refunded_cost", "committed_charge")
    }
    result = {
        "contract_version": (
            next(iter(versions))
            if len(versions) == 1
            else "mixed"
            if versions
            else COMPLETION_QUEUE_CONTRACT_VERSION
        ),
        "fabric_group_id": fabric_group_id,
        "observation": {
            "observed_at_ms": observed_at_ms,
            "stale": int(time.time() * 1000) - observed_at_ms
            > OBSERVATION_STALE_AFTER_MS,
            "stale_after_ms": OBSERVATION_STALE_AFTER_MS,
        },
        "policy": {
            "observed_generation": (
                next(iter(observed_generations))
                if dispatchers
                and complete_generation_observation
                and len(observed_generations) == 1
                else None
            ),
            "observed_digest": (
                next(iter(observed_digests))
                if dispatchers
                and complete_digest_observation
                and len(observed_digests) == 1
                else None
            ),
        },
        "drr": drr,
        "runtime_summary": {
            "registered_dispatchers": len(dispatchers),
            "running_dispatchers": sum(bool(row.get("running")) for row in dispatchers),
            "pending_request_count": pending if available else None,
            "pending_request_sample_count": sum(
                row.get("lifecycle_stage") in {"pending", "ready"} for row in requests
            ),
            "inflight_request_count": sum(
                int(row.get("inflight_request_count") or 0) for row in dispatchers
            ),
            "live_request_sample_limit": limit,
            "metadata_available": available,
            "sampling_available": all(
                row.get("contract_version") == "v3" for row in dispatchers
            ),
            **{
                name: sum(int(row.get(name) or 0) for row in dispatchers)
                for name in (
                    "execution_failures",
                    "malformed_requests_dropped",
                    "offline_producer_requests_dropped",
                    "offline_producer_replies_dropped",
                )
            },
        },
        "pool_config": pools,
        "pools": pools,
        "pool_count": max(
            [
                len(pools),
                *(int(row.get("pool_count") or 0) for row in dispatchers),
            ]
        ),
        "pools_truncated": any(
            bool(row.get("details_truncated")) for row in dispatchers
        ),
        "endpoint_groups": endpoint_groups,
        "endpoint_group_count": max(
            [
                len(endpoint_groups),
                *(int(row.get("endpoint_group_count") or 0) for row in dispatchers),
            ]
        ),
        "endpoint_groups_truncated": any(
            bool(row.get("details_truncated")) for row in dispatchers
        ),
        "live_requests": requests,
        "dispatchers": dispatchers,
    }
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_SNAPSHOT_BYTES:
        raise SnapshotUnavailable("snapshot_budget_exceeded")
    return result


def routing_resource_references(
    *,
    fabric_group_id: str,
    resource_type: str,
    resource_id: str,
    revision: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Read bounded Valkey references and fail closed when no store can answer."""
    with _REGISTRY_LOCK:
        runtimes = [
            runtime
            for runtime in _DISPATCHERS.values()
            if _fabric(runtime) == fabric_group_id
            and hasattr(runtime, "routing_resource_references")
        ]
    unique = []
    seen_stores: set[int] = set()
    for runtime in runtimes:
        store_identity = id(getattr(runtime, "store", runtime))
        if store_identity not in seen_stores:
            seen_stores.add(store_identity)
            unique.append(runtime)
    if not unique:
        raise SnapshotUnavailable("routing_reference_store_unavailable")
    totals = {"valkey": 0, "ready": 0, "active": 0}
    truncated = False
    for runtime in unique:
        try:
            result = runtime.routing_resource_references(
                resource_type=resource_type,
                resource_id=resource_id,
                revision=revision,
                limit=limit,
            )
        except Exception:
            raise SnapshotUnavailable("routing_reference_store_unavailable") from None
        for name in totals:
            value = result.get(name)
            if type(value) is not int or value < 0:
                raise SnapshotUnavailable("routing_reference_store_unavailable")
            totals[name] += value
        truncated = truncated or bool(result.get("truncated"))
    return {**totals, "truncated": truncated}


def dispatch_runtime_snapshot(*, fabric_group_id: str | None = None) -> dict[str, Any]:
    state = dispatch_runtime_live_state(fabric_group_id=fabric_group_id)
    return {
        "contract_version": state["contract_version"],
        "registered_dispatchers": state["runtime_summary"]["registered_dispatchers"],
        "running_dispatchers": state["runtime_summary"]["running_dispatchers"],
        "dispatchers": state["dispatchers"],
    }


async def read_runtime_snapshot(
    *, fabric_group_id: str, limit: int = 50, timeout_seconds: float = 1.0
) -> dict[str, Any]:
    """Bound concurrent work; timed-out reads finish off-loop before another starts."""
    if not _READ_LOCK.acquire(blocking=False):
        raise SnapshotUnavailable("runtime_read_busy")

    cancel = threading.Event()
    deadline = time.monotonic() + timeout_seconds

    def read() -> dict[str, Any]:
        try:
            return dispatch_runtime_live_state(
                limit,
                fabric_group_id=fabric_group_id,
                _cancel=cancel,
                _deadline=deadline,
            )
        except Exception:
            raise SnapshotUnavailable("runtime_read_failed") from None
        finally:
            _READ_LOCK.release()

    future = asyncio.get_running_loop().run_in_executor(_READ_WORKER, read)
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout_seconds)
    except (TimeoutError, asyncio.CancelledError) as exc:
        cancel.set()
        future.add_done_callback(
            lambda done: done.exception() if not done.cancelled() else None
        )
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise SnapshotUnavailable("runtime_read_timeout") from None
