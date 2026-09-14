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
_READ_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='llm-observe')
MAX_SNAPSHOT_BYTES = 262144
MAX_SNAPSHOT_ROWS = 250
MAX_SNAPSHOT_READ_UNITS = 1000


@dataclass
class SnapshotReadBudget:
    candidates_left: int
    deadline: float
    cancel: threading.Event | None = None
    units_left: int = MAX_SNAPSHOT_READ_UNITS

    def check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise SnapshotUnavailable('runtime_read_cancelled')
        if time.monotonic() >= self.deadline:
            raise SnapshotUnavailable('runtime_read_timeout')

    def before_read(self) -> None:
        self.check()
        if self.units_left <= 0:
            raise SnapshotUnavailable('snapshot_budget_exceeded')
        self.units_left -= 1

    def before_candidate(self) -> None:
        self.check()
        if self.candidates_left <= 0:
            raise SnapshotUnavailable('snapshot_budget_exceeded')
        self.candidates_left -= 1


_FIELDS = frozenset(
    '''contract_version fabric_group_id pool_id request_id attempt_id
endpoint_id config_revision dispatcher_id gateway_id model lifecycle_stage state_source
submitted_at admitted_at_ms oldest_pending_admitted_at_ms popped_at expires_at_ms payload_bytes cost execution_seq next_eligible
running draining role owner_generation enabled scheduler_policy quantum deficit inflight
min_concurrent max_concurrent max_burst_per_visit request_queue_depth oldest_pending_at
head_cost_units processed_batches processed_items last_processed_at last_batch_size
execution_failures malformed_requests_dropped offline_producer_requests_dropped
offline_producer_replies_dropped inflight_request_count pool_count queue_configured
total_concurrent_dispatch sampling_available metadata_unavailable
reserved_items reserved_bytes failures circuit next_probe probe_successes category open_until probe claim_id execution_limit
execution_bytes protected_slots borrowed_slots config_generation waiting_reason last_error last_error_at_ms request_queue_depth_error transport_failures'''.split()
)
_ERRORS = frozenset(
    '''connect_refused connect_timeout timeout outcome_unknown
store_unavailable invalid_response response_too_large endpoint_policy request_rejected
rate_limited authentication call_timeout connect_error invalid_request protocol_error provider_rejected read_error read_timeout route_or_model transport_unknown write_error write_timeout connect_unknown pool_pressure provider_unavailable v3_start_failed owner_lost cancelled expired queue_unavailable dispatch_error owner_uncertain start_unconfirmed policy_refresh_unavailable binding_conflict binding_migration_required'''.split()
)

_NUMBER_FIELDS = frozenset(
    """submitted_at admitted_at_ms oldest_pending_admitted_at_ms popped_at
expires_at_ms payload_bytes cost execution_seq next_eligible owner_generation quantum deficit inflight
min_concurrent max_concurrent max_burst_per_visit request_queue_depth oldest_pending_at head_cost_units
processed_batches processed_items last_processed_at last_batch_size execution_failures transport_failures
malformed_requests_dropped offline_producer_requests_dropped offline_producer_replies_dropped
inflight_request_count pool_count total_concurrent_dispatch metadata_unavailable reserved_items
reserved_bytes protected_slots borrowed_slots config_generation failures next_probe probe_successes open_until execution_limit execution_bytes last_error_at_ms""".split()
)


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in _FIELDS & row.keys():
        value = row[key]
        if key in _NUMBER_FIELDS and isinstance(value, str):
            result[key] = int(value) if value.isdecimal() and len(value) <= 16 else None
        elif key == 'model' and isinstance(value, str):
            result[key] = (
                value
                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}', value)
                and not value.lower().startswith(('data:', 'http:', 'https:'))
                else None
            )
        elif key in {'last_error', 'request_queue_depth_error', 'category'}:
            result[key] = (
                value if value in _ERRORS or value is None else 'runtime_error'
            )
        elif value is None or isinstance(value, (bool, int, float)):
            result[key] = value
        elif isinstance(value, str) and len(value) <= 128:
            result[key] = (
                None
                if value.lower().startswith(('data:', 'http:', 'https:'))
                else value
            )
    for key in ('counters', 'usage', 'skip_counts'):
        if isinstance(row.get(key), dict):
            result[key] = {
                name: value
                for name, value in list(row[key].items())[:64]
                if isinstance(name, str)
                and name.replace('_', '').isalnum()
                and len(name) <= 64
                and type(value) is int
            }
    return result


def _fabric(dispatcher: DispatchRuntime) -> str:
    store = getattr(dispatcher, 'store', None)
    if store is not None:
        return store.keys.fabric_id
    return str(
        getattr(getattr(dispatcher, 'config', None), 'fabric_group_id', '') or ''
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
    read_budget = SnapshotReadBudget(
        limit, _deadline if _deadline is not None else time.monotonic() + 1.0, _cancel
    )
    with _REGISTRY_LOCK:
        items = [
            (identity, runtime)
            for identity, runtime in _DISPATCHERS.items()
            if fabric_group_id is None or _fabric(runtime) == fabric_group_id
        ]
    if len(items) > MAX_SNAPSHOT_ROWS:
        raise SnapshotUnavailable('snapshot_budget_exceeded')
    rows_left, bytes_left = MAX_SNAPSHOT_ROWS, MAX_SNAPSHOT_BYTES

    def bounded(row: dict[str, Any], count: int = 1) -> dict[str, Any]:
        nonlocal rows_left, bytes_left
        rows_left -= count
        bytes_left -= len(json.dumps(row, allow_nan=False).encode())
        if rows_left < 0 or bytes_left < 0:
            raise SnapshotUnavailable('snapshot_budget_exceeded')
        return row

    dispatchers, pools, requests = [], [], []
    counted, sampled = set(), set()
    pending = 0
    available = True
    for identity, runtime in items:
        read_budget.check()
        try:
            raw = runtime.health(read_budget=read_budget)
        except Exception:
            raise SnapshotUnavailable('runtime_read_failed') from None
        if raw.get('request_queue_depth_error'):
            raise SnapshotUnavailable('runtime_read_failed')
        health = _clean(raw)
        health['dispatcher_id'] = identity
        fabric = _fabric(runtime)
        health['fabric_group_id'] = fabric
        lanes = raw.get('lanes') or [raw]
        clean_lanes = []
        for lane in lanes:
            read_budget.check()
            if len(pools) >= MAX_SNAPSHOT_ROWS:
                raise SnapshotUnavailable('snapshot_budget_exceeded')
            if 'request_queue_depth' in lane and lane['request_queue_depth'] is None:
                raise SnapshotUnavailable('runtime_read_failed')
            clean = _clean(lane)
            clean['fabric_group_id'] = fabric
            clean['scheduler_policy'] = health.get('scheduler_policy', 'fifo')
            clean['gateway_id'] = health.get('gateway_id')
            key = (fabric, clean.get('pool_id'))
            if key not in counted:
                counted.add(key)
                pools.append(bounded(clean))
                depth = clean.get('request_queue_depth')
                if isinstance(depth, int):
                    pending += depth
                else:
                    available = False
            clean_lanes.append(clean)
        if raw.get('lanes'):
            health['lanes'] = clean_lanes
        health['endpoints'] = {
            name: _clean(state)
            for name, state in list((raw.get('endpoints') or {}).items())[
                :MAX_SNAPSHOT_ROWS
            ]
        }
        dispatchers.append(
            bounded(health, 1 + len(health.get('lanes', [])) + len(health['endpoints']))
        )
        sample_key = (fabric, tuple(lane.get('pool_id') for lane in lanes))
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
            raise SnapshotUnavailable('runtime_read_failed') from None
        if hasattr(runtime, 'metadata_unavailable'):
            health['metadata_unavailable'] = runtime.metadata_unavailable
    versions = {
        row.get('contract_version', COMPLETION_QUEUE_CONTRACT_VERSION)
        for row in dispatchers
    }
    result = {
        'contract_version': (
            next(iter(versions))
            if len(versions) == 1
            else 'mixed'
            if versions
            else COMPLETION_QUEUE_CONTRACT_VERSION
        ),
        'fabric_group_id': fabric_group_id,
        'runtime_summary': {
            'registered_dispatchers': len(dispatchers),
            'running_dispatchers': sum(bool(row.get('running')) for row in dispatchers),
            'pending_request_count': pending if available else None,
            'pending_request_sample_count': sum(
                row.get('lifecycle_stage') in {'pending', 'ready'} for row in requests
            ),
            'inflight_request_count': sum(
                int(row.get('inflight_request_count') or 0) for row in dispatchers
            ),
            'live_request_sample_limit': limit,
            'metadata_available': available,
            'sampling_available': all(
                row.get('contract_version') == 'v3' for row in dispatchers
            ),
            **{
                name: sum(int(row.get(name) or 0) for row in dispatchers)
                for name in (
                    'execution_failures',
                    'malformed_requests_dropped',
                    'offline_producer_requests_dropped',
                    'offline_producer_replies_dropped',
                )
            },
        },
        'pool_config': pools,
        'live_requests': requests,
        'dispatchers': dispatchers,
    }
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_SNAPSHOT_BYTES:
        raise SnapshotUnavailable('snapshot_budget_exceeded')
    return result


def dispatch_runtime_snapshot(*, fabric_group_id: str | None = None) -> dict[str, Any]:
    state = dispatch_runtime_live_state(fabric_group_id=fabric_group_id)
    return {
        'contract_version': state['contract_version'],
        'registered_dispatchers': state['runtime_summary']['registered_dispatchers'],
        'running_dispatchers': state['runtime_summary']['running_dispatchers'],
        'dispatchers': state['dispatchers'],
    }


async def read_runtime_snapshot(
    *, fabric_group_id: str, limit: int = 50, timeout_seconds: float = 1.0
) -> dict[str, Any]:
    """Bound concurrent work; timed-out reads finish off-loop before another starts."""
    if not _READ_LOCK.acquire(blocking=False):
        raise SnapshotUnavailable('runtime_read_busy')

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
            raise SnapshotUnavailable('runtime_read_failed') from None
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
        raise SnapshotUnavailable('runtime_read_timeout') from None
