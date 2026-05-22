from __future__ import annotations

import threading
from typing import Any, Protocol

from marie.engine.completion_contract import COMPLETION_QUEUE_CONTRACT_VERSION


class DispatchRuntime(Protocol):
    def health(self) -> dict[str, Any]: ...

    def sample_pending_requests(self, limit: int) -> list[dict[str, Any]]: ...

    def inflight_requests_snapshot(self) -> list[dict[str, Any]]: ...


_REGISTRY_LOCK = threading.Lock()
_DISPATCHERS: dict[str, DispatchRuntime] = {}


def register_dispatcher(dispatcher_id: str, dispatcher: DispatchRuntime) -> None:
    with _REGISTRY_LOCK:
        _DISPATCHERS[dispatcher_id] = dispatcher


def unregister_dispatcher(dispatcher_id: str) -> None:
    with _REGISTRY_LOCK:
        _DISPATCHERS.pop(dispatcher_id, None)


def dispatch_runtime_snapshot() -> dict[str, Any]:
    with _REGISTRY_LOCK:
        dispatcher_items = list(_DISPATCHERS.items())

    dispatchers: list[dict[str, Any]] = []
    running_dispatchers = 0
    for dispatcher_id, dispatcher in dispatcher_items:
        try:
            health = dict(dispatcher.health())
        except Exception as exc:  # pragma: no cover - defensive
            health = {
                "dispatcher_id": dispatcher_id,
                "running": False,
                "last_error": f"health() failed: {exc}",
            }
        health.setdefault("dispatcher_id", dispatcher_id)
        if health.get("running"):
            running_dispatchers += 1
        dispatchers.append(health)

    dispatchers.sort(
        key=lambda item: (
            str(item.get("pool_id") or ""),
            str(item.get("dispatcher_id") or ""),
        )
    )
    return {
        "contract_version": COMPLETION_QUEUE_CONTRACT_VERSION,
        "registered_dispatchers": len(dispatchers),
        "running_dispatchers": running_dispatchers,
        "dispatchers": dispatchers,
    }


def dispatch_runtime_live_state(limit_per_pool: int = 50) -> dict[str, Any]:
    snapshot = dispatch_runtime_snapshot()

    with _REGISTRY_LOCK:
        dispatchers = list(_DISPATCHERS.values())

    pending_requests: list[dict[str, Any]] = []
    inflight_requests: list[dict[str, Any]] = []
    sampled_dispatch_keys: set[str] = set()
    pending_sample_counts_by_pool: dict[str, int] = {}

    for dispatcher in dispatchers:
        try:
            health = dispatcher.health()
        except Exception:  # pragma: no cover - defensive
            continue

        dispatcher_id = str(health.get("dispatcher_id") or "")
        pool_id = str(health.get("pool_id") or "")
        scheduler_policy = str(health.get("scheduler_policy") or "fifo")
        sample_key = (
            f"dispatcher:{dispatcher_id}"
            if scheduler_policy == "drr"
            else f"pool:{pool_id}"
        )
        if pool_id and sample_key not in sampled_dispatch_keys:
            sampled_dispatch_keys.add(sample_key)
            try:
                pool_requests = dispatcher.sample_pending_requests(limit_per_pool)
                pending_requests.extend(pool_requests)
                pending_sample_counts_by_pool[pool_id] = len(pool_requests)
            except Exception:
                pass

        try:
            inflight_requests.extend(dispatcher.inflight_requests_snapshot())
        except Exception:
            pass

    live_requests = pending_requests + inflight_requests
    live_requests.sort(
        key=lambda item: (
            str(item.get("state_source") or ""),
            float(item.get("submitted_at") or 0.0),
            str(item.get("request_id") or ""),
        )
    )

    dispatcher_rows = snapshot["dispatchers"]
    pending_request_count = 0
    counted_pools: set[str] = set()
    for item in dispatcher_rows:
        pool_id = str(item.get("pool_id") or "")
        if not pool_id or pool_id in counted_pools:
            continue
        counted_pools.add(pool_id)
        queue_depth = item.get("request_queue_depth")
        if isinstance(queue_depth, int):
            pending_request_count += queue_depth
        else:
            pending_request_count += pending_sample_counts_by_pool.get(pool_id, 0)

    pending_request_sample_count = len(pending_requests)
    return {
        "contract_version": COMPLETION_QUEUE_CONTRACT_VERSION,
        "runtime_summary": {
            "registered_dispatchers": snapshot["registered_dispatchers"],
            "running_dispatchers": snapshot["running_dispatchers"],
            "pending_request_count": pending_request_count,
            "pending_request_sample_count": pending_request_sample_count,
            "inflight_request_count": len(inflight_requests),
            "live_request_sample_limit_per_pool": limit_per_pool,
            "execution_failures": sum(
                int(item.get("execution_failures") or 0) for item in dispatcher_rows
            ),
            "malformed_requests_dropped": sum(
                int(item.get("malformed_requests_dropped") or 0)
                for item in dispatcher_rows
            ),
            "offline_producer_requests_dropped": sum(
                int(item.get("offline_producer_requests_dropped") or 0)
                for item in dispatcher_rows
            ),
            "offline_producer_replies_dropped": sum(
                int(item.get("offline_producer_replies_dropped") or 0)
                for item in dispatcher_rows
            ),
        },
        "pool_config": _pool_config_rows(dispatcher_rows),
        "live_requests": live_requests,
        "dispatchers": dispatcher_rows,
    }


def _pool_config_rows(dispatchers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dispatcher in dispatchers:
        scheduler_policy = str(dispatcher.get("scheduler_policy") or "fifo")
        if scheduler_policy == "drr":
            for lane in dispatcher.get("lanes") or []:
                rows.append(
                    {
                        "fabric_group_id": dispatcher.get("fabric_group_id"),
                        "gateway_id": dispatcher.get("gateway_id"),
                        "dispatcher_id": dispatcher.get("dispatcher_id"),
                        "scheduler_policy": scheduler_policy,
                        "pool_id": lane.get("pool_id"),
                        "display_name": lane.get("display_name"),
                        "enabled": True,
                        "quantum": lane.get("quantum"),
                        "deficit": lane.get("deficit"),
                        "inflight": lane.get("inflight"),
                        "request_queue_depth": lane.get("request_queue_depth"),
                        "oldest_pending_age_seconds": lane.get(
                            "oldest_pending_age_seconds"
                        ),
                        "head_cost_units": lane.get("head_cost_units"),
                        "min_concurrent": lane.get("min_concurrent"),
                        "max_concurrent": lane.get("max_concurrent"),
                        "max_burst_per_visit": lane.get("max_burst_per_visit"),
                        "endpoint_url": lane.get("endpoint_url"),
                    }
                )
            continue

        pool_id = dispatcher.get("pool_id")
        if not pool_id:
            continue
        rows.append(
            {
                "fabric_group_id": dispatcher.get("fabric_group_id"),
                "gateway_id": dispatcher.get("gateway_id"),
                "dispatcher_id": dispatcher.get("dispatcher_id"),
                "scheduler_policy": scheduler_policy,
                "pool_id": pool_id,
                "display_name": None,
                "enabled": bool(dispatcher.get("enabled", True)),
                "quantum": None,
                "deficit": None,
                "inflight": dispatcher.get("inflight_request_count"),
                "request_queue_depth": dispatcher.get("request_queue_depth"),
                "oldest_pending_age_seconds": None,
                "head_cost_units": None,
                "min_concurrent": None,
                "max_concurrent": None,
                "max_burst_per_visit": None,
                "endpoint_url": dispatcher.get("endpoint_url"),
            }
        )

    rows.sort(
        key=lambda item: (
            str(item.get("fabric_group_id") or ""),
            str(item.get("pool_id") or ""),
        )
    )
    return rows
