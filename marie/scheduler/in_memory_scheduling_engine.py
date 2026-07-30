import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from marie.scheduler.execution_planner import FlatJob
from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.util import (
    executor_name,
    frontier_candidate_window,
    is_control_flow_entrypoint,
)
from marie.utils.scheduler_trace import scheduler_trace


@dataclass(frozen=True, slots=True)
class InMemorySelection:
    """One regular-job selection result from a coherent frontier snapshot."""

    candidates: tuple[WorkInfo, ...]
    ranked: tuple[FlatJob, ...]
    requested: tuple[FlatJob, ...]
    selected: tuple[WorkInfo, ...]
    candidate_window: int
    slots_by_executor: tuple[tuple[str, int], ...]
    eligible_by_executor: tuple[tuple[str, int], ...]
    captured_by_executor: tuple[tuple[str, int], ...]
    eligible_by_dag: tuple[tuple[str, int], ...]
    captured_by_dag: tuple[tuple[str, int], ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(wi.id for wi in self.candidates)

    @property
    def ranked_ids(self) -> tuple[str, ...]:
        return tuple(wi.id for _, wi in self.ranked)

    @property
    def requested_ids(self) -> tuple[str, ...]:
        return tuple(wi.id for _, wi in self.requested)


@dataclass(frozen=True, slots=True)
class _SelectionSample:
    observed_at: float
    elapsed_ms: float
    outcome: str
    reason: str | None
    candidate_window: int
    dispatch_capacity: int
    candidates: int
    requested: int
    selected: int


class InMemorySchedulingEngine:
    """Own regular-job capture, ranking, capacity capping, and local take."""

    def __init__(
        self,
        frontier: MemoryFrontier,
        *,
        sla_priority_interval_seconds: int,
    ) -> None:
        self._frontier = frontier
        self._planner = GlobalPriorityExecutionPlanner(
            sla_priority_interval_seconds=sla_priority_interval_seconds
        )
        self._selection_samples: deque[_SelectionSample] = deque(maxlen=256)

    def diagnostics(self, window_seconds: float = 300.0) -> dict[str, Any]:
        """Return bounded rolling selection latency and volume diagnostics."""
        now = time.time()
        samples = [
            sample
            for sample in self._selection_samples
            if now - sample.observed_at <= window_seconds
        ]
        if not samples:
            return {
                "window_seconds": window_seconds,
                "sample_count": 0,
                "observed_seconds": 0.0,
                "cycles_per_second": None,
                "selected_per_second": None,
                "totals": {"candidates": 0, "requested": 0, "selected": 0},
                "latency_ms": {"p50": None, "p95": None, "max": None},
                "last": None,
            }

        observed_seconds = max(0.0, samples[-1].observed_at - samples[0].observed_at)
        rate_window = observed_seconds if observed_seconds >= 1.0 else None
        latencies = sorted(sample.elapsed_ms for sample in samples)
        selected = sum(sample.selected for sample in samples)
        last = samples[-1]
        return {
            "window_seconds": window_seconds,
            "sample_count": len(samples),
            "observed_seconds": observed_seconds,
            "cycles_per_second": (
                len(samples) / rate_window if rate_window is not None else None
            ),
            "selected_per_second": (
                selected / rate_window if rate_window is not None else None
            ),
            "totals": {
                "candidates": sum(sample.candidates for sample in samples),
                "requested": sum(sample.requested for sample in samples),
                "selected": selected,
            },
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": latencies[-1],
            },
            "last": {
                "age_seconds": max(0.0, now - last.observed_at),
                "outcome": last.outcome,
                "reason": last.reason,
                "candidate_window": last.candidate_window,
                "dispatch_capacity": last.dispatch_capacity,
                "candidates": last.candidates,
                "requested": last.requested,
                "selected": last.selected,
            },
        }

    def _record_selection(
        self,
        *,
        elapsed_ms: float,
        outcome: str,
        reason: str | None,
        candidate_window: int,
        dispatch_capacity: int,
        candidates: int,
        requested: int,
        selected: int,
    ) -> None:
        self._selection_samples.append(
            _SelectionSample(
                observed_at=time.time(),
                elapsed_ms=elapsed_ms,
                outcome=outcome,
                reason=reason,
                candidate_window=candidate_window,
                dispatch_capacity=dispatch_capacity,
                candidates=candidates,
                requested=requested,
                selected=selected,
            )
        )

    async def select_ready(
        self,
        *,
        slots_by_executor: dict[str, int],
        batch_size: int,
        dispatch_capacity: int,
        lease_ttl: float,
        resident_dag_ids: set[str],
        max_resident_dags: int,
        now: datetime | None = None,
    ) -> InMemorySelection:
        """Capture and locally claim one bounded batch of regular jobs."""
        slots = {
            executor: max(0, int(count))
            for executor, count in slots_by_executor.items()
        }
        slot_items = tuple(sorted(slots.items()))
        candidate_window = frontier_candidate_window(batch_size, slots)
        selection_started_at = time.time()
        selection_started = time.perf_counter()
        scheduler_trace(
            "scheduler_selection_started",
            candidate_window=candidate_window,
            dispatch_capacity=dispatch_capacity,
            slots_by_executor=dict(slot_items),
            resident_dags=len(resident_dag_ids),
            max_resident_dags=max_resident_dags,
        )

        if dispatch_capacity <= 0 or not any(slots.values()):
            reason = (
                "dispatch_capacity" if dispatch_capacity <= 0 else "executor_capacity"
            )
            elapsed_ms = (time.perf_counter() - selection_started) * 1000.0
            scheduler_trace(
                "scheduler_selection_completed",
                outcome="skipped",
                reason=reason,
                elapsed_ms=elapsed_ms,
                candidate_window=candidate_window,
                dispatch_capacity=dispatch_capacity,
                slots_by_executor=dict(slot_items),
                candidates=0,
                ranked=0,
                requested=0,
                selected=0,
                job_ids=[],
            )
            self._record_selection(
                elapsed_ms=elapsed_ms,
                outcome="skipped",
                reason=reason,
                candidate_window=candidate_window,
                dispatch_capacity=dispatch_capacity,
                candidates=0,
                requested=0,
                selected=0,
            )
            return InMemorySelection(
                (), (), (), (), candidate_window, slot_items, (), (), (), ()
            )

        has_resident_capacity = len(resident_dag_ids) < max_resident_dags

        def eligible(wi: WorkInfo) -> bool:
            metadata = wi.data.get("metadata", {}) if isinstance(wi.data, dict) else {}
            entrypoint = metadata.get("on", "") if isinstance(metadata, dict) else ""
            executor = executor_name(entrypoint)
            return (
                bool(executor)
                and not is_control_flow_entrypoint(entrypoint)
                and slots.get(executor, 0) > 0
                and (wi.dag_id in resident_dag_ids or has_resident_capacity)
            )

        capture_started = time.perf_counter()
        capture = await self._frontier.capture_ready(
            candidate_window,
            slots,
            filter_fn=eligible,
        )
        capture_ms = (time.perf_counter() - capture_started) * 1000.0
        candidates = tuple(capture.jobs)
        scheduler_trace(
            "scheduler_selection_capture_completed",
            elapsed_ms=capture_ms,
            candidate_window=candidate_window,
            eligible=sum(capture.eligible_by_executor.values()),
            captured=len(candidates),
            ready_heap_entries=capture.ready_heap_entries,
            ready_set_entries=capture.ready_set_entries,
            stale_heap_entries=capture.stale_heap_entries,
            job_ids=[wi.id for wi in candidates],
        )

        rank_started = time.perf_counter()
        planner_candidates = tuple(
            (wi.data.get("metadata", {}).get("on", ""), wi) for wi in candidates
        )
        ranked = tuple(
            self._planner.plan(
                planner_candidates,
                slots.copy(),
                resident_dag_ids,
                exclude_blocked=True,
                now=now,
                dag_remaining=capture.dag_remaining,
            )
        )
        rank_ms = (time.perf_counter() - rank_started) * 1000.0
        scheduler_trace(
            "scheduler_selection_rank_completed",
            elapsed_ms=rank_ms,
            candidates=len(candidates),
            ranked=len(ranked),
            job_ids=[wi.id for _, wi in ranked],
        )

        cap_started = time.perf_counter()
        requested = tuple(_limit_to_slots(ranked, slots)[:dispatch_capacity])
        cap_ms = (time.perf_counter() - cap_started) * 1000.0
        scheduler_trace(
            "scheduler_selection_cap_completed",
            elapsed_ms=cap_ms,
            ranked=len(ranked),
            requested=len(requested),
            dispatch_capacity=dispatch_capacity,
            slots_by_executor=dict(slot_items),
            job_ids=[wi.id for _, wi in requested],
        )

        selected: tuple[WorkInfo, ...] = ()
        take_ms = 0.0
        if requested:
            take_started = time.perf_counter()
            selected = tuple(
                await self._frontier.take(
                    [wi.id for _, wi in requested],
                    lease_ttl=lease_ttl,
                )
            )
            take_ms = (time.perf_counter() - take_started) * 1000.0
        scheduler_trace(
            "scheduler_selection_take_completed",
            outcome="completed" if requested else "skipped",
            elapsed_ms=take_ms,
            requested=len(requested),
            selected=len(selected),
            job_ids=[wi.id for wi in selected],
        )
        elapsed_ms = (time.perf_counter() - selection_started) * 1000.0
        scheduler_trace(
            "scheduler_selection_completed",
            outcome="completed",
            started_ts_unix=selection_started_at,
            elapsed_ms=elapsed_ms,
            capture_ms=capture_ms,
            rank_ms=rank_ms,
            cap_ms=cap_ms,
            take_ms=take_ms,
            candidate_window=candidate_window,
            candidates=len(candidates),
            ranked=len(ranked),
            requested=len(requested),
            selected=len(selected),
            ready_heap_entries=capture.ready_heap_entries,
            ready_set_entries=capture.ready_set_entries,
            stale_heap_entries=capture.stale_heap_entries,
            job_ids=[wi.id for wi in selected],
        )
        self._record_selection(
            elapsed_ms=elapsed_ms,
            outcome="completed",
            reason=None,
            candidate_window=candidate_window,
            dispatch_capacity=dispatch_capacity,
            candidates=len(candidates),
            requested=len(requested),
            selected=len(selected),
        )
        return InMemorySelection(
            candidates=candidates,
            ranked=ranked,
            requested=requested,
            selected=selected,
            candidate_window=candidate_window,
            slots_by_executor=slot_items,
            eligible_by_executor=tuple(sorted(capture.eligible_by_executor.items())),
            captured_by_executor=tuple(sorted(capture.captured_by_executor.items())),
            eligible_by_dag=tuple(sorted(capture.eligible_by_dag.items())),
            captured_by_dag=tuple(sorted(capture.captured_by_dag.items())),
        )


def _percentile(values: list[float], percentile: float) -> float:
    index = min(len(values) - 1, max(0, round(percentile * (len(values) - 1))))
    return values[index]


def _limit_to_slots(
    ranked: tuple[FlatJob, ...], slots_by_executor: dict[str, int]
) -> list[FlatJob]:
    remaining = slots_by_executor.copy()
    selected: list[FlatJob] = []

    for entrypoint, wi in ranked:
        executor = executor_name(entrypoint)
        if remaining.get(executor, 0) <= 0:
            continue
        remaining[executor] -= 1
        selected.append((entrypoint, wi))

    return selected
