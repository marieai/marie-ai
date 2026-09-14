from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

from marie.engine.completion_contract import (
    CompletionCallParams,
    QueuedCompletionEnvelope,
)
from marie.engine.llm_queue.queue_io import ListQueueClient, MalformedQueueRequest

MIN_REQUEST_COST_UNITS = 1
MAX_REQUEST_COST_UNITS = 16
PAGES_PER_COST_UNIT = 8
IMAGE_COST_UNITS = 2


@dataclass(frozen=True)
class DrrLaneConfig:
    pool_id: str
    quantum: int = 1
    min_concurrent: int = 0
    max_concurrent: Optional[int] = None
    max_burst_per_visit: Optional[int] = None
    display_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            'quantum',
            'min_concurrent',
            'max_concurrent',
            'max_burst_per_visit',
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise ValueError(f'{name} must be an integer')
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be a boolean')
        if not self.pool_id:
            raise ValueError("pool_id is required")
        if self.quantum < 1:
            raise ValueError("quantum must be at least 1")
        if self.min_concurrent < 0:
            raise ValueError("min_concurrent cannot be negative")
        if self.max_concurrent is not None and self.max_concurrent < 0:
            raise ValueError("max_concurrent cannot be negative")
        if (
            self.max_concurrent is not None
            and self.min_concurrent > self.max_concurrent
        ):
            raise ValueError("min_concurrent cannot exceed max_concurrent")
        if self.max_burst_per_visit is not None and self.max_burst_per_visit < 1:
            raise ValueError("max_burst_per_visit must be at least 1")


@dataclass(frozen=True)
class DrrDispatch:
    pool_id: str
    request: QueuedCompletionEnvelope
    cost_units: int
    deficit_after_dispatch: int


@dataclass(frozen=True)
class DrrLaneSnapshot:
    pool_id: str
    display_name: Optional[str]
    quantum: int
    deficit: int
    inflight: int
    queue_depth: Optional[int]
    oldest_pending_age_seconds: Optional[float]
    head_cost_units: Optional[int]
    min_concurrent: int
    max_concurrent: Optional[int]
    max_burst_per_visit: Optional[int]
    skip_counts: dict[str, int]
    malformed_requests_dropped: int


@dataclass
class _LaneState:
    config: DrrLaneConfig
    deficit: int = 0
    inflight: int = 0
    visit_started: bool = False
    visit_dispatches: int = 0
    skip_counts: Counter[str] = field(default_factory=Counter)
    malformed_requests_dropped: int = 0


class DrrLaneScheduler:
    def __init__(
        self,
        *,
        queue_client: ListQueueClient | None,
        lanes: list[DrrLaneConfig],
        total_concurrent_dispatch: int,
    ) -> None:
        if total_concurrent_dispatch < 1:
            raise ValueError("total_concurrent_dispatch must be at least 1")

        enabled_lanes = [lane for lane in lanes if lane.enabled]
        pool_ids = [lane.pool_id for lane in lanes]
        if len(pool_ids) != len(set(pool_ids)):
            raise ValueError("lane pool_id values must be unique")
        min_concurrent_total = sum(lane.min_concurrent for lane in enabled_lanes)
        if min_concurrent_total > total_concurrent_dispatch:
            raise ValueError(
                "sum of lane min_concurrent values cannot exceed total_concurrent_dispatch"
            )

        self.queue_client = queue_client
        self.total_concurrent_dispatch = total_concurrent_dispatch
        self._states = {lane.pool_id: _LaneState(config=lane) for lane in enabled_lanes}
        self._rotation: Deque[str] = deque()
        self._active: set[str] = set()
        self._global_inflight = 0
        self._backlogs: set[str] | None = None

    @property
    def inflight_count(self) -> int:
        return self._global_inflight

    def select_next(self) -> Optional[DrrDispatch]:
        if self._global_inflight >= self.total_concurrent_dispatch:
            return None

        self._refresh_rotation()
        visits_remaining = len(self._rotation)

        while visits_remaining > 0 and self._rotation:
            pool_id = self._rotation[0]
            state = self._states[pool_id]

            if not self._lane_has_backlog(pool_id):
                self._drop_current_lane(reset_deficit=True)
                visits_remaining -= 1
                continue

            can_launch, skip_reason = self._can_launch(state)
            if not can_launch:
                state.skip_counts[skip_reason] += 1
                self._finish_current_visit()
                visits_remaining -= 1
                continue

            if not state.visit_started:
                state.deficit += state.config.quantum
                state.visit_started = True
                state.visit_dispatches = 0

            try:
                request = self.queue_client.peek_request(pool_id)
            except MalformedQueueRequest as exc:
                if self.queue_client.discard_malformed_head(pool_id, exc.digest):
                    self._record_malformed_drop(state)
                return None

            if request is None:
                self._drop_current_lane(reset_deficit=True)
                visits_remaining -= 1
                continue

            cost_units = request_cost_units(request)
            if cost_units > state.deficit:
                state.skip_counts["insufficient_credit"] += 1
                self._finish_current_visit()
                visits_remaining -= 1
                continue

            try:
                popped = self.queue_client.try_pop_request(pool_id)
            except MalformedQueueRequest:
                self._record_malformed_drop(state)
                return None

            if popped is None:
                self._drop_current_lane(reset_deficit=True)
                visits_remaining -= 1
                continue

            state.deficit -= cost_units
            state.inflight += 1
            state.visit_dispatches += 1
            self._global_inflight += 1

            deficit_after_dispatch = state.deficit
            # The popped request must be returned before any further queue I/O.
            self._advance_after_dispatch(state, read_backlog=False)
            return DrrDispatch(
                pool_id=pool_id,
                request=popped,
                cost_units=cost_units,
                deficit_after_dispatch=deficit_after_dispatch,
            )

        return None

    def sync_reservations(
        self, reserved: dict[str, int], total: int, backlogged: set[str]
    ) -> None:
        """Use authoritative V3 reservations, including unresolved remote calls."""
        self._global_inflight = total
        self._backlogs = backlogged
        for pool, state in self._states.items():
            state.inflight = reserved.get(pool, 0)
            if pool not in backlogged:
                state.deficit = 0
                state.visit_started = False
                state.visit_dispatches = 0

    def select_metadata(self, costs: dict[str, int]) -> str | None:
        """Inspect a bounded round using metadata supplied by the fenced owner."""
        self._backlogs = set(costs)
        self._refresh_rotation()
        for _ in range(len(self._rotation)):
            pool = self._rotation[0]
            state = self._states[pool]
            if pool not in costs:
                self._drop_current_lane(reset_deficit=False)
                continue
            allowed, reason = self._can_launch(state)
            if not allowed:
                state.skip_counts[reason] += 1
                self._finish_current_visit()
                continue
            if not state.visit_started:
                state.deficit = min(1_000_000, state.deficit + state.config.quantum)
                state.visit_started = True
                state.visit_dispatches = 0
            if costs[pool] <= state.deficit:
                return pool
            state.skip_counts['insufficient_credit'] += 1
            self._finish_current_visit()
        # Skip virtual empty rounds arithmetically; work stays bounded for large costs.
        eligible = [
            self._states[pool]
            for pool in costs
            if self._can_launch(self._states[pool])[0]
        ]
        if eligible:
            rounds = min(
                max(
                    1,
                    math.ceil(
                        (costs[state.config.pool_id] - state.deficit)
                        / state.config.quantum
                    ),
                )
                for state in eligible
            )
            for state in eligible:
                state.deficit = min(
                    1_000_000, state.deficit + rounds * state.config.quantum
                )
                state.visit_started = True
            for pool in self._rotation:
                if (
                    pool in costs
                    and costs[pool] <= self._states[pool].deficit
                    and self._can_launch(self._states[pool])[0]
                ):
                    while self._rotation[0] != pool:
                        self._finish_current_visit()
                    return pool
        return None

    def claimed(self, pool_id: str, cost: int) -> None:
        """Charge only the cost returned by a successful atomic claim; perform no I/O."""
        state = self._states[pool_id]
        state.deficit -= cost
        state.inflight += 1
        state.visit_dispatches += 1
        self._global_inflight += 1
        self._advance_after_dispatch(state, read_backlog=False)

    def rejected(self, pool_id: str, reason: str) -> None:
        state = self._states[pool_id]
        state.skip_counts[reason] += 1
        self._finish_current_visit()

    def lane_metadata(self, pool_id: str) -> dict[str, Any]:
        state = self._states.get(pool_id)
        if state is None:
            return {}
        return dict(
            quantum=state.config.quantum,
            deficit=state.deficit,
            inflight=state.inflight,
            min_concurrent=state.config.min_concurrent,
            max_burst_per_visit=state.config.max_burst_per_visit,
            skip_counts=dict(state.skip_counts),
        )

    def release(self, pool_id: str) -> None:
        state = self._states[pool_id]
        if state.inflight <= 0:
            raise ValueError(f"lane {pool_id!r} has no inflight request to release")
        state.inflight -= 1
        self._global_inflight -= 1

    def lane_snapshots(
        self, *, before_read: Callable[[], None] | None = None
    ) -> list[DrrLaneSnapshot]:
        snapshots: list[DrrLaneSnapshot] = []
        now = time.time()
        for pool_id, state in sorted(self._states.items()):
            if before_read:
                before_read()
            try:
                queue_depth = self.queue_client.request_queue_depth(pool_id)
            except Exception:
                queue_depth = None
            oldest_pending_age_seconds = None
            head_cost_units = None
            snapshots.append(
                DrrLaneSnapshot(
                    pool_id=pool_id,
                    display_name=state.config.display_name,
                    quantum=state.config.quantum,
                    deficit=state.deficit,
                    inflight=state.inflight,
                    queue_depth=queue_depth,
                    oldest_pending_age_seconds=oldest_pending_age_seconds,
                    head_cost_units=head_cost_units,
                    min_concurrent=state.config.min_concurrent,
                    max_concurrent=state.config.max_concurrent,
                    max_burst_per_visit=state.config.max_burst_per_visit,
                    skip_counts=dict(state.skip_counts),
                    malformed_requests_dropped=state.malformed_requests_dropped,
                )
            )
        return snapshots

    def _refresh_rotation(self) -> None:
        for pool_id in self._states:
            if pool_id in self._active:
                continue
            if self._lane_has_backlog(pool_id):
                self._rotation.append(pool_id)
                self._active.add(pool_id)

    def _lane_has_backlog(self, pool_id: str) -> bool:
        if self._backlogs is not None:
            return pool_id in self._backlogs
        return self.queue_client.request_queue_depth(pool_id) > 0

    def _can_launch(self, state: _LaneState) -> tuple[bool, str]:
        if self._global_inflight >= self.total_concurrent_dispatch:
            return False, "global_capacity"

        max_concurrent = state.config.max_concurrent
        if max_concurrent is not None and state.inflight >= max_concurrent:
            return False, "lane_capacity"

        protected_needed_elsewhere = 0
        for pool_id, other in self._states.items():
            if other is state or not self._lane_has_backlog(pool_id):
                continue
            protected_needed_elsewhere += max(
                0,
                other.config.min_concurrent - other.inflight,
            )

        if self._global_inflight >= (
            self.total_concurrent_dispatch - protected_needed_elsewhere
        ):
            return False, "lane_capacity"

        return True, ""

    def _advance_after_dispatch(
        self, state: _LaneState, *, read_backlog: bool = True
    ) -> None:
        pool_id = state.config.pool_id
        if read_backlog and not self._lane_has_backlog(pool_id):
            self._drop_current_lane(reset_deficit=True)
            return

        burst_limit = state.config.max_burst_per_visit
        if burst_limit is not None and state.visit_dispatches >= burst_limit:
            self._finish_current_visit()
            return

        if state.deficit <= 0:
            self._finish_current_visit()

    def _finish_current_visit(self) -> None:
        pool_id = self._rotation.popleft()
        state = self._states[pool_id]
        state.visit_started = False
        state.visit_dispatches = 0
        self._rotation.append(pool_id)

    def _drop_current_lane(self, *, reset_deficit: bool) -> None:
        pool_id = self._rotation.popleft()
        self._active.discard(pool_id)
        state = self._states[pool_id]
        state.visit_started = False
        state.visit_dispatches = 0
        if reset_deficit:
            state.deficit = 0

    def _record_malformed_drop(self, state: _LaneState) -> None:
        state.malformed_requests_dropped += 1
        state.skip_counts["malformed_head"] += 1


def request_cost_units(request: QueuedCompletionEnvelope) -> int:
    explicit = _read_int(request.estimated_cost_units)
    metadata = request.metadata if isinstance(request.metadata, dict) else {}
    if explicit is None:
        explicit = _read_int(metadata.get("estimated_cost_units"))
    if explicit is not None:
        return _clamp_cost(explicit)

    image_count = _read_int(metadata.get("image_count"))
    if image_count is None:
        image_count = _count_message_images(request)

    chunk_page_count = (
        _read_int(metadata.get("chunk_page_count"))
        or _read_int(metadata.get("page_count"))
        or 0
    )
    cost = 1 + (IMAGE_COST_UNITS * max(0, image_count))
    if chunk_page_count > 0:
        cost += math.ceil(chunk_page_count / PAGES_PER_COST_UNIT)
    return _clamp_cost(cost)


def _read_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_message_images(request: QueuedCompletionEnvelope) -> int:
    image_count = 0
    for message in request.call.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        image_count += sum(
            1
            for item in content
            if isinstance(item, dict) and item.get("type") == "image_url"
        )
    return image_count


def _clamp_cost(value: int) -> int:
    return max(MIN_REQUEST_COST_UNITS, min(MAX_REQUEST_COST_UNITS, value))


def prepared_cost_units(
    call: CompletionCallParams, metadata: dict[str, Any] | None = None
) -> int:
    """Estimate the effective prepared call once, before releasing its body."""
    metadata = metadata or {}
    for name in (
        'estimated_cost_units',
        'image_count',
        'chunk_page_count',
        'page_count',
    ):
        value = metadata.get(name)
        if value is not None and (
            type(value) is not int or value < 0 or value > 1_000_000
        ):
            raise ValueError('Invalid scheduling cost metadata')
    explicit = metadata.get('estimated_cost_units')
    if explicit is not None:
        if not MIN_REQUEST_COST_UNITS <= explicit <= MAX_REQUEST_COST_UNITS:
            raise ValueError('Invalid scheduling cost override')
    prepared = call.to_create_kwargs()
    if prepared.get('extra_body'):
        prepared.update(prepared['extra_body'])
    messages = prepared.get('messages', [])
    images = sum(
        1
        for message in messages
        if isinstance(message, dict)
        for item in (
            message.get('content') if isinstance(message.get('content'), list) else []
        )
        if isinstance(item, dict) and item.get('type') == 'image_url'
    )
    images = max(images, metadata.get('image_count', 0))
    pages = metadata.get('chunk_page_count', metadata.get('page_count', 0))
    return _clamp_cost(
        max(
            explicit or MIN_REQUEST_COST_UNITS,
            1 + IMAGE_COST_UNITS * images + math.ceil(pages / PAGES_PER_COST_UNIT),
        )
    )
