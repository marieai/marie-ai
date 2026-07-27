import random
from typing import Callable, Sequence

from marie.job.common import JobStatus
from marie.logging_core.predefined import default_logger as logger
from marie.scheduler.state import WorkState
from marie.state.semaphore_store import EtcdStoreUnavailable, SemaphoreStore

CONTROL_FLOW_EXECUTORS = frozenset({"noop", "branch", "switch", "merger"})


def convert_job_status_to_work_state(job_status: JobStatus) -> WorkState:
    """
    Convert a JobStatus to a WorkState.
    :param job_status: The JobStatus to convert.
    :return: The corresponding WorkState.
    """
    if job_status == JobStatus.PENDING:
        return WorkState.CREATED
    elif job_status == JobStatus.RUNNING:
        return WorkState.ACTIVE
    elif job_status == JobStatus.SUCCEEDED:
        return WorkState.COMPLETED
    elif job_status == JobStatus.FAILED:
        return WorkState.FAILED
    elif job_status == JobStatus.STOPPED:
        return WorkState.CANCELLED
    else:
        raise ValueError(f"Unknown JobStatus: {job_status}")


def adjust_backoff(
    wait_time: float,
    idle_streak: int,
    scheduled: bool,
    min_poll_period: float,
    max_poll_period: float,
) -> float:
    """Return the next polling delay within the configured bounds."""
    if scheduled:
        return min(max(wait_time * 0.5, min_poll_period), max_poll_period)
    jitter = random.uniform(0.9, 1.1)
    wait_time *= (1.5 + 0.1 * idle_streak) * jitter
    return min(max(wait_time, min_poll_period), max_poll_period)


def has_available_slot(entrypoint: str, sem: SemaphoreStore) -> bool:
    """True if at least one slot is available for the executor in `entrypoint`."""
    return available_slots(entrypoint, sem) > 0


def available_slots(entrypoint: str, sem: SemaphoreStore) -> int:
    """
    Return available slots for the executor part of `entrypoint` (e.g. 'extract_executor://default').
    Reads from the semaphore store to ensure consistency with reserve()/renew().
    """
    executor = entrypoint.split("://", 1)[0]
    # single-slot query (O(1))
    return max(0, sem.available_slot_count(executor))


def available_slots_by_executor(sem: SemaphoreStore) -> dict[str, int]:
    """
    Snapshot available slots for all executors from the semaphore store.
    Equivalent to: capacities - used_count, based on holders/count keys.

    During an etcd outage this degrades to an empty snapshot (zero slots)
    instead of raising: callers keep cycling, control-flow nodes still
    dispatch, and executor jobs simply wait out the outage.
    """
    try:
        return sem.available_count_all()
    except EtcdStoreUnavailable as e:
        logger.warning(
            f"[WORK_DIST] Slot store unavailable (etcd outage?): {e} — "
            f"reporting zero executor slots for this cycle"
        )
        return {}


def executor_name(entrypoint: str) -> str:
    if "://" in entrypoint:
        return entrypoint.split("://", 1)[0].lower()
    return entrypoint.lower()


def is_control_flow_entrypoint(entrypoint: str) -> bool:
    return executor_name(entrypoint) in CONTROL_FLOW_EXECUTORS


def frontier_candidate_window(
    batch_size: int, slots_by_executor: dict[str, int]
) -> int:
    free_slots = sum(max(v, 0) for v in slots_by_executor.values())
    return max(batch_size, min(2000, 4 * free_slots + 64))


def frontier_slot_filter(
    slots_by_executor: dict[str, int],
) -> Callable[[object], bool]:
    def _eligible(wi: object) -> bool:
        data = getattr(wi, "data", {}) or {}
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        entrypoint = metadata.get("on", "") if isinstance(metadata, dict) else ""
        executor = executor_name(entrypoint)

        if not executor or executor in CONTROL_FLOW_EXECUTORS:
            return True

        return slots_by_executor.get(executor, 0) > 0

    return _eligible


def ordered_leased_jobs(
    planned: Sequence[tuple[str, object]], leased_ids: set[str]
) -> list[tuple[str, object]]:
    return [(entrypoint, wi) for entrypoint, wi in planned if wi.id in leased_ids]
