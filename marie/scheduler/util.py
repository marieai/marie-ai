import random
from collections import defaultdict

from marie.job.common import JobStatus
from marie.logging_core.predefined import default_logger as logger
from marie.scheduler.state import WorkState
from marie.state.semaphore_store import SemaphoreStore


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
    wait_time: float, idle_streak: int, scheduled: bool, min_poll_period: float
) -> float:
    """
    Adjusts the backoff time for a polling mechanism based on the provided wait time,
    the number of consecutive idle streaks, and whether the task is scheduled. The
    resulting wait time ensures it stays within predefined minimum and maximum periods.
    In cases where the task is scheduled, the wait time is reduced. For non-scheduled
    tasks, it considers random jitter and adjusts based on idle streaks.
    """
    if scheduled:
        return max(wait_time * 0.5, min_poll_period)
    jitter = random.uniform(0.9, 1.1)
    return min(wait_time * (1.5 + 0.1 * idle_streak), min_poll_period) * jitter


def has_available_slot(entrypoint: str, sem: SemaphoreStore) -> bool:
    """True if at least one slot is available for the executor in `entrypoint`."""
    return available_slots(entrypoint, sem) > 0


def available_slots(entrypoint: str, sem: SemaphoreStore) -> int:
    """
    Return available slots for the executor part of `entrypoint` (e.g. 'extract_executor://default').
    Reads from the semaphore store to ensure consistency with reserve()/renew().
    """
    executor = entrypoint.split("://", 1)[0]
    return max(0, sem.available_slot_count(executor))


def available_slots_by_executor(sem: SemaphoreStore) -> dict[str, int]:
    """
    Snapshot available slots for all executors from the semaphore store.
    Equivalent to: capacities - used_count, based on holders/count keys.

    Returns an empty dict if the semaphore store is unavailable.
    """
    try:
        return sem.available_count_all()
    except Exception as ex:
        logger.warning(f"Failed to read available slots (etcd unavailable?): {ex}")
        return {}


# # Example Usage
deployment_nodes = {
    'annotator_executor': [
        {
            'address': 'grpc://127.0.0.1:53543',
            'endpoint': '_jina_dry_run_',
            'executor': 'annotator_executor',
            'gateway': '127.0.0.1:53543',
        },
        {
            'address': 'grpc://127.0.0.1:53543',
            'endpoint': '/default',
            'executor': 'annotator_executor',
            'gateway': '127.0.0.1:53543',
        },
        {
            'address': 'grpc://127.0.0.1:53543',
            'endpoint': '/annotator/llm',
            'executor': 'annotator_executor',
            'gateway': '127.0.0.1:53543',
        },
        {
            'address': 'grpc://127.0.0.1:53543',
            'endpoint': '/annotator/table-llm',
            'executor': 'annotator_executor',
            'gateway': '127.0.0.1:53543',
        },
        {
            'address': 'grpc://127.0.0.1:53543',
            'endpoint': '/annotator/result-parser',
            'executor': 'annotator_executor',
            'gateway': '127.0.0.1:53543',
        },
    ]
}
#
# result = available_slots_by_entrypoint(deployment_nodes)
# print(result)
