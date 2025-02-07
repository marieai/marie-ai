from collections import defaultdict


def group_by_executor_and_status(deployments: list) -> dict:
    """Groups deployment objects by their executor and status into a nested dictionary format."""
    grouped = defaultdict(lambda: defaultdict(list))

    for item in deployments:
        executor = item["executor"]
        status = item["status"]

        grouped[executor][status].append(item)

    return {executor: dict(statuses) for executor, statuses in grouped.items()}


def get_counts_by_executor_and_status(deployments: list):
    """Returns a dictionary where the top-level keys are executors,
    and each key maps to a dictionary of status-to-count mappings."""
    counts = defaultdict(lambda: defaultdict(int))

    for item in deployments:
        executor = item["executor"]
        status = item["status"]

        counts[executor][status] += 1

    return {executor: dict(status_cnt) for executor, status_cnt in counts.items()}


def has_available_slot(entrypoint: str, deployments: dict) -> bool:
    """
    Determines if job slots are available for the specified entrypoint.

    :param entrypoint: A string in the format "executor://action" indicating
        which executor should handle the job.
    :returns: True if there is at least one slot available for the given executor,
        otherwise False.
    """
    return available_slots(entrypoint, deployments) > 0


def available_slots(entrypoint: str, deployments: dict) -> int:
    """
    Determines the number of job slots available for the specified entrypoint.

    :param entrypoint: A string in the format "executor://action" indicating
        which executor should handle the job.
    :returns: The number of available slots for the given executor.
    """
    executor = entrypoint.split("://")[0]
    grouped_by_executor = get_counts_by_executor_and_status(list(deployments.values()))

    ready_workers = 0
    if executor in grouped_by_executor:
        worker_status = grouped_by_executor[executor]
        ready_workers = int(worker_status.get("NOT_SERVING", 0)) + int(
            worker_status.get("SERVICE_UNKNOWN", 0)
        )

    return ready_workers


def available_slots_by_executor(deployments: dict) -> dict[str, int]:
    """
    Determines the number of job slots available for each executor(workers).

    :param deployments: A dictionary of deployments.
    :returns: A dictionary with executor names as keys and the number of available
        slots as values.
    """
    grouped_by_executor = get_counts_by_executor_and_status(list(deployments.values()))
    return {
        executor: int(worker_status.get("NOT_SERVING", 0))
        + int(worker_status.get("SERVICE_UNKNOWN", 0))
        for executor, worker_status in grouped_by_executor.items()
    }
