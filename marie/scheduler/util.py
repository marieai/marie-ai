from collections import defaultdict


def group_by_executor_and_status(deployments: list) -> dict:
    """Groups deployment objects by their executor and status into a nested dictionary format."""
    print('deployments:', deployments)
    print(type(deployments))
    grouped = defaultdict(lambda: defaultdict(list))

    for item in deployments:
        executor = item["executor"]
        status = item["status"]

        grouped[executor][status].append(item)

    return {executor: dict(statuses) for executor, statuses in grouped.items()}


def get_counts_by_executor_and_status(deployments: list):
    """Returns a dictionary where the top-level keys are executors,
    and each key maps to a dictionary of status-to-count mappings."""
    print('-----------------------')
    print('deployments:', deployments)

    counts = defaultdict(lambda: defaultdict(int))

    for item in deployments:
        executor = item["executor"]
        status = item["status"]

        counts[executor][status] += 1

    return {executor: dict(status_cnt) for executor, status_cnt in counts.items()}
