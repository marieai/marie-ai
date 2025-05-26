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


def available_slots_by_entrypoint(deployment_nodes: dict) -> dict[str, int]:
    """
    Determines available slots for each entrypoint based on deployment nodes.

    Parameters:
        deployment_nodes (dict): A dictionary where keys are executor names
                                 and values are lists of endpoint definitions.

    Returns:
        dict: A dictionary of entrypoints to available slots, grouped by executor.
    """
    flattened_result = {}

    for executor, nodes in deployment_nodes.items():
        entrypoints = [node['endpoint'] for node in nodes]

        for entrypoint in set(entrypoints):
            if entrypoint.startswith('/'):
                entrypoint = entrypoint[1:]
            key = f"{executor}://{entrypoint}"
            flattened_result[key] = flattened_result.get(key, 0) + 1
    return flattened_result


# # Example Usage
# deployment_nodes = {
#     'annotator_executor': [
#         {'address': 'grpc://127.0.0.1:53543', 'endpoint': '_jina_dry_run_', 'executor': 'annotator_executor',
#          'gateway': '127.0.0.1:53543'},
#         {'address': 'grpc://127.0.0.1:53543', 'endpoint': '/default', 'executor': 'annotator_executor',
#          'gateway': '127.0.0.1:53543'},
#         {'address': 'grpc://127.0.0.1:53543', 'endpoint': '/annotator/llm', 'executor': 'annotator_executor',
#          'gateway': '127.0.0.1:53543'},
#         {'address': 'grpc://127.0.0.1:53543', 'endpoint': '/annotator/table-llm', 'executor': 'annotator_executor',
#          'gateway': '127.0.0.1:53543'},
#         {'address': 'grpc://127.0.0.1:53543', 'endpoint': '/annotator/result-parser', 'executor': 'annotator_executor',
#          'gateway': '127.0.0.1:53543'}
#     ]
# }
#
# result = available_slots_by_entrypoint(deployment_nodes)
# print(result)


from rich.console import Console
from rich.table import Table

# JSON-like dictionary with job states
job_states_data = {
    "queues": {
        "extract": {
            "created": 0,
            "retry": 0,
            "active": 0,
            "completed": 48,
            "expired": 0,
            "cancelled": 0,
            "failed": 0,
            "all": 48,
        },
        "transform": {
            "created": 2,
            "retry": 1,
            "active": 0,
            "completed": 5,
            "expired": 0,
            "cancelled": 0,
            "failed": 3,
            "all": 11,
        },
    },
    "all": 59,
    "completed": 53,
}


from rich.console import Console
from rich.table import Table

# JSON-like dictionary with job states
job_states_data = {
    "queues": {
        "extract": {
            "created": 0,
            "retry": 0,
            "active": 0,
            "completed": 48,
            "expired": 0,
            "cancelled": 0,
            "failed": 0,
            "all": 48,
        },
        "transform": {
            "created": 2,
            "retry": 1,
            "active": 0,
            "completed": 5,
            "expired": 0,
            "cancelled": 0,
            "failed": 3,
            "all": 11,
        },
    },
    "all": 59,
    "completed": 53,
}

# Initialize the Rich Console
console = Console()

# Create a single Rich Table
table = Table(
    title="📊 Consolidated Job States for All Queues",
    border_style="green",
    title_style="bold white on blue",
    header_style="bold yellow",
    show_lines=True,  # Adds separating lines between rows for better readability
)

# Add the first column as "Queue"
table.add_column("Queue", justify="left", style="cyan", no_wrap=False, width=12)

# Dynamically add the metric columns with specific widths
metrics = list(next(iter(job_states_data["queues"].values())).keys())
for metric in metrics:
    table.add_column(
        metric.capitalize(), justify="center", style="magenta", no_wrap=False, width=26
    )

# Populate the table rows dynamically, one row for each queue
for queue_name, queue_data in job_states_data["queues"].items():
    table.add_row(
        queue_name.capitalize(),  # Queue name
        *[
            str(value) for value in queue_data.values()
        ],  # All metric values for this queue
    )

# Calculate the summary row
summary_values = {metric: 0 for metric in metrics}  # Initialize summary values
for queue_data in job_states_data["queues"].values():
    for metric, value in queue_data.items():
        summary_values[metric] += value  # Sum up the metrics

# Add the summary row to the table
table.add_row(
    "Summary",  # Label for the summary row
    *[
        str(summary_values[metric]) for metric in metrics
    ],  # Aggregated values for each metric
    style="bold green",  # Change style for the summary row
)

# Render the table in the console
console.print(table)

# Render the table in the console
console.print(table)
print(table)
