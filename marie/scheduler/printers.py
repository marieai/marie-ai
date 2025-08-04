import traceback
from typing import Any, Dict

from rich.console import Console
from rich.table import Table

from marie.logging_core.predefined import default_logger as logger


def print_slots_table(slots: dict[str, int], max_slots: dict[str, int] = None) -> None:
    console = Console()
    table = Table(
        title="⚙️  Available Slots",
        border_style="green",
        title_style="bold white on blue",
        header_style="bold yellow",
        show_lines=True,
    )
    table.add_column("Slot Type", justify="left", style="cyan", no_wrap=False)
    table.add_column("Count", justify="center", style="magenta", no_wrap=False)
    table.add_column("Max Seen", justify="center", style="green", no_wrap=False)

    keys = sorted(
        list(set(slots.keys()) | set(max_slots.keys() if max_slots else set()))
    )

    for slot_type in keys:
        row = [slot_type, str(slots.get(slot_type, "-"))]
        if max_slots:
            row.append(str(max_slots.get(slot_type, "-")))
        table.add_row(*row)

    console.print(table)


def print_state_summary(states_data: Dict[str, Any], entity_type: str = "job"):
    """
    Print a consolidated state summary table for jobs or dags.

    :param states_data: Dictionary containing state data from count_job_states or count_dag_states
    :param entity_type: Type of entity ("job" or "dag")
    """
    try:
        console = Console()

        # Configure table based on entity type
        entity_display = entity_type.capitalize()
        emoji = "💼" if entity_type == "job" else "📊"

        table = Table(
            title=f"{emoji} Consolidated {entity_display} States for All Queues",
            border_style="green",
            title_style="bold white on blue",
            header_style="bold yellow",
            show_lines=True,
        )

        table.add_column("Queue", justify="left", style="cyan", no_wrap=False, width=16)

        # Define metrics based on entity type
        if entity_type == "job":
            metrics = [
                "created",
                "active",
                "completed",
                "failed",
                "cancelled",
                "expired",
                "retry",
                "all",
            ]
        elif entity_type == "dag":
            metrics = [
                "created",
                "active",
                "completed",
                "failed",
                "cancelled",
                "all",
            ]
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        for metric in metrics:
            table.add_column(
                metric.capitalize(),
                justify="center",
                style="magenta",
                no_wrap=False,
                width=16,
            )

        if states_data.get("queues"):
            for queue_name, queue_data in states_data["queues"].items():
                row_values = [queue_name.capitalize()]
                for metric in metrics:
                    row_values.append(str(queue_data.get(metric, 0)))
                table.add_row(*row_values)

            # Calculate summary totals
            summary_values = {metric: 0 for metric in metrics}
            for queue_data in states_data["queues"].values():
                for metric, value in queue_data.items():
                    if metric in summary_values:
                        summary_values[metric] += value

            table.add_row(
                "Summary",
                *[str(summary_values[metric]) for metric in metrics],
                style="bold green",
            )
        else:
            table.add_row("No Data", *["0" for _ in metrics], style="bold red")

        console.print(table)
    except Exception as e:
        logger.error(f"Error printing {entity_type} state summary: {e}")
        logger.error(traceback.format_exc())


def print_job_state_summary(job_states_data: Dict[str, Any]):
    """
    Print a consolidated job state summary table.
    """
    print_state_summary(job_states_data, "job")


def print_dag_state_summary(dag_states_data: Dict[str, Any]):
    """
    Print a consolidated dag state summary table.
    """
    print_state_summary(dag_states_data, "dag")
