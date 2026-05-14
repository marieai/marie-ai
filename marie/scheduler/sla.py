from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from marie.scheduler.state import is_terminal_state

DEFAULT_INTERVAL_SECONDS = 15 * 60
MAX_SOFT_ESCALATION = 499
MAX_HARD_ESCALATION = 999


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def compute_sla_priority_bucket(
    now: Optional[datetime],
    soft_sla: Optional[datetime],
    hard_sla: Optional[datetime],
    *,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> int:
    """
    Match the old SQL refresh_job_priority() behavior.

    Ranges:
      0         : no SLA
      1-499     : approaching soft SLA
      500-999   : soft SLA missed
      1000-1999 : hard SLA missed
    """
    now_utc = _as_utc(now) or datetime.now(timezone.utc)
    interval_seconds = max(1, int(interval_seconds))
    soft_sla_utc = _as_utc(soft_sla)
    hard_sla_utc = _as_utc(hard_sla)

    if hard_sla_utc is not None and now_utc > hard_sla_utc:
        overdue_seconds = (now_utc - hard_sla_utc).total_seconds()
        escalation = min(MAX_HARD_ESCALATION, int(overdue_seconds // interval_seconds))
        return 1000 + escalation

    if soft_sla_utc is not None and now_utc > soft_sla_utc:
        overdue_seconds = (now_utc - soft_sla_utc).total_seconds()
        escalation = min(MAX_SOFT_ESCALATION, int(overdue_seconds // interval_seconds))
        return 500 + escalation

    if soft_sla_utc is not None:
        remaining_seconds = (soft_sla_utc - now_utc).total_seconds()
        urgency = 500 - int(-(-remaining_seconds // interval_seconds))
        return max(1, urgency)

    return 0


def sla_bucket_status(bucket: int) -> str:
    if bucket >= 1000:
        return "hard_missed"
    if bucket >= 500:
        return "soft_missed"
    if bucket > 0:
        return "approaching_soft"
    return "no_sla"


def summarize_sla_work_items(
    work_items: Iterable[Any],
    *,
    now: Optional[datetime] = None,
    top_n: int = 5,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """
    Summarize current SLA pressure for non-terminal work items.

    The scheduler uses this for observability only; dispatch ordering is still
    decided by the execution planner.
    """
    now_utc = _as_utc(now) or datetime.now(timezone.utc)
    summary = {
        "tracked": 0,
        "no_sla": 0,
        "approaching_soft": 0,
        "soft_missed": 0,
        "hard_missed": 0,
        "highest_bucket": 0,
        "top_urgent": [],
    }
    urgent_rows: list[dict[str, Any]] = []

    for wi in work_items:
        if wi is None or is_terminal_state(getattr(wi, "state", None)):
            continue

        bucket = compute_sla_priority_bucket(
            now_utc,
            wi.soft_sla,
            wi.hard_sla,
            interval_seconds=interval_seconds,
        )
        status = sla_bucket_status(bucket)
        summary["tracked"] += 1
        summary[status] += 1
        summary["highest_bucket"] = max(summary["highest_bucket"], bucket)

        if bucket <= 0:
            continue

        urgent_rows.append(
            {
                "id": getattr(wi, "id", None),
                "dag_id": getattr(wi, "dag_id", None),
                "name": getattr(wi, "name", None),
                "priority": int(getattr(wi, "priority", 0)),
                "job_level": int(getattr(wi, "job_level", 0)),
                "sla_bucket": bucket,
                "status": status,
            }
        )

    urgent_rows.sort(
        key=lambda row: (
            -int(row["sla_bucket"]),
            -int(row["priority"]),
            -int(row["job_level"]),
            str(row["id"]),
        )
    )
    if top_n > 0:
        summary["top_urgent"] = urgent_rows[:top_n]
    return summary
