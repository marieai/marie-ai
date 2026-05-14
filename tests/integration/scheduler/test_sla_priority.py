from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from marie.scheduler.sla import (
    compute_sla_priority_bucket,
    sla_bucket_status,
    summarize_sla_work_items,
)


def test_compute_sla_priority_bucket_matches_old_sql_ranges():
    now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)

    assert compute_sla_priority_bucket(now, None, None) == 0
    assert (
        compute_sla_priority_bucket(
            now,
            soft_sla=now + timedelta(minutes=10),
            hard_sla=now + timedelta(hours=1),
        )
        == 499
    )
    assert (
        compute_sla_priority_bucket(
            now,
            soft_sla=now - timedelta(minutes=10),
            hard_sla=now + timedelta(hours=1),
        )
        == 500
    )
    assert (
        compute_sla_priority_bucket(
            now,
            soft_sla=now - timedelta(hours=1),
            hard_sla=now - timedelta(minutes=10),
        )
        == 1000
    )


def test_compute_sla_priority_bucket_escalates_every_15_minutes():
    now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)

    soft_16 = compute_sla_priority_bucket(
        now,
        soft_sla=now - timedelta(minutes=16),
        hard_sla=now + timedelta(hours=1),
    )
    hard_31 = compute_sla_priority_bucket(
        now,
        soft_sla=now - timedelta(hours=2),
        hard_sla=now - timedelta(minutes=31),
    )

    assert soft_16 == 501
    assert hard_31 == 1002


def test_compute_sla_priority_bucket_can_use_second_scale_interval():
    now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)

    bucket = compute_sla_priority_bucket(
        now,
        soft_sla=now + timedelta(seconds=12),
        hard_sla=now + timedelta(minutes=1),
        interval_seconds=1,
    )

    assert bucket == 488


def test_sla_bucket_status_maps_ranges():
    assert sla_bucket_status(0) == "no_sla"
    assert sla_bucket_status(1) == "approaching_soft"
    assert sla_bucket_status(499) == "approaching_soft"
    assert sla_bucket_status(500) == "soft_missed"
    assert sla_bucket_status(999) == "soft_missed"
    assert sla_bucket_status(1000) == "hard_missed"


def test_summarize_sla_work_items_ignores_terminal_and_ranks_urgent():
    now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
    jobs = [
        NS(
            id="manual_urgent",
            dag_id="dag_1",
            name="manual_urgent",
            priority=100,
            job_level=2,
            state="created",
            soft_sla=now - timedelta(minutes=10),
            hard_sla=now + timedelta(hours=1),
        ),
        NS(
            id="hard_missed",
            dag_id="dag_2",
            name="hard_missed",
            priority=1,
            job_level=1,
            state="active",
            soft_sla=now - timedelta(hours=2),
            hard_sla=now - timedelta(minutes=31),
        ),
        NS(
            id="approaching",
            dag_id="dag_3",
            name="approaching",
            priority=5,
            job_level=3,
            state="created",
            soft_sla=now + timedelta(minutes=10),
            hard_sla=now + timedelta(hours=1),
        ),
        NS(
            id="done",
            dag_id="dag_4",
            name="done",
            priority=999,
            job_level=9,
            state="completed",
            soft_sla=now - timedelta(days=1),
            hard_sla=now - timedelta(hours=12),
        ),
    ]

    summary = summarize_sla_work_items(jobs, now=now, top_n=3)

    assert summary["tracked"] == 3
    assert summary["approaching_soft"] == 1
    assert summary["soft_missed"] == 1
    assert summary["hard_missed"] == 1
    assert summary["highest_bucket"] == 1002
    assert [row["id"] for row in summary["top_urgent"]] == [
        "hard_missed",
        "manual_urgent",
        "approaching",
    ]
