from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from marie.scheduler.repository import JobRepository


class OperationalConnection:
    def __init__(
        self,
        *,
        fetch: list[list[Any]] | None = None,
        fetchrow: list[Any] | None = None,
        fetchval: list[Any] | None = None,
    ) -> None:
        self.fetch_results = deque(fetch or [])
        self.fetchrow_results = deque(fetchrow or [])
        self.fetchval_results = deque(fetchval or [])
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.calls.append(("fetch", query, args))
        return self.fetch_results.popleft()

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.calls.append(("fetchrow", query, args))
        return self.fetchrow_results.popleft()

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.calls.append(("fetchval", query, args))
        return self.fetchval_results.popleft()


class OperationalPool:
    def __init__(self, connection: OperationalConnection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection

    def stats(self) -> dict[str, int]:
        return {
            "pool_min": 1,
            "pool_max": 20,
            "pool_size": 10,
            "pool_available": 7,
            "requests_waiting": 2,
        }


def repository(connection: OperationalConnection) -> JobRepository:
    return JobRepository({}, pool=OperationalPool(connection))


def job_row(*, state: str = "failed") -> tuple[Any, ...]:
    now = datetime.now(timezone.utc)
    created = now - timedelta(seconds=400)
    return (
        "06a69f16-ceb6-7c75-8000-37d2d2acb56e",
        "default",
        state,
        "92a69f16-ceb6-7c75-8000-37d2d2acb56e",
        "corr-index",
        "query-plan",
        2,
        1,
        1,
        3,
        created,
        created + timedelta(milliseconds=101),
        now - timedelta(seconds=2),
        now - timedelta(seconds=2),
        400,
        2,
        "gw-2",
        "84f69f16-ceb6-7c75-8000-37d2d2acb56e",
        "corr_index/rep-2",
        created + timedelta(milliseconds=72),
        now - timedelta(seconds=2),
        "failed",
        "failed",
        "executor",
        True,
    )


def job_page_row(*, state: str = "failed") -> tuple[Any, ...]:
    return (18_420, ["default", "claims"]) + job_row(state=state)


def dag_row() -> tuple[Any, ...]:
    now = datetime.now(timezone.utc)
    return (
        "92a69f16-ceb6-7c75-8000-37d2d2acb56e",
        "corr-index",
        "active",
        "query-plan",
        2,
        4,
        now - timedelta(minutes=20),
        now - timedelta(minutes=19),
        None,
        now - timedelta(seconds=4),
        1_200,
        4,
        4,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        ["default"],
        1,
        1,
        0,
        1,
        0,
        0,
    )


def dag_detail_row() -> tuple[Any, ...]:
    now = datetime.now(timezone.utc)
    return dag_row() + (
        [
            {
                "state": "created",
                "at": (now - timedelta(minutes=20)).isoformat(),
            },
            {"state": "active", "at": now.isoformat()},
        ],
    )


@pytest.mark.asyncio
async def test_operational_jobs_are_filtered_and_database_paged() -> None:
    connection = OperationalConnection(fetch=[[job_page_row()]])

    page = await repository(connection).list_operational_jobs(
        limit=25,
        offset=50,
        states=["active", "failed"],
        attention="failed",
        queue="default",
        search="corr",
        sort="attention",
    )

    assert page["page"] == {
        "limit": 25,
        "offset": 50,
        "total": 18_420,
        "has_next": True,
    }
    assert page["facets"]["queues"] == ["default", "claims"]
    assert page["items"][0]["attention"][0]["code"] == "FAILED"
    assert "data" not in page["items"][0]
    assert "output" not in page["items"][0]
    assert len(connection.calls) == 1
    method, query, params = connection.calls[0]
    assert method == "fetch"
    assert "marie_scheduler.list_operational_jobs" in query
    assert params == (
        25,
        50,
        ["active", "failed"],
        "failed",
        "default",
        "corr",
        "attention",
        None,
        300,
        900,
        600,
    )


@pytest.mark.asyncio
async def test_operational_job_detail_suppresses_payloads_and_raw_errors() -> None:
    now = datetime.now(timezone.utc)
    connection = OperationalConnection(
        fetchrow=[job_row()],
        fetch=[
            [("created", now - timedelta(seconds=3)), ("failed", now)],
            [
                (
                    "84f69f16-ceb6-7c75-8000-37d2d2acb56e",
                    "gw-2",
                    "scheduler-1",
                    "gateway-1",
                    "corr_index/rep-2",
                    "failed",
                    now - timedelta(seconds=2),
                    now,
                    "failed",
                    "failed",
                    "executor",
                    True,
                    None,
                    None,
                    now - timedelta(seconds=2),
                    now,
                )
            ],
        ],
    )

    detail = await repository(connection).get_operational_job(
        "06a69f16-ceb6-7c75-8000-37d2d2acb56e"
    )

    assert detail is not None
    assert detail["output_suppressed"] is True
    assert [event["state"] for event in detail["lifecycle"]] == [
        "created",
        "failed",
    ]
    assert detail["attempts"][0]["executor"] == "corr_index/rep-2"
    serialized = repr(detail)
    assert "dispatch_error" not in serialized
    assert "terminal_reject_reason" not in serialized
    assert "output" not in detail
    assert "data" not in detail


@pytest.mark.asyncio
async def test_execution_history_returns_paged_structured_worker_errors() -> None:
    now = datetime.now(timezone.utc)
    connection = OperationalConnection(
        fetch=[
            [
                (
                    73,
                    "92a69f16-ceb6-7c75-8000-37d2d2acb56e",
                    4102,
                    "06a69f16-ceb6-7c75-8000-37d2d2acb56e",
                    "corr_index",
                    now,
                    "U",
                    "FAILED",
                    "Job failed.",
                    "84f69f16-ceb6-7c75-8000-37d2d2acb56e",
                    "corr_index/rep-2",
                    "corr_indexing_executor",
                    "worker-2",
                    "/document/index",
                    "RuntimeError",
                    "Worker process stopped",
                    "worker.py",
                    "process_task",
                    "218",
                )
            ]
        ]
    )

    history = await repository(connection).list_operational_execution_history(
        job_id="06a69f16-ceb6-7c75-8000-37d2d2acb56e",
        limit=25,
        offset=25,
    )

    assert history is not None
    assert history["page"] == {
        "limit": 25,
        "offset": 25,
        "total": 73,
        "has_next": True,
    }
    assert history["scope"]["dag_id"] == (
        "92a69f16-ceb6-7c75-8000-37d2d2acb56e"
    )
    assert history["items"][0]["error"] == {
        "type": "RuntimeError",
        "message": "Worker process stopped",
        "file": "worker.py",
        "function": "process_task",
        "line": "218",
    }
    assert history["raw_runtime_environment_suppressed"] is True
    assert history["traceback_suppressed"] is True
    assert "runtime_env_json" not in repr(history)
    assert "traceback" not in history["items"][0]
    assert connection.calls[0][2] == (
        "06a69f16-ceb6-7c75-8000-37d2d2acb56e",
        None,
        25,
        25,
    )


@pytest.mark.asyncio
async def test_execution_history_requires_one_scope() -> None:
    connection = OperationalConnection(fetch=[])

    with pytest.raises(ValueError, match="exactly one"):
        await repository(connection).list_operational_execution_history()

    assert connection.calls == []


@pytest.mark.asyncio
async def test_operational_dags_use_database_rollups_and_pagination() -> None:
    connection = OperationalConnection(
        fetch=[[dag_row()], [("default",)]],
        fetchval=[71],
    )

    page = await repository(connection).list_operational_dags(
        limit=25,
        offset=25,
        states=["active"],
        attention="retrying",
        queue="default",
        search="corr",
        sort="updated",
    )

    assert page["page"]["total"] == 71
    assert page["items"][0]["jobs"] == {
        "total": 4,
        "created": 1,
        "retry": 1,
        "active": 1,
        "completed": 1,
        "skipped": 0,
        "expired": 0,
        "cancelled": 0,
        "failed": 0,
    }
    assert page["items"][0]["attention"][0]["code"] == "RUNNING_TOO_LONG"
    item_call = next(
        call
        for call in connection.calls
        if call[0] == "fetch" and "LIMIT %s OFFSET %s" in call[1]
    )
    assert item_call[2][-2:] == (25, 25)
    assert "serialized_dag" not in item_call[1]
    assert "ref_id" not in item_call[1]


@pytest.mark.asyncio
async def test_operational_dag_detail_keeps_rollup_and_bounded_job_page_separate() -> None:
    connection = OperationalConnection(
        fetchrow=[dag_detail_row()],
        fetch=[[(4, []) + job_row()]],
    )

    detail = await repository(connection).get_operational_dag(
        "92a69f16-ceb6-7c75-8000-37d2d2acb56e",
        job_limit=25,
        job_offset=0,
    )

    assert detail is not None
    assert detail["jobs"]["total"] == 4
    assert detail["job_page"]["page"]["total"] == 4
    assert len(detail["job_page"]["items"]) == 1
    assert detail["data_suppressed"] is True
    assert [event["state"] for event in detail["lifecycle"]] == [
        "created",
        "active",
    ]
    detail_call = next(call for call in connection.calls if call[0] == "fetchrow")
    assert "marie_scheduler.get_operational_dag" in detail_call[1]
    assert detail_call[2][1:] == (300, 900, 600)
    jobs_call = next(call for call in connection.calls if call[0] == "fetch")
    assert "marie_scheduler.list_operational_jobs" in jobs_call[1]
    assert jobs_call[2][6] == "timeline"
    assert jobs_call[2][7] == "92a69f16-ceb6-7c75-8000-37d2d2acb56e"


@pytest.mark.asyncio
async def test_operational_throughput_returns_bounded_reports() -> None:
    now = datetime.now(timezone.utc)
    system_total = (
        "window_total",
        now - timedelta(hours=24),
        now,
        False,
        12,
        10,
        1,
        0,
        1,
        83.33,
        120,
        100,
        4,
        1,
        0,
        15,
        96.0,
        0.42,
        4.17,
    )
    system_hour = (
        "hour",
        now - timedelta(hours=1),
        now,
        True,
        2,
        1,
        0,
        0,
        0,
        100.0,
        8,
        7,
        0,
        0,
        0,
        1,
        100.0,
        None,
        None,
    )
    planner_total = (
        "window_total",
        None,
        "query-plan",
        12,
        10,
        1,
        0,
        1,
        100,
        4,
        1,
        0,
    )
    task_total = (
        "window_total",
        None,
        "query-plan",
        "default",
        "extract",
        "/document/extract",
        True,
        75,
        2,
        0,
        0,
        3,
        0.42,
        0.91,
    )
    connection = OperationalConnection(
        fetch=[[system_total, system_hour], [planner_total], [task_total]],
    )

    detail = await repository(connection).get_operational_throughput(
        lookback_hours=24,
        planner=" query-plan ",
        planner_limit=10,
        task_limit=15,
    )

    assert detail["system"]["summary"]["avg_completed_plans_per_hour"] == 0.42
    assert detail["system"]["hourly"][0]["partial"] is True
    assert detail["planners"][0]["executor_tasks_completed"] == 100
    assert detail["tasks"][0]["p95_execution_seconds"] == 0.91
    assert detail["limits"] == {"planners": 10, "tasks": 15}
    assert [call[2] for call in connection.calls] == [
        (24, "query-plan"),
        (24, "query-plan", 10),
        (24, "query-plan", 15),
    ]
    assert all("LIMIT %s" in call[1] for call in connection.calls[1:])


@pytest.mark.asyncio
async def test_operational_attempts_are_filtered_and_database_paged() -> None:
    now = datetime.now(timezone.utc)
    attempt = (
        42,
        ["gateway-1"],
        ["corr_index/rep-2"],
        "84f69f16-ceb6-7c75-8000-37d2d2acb56e",
        "06a69f16-ceb6-7c75-8000-37d2d2acb56e",
        "default",
        "92a69f16-ceb6-7c75-8000-37d2d2acb56e",
        "gw-2",
        "scheduler-1",
        "gateway-1",
        "corr_index/rep-2",
        "activated",
        now - timedelta(minutes=20),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        now - timedelta(seconds=5),
        "retry",
        now - timedelta(minutes=20),
        now - timedelta(seconds=5),
        1_200.0,
        5.0,
        ["RECOVERED", "ACTIVE_TOO_LONG"],
    )
    connection = OperationalConnection(fetch=[[attempt]])

    page = await repository(connection).list_operational_attempts(
        limit=25,
        offset=25,
        states=["activated"],
        attention="recovered",
        gateway="gateway-1",
        executor="corr_index/rep-2",
        search="84f69f16",
        sort="updated",
    )

    assert page["page"] == {
        "limit": 25,
        "offset": 25,
        "total": 42,
        "has_next": True,
    }
    assert page["facets"] == {
        "gateways": ["gateway-1"],
        "executors": ["corr_index/rep-2"],
    }
    assert [item["code"] for item in page["items"][0]["attention"]] == [
        "RECOVERED",
        "ACTIVE_TOO_LONG",
    ]
    assert "dispatch_error" not in repr(page)
    assert "recovery_reason" not in repr(page)
    assert "marie_scheduler.list_operational_attempts" in connection.calls[0][1]


@pytest.mark.asyncio
async def test_operational_events_use_a_stable_cursor_and_hide_payloads() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        (
            "job:12",
            now,
            "bad",
            "scheduler.job",
            "JOB_FAILED",
            "job",
            "06a69f16-ceb6-7c75-8000-37d2d2acb56e",
            "06a69f16-ceb6-7c75-8000-37d2d2acb56e",
            "92a69f16-ceb6-7c75-8000-37d2d2acb56e",
            None,
            None,
            None,
            "Job state changed to failed",
        ),
        (
            "job:11",
            now - timedelta(seconds=1),
            "info",
            "scheduler.job",
            "JOB_ACTIVE",
            "job",
            "06a69f16-ceb6-7c75-8000-37d2d2acb56e",
            "06a69f16-ceb6-7c75-8000-37d2d2acb56e",
            "92a69f16-ceb6-7c75-8000-37d2d2acb56e",
            None,
            None,
            None,
            "Job state changed to active",
        ),
    ]
    connection = OperationalConnection(fetch=[rows])

    page = await repository(connection).list_operational_events(
        limit=1,
        window_seconds=900,
        severity="bad",
        search="06a69f16",
    )

    assert page["page"] == {
        "limit": 1,
        "has_next": True,
        "next_before_at": now.isoformat(),
        "next_before_id": "job:12",
    }
    assert page["items"][0]["code"] == "JOB_FAILED"
    assert "payload" not in repr(page)
    assert connection.calls[0][2] == (1, None, None, 900, "bad", None, "06a69f16")


@pytest.mark.asyncio
async def test_operational_flow_reports_unknown_dispatch_rates() -> None:
    now = datetime.now(timezone.utc)
    connection = OperationalConnection(
        fetchrow=[
            (
                now,
                30,
                28,
                24,
                23,
                20,
                2,
                40,
                12,
                120.0,
                0.5,
                2.5,
                8.0,
                1.2,
                4.8,
                10.0,
                [
                    {
                        "name": "default",
                        "arrivals": 30,
                        "terminals": 20,
                        "failures": 2,
                        "ready": 40,
                        "active": 12,
                        "oldest_ready_seconds": 120.0,
                    }
                ],
            )
        ]
    )

    flow = await repository(connection).get_operational_flow(window_seconds=300)

    assert flow["pressure"]["state"] == "growing"
    assert flow["rates"]["dispatch_per_second"] is None
    assert flow["rates"]["lease_per_second"] is None
    assert flow["queues"][0]["state"] == "growing"
    assert flow["stages"][0]["p95_seconds"] == 2.5


@pytest.mark.asyncio
async def test_operational_database_health_exposes_aggregates_only() -> None:
    connection = OperationalConnection(fetchrow=[(8, 2, 18.0)])

    health = await repository(connection).get_operational_database_health()

    assert health["schema_version"] == 89
    assert health["pool"] == {
        "minimum": 1,
        "maximum": 20,
        "size": 10,
        "available": 7,
        "used": 3,
        "waiters": 2,
    }
    assert health["blocked_sessions"] == 2
    assert "query" not in health
    assert "connection" not in health
