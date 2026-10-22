import asyncio
import json
import multiprocessing
import os
import statistics
import time
from contextlib import asynccontextmanager
from queue import Empty

import pytest

from tests.integration.scheduler._queue_consumer_harness import (
    owned_database,
    run_consumer,
    seed_dag,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MARIE_SCHEDULER_QUEUE_ISOLATION_INTEGRATION") != "1",
    reason="Set MARIE_SCHEDULER_QUEUE_ISOLATION_INTEGRATION=1 for owned-server process tests",
)


class Consumer:
    def __init__(self, config, scope, name, **kwargs):
        context = multiprocessing.get_context("spawn")
        self.output = context.Queue()
        self.commands = context.Queue()
        self.stopped = context.Event()
        self.records = []
        self.process = context.Process(
            target=run_consumer,
            args=(config, scope, name, self.output, self.commands, self.stopped),
            kwargs=kwargs,
        )
        self.process.start()

    def drain(self, check_errors=True):
        while True:
            try:
                self.records.append(self.output.get_nowait())
            except Empty:
                break
        errors = [row for row in self.records if row["phase"] == "error"]
        if check_errors:
            assert not errors, errors

    async def wait(self, phase, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.drain()
            rows = [row for row in self.records if row["phase"] == phase]
            if rows:
                return rows[-1]
            assert self.process.is_alive(), self.process.exitcode
            await asyncio.sleep(0.025)
        raise AssertionError((phase, self.records))

    async def close(self):
        if self.process.is_alive():
            self.stopped.set()
        deadline = time.monotonic() + 5
        while self.process.is_alive() and time.monotonic() < deadline:
            self.drain(check_errors=False)
            await asyncio.sleep(0.025)
        if self.process.is_alive():
            self.process.terminate()
            await asyncio.to_thread(self.process.join, 3)
        try:
            self.drain()
        finally:
            self.output.close()
            self.commands.close()


@asynccontextmanager
async def consumers(config, scopes, **kwargs):
    workers = [
        Consumer(config, scope, f"consumer-{i}", **kwargs)
        for i, scope in enumerate(scopes)
    ]
    try:
        for worker in workers:
            await worker.wait("started")
        yield workers
    finally:
        outcomes = await asyncio.gather(
            *(worker.close() for worker in workers), return_exceptions=True
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome


async def wait_jobs(conn, ids, timeout=20, workers=()):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for worker in workers:
            worker.drain()
        count = await conn.fetchval(
            "SELECT count(*) FROM marie_scheduler.job WHERE id=ANY(%s::uuid[]) AND state='completed'",
            ids,
        )
        if count == len(ids):
            return
        await asyncio.sleep(0.03)
    rows = await conn.fetch(
        "SELECT id,state,run_owner FROM marie_scheduler.job WHERE id=ANY(%s::uuid[])",
        ids,
    )
    raise AssertionError(rows)


async def test_disjoint_consumers_before_and_after_startup():
    async with owned_database() as (repo, conn, config):
        _, a = await seed_dag(repo, conn, "extract")
        _, b = await seed_dag(repo, conn, "kb_indexing")
        async with consumers(config, [["extract"], ["kb_indexing"]]) as workers:
            _, later = await seed_dag(
                repo, conn, "kb_indexing", entrypoint="noop://default"
            )
            await wait_jobs(conn, a + b + later)
            for worker, queue in zip(workers, ["extract", "kb_indexing"]):
                worker.drain()
                assert {
                    row["queue"]
                    for row in worker.records
                    if row["phase"] in {"admission", "lease", "dispatch"}
                } == {queue}
            assert (
                await conn.fetchval("SELECT count(*) FROM marie_scheduler.job_attempt")
                == 3
            )


async def test_omitted_queue_names_consumes_existing_and_new_queues():
    async with owned_database() as (repo, conn, config):
        _, first = await seed_dag(repo, conn, "extract")
        async with consumers(config, [None]) as (worker,):
            await wait_jobs(conn, first)
            _, second = await seed_dag(repo, conn, "new_queue")
            await wait_jobs(conn, second)
            worker.commands.put(dict(action="restart"))
            assert (await worker.wait("restarted"))["scope"] is None
            _, third = await seed_dag(repo, conn, "later_queue")
            await wait_jobs(conn, third)


@pytest.mark.parametrize("entrypoint", ["extract://default", "noop://default"])
async def test_foreign_hydration_and_frontier_injection(entrypoint):
    async with owned_database() as (repo, conn, config):
        dag, ids = await seed_dag(repo, conn, "kb_indexing", entrypoint=entrypoint)
        async with consumers(config, [["extract"]]) as (worker,):
            worker.commands.put(dict(action="inject", dag=dag, queue="kb_indexing"))
            result = await worker.wait("injected")
            assert not result["hydrated"] and not result["resident"]
            assert (
                await conn.fetchval(
                    "SELECT state::text FROM marie_scheduler.job WHERE id=%s", ids[0]
                )
                == "created"
            )
            assert (
                await conn.fetchval("SELECT count(*) FROM marie_scheduler.job_attempt")
                == 0
            )


async def test_mixed_parent_regular_injection_refused():
    async with owned_database() as (repo, conn, config):
        dag, ids = await seed_dag(repo, conn, "extract", children=2)
        await conn.execute(
            "UPDATE marie_scheduler.job SET name='foreign',state='completed' WHERE id=%s",
            ids[1],
        )
        async with consumers(config, [["extract"]]) as (worker,):
            worker.commands.put(dict(action="inject", dag=dag, queue="extract"))
            result = await worker.wait("injected")
            assert not result["hydrated"] and not result["resident"]
            assert (
                await conn.fetchval("SELECT count(*) FROM marie_scheduler.job_attempt")
                == 0
            )


async def test_storage_sync_stays_within_subscriptions():
    async with owned_database() as (repo, conn, config):
        _, ids = await seed_dag(repo, conn, "extract")
        async with consumers(config, [["extract"]], storage_only=True):
            await wait_jobs(conn, ids)
            assert (
                await conn.fetchval(
                    "SELECT attempt_state FROM marie_scheduler.job_attempt LIMIT 1"
                )
                == "completed"
            )


@pytest.mark.parametrize("scope", [None, ["extract"]])
async def test_submission_through_nonconsuming_gateway(scope):
    async with owned_database() as (repo, conn, config):
        async with consumers(config, [scope, ["kb_indexing"]]) as (
            producer,
            subscriber,
        ):
            producer.commands.put(dict(action="submit", queue="kb_indexing"))
            result = await producer.wait("submitted")
            assert result["registered"] and result["scope"] == scope
            ids = [
                str(row[0])
                for row in await conn.fetch(
                    "SELECT id FROM marie_scheduler.job WHERE dag_id=%s", result["dag"]
                )
            ]
            await wait_jobs(conn, ids)
            producer.drain()
            if scope is not None:
                assert not [r for r in producer.records if r["phase"] == "dispatch"]


async def test_foreign_backlog_does_not_fill_candidate_window():
    async with owned_database() as (repo, conn, config):
        for _ in range(105):
            await seed_dag(repo, conn, "extract", priority=100)
        _, allowed = await seed_dag(repo, conn, "kb_indexing")
        async with consumers(config, [["kb_indexing"]]):
            await wait_jobs(conn, allowed)
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM marie_scheduler.job WHERE name='extract' AND state='created'"
                )
                == 105
            )


@pytest.mark.parametrize("scopes", [[None, ["extract"]], [["extract", "kb_indexing"]]])
async def test_overlapping_and_combined_consumers(scopes):
    async with owned_database() as (repo, conn, config):
        _, a = await seed_dag(repo, conn, "extract", children=3)
        _, b = await seed_dag(repo, conn, "kb_indexing", children=3)
        async with consumers(config, scopes) as workers:
            await wait_jobs(conn, a + b)
            assert (
                await conn.fetchval("SELECT count(*) FROM marie_scheduler.job_attempt")
                == 6
            )
            for worker, scope in zip(workers, scopes):
                worker.drain()
                if scope is not None:
                    assert all(
                        r["queue"] in scope
                        for r in worker.records
                        if r["phase"] in {"admission", "lease", "dispatch"}
                    )


async def test_restart_narrower_subscription_and_missing_subscriber():
    async with owned_database() as (repo, conn, config):
        async with consumers(config, [None]):
            _, first = await seed_dag(repo, conn, "kb_indexing")
            await wait_jobs(conn, first)
        _, pending = await seed_dag(repo, conn, "kb_indexing")
        _, active = await seed_dag(repo, conn, "kb_indexing")
        await repo.lease_jobs(
            active, owner="dead", ttl_seconds=1, job_name="kb_indexing"
        )
        await repo.activate_from_lease(active, "dead", 1, "dead")
        await conn.execute(
            "UPDATE marie_scheduler.job SET run_lease_expires_at=now()-interval '1 minute' WHERE id=%s",
            active[0],
        )
        async with consumers(config, [["extract"]]) as (restricted,):
            restricted.commands.put(dict(action="recover"))
            await restricted.wait("recovered")
            states = await conn.fetch(
                "SELECT state::text FROM marie_scheduler.job WHERE id=ANY(%s::uuid[])",
                pending + active,
            )
            assert sorted(r[0] for r in states) == ["active", "created"]
            async with consumers(config, [["kb_indexing"]]):
                await wait_jobs(conn, pending + active)


@pytest.mark.parametrize(
    "mode,scopes",
    [
        ("single", [None]),
        ("disjoint", [["extract"], ["kb_indexing"]]),
        ("overlap", [["extract"], ["extract"]]),
    ],
)
async def test_scheduler_throughput_and_dispatch_latency(mode, scopes, tmp_path):
    async with owned_database() as (repo, conn, config):
        ids = []
        for i in range(100):
            queue = "extract" if mode == "overlap" or i % 2 == 0 else "kb_indexing"
            _, jobs = await seed_dag(repo, conn, queue, children=4)
            ids.extend(jobs)
        async with consumers(config, scopes) as workers:
            for i in range(10):
                queue = "extract" if mode == "overlap" or i % 2 == 0 else "kb_indexing"
                _, jobs = await seed_dag(repo, conn, queue, children=4)
                ids.extend(jobs)
            await wait_jobs(conn, ids, timeout=60, workers=workers)
            completed_at = time.monotonic()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                for worker in workers:
                    worker.drain()
                if sum(
                    row["phase"] == "dispatch"
                    for worker in workers
                    for row in worker.records
                ) >= len(ids):
                    break
                await asyncio.sleep(0.025)
            records = [row for worker in workers for row in worker.records]
            dispatched = [row for row in records if row["phase"] == "dispatch"]
            assert len(dispatched) == len(ids)
            assert {row["job"] for row in dispatched} == set(ids)
            admitted = {}
            for row in records:
                if row["phase"] == "admission":
                    key = (row["instance"], row["dag"])
                    admitted[key] = min(
                        admitted.get(key, float("inf")), row["observed_at"]
                    )
            first_dispatch = {}
            for row in dispatched:
                previous = first_dispatch.get(row["dag"])
                if previous is None or row["observed_at"] < previous["observed_at"]:
                    first_dispatch[row["dag"]] = row
            latency = sorted(
                (row["observed_at"] - admitted[(row["instance"], row["dag"])]) * 1000
                for row in first_dispatch.values()
            )
            assert len(latency) == 110 and min(latency) >= 0
            report = dict(
                mode=mode,
                completed_jobs=len(ids),
                completed_jobs_per_second=len(ids)
                / (completed_at - min(row["observed_at"] for row in dispatched)),
                dag_latency_samples=len(latency),
                candidate_to_first_dispatch_p50_ms=statistics.median(latency),
                candidate_to_first_dispatch_p95_ms=latency[
                    int(0.95 * (len(latency) - 1))
                ],
                records=records,
            )
            (tmp_path / "scheduler-performance.json").write_text(
                json.dumps(report, indent=2)
            )


@pytest.mark.parametrize("phase", ["acquisition", "activation"])
async def test_same_queue_failover(phase):
    async with owned_database() as (repo, conn, config):
        _, ids = await seed_dag(repo, conn, "extract")
        dead = Consumer(config, ["extract"], "dead", crash_after=phase)
        try:
            await dead.wait("lease" if phase == "acquisition" else "dispatch")
            dead.process.terminate()
            await asyncio.to_thread(dead.process.join, 3)
            await conn.execute(
                "UPDATE marie_scheduler.job SET lease_expires_at=now()-interval '1 minute',run_lease_expires_at=now()-interval '1 minute'"
            )
            async with consumers(config, [["kb_indexing"], ["extract"]]) as workers:
                await wait_jobs(conn, ids)
                workers[0].drain()
                assert not [r for r in workers[0].records if r["phase"] == "dispatch"]
                if phase == "activation":
                    states = await conn.fetch(
                        "SELECT attempt_state FROM marie_scheduler.job_attempt ORDER BY activated_at"
                    )
                    assert sorted(row[0] for row in states) == [
                        "completed",
                        "recovered_retry",
                    ]
        finally:
            await dead.close()
