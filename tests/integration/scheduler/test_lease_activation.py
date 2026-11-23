import asyncio
from uuid import UUID, uuid4

import psycopg
import pytest

from marie.scheduler.repository.plans import cancel_jobs
from marie.storage.database.postgres_pool import AsyncPostgresConnection
from tests.integration.scheduler._queue_consumer_harness import owned_database, seed_dag


@pytest.mark.parametrize("executor", [True, False], ids=["executor", "control_flow"])
@pytest.mark.parametrize("conflict", ["cancel", "replace_owner", "expire_while_locked"])
async def test_activation_rechecks_lease_after_waiting(
    executor: bool, conflict: str
) -> None:
    async with owned_database() as (repo, conn, config):
        _, jobs = await seed_dag(repo, conn, "extract")
        job_id = jobs[0]
        activating_pid = await conn.fetchval("SELECT pg_backend_pid()")
        async with await psycopg.AsyncConnection.connect(
            host=config["hostname"],
            port=config["port"],
            dbname=config["database"],
            user=config["username"],
            autocommit=True,
        ) as raw:
            blocker = AsyncPostgresConnection(raw)
            ttl = 2 if conflict == "expire_while_locked" else 60
            assert await repo.lease_jobs(jobs, "owner", ttl, "extract") == {job_id}
            completed_on = None
            async with asyncio.TaskGroup() as tasks:
                async with blocker.transaction():
                    if conflict == "cancel":
                        assert (
                            await blocker.fetchval(
                                cancel_jobs("marie_scheduler", "extract", jobs)
                            )
                            == 1
                        )
                        completed_on = await blocker.fetchval(
                            "SELECT completed_on FROM marie_scheduler.job WHERE id=%s",
                            job_id,
                        )
                        assert completed_on is not None
                    elif conflict == "replace_owner":
                        await blocker.execute(
                            "UPDATE marie_scheduler.job SET lease_owner='successor' WHERE id=%s",
                            job_id,
                        )
                    else:
                        await blocker.execute(
                            "SELECT id FROM marie_scheduler.job WHERE id=%s FOR UPDATE",
                            job_id,
                        )
                    activation = tasks.create_task(
                        repo.activate_from_lease(
                            jobs,
                            "owner",
                            60,
                            "gateway",
                            run_attempt_ids=(
                                {job_id: str(uuid4())} if executor else None
                            ),
                        )
                    )
                    async with asyncio.timeout(5):
                        while not await blocker.fetchval(
                            "SELECT pg_backend_pid() = ANY(pg_blocking_pids(%s))",
                            activating_pid,
                        ):
                            await asyncio.sleep(0.01)
                    if conflict == "expire_while_locked":
                        await blocker.execute(
                            """SELECT pg_sleep(GREATEST(0, EXTRACT(EPOCH FROM
                                   lease_expires_at - clock_timestamp())))
                               FROM marie_scheduler.job WHERE id=%s""",
                            job_id,
                        )

            assert activation.result() == {}
            state = "cancelled" if conflict == "cancel" else "created"
            lease_owner = {
                "cancel": None,
                "replace_owner": "successor",
                "expire_while_locked": "owner",
            }
            job = await conn.fetchrow(
                """SELECT state::text, completed_on, lease_owner, run_owner, run_attempt_id
                   FROM marie_scheduler.job WHERE id=%s""",
                job_id,
            )
            assert job == (state, completed_on, lease_owner[conflict], None, None)
            attempts = await conn.fetchval(
                "SELECT count(*) FROM marie_scheduler.job_attempt WHERE job_id=%s",
                job_id,
            )
            assert attempts == 0


@pytest.mark.parametrize("executor", [True, False], ids=["executor", "control_flow"])
async def test_activation_audits_only_eligible_jobs(executor: bool) -> None:
    async with owned_database() as (repo, conn, _):
        _, jobs = await seed_dag(repo, conn, "extract", children=5)
        created, retry, expired, foreign, terminal = jobs
        assert await repo.lease_jobs(jobs, "owner", 60, "extract") == set(jobs)
        await conn.execute(
            "UPDATE marie_scheduler.job SET state='retry' WHERE id=%s", retry
        )
        await conn.execute(
            "UPDATE marie_scheduler.job SET lease_expires_at=clock_timestamp() WHERE id=%s",
            expired,
        )
        await conn.execute(
            "UPDATE marie_scheduler.job SET lease_owner='other' WHERE id=%s", foreign
        )
        await conn.execute(
            "UPDATE marie_scheduler.job SET state='cancelled' WHERE id=%s", terminal
        )
        supplied = {job_id: str(uuid4()) for job_id in jobs} if executor else None
        activated = await repo.activate_from_lease(
            jobs, "owner", 60, "gateway", supplied
        )

        assert set(activated) == {created, retry}
        assert len({UUID(attempt) for attempt in activated.values()}) == 2
        if supplied is not None:
            assert activated == {job: supplied[job] for job in (created, retry)}
        audits = await conn.fetch(
            """SELECT job_id::text, run_attempt_id::text FROM marie_scheduler.job_attempt
               WHERE attempt_state='activated'"""
        )
        active = await conn.fetch(
            """SELECT id::text, run_attempt_id::text FROM marie_scheduler.job
               WHERE state='active' AND run_owner='owner'"""
        )
        assert dict(audits) == dict(active) == activated


async def test_fresh_database_installs_current_activation_signatures() -> None:
    async with owned_database() as (repo, conn, _):
        signatures = await conn.fetch(
            """SELECT oidvectortypes(proargtypes) FROM pg_proc
               WHERE pronamespace = 'marie_scheduler'::regnamespace
                 AND proname = 'activate_from_lease'"""
        )
        assert {row[0] for row in signatures} == {
            "uuid[], uuid[], text, interval, text",
            "uuid[], text, interval, text",
        }
        await repo.validate_durable_scheduler_schema()
