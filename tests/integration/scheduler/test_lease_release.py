from marie.scheduler.psql import PostgreSQLJobScheduler
from tests.integration.scheduler._queue_consumer_harness import owned_database, seed_dag


async def test_late_release_preserves_the_successors_claim() -> None:
    async with owned_database() as (repo, conn, _):
        scheduler = object.__new__(PostgreSQLJobScheduler)
        scheduler.repository = repo
        scheduler.lease_owner = "scheduler-a"
        _, jobs = await seed_dag(repo, conn, "extract", children=2)
        retained, reclaimed = jobs
        claimed = await repo.lease_jobs(jobs, scheduler.lease_owner, 60, "extract")
        assert claimed == set(jobs)
        await conn.execute(
            "UPDATE marie_scheduler.job SET lease_expires_at=clock_timestamp() WHERE id=%s",
            reclaimed,
        )
        successor_claim = await repo.lease_jobs(
            [reclaimed], "scheduler-b", 60, "extract"
        )
        assert successor_claim == {reclaimed}

        assert await scheduler._release_lease_db([reclaimed]) == set()
        assert await scheduler._release_lease_db(jobs) == {retained}
        released = await conn.fetchrow(
            "SELECT lease_owner, lease_expires_at FROM marie_scheduler.job WHERE id=%s",
            retained,
        )
        assert released == (None, None)
        activated = await repo.activate_from_lease(
            [reclaimed], "scheduler-b", 60, "gateway-b"
        )
        assert set(activated) == {reclaimed}
