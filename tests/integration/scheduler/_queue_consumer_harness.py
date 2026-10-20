import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from marie.scheduler.repository import JobRepository
from marie.storage.database.postgres_pool import AsyncPostgresConnection


@asynccontextmanager
async def owned_database() -> (
    AsyncIterator[tuple[JobRepository, AsyncPostgresConnection, dict]]
):
    port = os.environ.get("MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT")
    if not port:
        pytest.skip(
            "Set MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT to an owned disposable server"
        )
    database = "queue_isolation_" + uuid4().hex
    connection_args = dict(
        host="127.0.0.1", port=int(port), user="postgres", autocommit=True
    )
    async with await psycopg.AsyncConnection.connect(
        dbname="postgres", **connection_args
    ) as admin:
        await admin.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
        )
        try:
            async with await psycopg.AsyncConnection.connect(
                dbname=database, **connection_args
            ) as raw:
                conn = AsyncPostgresConnection(raw)
                await conn.execute(
                    """
                    CREATE EXTENSION vector;
                    CREATE EXTENSION pg_trgm;
                    CREATE SCHEMA cron;
                    CREATE TABLE cron.job(jobid bigint, jobname text);
                    CREATE FUNCTION cron.unschedule(bigint) RETURNS boolean
                        LANGUAGE sql AS 'SELECT true';
                """
                )

                class Pool:
                    @asynccontextmanager
                    async def acquire(self) -> AsyncIterator[AsyncPostgresConnection]:
                        yield conn

                await conn.execute(
                    """CREATE TABLE public.queue_test_runtime (
                    id uuid PRIMARY KEY, status text, entrypoint text,
                    run_owner text, run_attempt_id uuid, end_time bigint
                )"""
                )
                config = dict(
                    hostname="127.0.0.1",
                    port=int(port),
                    username="postgres",
                    password="",
                    database=database,
                )
                repository = JobRepository(config=config, pool=Pool())
                await repository.create_tables()
                yield repository, conn, config
        finally:
            await admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                    sql.Identifier(database)
                )
            )


async def seed_dag(
    repository: JobRepository,
    conn: AsyncPostgresConnection,
    queue: str,
    *,
    priority: int = 0,
    projection: str | None = "same",
    children: int = 1,
    entrypoint: str = "extract://default",
) -> tuple[str, list[str]]:
    await repository.create_queue(queue)
    dag_id = str(uuid4())
    job_ids = [str(uuid4()) for _ in range(children)]
    plan = {
        "nodes": [
            dict(
                task_id=job_id,
                query_str="Queue isolation fixture",
                dependencies=[],
                definition=dict(
                    method=(
                        "NOOP" if entrypoint.startswith("noop") else "EXECUTOR_ENDPOINT"
                    ),
                    endpoint=entrypoint,
                    params={},
                ),
            )
            for job_id in job_ids
        ]
    }
    await conn.execute(
        """INSERT INTO marie_scheduler.dag
        (id, name, state, serialized_dag, submission_name, priority)
        VALUES (%s, %s, 'created', %s, %s, %s)""",
        dag_id,
        dag_id + "_dag",
        Jsonb(plan),
        queue if projection == "same" else projection,
        priority,
    )
    for job_id in job_ids:
        await conn.execute(
            """INSERT INTO marie_scheduler.job
            (id, dag_id, name, state, priority, data, start_after, keep_until)
            VALUES (%s, %s, %s, 'created', %s, %s, now(), now() + interval '1 day')""",
            job_id,
            dag_id,
            queue,
            priority,
            Jsonb({"metadata": {"on": entrypoint}}),
        )
    return dag_id, job_ids


def run_consumer(
    config: dict,
    scope: list[str] | None,
    name: str,
    output,
    commands,
    stopped,
    crash_after: str | None = None,
    storage_only: bool = False,
) -> None:
    import asyncio
    import tempfile
    import time
    import traceback
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    from queue import Empty
    from types import SimpleNamespace

    from marie.job.common import JobInfo, JobStatus
    from marie.job.event_publisher import EventPublisher
    from marie.scheduler import psql
    from marie.state.semaphore_store import SemaphoreReleaseResult

    destination = output

    class Records:
        def put(self, record):
            record.setdefault("observed_at", time.monotonic())
            destination.put(record)

    output = Records()

    class Semaphores:
        def __init__(self, *args, **kwargs):
            self.holders = {}

        def reconcile_all(self, **kwargs):
            return {}

        def available_count_all(self):
            return {"extract": max(0, 8 - len(self.holders))}

        def reserve_many(self, executor, ids, **kwargs):
            reserved = set(ids) - self.holders.keys()
            for job_id in reserved:
                self.holders[job_id] = kwargs["run_attempt_id_by_ticket"][job_id]
            return reserved

        def renew(self, executor, job_id, **kwargs):
            return self.holders.get(job_id) == kwargs["run_attempt_id"]

        def release_owned(self, executor, job_id, **kwargs):
            if self.holders.get(job_id) == kwargs.get("run_attempt_id"):
                self.holders.pop(job_id)
            return SemaphoreReleaseResult(True, "test_release", 1)

    async def run() -> None:
        psql.SemaphoreStore = Semaphores
        psql.SYNC_POLL_PERIOD = 0.05
        original_start_sync = psql.DAGManagementService.start_sync

        async def start_sync(service, interval=30):
            await original_start_sync(service, sync_interval=0.05)

        psql.DAGManagementService.start_sync = start_sync
        from marie.query_planner import mapper
        from marie.query_planner.base import (
            ExecutorEndpointQueryDefinition,
            Query,
            QueryPlan,
            QueryPlanRegistry,
        )
        from marie.scheduler import planner_util
        from marie.scheduler.models import WorkInfo
        from marie.scheduler.state import WorkState

        mapper.__default_extract_dir__ = artifacts.name
        planner_util.__default_extract_dir__ = artifacts.name
        mapper_path = Path(artifacts.name) / "base"
        mapper_path.mkdir()
        (mapper_path / "mapper.yml").write_text(
            "model_executors: {}\nserverless_executor: extract\n"
        )
        original_plan_to_yaml = planner_util.plan_to_yaml
        planner_util.plan_to_yaml = lambda plan: original_plan_to_yaml(
            plan, str(Path(artifacts.name) / "plan.yaml")
        )

        class Manager:
            _etcd_client = None

            def __init__(self):
                self.event_publisher = EventPublisher(
                    worker_count=1, publish_blocking=True
                )
                self.tasks = set()

            async def handle_job_status_notification(self, payload):
                pass

            def job_info_client(self):
                return self

            async def get_info(self, job_id):
                async with scheduler.repository._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT status,entrypoint,run_owner,run_attempt_id,end_time FROM public.queue_test_runtime WHERE id=%s",
                        job_id,
                    )
                return (
                    JobInfo(
                        status=JobStatus(row[0]),
                        entrypoint=row[1],
                        run_owner=row[2],
                        run_attempt_id=str(row[3]),
                        end_time=row[4],
                    )
                    if row
                    else None
                )

            async def get_job_status(self, job_id):
                info = await self.get_info(job_id)
                return info.status if info else None

            async def submit_job(
                self,
                *,
                entrypoint,
                submission_id,
                metadata,
                confirmation_event,
                run_owner,
                run_attempt_id,
                **kwargs,
            ):
                item = await scheduler.repository.get_job_by_id(submission_id)
                assert (
                    item.run_owner == run_owner
                    and str(item.run_attempt_id) == run_attempt_id
                )
                async with scheduler.repository._pool.acquire() as conn:
                    await conn.execute(
                        """INSERT INTO public.queue_test_runtime
                        VALUES (%s,'PENDING',%s,%s,%s,NULL) ON CONFLICT(id) DO UPDATE
                        SET status='PENDING',entrypoint=excluded.entrypoint,run_owner=excluded.run_owner,
                            run_attempt_id=excluded.run_attempt_id,end_time=NULL""",
                        submission_id,
                        entrypoint,
                        run_owner,
                        run_attempt_id,
                    )
                output.put(
                    dict(
                        phase="dispatch",
                        instance=name,
                        queue=item.name,
                        job=submission_id,
                        dag=item.dag_id,
                        owner=run_owner,
                        attempt=run_attempt_id,
                    )
                )
                confirmation_event.set()
                if crash_after == "activation":
                    await asyncio.Event().wait()
                task = asyncio.create_task(self.finish(submission_id))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
                return submission_id

            async def finish(self, job_id):
                async with scheduler.repository._pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE public.queue_test_runtime SET status='RUNNING' WHERE id=%s",
                        job_id,
                    )
                info = await self.get_info(job_id)
                await self.event_publisher.publish(
                    JobStatus.RUNNING,
                    dict(
                        job_id=job_id,
                        run_owner=info.run_owner,
                        run_attempt_id=info.run_attempt_id,
                    ),
                )
                await asyncio.sleep(0.03)
                async with scheduler.repository._pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE public.queue_test_runtime SET status='SUCCEEDED',end_time=(extract(epoch FROM now()-interval '10 minutes')*1000)::bigint WHERE id=%s",
                        job_id,
                    )
                if not storage_only:
                    await self.event_publisher.publish(
                        JobStatus.SUCCEEDED,
                        dict(
                            job_id=job_id,
                            run_owner=info.run_owner,
                            run_attempt_id=info.run_attempt_id,
                        ),
                    )

                scheduler._semaphore_store.release_owned(
                    info.entrypoint.split("://", 1)[0],
                    job_id,
                    run_attempt_id=info.run_attempt_id,
                    owner=job_id,
                )

        settings = dict(
            config,
            gateway_instance_id=name,
            maintenance_interval=1,
            lease_ttl_seconds=2,
            run_ttl_seconds=6,
            run_lease_renewal_interval_seconds=1,
            min_pool_size=1,
            max_pool_size=5,
            dag_manager={"max_concurrent_dags": 8},
        )
        if scope is not None:
            settings["queue_names"] = scope
        manager = Manager()
        scheduler = psql.PostgreSQLJobScheduler(settings, manager)
        QueryPlanRegistry.register(
            "queue-isolation-test",
            lambda info: QueryPlan(
                nodes=[
                    Query(
                        task_id=str(uuid4()),
                        query_str="Test submission",
                        definition=ExecutorEndpointQueryDefinition(
                            endpoint="extract://default", params={}
                        ),
                    )
                ]
            ),
        )
        original_lease = scheduler._lease_jobs_db

        async def lease(queue, ids):
            leased = await original_lease(queue, ids)
            for job_id in leased:
                output.put(
                    dict(
                        phase="lease",
                        instance=name,
                        queue=queue,
                        job=job_id,
                        owner=scheduler.lease_owner,
                    )
                )
            if leased and crash_after == "acquisition":
                await asyncio.Event().wait()
            return leased

        scheduler._lease_jobs_db = lease
        original_admit = scheduler.dag_service._admit_dag

        async def admit(dag_id, dag, **kwargs):
            result = await original_admit(dag_id, dag, **kwargs)
            admitted_at = time.monotonic()
            if result[0]:
                identity = await scheduler.repository.get_dag_queue_identity(dag_id)
                output.put(
                    dict(
                        phase="admission",
                        instance=name,
                        dag=dag_id,
                        queue=identity[0],
                        observed_at=admitted_at,
                    )
                )
            return result

        scheduler.dag_service._admit_dag = admit
        await scheduler.start()
        output.put(
            dict(
                phase="started",
                instance=name,
                owner=scheduler.lease_owner,
                scope=(
                    None
                    if scheduler.subscribed_queues is None
                    else sorted(scheduler.subscribed_queues)
                ),
            )
        )
        try:
            while not stopped.is_set():
                try:
                    command = commands.get_nowait()
                except Empty:
                    await asyncio.sleep(0.02)
                    continue
                action = command["action"]
                if action == "inject":
                    dag_id = command["dag"]
                    accepted = await scheduler.dag_service.hydrate_single_dag(dag_id)
                    await scheduler.dag_service.stop_sync()
                    dag = await scheduler.repository.get_dag_by_id(dag_id)
                    jobs = await scheduler.repository.list_jobs(
                        queue=command["queue"], limit=0
                    )
                    jobs = [item for item in jobs if item.dag_id == dag_id]
                    scheduler.active_dags[dag_id] = dag
                    await scheduler.frontier.add_dag(dag, jobs)
                    await scheduler.run_dispatch_cycle(999)
                    output.put(
                        dict(
                            phase="injected",
                            instance=name,
                            dag=dag_id,
                            hydrated=accepted,
                            resident=dag_id in scheduler.active_dags,
                        )
                    )
                elif action == "restart":
                    await scheduler.stop()
                    await scheduler.start()
                    output.put(
                        dict(
                            phase="restarted",
                            instance=name,
                            scope=(
                                None
                                if scheduler.subscribed_queues is None
                                else sorted(scheduler.subscribed_queues)
                            ),
                        )
                    )
                elif action == "recover":
                    await scheduler.maintenance_service.expire()
                    output.put(dict(phase="recovered", instance=name))
                elif action == "submit":
                    now = datetime.now(timezone.utc)
                    item = WorkInfo(
                        id=str(uuid4()),
                        name=command["queue"],
                        data={"metadata": {"planner": "queue-isolation-test"}},
                        state=WorkState.CREATED,
                        retry_limit=2,
                        retry_delay=0,
                        retry_backoff=False,
                        start_after=now,
                        expire_in_seconds=60,
                        keep_until=now + timedelta(days=1),
                        policy="ALLOW_ALL",
                    )
                    dag_id = await scheduler.submit_job(item)
                    output.put(
                        dict(
                            phase="submitted",
                            instance=name,
                            dag=dag_id,
                            scope=(
                                None
                                if scheduler.subscribed_queues is None
                                else sorted(scheduler.subscribed_queues)
                            ),
                            registered=command["queue"] in scheduler.known_queues,
                        )
                    )
        finally:
            if manager.tasks:
                await asyncio.gather(*tuple(manager.tasks))
            await scheduler.stop(timeout=3)
            await manager.event_publisher.stop()

    try:
        with tempfile.TemporaryDirectory(prefix="queue-consumer-") as directory:
            artifacts = SimpleNamespace(name=directory)
            asyncio.run(run())
    except BaseException:
        output.put(dict(phase="error", instance=name, error=traceback.format_exc()))
        raise
