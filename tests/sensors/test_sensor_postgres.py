"""Qualification against an explicitly selected disposable PostgreSQL database."""

import asyncio
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from psycopg.rows import tuple_row

from marie.logging_core.logger import MarieLogger
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.repository.async_job_repository import AsyncJobRepository
from marie.scheduler.services.dag_submission_service import DagSubmissionService
from marie.sensors.daemon.worker import SensorWorker, build_work_info
from marie.sensors.definitions.schedule_sensor import ScheduleSensor
from marie.sensors.state.psql_storage import PostgreSQLSensorStorage
from marie.sensors.types import RunRequest, TickStatus
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Set disposable PostgreSQL test port',
)


@pytest.fixture
async def runtime(monkeypatch):
    config = dict(
        hostname='127.0.0.1',
        port=int(os.environ['MARIE_SENSOR_TEST_PORT']),
        username='postgres',
        password='',
        database='f10c_qualification',
        max_connections=12,
    )
    pool = AsyncPostgresConnectionPool()
    await pool.initialize(config, autocommit=True)
    repo_pool = AsyncPostgresConnectionPool()
    await repo_pool.initialize(config, row_factory=tuple_row, autocommit=True)
    await pool.execute('CREATE SCHEMA IF NOT EXISTS marie_scheduler')
    directory = Path(__file__).parents[2] / 'config/psql/schema'
    for name in [
        '002_enums.sql',
        '004_queue.sql',
        '005_job.sql',
        '007_dag.sql',
        '062_job_search_document.sql',
        '051_sensor_enums.sql',
        '052_sensor_tables.sql',
        '093_sensor_evaluation_indexes.sql',
    ]:
        await pool.execute(
            (directory / name).read_text().replace('{schema}', 'marie_scheduler')
        )
    await pool.execute(
        'ALTER TABLE marie_scheduler.job ADD COLUMN IF NOT EXISTS llm_routing_ready BOOLEAN NOT NULL DEFAULT TRUE'
    )
    await pool.execute(
        "INSERT INTO marie_scheduler.queue(name) VALUES ('mock_connector_tool') ON CONFLICT DO NOTHING"
    )
    await pool.execute(
        'TRUNCATE marie_scheduler.sensor, marie_scheduler.job, marie_scheduler.dag CASCADE'
    )
    storage = PostgreSQLSensorStorage(pool)
    repository = AsyncJobRepository({}, pool=repo_pool)
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.repository = repository

    async def ready(*args):
        return True

    service = DagSubmissionService(
        repository=repository,
        dag_admission_callback=ready,
        known_queues={'mock_connector_tool'},
        notify_callback=ready,
        is_running=lambda: True,
        submission_processed_callback=None,
        logger=MarieLogger('controlled-admission'),
    )
    scheduler.submit_job = service.persist
    monkeypatch.setattr('marie.scheduler.planner_util.plan_to_yaml', lambda plan: None)
    from marie.query_planner.mock_plans import (
        connector,  # register the controlled planner
    )

    worker = SensorWorker({}, job_scheduler=scheduler)
    worker.set_storage(storage)
    from marie.sensors.types import SensorType

    worker._registry.register(SensorType.SCHEDULE, ScheduleSensor)
    try:
        yield SimpleNamespace(
            pool=pool, storage=storage, scheduler=scheduler, worker=worker
        )
    finally:
        await repo_pool.close()
        await pool.close()


async def add_source(runtime):
    sid = await runtime.storage.create_sensor(
        {
            'external_id': str(uuid4()),
            'name': 'controlled source',
            'sensor_type': 'schedule',
            'status': 'active',
            'target_job_name': 'mock_connector_tool',
            'config': {'cron': '* * * * *', 'timezone': 'UTC'},
        }
    )
    await runtime.storage.set_cursor(
        sid, (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    )
    return await runtime.storage.get_sensor(sid)


async def test_concurrent_evaluations_admit_one_dag(runtime):
    source = await add_source(runtime)
    await asyncio.gather(
        *(runtime.worker._evaluate_sensor_with_tick(source) for _ in range(8))
    )
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 1
    assert (await runtime.storage.get_sensor(source['id']))['cursor'] != source[
        'cursor'
    ]
    assert (await runtime.storage.get_sensor(source['id']))['last_run_key']


async def test_crash_after_durable_admission_and_partial_recovery(runtime, monkeypatch):
    source = await add_source(runtime)
    request = RunRequest(run_key='first', job_name='mock_connector_tool')
    other = RunRequest(run_key='second', job_name='mock_connector_tool')
    first_id = build_work_info(
        request, source['id'], 'n', 'mock_connector_tool', None
    ).id
    second_id = build_work_info(
        other, source['id'], 'n', 'mock_connector_tool', None
    ).id
    tick = await runtime.storage.create_tick(
        source['id'],
        TickStatus.STARTED,
        [request.to_dict(), other.to_dict()],
        [first_id, second_id],
        cursor='2026-11-05T00:00:00+00:00',
    )
    original = runtime.storage.record_run_key

    async def crash(*args):
        raise RuntimeError('interrupted after scheduler acceptance')

    monkeypatch.setattr(runtime.storage, 'record_run_key', crash)
    with pytest.raises(RuntimeError):
        await runtime.worker._submit_run_request(source, request)
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 1
    monkeypatch.setattr(runtime.storage, 'record_run_key', original)
    # A new worker reads durable STARTED requests and reconciles the accepted DAG.
    restarted = SensorWorker({}, job_scheduler=runtime.scheduler)
    restarted.set_storage(runtime.storage)
    await restarted._evaluate_sensor_with_tick(source)
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 2
    assert (await runtime.storage.get_tick(tick))['status'] == 'success'
    assert len((await runtime.storage.get_tick(tick))['run_ids']) == 2
    assert (await runtime.storage.get_sensor(source['id']))[
        'cursor'
    ] == '2026-11-05T00:00:00+00:00'


async def test_failed_admission_preserves_cursor_and_receipt_ownership(runtime):
    source = await add_source(runtime)
    runtime.worker._job_scheduler = None
    await runtime.worker._evaluate_sensor_with_tick(source)
    assert (await runtime.storage.get_sensor(source['id']))['cursor'] == source[
        'cursor'
    ]
    pending = await runtime.storage.get_pending_tick(source['id'])
    assert pending['status'] == 'failed'
    assert pending['cursor'] != source['cursor']
    runtime.worker._job_scheduler = runtime.scheduler
    await runtime.worker._evaluate_sensor_with_tick(source)
    assert (await runtime.storage.get_tick(pending['id']))['status'] == 'success'
    dag_id = await runtime.pool.fetchval('SELECT id FROM marie_scheduler.dag LIMIT 1')
    with pytest.raises(ValueError, match='different occurrence'):
        await runtime.scheduler.get_sensor_job_state(
            str(dag_id), str(uuid4()), 'different', 'mock_connector_tool'
        )


async def test_250_sources_progress_and_keep_plugin_work_outside_sensor(
    runtime, monkeypatch, tmp_path
):
    from docarray import DocList
    from docarray.documents import TextDoc

    from marie.executor.extensions.plugin_daemon_executor import (
        MariePluginDaemonExecutor,
    )

    for _ in range(250):
        await add_source(runtime)
    cycles = []
    queued_after_cycle = []
    for _ in range(3):
        start = time.perf_counter()
        await runtime.worker._evaluate_all_sensors()
        cycles.append(round((time.perf_counter() - start) * 1000, 2))
        queued_after_cycle.append(
            await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag")
        )
    assert (
        await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 250
    )
    assert (
        await runtime.pool.fetchval(
            'SELECT count(*) FROM marie_scheduler.sensor WHERE last_run_key IS NOT NULL'
        )
        == 250
    )
    # Restart with queued work and the same due schedules; no overlapping source runs.
    restarted = SensorWorker({}, job_scheduler=runtime.scheduler)
    restarted.set_storage(runtime.storage)
    await runtime.pool.execute(
        "UPDATE marie_scheduler.sensor SET last_tick_at=NOW()-interval '1 hour'"
    )
    await restarted._evaluate_all_sensors()
    assert (
        await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 250
    )
    rows = await runtime.pool.fetch(
        'SELECT id, serialized_dag FROM marie_scheduler.dag ORDER BY id'
    )
    executor = MariePluginDaemonExecutor(
        start=False, daemon_url='http://127.0.0.1:1', env={}
    )
    calls = []
    sensor_thread = threading.get_ident()

    def slow_plugin(*args, **kwargs):
        calls.append(threading.get_ident())
        time.sleep(0.02)
        return []

    monkeypatch.setattr(executor, '_invoke_daemon', slow_plugin)
    active = maximum = completed = 0
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    health_latencies = []

    async def consume():
        nonlocal active, maximum, completed
        while not queue.empty():
            row = queue.get_nowait()
            plan = row['serialized_dag']
            if isinstance(plan, str):
                plan = json.loads(plan)
            definition = next(
                node['definition']
                for node in plan['nodes']
                if node['definition'].get('endpoint')
                == 'plugin_daemon_executor://execute'
            )
            active += 1
            maximum = max(maximum, active)
            await executor.invoke_plugin(
                DocList[TextDoc](),
                parameters={
                    'job_id': str(row['id']),
                    'payload': {'op_params': definition['params']},
                },
            )
            active -= 1
            completed += 1

    async def heartbeat():
        while completed < 250:
            start = time.perf_counter()
            await asyncio.sleep(0.001)
            health_latencies.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    await asyncio.gather(heartbeat(), *(consume() for _ in range(4)))
    assert completed == len(calls) == 250
    assert maximum == 4
    assert all(thread != sensor_thread for thread in calls)
    lag = await runtime.pool.fetchrow("""SELECT
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM (d.created_on-(j.data->'run_config'->>'scheduled_at')::timestamptz))*1000) AS p50,
        max(extract(epoch FROM (d.created_on-(j.data->'run_config'->>'scheduled_at')::timestamptz))*1000) AS maximum
        FROM marie_scheduler.dag d JOIN marie_scheduler.job j ON j.dag_id=d.id""")
    assert queued_after_cycle == [100, 200, 250]
    metrics = {
        'sources': 250,
        'cycle_limit': 100,
        'evaluation_concurrency': 10,
        'cycle_ms': cycles,
        'queued_after_cycle': queued_after_cycle,
        'schedule_to_admission_p50_ms': round(float(lag['p50']), 2),
        'schedule_to_admission_max_ms': round(float(lag['maximum']), 2),
        'queued_before_consumption': 250,
        'controlled_executor_concurrency': maximum,
        'completed': completed,
        'execution_seconds': round(time.perf_counter() - start, 3),
        'event_loop_heartbeat_max_ms': round(max(health_latencies), 3),
        'qualification': 'local PostgreSQL and controlled executor invocation; no live gateway network/daemon capacity claim',
    }
    artifact = os.environ.get('MARIE_SENSOR_METRICS_FILE')
    if artifact:
        Path(artifact).write_text(json.dumps(metrics, indent=2) + '\n')


async def test_target_edit_after_first_fire_does_not_wedge(runtime, monkeypatch):
    import marie.sensors.definitions.schedule_sensor as schedule
    from tests.sensors.test_schedule_sensor import frozen

    source = await add_source(runtime)
    await runtime.worker._evaluate_sensor_with_tick(source)
    await runtime.pool.execute("UPDATE marie_scheduler.dag SET state='completed'")
    await runtime.pool.execute(
        "INSERT INTO marie_scheduler.queue(name) VALUES ('controlled_b') ON CONFLICT DO NOTHING"
    )
    await runtime.storage.update_sensor(
        source['id'], {'target_job_name': 'controlled_b'}
    )
    # The new queue uses the same controlled planner.
    await runtime.storage.update_sensor(
        source['id'],
        {
            'config': {
                'cron': '* * * * *',
                'timezone': 'UTC',
                'run_config': {'planner': 'mock_connector_tool'},
            }
        },
    )
    frozen(monkeypatch, datetime.now(timezone.utc) + timedelta(minutes=2))
    await runtime.worker._evaluate_sensor_with_tick(
        await runtime.storage.get_sensor(source['id'])
    )
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 2
    assert (await runtime.storage.get_sensor(source['id']))['failure_count'] == 0


async def test_locked_storage_lookup_does_not_exhaust_pool(runtime, monkeypatch):
    from marie.sensors.definitions.kb_document_sensor import KbDocumentSensor
    from marie.sensors.types import SensorResult

    await runtime.pool.execute(
        (Path(__file__).parents[2] / 'config/psql/schema/094_kb_item_state.sql').read_text().replace('{schema}', 'marie_scheduler')
    )
    count = 0
    ready = asyncio.Event()

    class LookupSensor:
        def __init__(self, sensor):
            pass

        async def evaluate(self, context):
            nonlocal count
            count += 1
            if count == 10:
                ready.set()
            await ready.wait()
            await KbDocumentSensor({'id':str(uuid4()),'name':'lookup','config':{}}).evaluate(context)
            return SensorResult.skip('No configured binding')

    monkeypatch.setattr(
        runtime.worker._registry, 'get_evaluator', lambda *args, **kwargs: LookupSensor
    )
    for _ in range(10):
        await add_source(runtime)
    # Occupy the remaining two of the fixture's 12 connections, reproducing the
    # deployed default of ten available connections and ten simultaneous evaluations.
    async with runtime.pool.acquire(), runtime.pool.acquire():
        await asyncio.wait_for(runtime.worker._evaluate_all_sensors(), timeout=1)
    assert count == 10
    assert (
        await runtime.pool.fetchval(
            "SELECT count(*) FROM marie_scheduler.sensor_tick WHERE status='skipped'"
        )
        == 10
    )


async def test_inflight_tick_preserves_concurrent_configuration_cursor_reset(
    runtime, monkeypatch
):
    from marie.sensors.types import SensorResult

    ready, release = asyncio.Event(), asyncio.Event()

    class PausedEvaluator:
        def __init__(self, sensor):
            pass

        async def evaluate(self, context):
            ready.set()
            await release.wait()
            return SensorResult.skip('controlled pause', cursor=context.cursor)

    monkeypatch.setattr(
        runtime.worker._registry,
        'get_evaluator',
        lambda *args, **kwargs: PausedEvaluator,
    )
    source = await add_source(runtime)
    task = asyncio.create_task(runtime.worker._evaluate_sensor_with_tick(source))
    await ready.wait()
    edited = await runtime.storage.update_sensor(
        source['id'], {'config': {'cron': '0 2 * * 6', 'timezone': 'America/Chicago'}}
    )
    release.set()
    await task
    assert (await runtime.storage.get_sensor(source['id']))['cursor'] == edited[
        'cursor'
    ]


async def test_target_edit_reconciles_an_accepted_pending_occurrence(
    runtime, monkeypatch
):
    from tests.sensors.test_schedule_sensor import frozen

    source = await add_source(runtime)
    original = runtime.storage.record_run_key

    async def fail_receipt(*args):
        raise RuntimeError('receipt interrupted')

    monkeypatch.setattr(runtime.storage, 'record_run_key', fail_receipt)
    await runtime.worker._evaluate_sensor_with_tick(source)
    assert await runtime.storage.get_pending_tick(source['id'])
    assert (await runtime.storage.get_sensor(source['id']))['last_run_key'] is None
    monkeypatch.setattr(runtime.storage, 'record_run_key', original)
    await runtime.pool.execute(
        "INSERT INTO marie_scheduler.queue(name) VALUES ('controlled_b') ON CONFLICT DO NOTHING"
    )
    await runtime.storage.update_sensor(
        source['id'],
        {
            'target_job_name': 'controlled_b',
            'config': {
                'cron': '* * * * *',
                'timezone': 'UTC',
                'run_config': {'planner': 'mock_connector_tool'},
            },
        },
    )
    await runtime.worker._evaluate_sensor_with_tick(
        await runtime.storage.get_sensor(source['id'])
    )
    assert (await runtime.storage.get_sensor(source['id']))['last_run_key']
    frozen(monkeypatch, datetime.now(timezone.utc) + timedelta(minutes=2))
    await runtime.worker._evaluate_sensor_with_tick(
        await runtime.storage.get_sensor(source['id'])
    )
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 1
    await runtime.pool.execute("UPDATE marie_scheduler.dag SET state='completed'")
    await runtime.worker._evaluate_sensor_with_tick(
        await runtime.storage.get_sensor(source['id'])
    )
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 2


async def test_successful_later_tick_does_not_resurrect_legacy_failed_request(runtime):
    source = await add_source(runtime)
    request = RunRequest(job_name='mock_connector_tool')
    old = await runtime.storage.create_tick(
        source['id'], TickStatus.FAILED, [request.to_dict()], [str(uuid4())]
    )
    await runtime.storage.record_tick(
        source['id'],
        TickStatus.SKIPPED,
        cursor=datetime.now(timezone.utc).isoformat(),
        skip_reason='later completed evaluation',
    )
    assert await runtime.storage.get_pending_tick(source['id']) is None
    await runtime.worker._evaluate_sensor_with_tick(
        await runtime.storage.get_sensor(source['id'])
    )
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 0
    assert (await runtime.storage.get_tick(old))['status'] == 'failed'
