import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from marie.sensors.daemon.worker import SensorWorker, build_work_info
from marie.sensors.types import RunRequest, TickStatus


def fixture():
    worker = SensorWorker({})
    worker._storage = AsyncMock()
    worker._storage.has_run_key.return_value = False
    worker._storage.get_run_key_job.return_value = None
    sensor = {
        'id': str(uuid4()),
        'name': 'controlled',
        'sensor_type': 'schedule',
        'target_job_name': 'connector',
        'config': {},
    }
    request = RunRequest(run_key='occurrence-a', job_name='connector')
    return worker, sensor, request


async def test_missing_scheduler_is_failure():
    worker, sensor, request = fixture()
    with pytest.raises(RuntimeError, match='scheduler'):
        await worker._submit_run_request(sensor, request)


async def test_missing_target_is_failure():
    worker, sensor, request = fixture()
    request.job_name = None
    sensor['target_job_name'] = None
    with pytest.raises(ValueError, match='target'):
        await worker._submit_run_request(sensor, request)


def test_occurrence_identity_is_stable_and_binding_scoped():
    _, sensor, request = fixture()

    def build(sid):
        return build_work_info(request, sid, 'n', 'connector', None).id

    assert build(sensor['id']) == build(sensor['id'])
    assert build(sensor['id']) != build(str(uuid4()))


def test_sensor_admission_preserves_existing_duplicate_policy():
    _, sensor, request = fixture()
    assert build_work_info(request, sensor['id'], 'n', 'connector', None).policy is None


async def test_tags_cannot_override_occurrence_identity():
    _, sensor, request = fixture()
    request.tags = {'sensor_id': 'other', 'run_key': 'other'}
    data = build_work_info(request, sensor['id'], 'n', 'connector', None).data
    assert data['sensor_id'] == sensor['id']
    assert data['run_key'] == request.run_key


async def test_retry_after_record_failure_reuses_durable_id():
    worker, sensor, request = fixture()
    ids = set()

    async def accept(work):
        ids.add(work.id)
        return work.id

    worker._job_scheduler = AsyncMock()
    worker._job_scheduler.submit_job.side_effect = accept
    worker._job_scheduler.submit_sensor_job.side_effect = accept
    worker._storage.record_run_key.side_effect = [
        RuntimeError('crash after acceptance'),
        None,
    ]
    with pytest.raises(RuntimeError):
        await worker._submit_run_request(sensor, request)
    await worker._submit_run_request(sensor, request)
    assert len(ids) == 1


async def test_partial_recovery_is_not_success():
    worker, sensor, request = fixture()
    worker._submit_run_request = AsyncMock(
        side_effect=[str(uuid4()), RuntimeError('unavailable')]
    )
    tick = {
        'id': str(uuid4()),
        'sensor_id': sensor['id'],
        'run_requests': [request.to_dict(), {**request.to_dict(), 'run_key': 'b'}],
        'cursor': 'new',
    }
    await worker._recover_tick(tick)
    recorded = worker._storage.record_tick.call_args.kwargs
    assert recorded['status'] == TickStatus.FAILED
    assert recorded['cursor'] is None
    assert len(recorded['run_ids']) == 1


async def test_cycle_filters_ineligible_prefix_before_limit():
    worker, _, _ = fixture()
    now = datetime.now(timezone.utc)
    worker._storage.get_active_sensors.return_value = [
        {'id': str(i), 'last_tick_at': now, 'minimum_interval_seconds': 3600}
        for i in range(100)
    ] + [{'id': 'due', 'last_tick_at': now - timedelta(days=1)}]
    worker._evaluate_sensor_with_tick = AsyncMock()
    await worker._evaluate_all_sensors()
    worker._evaluate_sensor_with_tick.assert_awaited_once()


async def test_cycle_bounds_evaluation_concurrency():
    worker, _, _ = fixture()
    worker._settings.max_concurrent_evaluations = 3
    worker._storage.get_active_sensors.return_value = [
        {'id': str(i)} for i in range(100)
    ]
    active = maximum = count = 0

    async def evaluate(sensor):
        nonlocal active, maximum, count
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.001)
        active -= 1
        count += 1

    worker._evaluate_sensor_with_tick = evaluate
    await worker._evaluate_all_sensors()
    assert maximum == 3
    assert count == 100


async def test_scheduler_confirms_an_existing_occurrence_without_resubmitting():
    from marie.scheduler.psql import PostgreSQLJobScheduler

    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.repository = AsyncMock()
    scheduler.repository.get_sensor_job_state.return_value = 'created'
    scheduler.submit_job = AsyncMock()
    _, sensor, request = fixture()
    work = build_work_info(request, sensor['id'], 'n', 'connector', None)
    assert await scheduler.submit_sensor_job(work) == work.id
    scheduler.submit_job.assert_not_awaited()


async def test_scheduler_does_not_turn_refusal_into_success():
    from marie.scheduler.psql import PostgreSQLJobScheduler

    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.repository = AsyncMock()
    scheduler.repository.get_sensor_job_state.return_value = None
    scheduler.submit_job = AsyncMock(side_effect=ValueError('refused'))
    _, sensor, request = fixture()
    work = build_work_info(request, sensor['id'], 'n', 'connector', None)
    with pytest.raises(ValueError, match='refused'):
        await scheduler.submit_sensor_job(work)
