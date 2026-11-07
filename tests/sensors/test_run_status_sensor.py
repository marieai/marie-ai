from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.daemon.worker import SensorWorker, build_job_data
from marie.sensors.definitions.run_status_sensor import RunStatusSensor
from marie.sensors.types import SensorType


def completion(**updates):
    return {
        'id': str(uuid4()),
        'dag_id': str(uuid4()),
        'completion_id': '41',
        'name': 'extract',
        'state': 'completed',
        'completed_at': '2026-11-06T15:00:00+00:00',
        'data': {'metadata': {
            'uri': 's3://fixture/input.pdf',
            'ref_id': 'saved-output',
            'ref_type': 'document',
            'project_id': 'project-one',
            'planner': 'extract',
            'on': 'extract_executor://document/extract',
            'op_params': {'layout': 'old'},
        }},
        'output': {'games': [{'id': 'g1', 'moves': ['e4', 'e5']}]},
        **updates,
    }


def sensor(config=None):
    return RunStatusSensor({
        'id': str(uuid4()),
        'name': 'follow extraction',
        'target_job_name': 'kb_indexing',
        'config': {
            'monitored_job_names': ['extract'],
            'status_types': ['completed'],
            'pipeline_mode': True,
            'target_planner': 'kb_index_content',
            **(config or {}),
        },
    })


def context(instance, jobs):
    return SensorEvaluationContext(
        sensor_id=instance.sensor_id,
        sensor_name=instance.name,
        sensor_type=SensorType.RUN_STATUS,
        completed_jobs=jobs,
    )


def test_studio_configuration_names_are_supported():
    instance = RunStatusSensor({
        'id': str(uuid4()),
        'target_job_name': 'kb_indexing',
        'config': {'watchJobNames': ['extract'], 'triggerOnStatuses': ['completed']},
    })
    instance.validate_config()
    assert instance.monitored_job_names == ['extract']
    assert instance.normalized_statuses == {'completed'}


async def test_pipeline_maps_input_and_preserves_typed_output_without_reusing_executor():
    instance = sensor()
    job = completion()
    before = deepcopy(job)
    result = await instance.evaluate(context(instance, [job]))
    request = result.run_requests[0]
    data = build_job_data(request, instance.sensor_id, instance.name)
    metadata = data['metadata']
    assert request.job_name == 'kb_indexing'
    assert metadata['planner'] == 'kb_index_content'
    assert metadata['uri'] == job['data']['metadata']['uri']
    assert metadata['ref_id'] == 'saved-output'
    assert metadata['upstream']['output'] == job['output']
    assert metadata['upstream']['completion_id'] == '41'
    assert 'on' not in metadata
    assert 'op_params' not in metadata
    assert job == before


async def test_resetting_the_same_dag_produces_a_distinct_completion_run_key():
    instance = sensor()
    first = completion()
    second = {**first, 'completion_id': '42'}
    first_request = (await instance.evaluate(context(instance, [first]))).run_requests[0]
    second_request = (await instance.evaluate(context(instance, [second]))).run_requests[0]
    assert first_request.run_key != second_request.run_key
    assert first_request.run_key.endswith(':41')


async def test_source_scope_filters_unrelated_project_completions():
    instance = sensor({'source_scope': {'project_ids': ['project-one']}})
    allowed = completion()
    other = completion(data={'metadata': {'project_id': 'project-two'}})
    result = await instance.evaluate(context(instance, [other, allowed]))
    assert len(result.run_requests) == 1
    assert result.run_requests[0].run_config['trigger_job_id'] == allowed['id']


async def test_worker_loads_durable_workflow_completions_from_storage():
    worker = SensorWorker({})
    jobs = [completion()]
    storage = SimpleNamespace(get_completed_workflows=AsyncMock(return_value=jobs))
    worker._storage = storage
    definition = {'id': str(uuid4()), 'sensor_type': 'run_status'}
    assert await worker._load_completed_jobs(definition, 'old-cursor') == jobs
    storage.get_completed_workflows.assert_awaited_once()


async def test_failed_workflow_does_not_trigger_a_completed_only_sensor():
    instance = sensor()
    result = await instance.evaluate(context(instance, [completion(state='failed')]))
    assert not result.run_requests
