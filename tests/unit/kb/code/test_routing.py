from unittest.mock import AsyncMock

from marie.query_planner.base import PlannerInfo
from marie.query_planner.kb_repository_planner import query_planner_kb_repository
from marie.sensors.daemon.worker import repository_sensor


def test_repository_plan_targets_executor_not_gateway_or_connector():
    plan = query_planner_kb_repository(
        PlannerInfo(
            name='kb_repository_refresh',
            base_id='0193391e-0a32-7000-8000-000000000000',
            current_id=0,
        )
    )
    assert (
        plan.nodes[1].definition.endpoint == 'repository_executor://refresh_repository'
    )


def test_only_native_repository_source_schedules_pass_target_guard():
    sensor = {
        'target_job_name': 'kb_repository_refresh',
        'config': {
            'kbIndexId': 'index',
            'kbSourceId': 'source',
            'run_config': {
                'planner': 'kb_repository_refresh',
                'index_id': 'index',
                'source_id': 'source',
                'ref_type': 'kb_repository',
                'ref_id': 'source',
            },
        },
    }
    assert repository_sensor(sensor)
    sensor['config']['run_config']['source_id'] = 'other'
    assert not repository_sensor(sensor)
    assert not repository_sensor(
        {'target_job_name': 'kb_datasource_sync', 'config': {'kbSourceId': 'source'}}
    )


def test_gateway_routes_code_to_cpu_executor_and_keeps_search_coverage(monkeypatch):
    from marie.auth.api_key_manager import APIKeyManager

    monkeypatch.setattr(APIKeyManager, "is_valid", staticmethod(lambda token: True))
    from tests.unit.kb.test_kb_gateway_routes import AUTH, _client, _streamer_returning

    payload = {
        'results': [],
        'truncated': True,
        'indexed': {'commit': 'c' * 40},
        'backend': 'zoekt',
    }
    streamer = _streamer_returning(payload)
    client = _client(
        streamer, {'executor': 'vectors', 'code_executor': 'repository_executor'}
    )
    response = client.post(
        '/api/v1/kb/code_search',
        json={
            'tenant_id': 'tenant',
            'index_id': 'index',
            'source_id': 'source',
            'organization_id': 'org',
            'query': 'pattern',
        },
        headers=AUTH,
    )
    assert response.status_code == 200 and response.json()['result'] == payload
    request = streamer.process_single_data.call_args.kwargs['request']
    assert request.header.target_executor == 'repository_executor'
    assert request.header.exec_endpoint == '/code_search'
