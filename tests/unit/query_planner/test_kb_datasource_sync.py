from uuid import uuid4

from marie.query_planner.base import ExecutorEndpointQueryDefinition, PlannerInfo
from marie.query_planner.kb_datasource_sync_planner import (
    query_planner_kb_datasource_sync,
)


def test_datasource_sync_routes_one_normal_plugin_job():
    plan = query_planner_kb_datasource_sync(
        PlannerInfo(name='kb_datasource_sync', base_id=str(uuid4()), metadata=None)
    )
    operations = [
        node
        for node in plan.nodes
        if isinstance(node.definition, ExecutorEndpointQueryDefinition)
    ]
    assert len(operations) == 1
    assert operations[0].definition.endpoint == 'plugin_daemon_executor://execute'
    assert operations[0].definition.params == {'operation': 'kb_datasource_sync'}
    assert plan.nodes[-1].dependencies == [operations[0].task_id]


def test_scheduled_binding_keeps_scope_and_uses_existing_replace_policy():
    from marie.sensors.daemon.worker import build_work_info
    from marie.sensors.types import RunRequest

    scope = {
        f'{key}_id': str(uuid4())
        for key in ('tenant', 'index', 'source', 'organization', 'workspace', 'user')
    }
    request = RunRequest(
        job_name='kb_datasource_sync',
        run_key='weekend-1',
        run_config={
            **scope,
            'planner': 'kb_datasource_sync',
            'ref_type': 'kb_datasource',
            'ref_id': scope['source_id'],
        },
    )
    work = build_work_info(request, str(uuid4()), 'weekend', 'kb_datasource_sync', None)
    assert work.policy == 'replace'
    assert all(work.data['metadata'][key] == value for key, value in scope.items())
    assert work.retry_limit == 3


def test_mock_deployment_provides_existing_datasource_runtime_resources():
    from pathlib import Path

    import yaml

    config = yaml.safe_load(
        (
            Path(__file__).parents[3]
            / 'config/service/mock/marie-mock-scheduler-test.yml'
        ).read_text()
    )
    nodes = config.get('executors', config.get('deployments', []))
    plugin = next(node for node in nodes if node['name'] == 'plugin_daemon_executor')
    storage = plugin['uses']['with']['storage']
    assert (
        storage['psql']['schema'] == 'marie_scheduler'
        and storage['psql']['enabled'] is False
    )
    assert storage['s3']['bucket_name'] == config['storage']['s3']['bucket_name']
    assert {
        'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID',
        'MARIE_PLUGIN_DAEMON_SIGNING_SECRET',
        'MARIE_EXTENSION_PACKAGE_STORE',
    } <= plugin['env'].keys()
