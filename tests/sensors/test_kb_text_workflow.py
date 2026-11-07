"""Saved workflow output through real scheduler, PostgreSQL and disposable S3."""

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from marie.auth.api_key_manager import APIKeyManager
from marie.extraction.artifacts import saved_extraction_reference
from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.kb.gateway_routes import register_kb_routes
from marie.kb.text import TextIndexStore
from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
from marie.sensors.types import SensorType
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.utils.asset_util import s3_asset_path, split_filename
from marie.utils.json import to_json
from tests.sensors.test_kb_workflow_handoff import extraction, handoff
from tests.sensors.test_run_status_postgres import completions
from tests.sensors.test_sensor_postgres import runtime
from tests.unit.executor.kb.test_vector_store_executor_embed import _make_executor

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get('MARIE_SENSOR_TEST_PORT')
        and os.environ.get('MARIE_KB_TEST_S3_CONFIG')
    ),
    reason='Requires owned disposable PostgreSQL and S3',
)


@pytest.fixture
def s3(monkeypatch):
    config = json.loads(Path(os.environ['MARIE_KB_TEST_S3_CONFIG']).read_text())
    handler = S3StorageHandler(
        {
            'S3_ENDPOINT_URL': config['endpoint'],
            'S3_ACCESS_KEY_ID': config['access_key'],
            'S3_SECRET_ACCESS_KEY': config['secret_key'],
            'S3_STORAGE_BUCKET_NAME': 'marie',
        }
    )
    monkeypatch.setattr(
        StorageManager, '_PATH_HANDLERS', OrderedDict({'s3://': handler})
    )
    client = handler.s3.meta.client
    existing = {bucket['Name'] for bucket in client.list_buckets()['Buckets']}
    for bucket in ['marie', 'fixture']:
        if bucket not in existing:
            client.create_bucket(Bucket=bucket)
    return client


@pytest.mark.parametrize("input_kind", ["extraction", "text"])
async def test_saved_extraction_fans_out_to_full_text_search_and_verified_source(
    handoff, s3, monkeypatch, input_kind
):
    rt, definition, items, scopes, uri = handoff
    directory = Path(__file__).parents[2] / 'config/psql/schema'
    await rt.pool.execute(
        (directory / '099_kb_text_index.sql')
        .read_text()
        .replace('{schema}', 'marie_scheduler')
    )
    await rt.pool.execute('TRUNCATE marie_scheduler.kb_text_section')
    await rt.pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET backend='postgresql_text',store_name='kb_text_section',run_params=run_params || '{\"retrieval_backend\":\"postgresql_text\"}'::jsonb"
    )
    metadata = {
        'ocr': [
            {
                'lines': [
                    {'text': '# Benefits'},
                    {'text': 'Emergency treatment is covered.'},
                ]
            }
        ]
    }
    ref = saved_extraction_reference('saved-output', 'document', metadata)
    filename, _, _ = split_filename(ref['ref_id'])
    artifact = urlparse(
        f"{s3_asset_path(ref['ref_id'],ref['ref_type'])}/{filename}.meta.json"
    )
    if input_kind == 'text':
        payload = b'# Benefits\nEmergency treatment is covered.'
        artifact = urlparse('s3://fixture/saved-output.md')
        ref = {
            'kind': 'text',
            'uri': artifact.geturl(),
            'media_type': 'text/markdown',
            'sha256': hashlib.sha256(payload).hexdigest(),
        }
    else:
        payload = to_json(metadata).encode()
    s3.put_object(
        Bucket=artifact.netloc,
        Key=artifact.path.lstrip('/'),
        Body=payload,
    )
    dag = await extraction(rt, scopes[0], uri)
    await rt.pool.execute(
        "UPDATE marie_scheduler.job SET output=$1::jsonb WHERE dag_id=$2 AND starts_with(data->'metadata'->>'on','extract_executor:')",
        json.dumps({'input_ref': ref}),
        UUID(dag),
    )
    await rt.pool.execute(
        "UPDATE marie_scheduler.dag SET state='completed',completed_on=clock_timestamp() WHERE id=$1",
        UUID(dag),
    )
    rt.worker._registry.register(
        SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow'
    )
    await rt.worker._evaluate_sensor_with_tick(definition)
    jobs = await rt.pool.fetch(
        "SELECT * FROM marie_scheduler.job WHERE data->'metadata'->>'on'='vector_store_executor://index_text'"
    )
    assert len(jobs) == 2
    executor = _make_executor()
    executor._text_store = TextIndexStore(rt.pool)
    for job in jobs:
        run = str(uuid4())
        await rt.pool.execute(
            "UPDATE marie_scheduler.job SET state='active',run_owner='qualification',run_attempt_id=$2,run_lease_expires_at=NOW()+INTERVAL '5 minutes' WHERE id=$1",
            job['id'],
            UUID(run),
        )
        info = JobInfo(
            status=JobStatus.PENDING,
            entrypoint='vector_store_executor://index_text',
            metadata={
                **job['data'],
                'dag_id': str(job['dag_id']),
                'run_owner': 'qualification',
                'run_attempt_id': run,
            },
        )
        params, asset = await GatewayJobDistributor()._build_payload(
            str(job['id']), info
        )
        from docarray import DocList

        from marie.api.docs import AssetKeyDoc

        await executor.index_text(DocList[AssetKeyDoc]([asset]), params)
    app = FastAPI()

    async def forward(request):
        endpoint = request.header.exec_endpoint.lstrip('/')
        result = await getattr(executor, endpoint)([], dict(request.parameters))
        return SimpleNamespace(status=None, parameters={'__results__': {'kb': result}})

    register_kb_routes(
        app, {'executor': 'kb'}, lambda: SimpleNamespace(process_single_data=forward)
    )
    monkeypatch.setattr(
        APIKeyManager, 'is_valid', staticmethod(lambda token: token == 'fixture')
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url='http://fixture',
        headers={'Authorization': 'Bearer fixture'},
    ) as client:
        for scope in scopes:
            response = await client.post(
                '/api/v1/kb/text_search',
                json={
                    'tenant_id': scope.tenant_id,
                    'index_name': scope.index_id,
                    'query': 'emergency',
                },
            )
            assert response.status_code == 200, response.text
            rows = response.json()['result']
            assert len(rows) == 1 and rows[0]['item_id'] == scope.item_id
            response = await client.post(
                '/api/v1/kb/open_text',
                json={
                    'tenant_id': scope.tenant_id,
                    'index_id': scope.index_id,
                    'source_id': scope.source_id,
                    **rows[0],
                },
            )
            assert response.status_code == 200, response.text
            result = response.json()['result']
            assert result['outcome'] == 'ready' and 'Emergency' in result['passage']
    executor._embeddings.embed_text.assert_not_called()
    await rt.worker._evaluate_sensor_with_tick(
        await rt.storage.get_sensor(definition['id'])
    )
    assert (
        await rt.pool.fetchval(
            "SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'"
        )
        == 2
    )
    s3.put_object(
        Bucket=artifact.netloc, Key=artifact.path.lstrip('/'), Body=b'{"ocr":[]}'
    )
    scope = scopes[0]
    row = (
        await executor._text_store.search(scope.tenant_id, scope.index_id, 'emergency')
    )[0]
    result = await executor.open_text(
        [],
        {
            'tenant_id': scope.tenant_id,
            'index_id': scope.index_id,
            'source_id': scope.source_id,
            **row,
        },
    )
    assert result['outcome'] == 'changed' and 'passage' not in result
