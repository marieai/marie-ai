import json
import os
import zipfile
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from docarray import DocList
from docarray.documents import TextDoc

from marie.executor.extensions.plugin_daemon_executor import MariePluginDaemonExecutor
from marie.kb.datasource import admit_datasource_items
from marie.kb.item_state import KbItemStore
from marie.plugins.datasource import load_datasource_binding
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable KB PostgreSQL'
)


@pytest.fixture
async def sync(tmp_path, monkeypatch):
    pool = AsyncPostgresConnectionPool()
    await pool.initialize(
        dict(
            hostname='127.0.0.1',
            port=int(os.environ['MARIE_KB_TEST_PORT']),
            username='postgres',
            password='',
            database='postgres',
        ),
        autocommit=True,
    )
    await pool.execute(
        'DROP SCHEMA IF EXISTS marie_studio CASCADE; CREATE SCHEMA marie_studio; CREATE SCHEMA IF NOT EXISTS marie_scheduler'
    )
    await pool.execute(
        'DROP TABLE IF EXISTS marie_scheduler.kb_text_section CASCADE; DROP TABLE IF EXISTS marie_scheduler.kb_item_state CASCADE; DROP TABLE IF EXISTS marie_scheduler.job CASCADE'
    )
    for name in (
        '094_kb_item_state.sql',
        '098_kb_workflow_inputs.sql',
        '099_kb_text_index.sql',
        '100_kb_indexing_dag_names.sql',
    ):
        await pool.execute(
            (Path(__file__).parents[4] / 'config/psql/schema' / name)
            .read_text()
            .replace('{schema}', 'marie_scheduler')
        )
    await pool.execute(
        '''CREATE TABLE marie_studio.tenant(id uuid PRIMARY KEY,organization_id uuid,status text);
    CREATE TABLE IF NOT EXISTS marie_scheduler.resource_workflow_binding(resource_type text,resource_id uuid,tenant_id uuid,run_params jsonb);
    TRUNCATE marie_scheduler.resource_workflow_binding;
    CREATE TABLE marie_studio."user"(id uuid PRIMARY KEY,is_active boolean);
    CREATE TABLE marie_studio.organization_membership(user_id uuid,organization_id uuid,status text);
    CREATE TABLE marie_studio.kb_index(id uuid PRIMARY KEY,tenant_id uuid,workspace_id uuid,metadata jsonb,
      chunk_size int,chunk_overlap int,segmentation_mode text,multimodal_enabled boolean,
      embedding_model text DEFAULT 'runtime-model',embedding_dim int DEFAULT 1024);
    CREATE TABLE marie_studio.kb_source(id uuid PRIMARY KEY,index_id uuid,source_type text,status text,
      extension_install_id uuid,datasource_ref text,dataset_inputs jsonb,metadata jsonb,created_by uuid,
      error_message text,updated_at timestamptz DEFAULT now());
    CREATE TABLE marie_studio.extension_installs(id uuid PRIMARY KEY,package_id uuid,organization_id uuid,
      workspace_id uuid,enabled boolean,status text);
    CREATE TABLE marie_studio.extension_packages(id uuid PRIMARY KEY,package_ref text,source_digest text,
      trust_level text,manifest_snapshot jsonb);
    CREATE TABLE marie_studio.extension_datasources(package_id uuid,datasource_ref text,output_schema jsonb,declaration jsonb);
    CREATE TABLE marie_studio.extension_providers(id uuid,package_id uuid,provider_ref text,type text);
    CREATE TABLE marie_studio.extension_credential_requirements(id uuid,provider_id uuid,name text,required boolean);
    CREATE TABLE marie_studio.extension_credential_bindings(id uuid,requirement_id uuid,organization_id uuid,
      install_id uuid,status text,workspace_id uuid,owner_type text,owner_id uuid,secret_ref text);
    CREATE TABLE marie_scheduler.job(id uuid PRIMARY KEY,dag_id uuid,data jsonb,state text,run_attempt_id uuid,
      run_owner text,run_lease_expires_at timestamptz,created_on timestamptz DEFAULT now());'''
    )
    ids = {
        key: uuid4()
        for key in (
            'tenant',
            'organization',
            'workspace',
            'index',
            'source',
            'install',
            'package',
            'user',
            'job',
            'dag',
            'attempt',
        )
    }
    await pool.execute('INSERT INTO marie_studio."user" VALUES($1,true)', ids['user'])
    await pool.execute(
        "INSERT INTO marie_studio.organization_membership VALUES($1,$2,'active')",
        ids['user'],
        ids['organization'],
    )
    root = tmp_path / 'packages'
    folder = root / str(ids['organization'])
    folder.mkdir(parents=True)
    archive = tmp_path / 'sample.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr('marie-extension.yaml', 'metadata:\n  id: ext.test.datasource\n')
    digest = sha256(archive.read_bytes()).hexdigest()
    archive.rename(folder / f'{digest}.zip')
    await pool.execute(
        'INSERT INTO marie_studio.tenant VALUES ($1,$2,\'active\')',
        ids['tenant'],
        ids['organization'],
    )
    await pool.execute(
        "INSERT INTO marie_studio.kb_index VALUES ($1,$2,$3,'{\"retrievalBackend\":\"postgresql_text\"}',1024,200,'basic',false)",
        ids['index'],
        ids['tenant'],
        ids['workspace'],
    )
    await pool.execute(
        "INSERT INTO marie_studio.kb_source VALUES ($1,$2,'datasource','pending',$3,'datasources/wiki','{\"space\":\"ENG\"}','{}',$4,NULL,now())",
        ids['source'],
        ids['index'],
        ids['install'],
        ids['user'],
    )
    await pool.execute(
        "INSERT INTO marie_studio.extension_installs VALUES ($1,$2,$3,$4,true,'enabled')",
        ids['install'],
        ids['package'],
        ids['organization'],
        ids['workspace'],
    )
    await pool.execute(
        "INSERT INTO marie_studio.extension_packages VALUES ($1,'ext.test.datasource',$2,'community','{}')",
        ids['package'],
        digest,
    )
    await pool.execute(
        "INSERT INTO marie_studio.extension_datasources VALUES ($1,'datasources/wiki','{\"type\":\"object\"}','{\"indexingAdapter\":\"document\"}')",
        ids['package'],
    )
    metadata = {
        f'{key}_id': str(ids[key])
        for key in ('tenant', 'organization', 'workspace', 'index', 'source', 'user')
    }
    metadata['planner'] = 'kb_datasource_sync'
    await pool.execute(
        "INSERT INTO marie_scheduler.job VALUES ($1,$2,$3::jsonb,'active',$4,'worker',now()+interval '5 minutes',now())",
        ids['job'],
        ids['dag'],
        json.dumps({'metadata': metadata}),
        ids['attempt'],
    )
    parameters = {
        'job_id': str(ids['job']),
        'run_attempt_id': str(ids['attempt']),
        'run_owner': 'worker',
        'operation': 'kb_datasource_sync',
    }
    binding = await load_datasource_binding(pool, parameters, root)
    binding['bucket_name'] = 'saved'
    worker = MariePluginDaemonExecutor(
        storage={'s3': {'enabled': False, 'bucket_name': 'saved'}},
        daemon_url='http://unused',
        start=False,
        env={
            'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID': 'test',
            'MARIE_PLUGIN_DAEMON_SIGNING_SECRET': 'test-only',
            'MARIE_EXTENSION_PACKAGE_STORE': str(root),
        },
    )
    worker._datasource_pool = pool
    uri = f"s3://saved/tenants/{ids['tenant']}/wiki.md"
    output = [
        {
            'id': 'wiki:42',
            'uri': uri,
            'input_ref': {
                'kind': 'text',
                'uri': uri,
                'media_type': 'text/markdown',
                'sha256': 'a' * 64,
            },
        }
    ]
    calls = []

    def invoke(spec, payload, request_id, **kwargs):
        calls.append((spec, payload, kwargs))
        return [{'type': 'stream', 'data': {'records': output}}, {'type': 'end'}]

    monkeypatch.setattr(worker, '_invoke_daemon', invoke)
    yield pool, ids, binding, worker, parameters, output, calls
    await pool.close()


async def test_scheduled_datasource_executes_and_registers_saved_output(sync):
    pool, ids, binding, worker, parameters, output, calls = sync
    result = await worker.invoke_plugin(DocList[TextDoc](), parameters)
    assert json.loads(result[0].text)['data'] == {'records_admitted': 1}
    assert calls[0][1]['configuration'] == {'space': 'ENG'}
    assert calls[0][1]['source_id'] == str(ids['source'])
    item = (await KbItemStore(pool).pending('datasource', 100))[0]
    assert item['item_id'] == 'wiki:42'
    assert item['input_ref']['uri'] == output[0]['uri']
    assert item['input_ref']['processing_workflow'] == 'kb_index_content'


async def test_presentation_save_does_not_invalidate_datasource_delivery(sync):
    pool, ids, binding, _, _, output, _ = sync
    await pool.execute(
        "UPDATE marie_studio.kb_index SET metadata=metadata || '{\"configurationRevision\":1,\"preferences\":{\"display\":\"compact\"}}'::jsonb"
    )
    assert await admit_datasource_items(pool, binding, output) == 1


@pytest.mark.parametrize(
    'run_params', [{'parse_mode': 'agent', 'cache_options': {'disabled': False}}, None]
)
async def test_inactive_parser_defaults_do_not_invalidate_datasource_delivery(
    sync, run_params
):
    pool, ids, binding, _, _, output, _ = sync
    await pool.execute(
        "INSERT INTO marie_scheduler.resource_workflow_binding VALUES('kb_index',$1,$2,$3::jsonb)",
        ids['index'],
        ids['tenant'],
        json.dumps(run_params),
    )
    assert await admit_datasource_items(pool, binding, output) == 1


async def test_processing_save_rejects_outdated_datasource_delivery(sync):
    pool, ids, binding, _, _, output, _ = sync
    await pool.execute('UPDATE marie_studio.kb_index SET chunk_size=512')
    with pytest.raises(ValueError, match='changed'):
        await admit_datasource_items(pool, binding, output)


async def test_new_profile_metadata_does_not_rebuild_an_unchanged_legacy_item(sync):
    pool, _, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET run_params=run_params - 'embedding_model' - 'embedding_dim'"
    )
    before = await pool.fetchrow(
        'SELECT indexing_dag_id FROM marie_scheduler.kb_item_state'
    )
    await admit_datasource_items(pool, binding, output)
    after = await pool.fetchrow(
        'SELECT indexing_dag_id FROM marie_scheduler.kb_item_state'
    )
    assert after == before


async def test_retry_is_idempotent_and_unchanged_items_keep_their_indexing_dag(sync):
    pool, ids, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    first = await pool.fetchrow('SELECT * FROM marie_scheduler.kb_item_state')
    await admit_datasource_items(pool, binding, output)
    current = await pool.fetchrow('SELECT * FROM marie_scheduler.kb_item_state')
    assert current['indexing_dag_id'] == first['indexing_dag_id']
    assert current['input_ref'] == first['input_ref']


async def test_changed_output_retains_previous_searchable_result(sync):
    pool, ids, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET state='ready',indexed_dag_id=indexing_dag_id"
    )
    original = await pool.fetchrow('SELECT * FROM marie_scheduler.kb_item_state')
    output[0]['input_ref']['sha256'] = 'b' * 64
    await admit_datasource_items(pool, binding, output)
    current = await pool.fetchrow('SELECT * FROM marie_scheduler.kb_item_state')
    assert current['state'] == 'pending'
    assert current['indexed_dag_id'] == original['indexed_dag_id']
    assert current['indexing_dag_id'] != original['indexing_dag_id']


async def test_partial_or_invalid_delivery_does_not_remove_existing_items(sync):
    pool, _, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    await admit_datasource_items(pool, binding, [])
    assert (
        await pool.fetchval('SELECT count(*) FROM marie_scheduler.kb_item_state') == 1
    )
    with pytest.raises(ValueError):
        await admit_datasource_items(
            pool, binding, [output[0], {'id': 'domain', 'moves': ['e4']}]
        )
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM marie_scheduler.kb_item_state WHERE state <> 'removed'"
        )
        == 1
    )


async def test_only_explicit_owned_deletes_withdraw_results(sync):
    pool, _, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    await admit_datasource_items(pool, binding, [{'id': 'wiki:42', 'deleted': True}])
    assert (
        await pool.fetchval('SELECT state FROM marie_scheduler.kb_item_state')
        == 'removed'
    )


@pytest.mark.parametrize('change', ['lease', 'disabled', 'inputs'])
async def test_changed_execution_cannot_publish(sync, change):
    pool, ids, binding, _, _, output, _ = sync
    if change == 'lease':
        await pool.execute(
            "UPDATE marie_scheduler.job SET run_lease_expires_at=now()-interval '1 second'"
        )
    elif change == 'disabled':
        await pool.execute("UPDATE marie_studio.kb_source SET status='disabled'")
    else:
        await pool.execute("UPDATE marie_studio.kb_source SET dataset_inputs='{}'")
    with pytest.raises(ValueError):
        await admit_datasource_items(pool, binding, output)
    assert (
        await pool.fetchval('SELECT count(*) FROM marie_scheduler.kb_item_state') == 0
    )


async def test_delivered_items_use_existing_indexing_sensor_and_are_searchable(
    sync, monkeypatch
):
    from marie.kb.item_state import KbItemScope
    from marie.kb.text import TextIndexStore
    from marie.sensors.context import SensorEvaluationContext
    from marie.sensors.definitions.kb_document_sensor import KbDocumentSensor
    from marie.sensors.types import SensorType

    pool, ids, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    sensor = KbDocumentSensor(
        {
            'id': str(uuid4()),
            'name': 'kb-indexing',
            'config': {},
            'target_job_name': 'kb_indexing',
        }
    )
    context = SensorEvaluationContext(
        sensor_id=sensor.sensor_id,
        sensor_name=sensor.name,
        sensor_type=SensorType.DATA_SINK,
        resources={'postgres_pool': pool},
    )
    result = await sensor.evaluate(context)
    assert len(result.run_requests) == 1
    parameters = result.run_requests[0].run_config
    assert parameters['planner'] == 'kb_index_content'
    store = TextIndexStore(pool)
    scope = KbItemScope(
        str(ids['tenant']), str(ids['index']), str(ids['source']), 'wiki:42'
    )
    await store.replace(
        scope,
        parameters['kb_indexing_dag_id'],
        'Engineering retry policy',
        output[0]['input_ref'],
    )
    matches = await store.search(str(ids['tenant']), str(ids['index']), 'retry')
    assert len(matches) == 1
    assert matches[0]['item_id'] == 'wiki:42'


async def test_two_bindings_keep_the_same_external_item_id_separate(sync):
    pool, ids, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    other = uuid4()
    await pool.execute(
        '''INSERT INTO marie_studio.kb_source
       SELECT $1,index_id,source_type,status,extension_install_id,datasource_ref,
              dataset_inputs,metadata,created_by,error_message,updated_at FROM marie_studio.kb_source WHERE id=$2''',
        other,
        ids['source'],
    )
    other_binding = {**binding, 'source_id': other}
    job_id = uuid4()
    await pool.execute(
        '''INSERT INTO marie_scheduler.job SELECT $1,dag_id,
        jsonb_set(data,'{metadata,source_id}',to_jsonb($2::text)),state,run_attempt_id,run_owner,
        run_lease_expires_at,created_on FROM marie_scheduler.job WHERE id=$3''',
        job_id,
        str(other),
        ids['job'],
    )
    other_binding['execution'] = {**binding['execution'], 'job_id': str(job_id)}
    await admit_datasource_items(pool, other_binding, output)
    await admit_datasource_items(pool, binding, [{'id': 'wiki:42', 'deleted': True}])
    rows = await pool.fetch('SELECT source_id,state FROM marie_scheduler.kb_item_state')
    assert {row['source_id']: row['state'] for row in rows} == {
        str(ids['source']): 'removed',
        str(other): 'pending',
    }


async def test_a_late_older_sync_cannot_replace_newer_delivered_output(sync):
    pool, ids, binding, _, _, output, _ = sync
    previous_job = uuid4()
    previous_dag = uuid4()
    await pool.execute(
        '''INSERT INTO marie_scheduler.job SELECT $1,$2,data,state,run_attempt_id,run_owner,
       run_lease_expires_at,created_on-interval '1 hour' FROM marie_scheduler.job WHERE id=$3''',
        previous_job,
        previous_dag,
        ids['job'],
    )
    older = {
        **binding,
        'dag_id': str(previous_dag),
        'execution': {**binding['execution'], 'job_id': str(previous_job)},
    }
    await admit_datasource_items(pool, binding, output)
    with pytest.raises(ValueError, match='newer|superseded'):
        await admit_datasource_items(pool, older, [])


async def test_datasource_cannot_admit_another_tenants_saved_artifact(sync):
    pool, _, binding, _, _, output, _ = sync
    output[0]['uri'] = f's3://saved/tenants/{uuid4()}/private.md'
    output[0]['input_ref']['uri'] = output[0]['uri']
    with pytest.raises(ValueError, match='scope|tenant'):
        await admit_datasource_items(pool, binding, output)
    assert (
        await pool.fetchval('SELECT count(*) FROM marie_scheduler.kb_item_state') == 0
    )


async def test_unchanged_failed_indexing_gets_a_fresh_dag_and_keeps_prior_results(sync):
    pool, _, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET state='error',error='index failed',indexed_dag_id=indexing_dag_id,admitted_at=now()"
    )
    previous = await pool.fetchrow('SELECT * FROM marie_scheduler.kb_item_state')
    await admit_datasource_items(pool, binding, output)
    current = await pool.fetchrow('SELECT * FROM marie_scheduler.kb_item_state')
    assert (
        current['state'] == 'pending'
        and current['error'] is None
        and current['admitted_at'] is None
    )
    assert current['indexing_dag_id'] != previous['indexing_dag_id']
    assert current['indexed_dag_id'] == previous['indexed_dag_id']


async def test_cleaned_up_external_id_can_be_delivered_again(sync):
    pool, _, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, binding, output)
    await admit_datasource_items(pool, binding, [{'id': 'wiki:42', 'deleted': True}])
    from marie.kb.item_state import KbItemScope
    from marie.kb.text import TextIndexStore

    scope = KbItemScope(
        str(binding['tenant_id']),
        str(binding['index_id']),
        str(binding['source_id']),
        'wiki:42',
    )
    await TextIndexStore(pool).delete_item(scope)
    assert (
        await pool.fetchval('SELECT cleanup_state FROM marie_scheduler.kb_item_state')
        == 'complete'
    )
    await admit_datasource_items(pool, binding, output)
    current = await pool.fetchrow('SELECT * FROM marie_scheduler.kb_item_state')
    assert current['state'] == 'pending' and current['cleanup_state'] == 'none'
    assert current['indexed_dag_id'] is None and current['record_count'] == 0


async def test_fabric_routing_does_not_change_content_identity(sync):
    pool, _, binding, _, _, output, _ = sync
    await admit_datasource_items(pool, {**binding, 'fabric_group_id': 'one'}, output)
    previous = await pool.fetchval(
        'SELECT indexing_dag_id FROM marie_scheduler.kb_item_state'
    )
    await admit_datasource_items(pool, {**binding, 'fabric_group_id': 'two'}, output)
    assert (
        await pool.fetchval('SELECT indexing_dag_id FROM marie_scheduler.kb_item_state')
        == previous
    )


async def test_bucket_must_match_the_existing_storage_connection(sync):
    pool, _, binding, _, _, output, _ = sync
    output[0]['uri'] = output[0]['uri'].replace('s3://saved/', 's3://other/')
    output[0]['input_ref']['uri'] = output[0]['uri']
    with pytest.raises(ValueError, match='bucket|storage'):
        await admit_datasource_items(pool, binding, output)


async def test_disabled_schedule_principal_cannot_run(sync):
    pool, ids, _, worker, parameters, _, calls = sync
    await pool.execute(
        'CREATE TABLE IF NOT EXISTS marie_studio."user"(id uuid PRIMARY KEY,is_active boolean)'
    )
    await pool.execute(
        'CREATE TABLE IF NOT EXISTS marie_studio.organization_membership(user_id uuid,organization_id uuid,status text)'
    )
    await pool.execute(
        'UPDATE marie_studio."user" SET is_active=false WHERE id=$1', ids['user']
    )
    with pytest.raises(ValueError, match='principal|member'):
        await worker.invoke_plugin(DocList[TextDoc](), parameters)
    assert calls == []


async def test_full_sync_uses_the_installed_runtime_timeout(sync):
    pool, _, _, worker, parameters, _, calls = sync
    await pool.execute(
        '''UPDATE marie_studio.extension_packages SET manifest_snapshot=
      '{"runtimePolicy":{"timeoutMs":1500,"maxConcurrent":1,"maxMemoryBytes":67108864,"networkPolicy":"none"}}' '''
    )
    await worker.invoke_plugin(DocList[TextDoc](), parameters)
    assert calls[0][2]['runtime_policy']['timeoutMs'] == 1500


async def test_configured_worker_initializes_its_existing_psql_pool(sync, monkeypatch):
    _, _, _, old_worker, parameters, _, _ = sync
    worker = MariePluginDaemonExecutor(
        daemon_url='http://unused',
        start=False,
        env=old_worker._env,
        storage={
            'psql': {
                'enabled': False,
                'hostname': '127.0.0.1',
                'port': int(os.environ['MARIE_KB_TEST_PORT']),
                'username': 'postgres',
                'password': '',
                'database': 'postgres',
            },
            's3': {'enabled': False, 'bucket_name': 'saved'},
        },
    )
    monkeypatch.setattr(worker, '_invoke_daemon', old_worker._invoke_daemon)
    assert worker._datasource_pool is None
    try:
        result = await worker.invoke_plugin(DocList[TextDoc](), parameters)
        assert json.loads(result[0].text)['data'] == {'records_admitted': 1}
    finally:
        if worker._datasource_pool:
            await worker._datasource_pool.close()


@pytest.mark.parametrize('backend', ['postgresql_text', 'pgvector'])
@pytest.mark.parametrize('reported_delete', [True, False])
async def test_processing_cleans_only_withdrawn_index_output(
    sync, backend, reported_delete
):
    from marie.kb.item_state import KbItemScope

    pool, ids, binding, _, _, output, _ = sync
    binding['index_metadata'] = {'retrievalBackend': backend}
    await pool.execute(
        'UPDATE marie_studio.kb_index SET metadata=$1::jsonb',
        json.dumps(binding['index_metadata']),
    )
    await admit_datasource_items(
        pool, binding, [output[0], {**output[0], 'id': 'wiki:keep'}]
    )
    other_source = str(uuid4())
    await pool.execute(
        '''INSERT INTO marie_scheduler.kb_item_state
        (tenant_id,index_id,source_id,item_id,indexing_dag_id,owner_type,owner_id,uri,run_params,backend,store_name,input_ref)
        SELECT tenant_id,index_id,$1,item_id,indexing_dag_id,owner_type,$1,uri,run_params,backend,store_name,input_ref
        FROM marie_scheduler.kb_item_state WHERE item_id='wiki:42' ''',
        other_source,
    )
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET state='ready',indexed_dag_id=indexing_dag_id"
    )
    if backend == 'postgresql_text':
        await pool.execute(
            '''INSERT INTO marie_scheduler.kb_text_section
            (tenant_id,index_id,source_id,item_id,section_id,indexed_dag_id,content,source_digest,input_ref,start_offset,end_offset)
            SELECT tenant_id,index_id,source_id,item_id,repeat('b',64),indexed_dag_id,'saved text',repeat('a',64),input_ref,0,10
            FROM marie_scheduler.kb_item_state'''
        )
        table, item_column = 'marie_scheduler.kb_text_section', 'item_id'
    else:
        await pool.execute('DROP TABLE IF EXISTS kb_vectors CASCADE')
        await pool.execute(
            'CREATE TABLE kb_vectors(index_name text,source_id text,ref_doc_id text,content text)'
        )
        await pool.execute(
            '''INSERT INTO kb_vectors
            SELECT index_id::text,source_id,item_id,'saved vector' FROM marie_scheduler.kb_item_state'''
        )
        table, item_column = 'kb_vectors', 'ref_doc_id'
    if reported_delete:
        delivery = [{'id': 'wiki:42', 'deleted': True}]
    else:
        scope = KbItemScope(
            str(ids['tenant']), str(ids['index']), str(ids['source']), 'wiki:42'
        )
        await KbItemStore(pool).remove(scope)
        delivery = []
    await admit_datasource_items(pool, binding, delivery)
    assert await pool.fetchval(f'SELECT count(*) FROM {table}') == 2
    assert (
        await pool.fetchval(
            f'SELECT count(*) FROM {table} WHERE source_id=$1 AND {item_column}=$2',
            str(ids['source']),
            'wiki:42',
        )
        == 0
    )
    row = await pool.fetchrow(
        '''SELECT cleanup_state,uri FROM marie_scheduler.kb_item_state
        WHERE source_id=$1 AND item_id='wiki:42' ''',
        str(ids['source']),
    )
    assert row == {'cleanup_state': 'complete', 'uri': output[0]['uri']}


async def test_removal_before_vector_indexing_needs_no_vector_table(sync):
    pool, ids, binding, _, _, output, _ = sync
    binding['index_metadata'] = {'retrievalBackend': 'pgvector'}
    await pool.execute(
        'UPDATE marie_studio.kb_index SET metadata=$1::jsonb',
        json.dumps(binding['index_metadata']),
    )
    await pool.execute('DROP TABLE IF EXISTS kb_vectors CASCADE')
    await admit_datasource_items(pool, binding, output)
    await admit_datasource_items(pool, binding, [{'id': 'wiki:42', 'deleted': True}])
    assert (
        await pool.fetchval('SELECT cleanup_state FROM marie_scheduler.kb_item_state')
        == 'complete'
    )
