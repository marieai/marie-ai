import asyncio
import json
import os
from uuid import UUID, uuid4

import pytest

from marie.kb.catalog import CatalogError, KbCatalog
from tests.unit.kb.test_catalog import catalog, index_and_source, scope

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Requires owned disposable PostgreSQL',
)


@pytest.mark.parametrize(
    'item_state,admitted,stage_state,current_output,registered,expected',
    [
        ('pending', True, None, False, True, 1),
        ('pending', True, 'completed', False, True, 1),
        ('removed', True, 'failed', False, True, 0),
        ('ready', True, 'failed', True, True, 0),
        ('pending', False, None, False, False, 1),
        ('pending', False, None, False, True, 0),
        ('error', False, None, False, True, 1),
    ],
)
async def test_bulk_reindex_matches_current_status_guards(
    catalog, item_state, admitted, stage_state, current_output, registered, expected
):
    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    document = (
        await service.call(
            'admit_documents',
            {
                **identity,
                'indexId': index['id'],
                'bucket': 'fixture',
                'files': [
                    dict(
                        sourceId=upload['source']['id'],
                        name='one.pdf',
                        s3Key=upload['s3Key'],
                        size=12,
                    )
                ],
            },
        )
    )[0]
    await pool.execute(
        f'''UPDATE {schema}.kb_item_state SET state=$1,
        admitted_at=CASE WHEN $2 THEN clock_timestamp() ELSE NULL END,
        indexed_dag_id=CASE WHEN $3 THEN indexing_dag_id ELSE NULL END''',
        item_state,
        admitted,
        current_output,
    )
    await pool.execute(
        f'''CREATE TABLE {schema}.job(id uuid,dag_id uuid,name text,state text,started_on timestamptz,completed_on timestamptz);
        CREATE TABLE {schema}.job_search_document(job_id uuid,dag_id uuid,queue_name text,node_label text);
        CREATE TABLE {schema}.kv_store_worker(namespace text,key text,value jsonb);'''
    )
    if stage_state:
        job = uuid4()
        await pool.execute(
            f'INSERT INTO {schema}.job(id,dag_id,name,state) VALUES($1,$2,$3,$4)',
            job,
            UUID(document['dagId']),
            'index',
            stage_state,
        )
        await pool.execute(
            f"INSERT INTO {schema}.job_search_document VALUES($1,$2,'index','INDEX text')",
            job,
            UUID(document['dagId']),
        )
    if not registered:
        await pool.execute(f'DELETE FROM {schema}.kb_item_state')
        await pool.execute(
            f"UPDATE {schema}.kb_document SET metadata=metadata - 'kbItem'"
        )
    results = await service.call(
        'reindex_failed', {**identity, 'indexId': index['id'], 'bucket': 'fixture'}
    )
    assert len(results) == expected


@pytest.mark.parametrize(
    'policy',
    [
        'enabled',
        'disabled',
        'status',
        'blocked',
        'missing',
        'organization',
        'workspace',
    ],
)
async def test_datasource_citation_requires_current_extension_authority(
    catalog, policy
):
    service, pool, schema = catalog
    identity = scope()
    platform = f'kb_platform_{uuid4().hex}'
    await pool.execute(
        f'''CREATE SCHEMA {platform};
        CREATE TABLE {platform}.extension_packages(id uuid PRIMARY KEY,trust_level text);
        CREATE TABLE {platform}.extension_installs(id uuid PRIMARY KEY,package_id uuid,organization_id uuid,workspace_id uuid,
          enabled boolean,status text);'''
    )
    try:
        service = KbCatalog(pool, schema=schema, platform_schema=platform)
        workspace, install, package, dag = uuid4(), uuid4(), uuid4(), uuid4()
        index = await service.call(
            'create_index',
            {
                **identity,
                'name': 'datasource',
                'workspaceId': str(workspace),
                'retrievalBackend': 'postgresql_text',
            },
        )
        source = (
            await service.call(
                'save_source',
                {
                    **identity,
                    'indexId': index['id'],
                    'expectedConfigurationRevision': 0,
                    'name': 'Wiki',
                    'sourceType': 'datasource',
                    'extensionInstallId': str(install),
                    'datasourceRef': 'wiki/pages',
                },
            )
        )['source']
        await pool.execute(
            f"INSERT INTO {platform}.extension_packages VALUES($1,$2)",
            package,
            'blocked' if policy == 'blocked' else 'community',
        )
        if policy != 'missing':
            await pool.execute(
                f'INSERT INTO {platform}.extension_installs VALUES($1,$2,$3,$4,$5,$6)',
                install,
                package,
                (
                    uuid4()
                    if policy == 'organization'
                    else UUID(identity['organization_id'])
                ),
                uuid4() if policy == 'workspace' else workspace,
                policy != 'disabled',
                'disabled' if policy == 'status' else 'enabled',
            )
        await pool.execute(
            f'''INSERT INTO {schema}.kb_item_state(tenant_id,index_id,source_id,item_id,owner_type,owner_id,uri,
            indexing_dag_id,indexed_dag_id,state,backend,store_name)
            VALUES($1,$2,$3,'record','datasource',$3,'s3://fixture/record',$4,$4,'ready','postgresql_text','kb_text_section')''',
            UUID(identity['tenant_id']),
            UUID(index['id']),
            source['id'],
            dag,
        )
        await pool.execute(
            f'''INSERT INTO {schema}.kb_text_section(tenant_id,index_id,source_id,item_id,section_id,indexed_dag_id,
            content,source_digest,input_ref,start_offset,end_offset) VALUES($1,$2,$3,'record','section',$4,'saved text',$5,'{{}}',0,10)''',
            UUID(identity['tenant_id']),
            UUID(index['id']),
            source['id'],
            dag,
            'a' * 64,
        )
        query = {
            **identity,
            'indexId': index['id'],
            'citation': dict(
                sourceId=source['id'],
                itemId='record',
                sectionId='section',
                indexedDagId=str(dag),
                sourceDigest='a' * 64,
            ),
        }
        if policy == 'enabled':
            assert (await service.call('check_text_citation', query))['source'][
                'id'
            ] == source['id']
        else:
            with pytest.raises(CatalogError) as failure:
                await service.call('check_text_citation', query)
            assert failure.value.status_code in (403, 404)
    finally:
        await pool.execute(f'DROP SCHEMA {platform} CASCADE')


async def upstream_submission(catalog):
    service, pool, schema = catalog
    service.platform_schema = schema
    identity = scope()
    workspace, template, submission, owner = uuid4(), uuid4(), uuid4(), uuid4()
    await pool.execute(
        f'''CREATE TABLE {schema}.workspaces(id uuid,organization_id uuid);
        CREATE TABLE {schema}.query_plan_templates(id uuid,name text,published_to_groups text[],workspace_id uuid);
        CREATE TABLE {schema}.submissions(id uuid PRIMARY KEY,tenant_id uuid,name text,status text,
            enable_semantic_search boolean,kb_index_id uuid,query_plan_template_id uuid,updated_at timestamptz DEFAULT now());
        CREATE TABLE {schema}.submission_documents(id uuid PRIMARY KEY,submission_id uuid,file_name text,file_size bigint,
            content_type text,storage_key text,indexing_status text);
        CREATE TABLE {schema}.submission_kb_indexes(id uuid PRIMARY KEY DEFAULT gen_random_uuid(),submission_id uuid,kb_index_id uuid);
        CREATE TABLE {schema}.dag(id uuid,ref_id text,ref_type text,project_id text,planner text,created_on timestamptz DEFAULT now(),completed_on timestamptz,state text);
        CREATE TABLE {schema}.job(id uuid,dag_id uuid,data jsonb);'''
    )
    await pool.execute(
        f'INSERT INTO {schema}.workspaces VALUES($1,$2)',
        workspace,
        UUID(identity['organization_id']),
    )
    await pool.execute(
        f"INSERT INTO {schema}.query_plan_templates VALUES($1,'extract',ARRAY['fabric-a'],$2)",
        template,
        workspace,
    )
    index = await service.call(
        'create_index', {**identity, 'name': 'upstream', 'workspaceId': str(workspace)}
    )
    await pool.execute(
        f"INSERT INTO {schema}.submissions(id,tenant_id,name,status,enable_semantic_search,kb_index_id,query_plan_template_id) VALUES($1,$2,'Batch','open',true,$3,$4)",
        submission,
        UUID(identity['tenant_id']),
        UUID(index['id']),
        template,
    )
    key = f"tenants/{identity['tenant_id']}/submissions/{submission}/one.pdf"
    await pool.execute(
        f"INSERT INTO {schema}.submission_documents VALUES($1,$2,'one.pdf',12,'application/pdf',$3,NULL)",
        owner,
        submission,
        key,
    )
    await pool.execute(
        f'INSERT INTO {schema}.submission_kb_indexes(submission_id,kb_index_id) VALUES($1,$2)',
        submission,
        UUID(index['id']),
    )
    source = (
        await service.call(
            'save_source',
            {
                **identity,
                'indexId': index['id'],
                'expectedConfigurationRevision': 0,
                'sourceType': 'submission',
                'name': 'Batch',
                'metadata': {'submissionId': str(submission)},
            },
        )
    )['source']
    admission = {
        **identity,
        'indexId': index['id'],
        'bucket': 'fixture',
        'files': [
            dict(
                sourceId=source['id'],
                name='one.pdf',
                s3Key=key,
                size=12,
                origin=dict(
                    kind='submission',
                    submissionId=str(submission),
                    submissionDocumentId=str(owner),
                ),
                upstreamPlanner='extract',
                upstreamFabrics=['fabric-a'],
            )
        ],
    }
    document = (await service.call('admit_documents', admission))[0]
    return service, pool, schema, identity, index, submission, document, admission


async def test_registration_and_upstream_reindex_share_parent_before_advisory_order(
    catalog, monkeypatch
):
    service, pool, schema, identity, index, submission, document, admission = (
        await upstream_submission(catalog)
    )
    held, resume = asyncio.Event(), asyncio.Event()
    original = service._submission

    async def parent(connection, command):
        result = await original(connection, command)
        if asyncio.current_task().get_name() == 'registration':
            held.set()
            await resume.wait()
        return result

    monkeypatch.setattr(service, '_submission', parent)
    registration = asyncio.create_task(
        service.call(
            'register_submission_documents',
            {**identity, 'submissionId': str(submission), 'bucket': 'fixture'},
        ),
        name='registration',
    )
    await asyncio.wait_for(held.wait(), 1)
    retry = asyncio.create_task(
        service.call(
            'reindex_document',
            {**identity, 'documentId': document['id'], 'bucket': 'fixture'},
        ),
        name='retry',
    )
    await asyncio.sleep(0.05)
    resume.set()
    results = await asyncio.wait_for(asyncio.gather(registration, retry), 0.8)
    assert results[0][0]['id'] == document['id']
    assert (
        results[1]['metadata']['kbItem']['indexingDagId']
        != document['metadata']['kbItem']['indexingDagId']
    )


async def test_disabling_submission_memberships_does_not_require_retired_workflow_authority(
    catalog,
):
    service, pool, schema, identity, index, submission, document, admission = (
        await upstream_submission(catalog)
    )
    await pool.execute(f'DELETE FROM {schema}.query_plan_templates')
    result = await service.call(
        'set_submission_destinations',
        {
            **identity,
            'submissionId': str(submission),
            'kbIndexIds': [index['id']],
            'enableSemanticSearch': False,
            'bucket': 'fixture',
        },
    )
    assert not result['submission']['enableSemanticSearch']
    assert result['items'][0]['state'] == 'removed'


async def test_destination_edit_does_not_hold_index_before_upstream_advisory(
    catalog, monkeypatch
):
    service, pool, schema, identity, index, submission, document, admission = (
        await upstream_submission(catalog)
    )
    held, resume = asyncio.Event(), asyncio.Event()
    original = service._index

    async def index_lock(*args, **kwargs):
        result = await original(*args, **kwargs)
        if asyncio.current_task().get_name() == 'destinations' and kwargs.get('lock'):
            held.set()
            await resume.wait()
        return result

    monkeypatch.setattr(service, '_index', index_lock)
    edit = asyncio.create_task(
        service.call(
            'set_submission_destinations',
            {
                **identity,
                'submissionId': str(submission),
                'kbIndexIds': [index['id']],
                'enableSemanticSearch': True,
                'bucket': 'fixture',
            },
        ),
        name='destinations',
    )
    await asyncio.wait_for(held.wait(), 1)
    registration = asyncio.create_task(
        service.call('admit_documents', admission), name='direct'
    )
    await asyncio.sleep(0.05)
    resume.set()
    results = await asyncio.wait_for(asyncio.gather(edit, registration), 0.8)
    assert results[1][0]['id'] == document['id']


@pytest.mark.parametrize(
    'operation', ['withdraw_document', 'admit_documents', 'reindex_document']
)
async def test_item_completion_can_update_document_while_catalog_waits_for_item(
    catalog, monkeypatch, operation
):
    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    admission = {
        **identity,
        'indexId': index['id'],
        'bucket': 'fixture',
        'files': [
            dict(
                sourceId=upload['source']['id'],
                name='one.pdf',
                s3Key=upload['s3Key'],
                size=12,
            )
        ],
    }
    document = (await service.call('admit_documents', admission))[0]
    index_held = asyncio.Event()
    original = service._index

    async def index_lock(*args, **kwargs):
        result = await original(*args, **kwargs)
        if asyncio.current_task().get_name() == 'catalog-mutation' and kwargs.get(
            'lock'
        ):
            index_held.set()
        return result

    monkeypatch.setattr(service, '_index', index_lock)
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.fetch(f'SELECT * FROM {schema}.kb_item_state FOR UPDATE')
            command = (
                admission
                if operation == 'admit_documents'
                else {**identity, 'documentId': document['id']}
            )
            mutation = asyncio.create_task(
                service.call(operation, command), name='catalog-mutation'
            )
            await asyncio.wait_for(index_held.wait(), 1)
            await asyncio.sleep(0.05)
            await asyncio.wait_for(
                connection.execute(
                    f"UPDATE {schema}.kb_document SET stage='runtime-complete' WHERE id=$1",
                    UUID(document['id']),
                ),
                0.5,
            )
    await asyncio.wait_for(mutation, 1)
    current = await service.call(
        'get_document', {**identity, 'documentId': document['id']}
    )
    assert (
        current['metadata']['kbItem']['indexingDagId']
        == current['itemState']['indexing_dag_id']
    )


async def test_bulk_retry_acquires_all_origin_advisories_before_first_document_mutation(
    catalog, monkeypatch
):
    service, pool, schema = catalog
    identity = scope()
    index, one = await index_and_source(service, identity)
    two = await service.call(
        'prepare_upload', {**identity, 'indexId': index['id'], 'fileName': 'two.pdf'}
    )
    await pool.execute(
        f'''CREATE TABLE {schema}.dag(id uuid,ref_id text,ref_type text,project_id text,planner text,created_on timestamptz DEFAULT now(),completed_on timestamptz,state text);
        CREATE TABLE {schema}.job(id uuid,dag_id uuid,data jsonb);'''
    )
    admissions = {}
    for position, upload in enumerate([one, two]):
        command = {
            **identity,
            'indexId': index['id'],
            'bucket': 'fixture',
            'files': [
                dict(
                    sourceId=upload['source']['id'],
                    name='one.pdf' if position == 0 else 'two.pdf',
                    s3Key=upload['s3Key'],
                    size=12,
                    upstreamPlanner=f'extract-{position}',
                )
            ],
        }
        document = (await service.call('admit_documents', command))[0]
        admissions[document['id']] = command
    await pool.execute(
        f"UPDATE {schema}.kb_item_state SET state='error',error='retry required'"
    )
    first, resume = asyncio.Event(), asyncio.Event()
    processed = []
    original = service._reindex_document

    async def retry(connection, command):
        result = await original(connection, command)
        if asyncio.current_task().get_name() == 'bulk' and not processed:
            processed.append(str(command.documentId))
            first.set()
            await resume.wait()
        return result

    monkeypatch.setattr(service, '_reindex_document', retry)
    bulk = asyncio.create_task(
        service.call(
            'reindex_failed', {**identity, 'indexId': index['id'], 'bucket': 'fixture'}
        ),
        name='bulk',
    )
    await asyncio.wait_for(first.wait(), 1)
    other = next(
        command
        for document_id, command in admissions.items()
        if document_id != processed[0]
    )
    registration = asyncio.create_task(
        service.call('admit_documents', other), name='direct'
    )
    await asyncio.sleep(0.05)
    resume.set()
    results = await asyncio.wait_for(asyncio.gather(bulk, registration), 0.8)
    assert len(results[0]) == 2


@pytest.mark.parametrize(
    'operation', ['update_index_configuration', 'withdraw_index', 'withdraw_source']
)
async def test_multi_item_mutations_wait_in_runtime_item_key_order(
    catalog, monkeypatch, operation
):
    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    documents = await service.call(
        'admit_documents',
        {
            **identity,
            'indexId': index['id'],
            'bucket': 'fixture',
            'files': [
                dict(
                    sourceId=upload['source']['id'],
                    name=name,
                    s3Key=upload['s3Key'].rsplit('/', 1)[0] + '/' + name,
                    size=12,
                )
                for name in ('b.pdf', 'a.pdf')
            ],
        },
    )
    for key, document in zip(('b', 'a'), documents):
        await pool.execute(
            f"UPDATE {schema}.kb_item_state SET item_id=$1 WHERE item_id=$2",
            key,
            document['metadata']['kbItem']['itemId'],
        )
        await pool.execute(
            f"UPDATE {schema}.kb_document SET metadata=jsonb_set(metadata,'{{kbItem,itemId}}',$2::jsonb) WHERE id=$1",
            UUID(document['id']),
            json.dumps(key),
        )
    held = asyncio.Event()
    original = service._index

    async def index_lock(*args, **kwargs):
        result = await original(*args, **kwargs)
        if asyncio.current_task().get_name() == 'batch' and kwargs.get('lock'):
            held.set()
        return result

    monkeypatch.setattr(service, '_index', index_lock)
    command = {**identity, 'indexId': index['id']}
    if operation == 'update_index_configuration':
        command.update(expectedConfigurationRevision=0, data={'chunkSize': 2048})
    if operation == 'withdraw_source':
        command['sourceId'] = upload['source']['id']
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.fetch(
                f"SELECT * FROM {schema}.kb_item_state WHERE item_id='a' FOR UPDATE"
            )
            mutation = asyncio.create_task(
                service.call(operation, command), name='batch'
            )
            await asyncio.wait_for(held.wait(), 1)
            await asyncio.sleep(0.05)
            await asyncio.wait_for(
                connection.fetch(
                    f"SELECT * FROM {schema}.kb_item_state WHERE item_id='b' FOR UPDATE"
                ),
                0.5,
            )
    await asyncio.wait_for(mutation, 1)


async def test_source_schedule_read_preserves_native_metrics_history_and_workspace_scope(
    catalog,
):
    service, pool, schema = catalog
    identity = scope()
    workspace = uuid4()
    await pool.execute(
        f'''CREATE TABLE {schema}.sensor(id uuid PRIMARY KEY DEFAULT gen_random_uuid(),external_id uuid UNIQUE,
        name text,config jsonb,target_job_name text,target_dag_id uuid,sensor_type text,status text,cursor text,
        last_tick_at timestamptz,last_run_key text,failure_count int DEFAULT 0,last_error text,minimum_interval_seconds int DEFAULT 30,
        created_at timestamptz DEFAULT now(),updated_at timestamptz DEFAULT now());
        CREATE TABLE {schema}.sensor_tick(id uuid PRIMARY KEY DEFAULT gen_random_uuid(),sensor_id uuid,status text,
            started_at timestamptz,completed_at timestamptz,duration_ms int,error_message text);'''
    )
    index = await service.call(
        'create_index', {**identity, 'name': 'scheduled', 'workspaceId': str(workspace)}
    )
    source = (
        await service.call(
            'save_source',
            {
                **identity,
                'indexId': index['id'],
                'expectedConfigurationRevision': 0,
                'name': 'Repo',
                'sourceType': 'repository',
                'schedule': {'cron': '0 * * * *', 'timezone': 'UTC'},
            },
        )
    )['source']
    sensor = await pool.fetchrow(
        f"UPDATE {schema}.sensor SET failure_count=3,last_error='timeout',cursor='checkpoint',last_tick_at='2026-11-09T18:00Z',minimum_interval_seconds=90 RETURNING *"
    )
    await pool.execute(
        f'''INSERT INTO {schema}.sensor_tick(sensor_id,status,started_at,duration_ms)
        SELECT $1,'success','2026-11-09T18:00Z'::timestamptz + make_interval(secs=>n),n FROM generate_series(1,12) n''',
        sensor['id'],
    )
    request = {key: value for key, value in identity.items() if key != 'tenant_id'}
    request['workspaceId'] = str(workspace)
    rows = await service.call('source_schedules', request)
    assert len(rows) == 1
    assert rows[0]['externalId'] == source['triggerConfigId']
    assert rows[0]['indexId'] == index['id'] and rows[0]['sourceId'] == source['id']
    assert rows[0]['workspaceId'] == str(workspace)
    assert rows[0]['failureCount'] == 3 and rows[0]['lastError'] == 'timeout'
    assert rows[0]['cursor'] == 'checkpoint' and rows[0]['minimumIntervalSeconds'] == 90
    assert rows[0]['lastTickAt'] == '2026-11-09T18:00:00+00:00'
    assert len(rows[0]['executions']) == 10
    assert [tick['durationMs'] for tick in rows[0]['executions']] == list(
        range(12, 2, -1)
    )
    for mismatch in (
        {'organization_id': str(uuid4())},
        {'workspaceId': str(uuid4())},
        {'tenant_id': str(uuid4())},
    ):
        assert await service.call('source_schedules', {**request, **mismatch}) == []


@pytest.mark.parametrize('operation', ['configuration', 'source', 'index'])
async def test_historical_submission_origin_does_not_block_catalog_cleanup(
    catalog, operation
):
    service, pool, schema, identity, index, submission, document, admission = (
        await upstream_submission(catalog)
    )
    await pool.execute(
        f'ALTER TABLE {schema}.submission_documents ADD COLUMN indexing_error text, ADD COLUMN updated_at timestamptz DEFAULT now()'
    )
    origin = document['metadata']['origin']
    await service.call(
        'withdraw_submission',
        {
            **identity,
            'submissionId': str(submission),
            'documentId': origin['submissionDocumentId'],
        },
    )
    await pool.execute(f"UPDATE {schema}.kb_item_state SET cleanup_state='complete'")
    await pool.execute(
        f'DELETE FROM {schema}.submission_documents WHERE submission_id=$1', submission
    )
    await pool.execute(
        f'DELETE FROM {schema}.submission_kb_indexes WHERE submission_id=$1', submission
    )
    await pool.execute(f'DELETE FROM {schema}.submissions WHERE id=$1', submission)
    for mutation, command in [
        ('admit_documents', admission),
        (
            'reindex_document',
            {**identity, 'documentId': document['id'], 'bucket': 'fixture'},
        ),
    ]:
        with pytest.raises(CatalogError) as forbidden:
            await service.call(mutation, command)
        assert forbidden.value.status_code in (403, 404)
    if operation == 'configuration':
        result = await service.call(
            'update_index_configuration',
            {
                **identity,
                'indexId': index['id'],
                'expectedConfigurationRevision': 1,
                'data': {'chunkSize': 2048},
            },
        )
        assert result['configurationRevision'] == 2
    elif operation == 'source':
        await service.call(
            'withdraw_source',
            {**identity, 'indexId': index['id'], 'sourceId': document['sourceId']},
        )
        assert await service.call(
            'finalize_source_removal',
            {**identity, 'indexId': index['id'], 'sourceId': document['sourceId']},
        ) == {'deleted': True}
    else:
        await service.call('withdraw_index', {**identity, 'indexId': index['id']})
        assert await service.call(
            'finalize_index_removal', {**identity, 'indexId': index['id']}
        ) == {'deleted': True}


async def test_index_finalization_retries_if_a_new_parent_appears_before_index_lock(
    catalog, monkeypatch
):
    service, pool, schema, identity, index, submission, document, admission = (
        await upstream_submission(catalog)
    )
    await service.call('withdraw_index', {**identity, 'indexId': index['id']})
    await pool.execute(f"UPDATE {schema}.kb_item_state SET cleanup_state='complete'")
    held, resume = asyncio.Event(), asyncio.Event()
    original = service._index

    async def index_lock(*args, **kwargs):
        if asyncio.current_task().get_name() == 'finalize' and kwargs.get('lock'):
            held.set()
            await resume.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, '_index', index_lock)
    finalization = asyncio.create_task(
        service.call('finalize_index_removal', {**identity, 'indexId': index['id']}),
        name='finalize',
    )
    await asyncio.wait_for(held.wait(), 1)
    await pool.execute(
        f'''INSERT INTO {schema}.submissions(id,tenant_id,name,status,enable_semantic_search,kb_index_id)
        VALUES($1,$2,'Late legacy reference','open',false,$3)''',
        uuid4(),
        UUID(identity['tenant_id']),
        UUID(index['id']),
    )
    resume.set()
    with pytest.raises(CatalogError, match='membership changed'):
        await asyncio.wait_for(finalization, 1)
    assert await service.call(
        'finalize_index_removal', {**identity, 'indexId': index['id']}
    ) == {'deleted': True}
