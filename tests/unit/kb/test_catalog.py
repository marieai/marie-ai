import asyncio
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Requires an owned disposable PostgreSQL database',
)
SQL_DIRECTORY = Path(__file__).parents[3] / 'config/psql/schema'


@pytest.fixture
async def catalog():
    from marie.kb.catalog import KbCatalog

    schema = f'kb_catalog_{uuid4().hex}'
    pool = AsyncPostgresConnectionPool()
    await pool.initialize(
        dict(
            hostname='127.0.0.1',
            port=int(os.environ['MARIE_SENSOR_TEST_PORT']),
            username='postgres',
            password='',
            database='f10c_qualification',
            max_connections=4,
        ),
        autocommit=True,
    )
    await pool.execute(f'CREATE SCHEMA {schema}')
    try:
        for name in [
            '072_resource_workflow_binding.sql',
            '094_kb_item_state.sql',
            '099_kb_text_index.sql',
            '101_kb_catalog.sql',
        ]:
            await pool.execute(
                (SQL_DIRECTORY / name).read_text().replace('{schema}', schema)
            )
        yield KbCatalog(pool, schema=schema), pool, schema
    finally:
        await pool.execute(f'DROP SCHEMA {schema} CASCADE')
        await pool.close()


def scope():
    return dict(
        tenant_id=str(uuid4()), organization_id=str(uuid4()), actor_id=str(uuid4())
    )


async def index_and_source(service, identity):
    index = await service.call('create_index', {**identity, 'name': 'research'})
    upload = await service.call(
        'prepare_upload',
        {
            **identity,
            'indexId': index['id'],
            'fileName': 'one.pdf',
        },
    )
    return index, upload


async def test_catalog_is_standalone_and_scopes_every_lookup(catalog):
    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    assert index['configurationRevision'] == 0
    assert index['workflowBinding']['workflowName'] == 'kb_indexing'
    assert upload['s3Key'].startswith(
        f"tenants/{identity['tenant_id']}/kb-indexes/{index['id']}/"
    )
    assert len(await service.call('list_indexes', identity)) == 1
    from marie.kb.catalog import CatalogError

    for mismatch in [{'tenant_id': str(uuid4())}, {'organization_id': str(uuid4())}]:
        with pytest.raises(CatalogError) as error:
            await service.call(
                'get_index', {**identity, **mismatch, 'indexId': index['id']}
            )
        assert error.value.status_code == 404
    await pool.execute(
        (SQL_DIRECTORY / '101_kb_catalog.sql').read_text().replace('{schema}', schema)
    )
    assert (await service.call('get_index', {**identity, 'indexId': index['id']}))[
        'name'
    ] == 'research'


async def test_configuration_conflict_and_admission_are_atomic(catalog):
    from marie.kb.catalog import CatalogError

    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    file = dict(
        sourceId=upload['source']['id'], name='one.pdf', s3Key=upload['s3Key'], size=12
    )
    command = {**identity, 'indexId': index['id'], 'bucket': 'fixture', 'files': [file]}
    first = (await service.call('admit_documents', command))[0]
    duplicate = (await service.call('admit_documents', command))[0]
    assert first['id'] == duplicate['id']
    assert first['metadata']['kbItem'] == duplicate['metadata']['kbItem']
    with pytest.raises(CatalogError):
        await service.call(
            'admit_documents',
            {
                **command,
                'files': [file, {**file, 'sourceId': str(uuid4())}],
            },
        )
    assert await pool.fetchval(f'SELECT count(*) FROM {schema}.kb_document') == 1
    updated = await service.call(
        'update_index_configuration',
        {
            **identity,
            'indexId': index['id'],
            'expectedConfigurationRevision': 0,
            'data': {'chunkSize': 2048},
        },
    )
    assert updated['configurationRevision'] == 1
    assert updated['workflowBinding']['runParams']['chunk_size'] == 2048
    current = await service.call(
        'get_document', {**identity, 'indexId': index['id'], 'documentId': first['id']}
    )
    assert (
        current['metadata']['kbItem']['indexingDagId']
        != first['metadata']['kbItem']['indexingDagId']
    )
    with pytest.raises(CatalogError) as error:
        await service.call(
            'update_index_configuration',
            {
                **identity,
                'indexId': index['id'],
                'expectedConfigurationRevision': 0,
                'data': {'chunkSize': 4096},
            },
        )
    assert error.value.status_code == 409
    assert (await service.call('get_index', {**identity, 'indexId': index['id']}))[
        'chunkSize'
    ] == 2048


async def test_removal_preserves_catalog_until_cleanup_and_rejects_stale_reindex(
    catalog,
):
    from marie.kb.catalog import CatalogError

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
    query = {**identity, 'indexId': index['id'], 'documentId': document['id']}
    removed = await service.call('withdraw_document', query)
    assert removed['document']['metadata']['removalPending']
    assert removed['items'][0]['state'] == 'removed'
    with pytest.raises(CatalogError) as error:
        await service.call('finalize_document_removal', query)
    assert error.value.status_code == 409
    with pytest.raises(CatalogError):
        await service.call('reindex_document', query)
    assert (await service.call('withdraw_document', query))['items'][0][
        'cleanup_state'
    ] == 'pending'
    await pool.execute(f"UPDATE {schema}.kb_item_state SET cleanup_state='complete'")
    assert await service.call('finalize_document_removal', query) == {'deleted': True}
    assert await pool.fetchval(f'SELECT count(*) FROM {schema}.kb_document') == 0


async def test_submission_admission_fans_out_atomically_and_revokes_memberships(
    catalog,
):
    from marie.kb.catalog import CatalogError

    service, pool, schema = catalog
    await pool.execute(
        f'''CREATE TABLE {schema}.submissions (
        id uuid PRIMARY KEY,tenant_id uuid,name text,status text DEFAULT 'open',
        enable_semantic_search boolean DEFAULT false,kb_index_id uuid,query_plan_template_id uuid,
        total_documents integer DEFAULT 0,updated_at timestamptz DEFAULT now());
        CREATE TABLE {schema}.submission_kb_indexes (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),submission_id uuid,kb_index_id uuid,
        created_at timestamptz DEFAULT now(),UNIQUE(submission_id,kb_index_id));
        CREATE TABLE {schema}.submission_documents (
        id uuid PRIMARY KEY,submission_id uuid,file_name text,file_size bigint,content_type text,
        storage_key text,status text DEFAULT 'pending',indexing_status text,indexing_error text,
        created_at timestamptz DEFAULT now(),updated_at timestamptz DEFAULT now());'''
    )
    identity = scope()
    one = await service.call('create_index', {**identity, 'name': 'one'})
    two = await service.call('create_index', {**identity, 'name': 'two'})
    submission_id, document_id = uuid4(), uuid4()
    await pool.execute(
        f'INSERT INTO {schema}.submissions(id,tenant_id,name) VALUES($1,$2,$3)',
        submission_id,
        UUID(identity['tenant_id']),
        'Batch',
    )
    command = {
        **identity,
        'submissionId': str(submission_id),
        'kbIndexIds': [one['id'], two['id']],
        'enableSemanticSearch': True,
        'bucket': 'fixture',
    }
    await service.call('set_submission_destinations', command)
    from marie.kb.catalog import CatalogError

    blocked = max(one['id'], two['id'])
    await pool.execute(
        f"UPDATE {schema}.kb_index SET metadata=metadata || '{{\"removalPending\":true}}'::jsonb WHERE id=$1",
        UUID(blocked),
    )
    with pytest.raises(CatalogError):
        await service.call(
            'admit_submission_documents',
            {
                **identity,
                'submissionId': str(submission_id),
                'bucket': 'fixture',
                'files': [
                    dict(
                        id=str(document_id),
                        fileName='one.pdf',
                        fileSize=12,
                        storageKey=f"tenants/{identity['tenant_id']}/submissions/{submission_id}/one.pdf",
                    )
                ],
            },
        )
    for table in ('submission_documents', 'kb_document', 'kb_item_state'):
        assert await pool.fetchval(f'SELECT count(*) FROM {schema}.{table}') == 0
    await pool.execute(
        f"UPDATE {schema}.kb_index SET metadata=metadata - 'removalPending' WHERE id=$1",
        UUID(blocked),
    )
    admitted = await service.call(
        'admit_submission_documents',
        {
            **identity,
            'submissionId': str(submission_id),
            'bucket': 'fixture',
            'files': [
                dict(
                    id=str(document_id),
                    fileName='one.pdf',
                    fileSize=12,
                    storageKey=f"tenants/{identity['tenant_id']}/submissions/{submission_id}/one.pdf",
                )
            ],
        },
    )
    assert len(admitted['documents']) == 1
    assert len(admitted['kbDocuments']) == 2
    assert await pool.fetchval(f'SELECT count(*) FROM {schema}.kb_item_state') == 2
    with pytest.raises(CatalogError):
        await service.call(
            'set_submission_destinations',
            {**command, 'kbIndexIds': [one['id'], str(uuid4())]},
        )
    assert (
        await pool.fetchval(f'SELECT count(*) FROM {schema}.submission_kb_indexes') == 2
    )
    withdrawn = await service.call(
        'set_submission_destinations', {**command, 'kbIndexIds': [two['id']]}
    )
    assert len(withdrawn['items']) == 1
    assert withdrawn['items'][0]['index_id'] == one['id']
    assert (
        await pool.fetchval(
            f"SELECT count(*) FROM {schema}.kb_item_state WHERE state='removed'"
        )
        == 1
    )
    listing = {**identity, 'indexId': one['id']}
    assert len(await service.call('list_documents', listing)) == 1
    await pool.execute(
        f"UPDATE {schema}.kb_item_state SET cleanup_state='complete' WHERE state='removed'"
    )
    assert await service.call('list_documents', listing) == []
    unlinked = await service.call('get_index', listing)
    assert unlinked['documentCount'] == 0
    assert unlinked['sources'][0]['documentCount'] == 0
    assert (
        len(await service.call('list_documents', {**identity, 'indexId': two['id']}))
        == 1
    )
    restored = await service.call('set_submission_destinations', command)
    original = next(
        document
        for document in admitted['kbDocuments']
        if document['indexId'] == one['id']
    )
    replacement = next(
        document
        for document in restored['documents']
        if document['indexId'] == one['id']
    )
    assert (
        replacement['metadata']['kbItem']['itemId']
        != original['metadata']['kbItem']['itemId']
    )
    assert await pool.fetchval(f'SELECT count(*) FROM {schema}.kb_item_state') == 3
    assert len(await service.call('list_documents', listing)) == 1
    assert (await service.call('get_index', listing))['documentCount'] == 1
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.fetchval(
                f'SELECT id FROM {schema}.submissions WHERE id=$1 FOR UPDATE',
                submission_id,
            )
            registration = asyncio.create_task(
                service.call(
                    'register_submission_documents',
                    {
                        **identity,
                        'submissionId': str(submission_id),
                        'bucket': 'fixture',
                    },
                )
            )
            await asyncio.sleep(0.05)
            assert not registration.done()
            await connection.execute(
                f'UPDATE {schema}.submissions SET enable_semantic_search=false WHERE id=$1',
                submission_id,
            )
            await connection.execute(
                f"UPDATE {schema}.kb_item_state SET state='removed',cleanup_state='pending'"
            )
    assert await registration == []
    assert (
        await pool.fetchval(
            f"SELECT count(*) FROM {schema}.kb_item_state WHERE state <> 'removed'"
        )
        == 0
    )


async def test_source_schedule_is_atomic_and_active_edits_conflict(catalog):
    from marie.kb.catalog import CatalogError

    service, pool, schema = catalog
    await pool.execute(
        f'''CREATE TABLE {schema}.sensor (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),external_id uuid UNIQUE,name text,config jsonb,
        target_job_name text,target_dag_id uuid,cursor text,minimum_interval_seconds int DEFAULT 30,
        sensor_type text,status text DEFAULT 'inactive',updated_at timestamptz DEFAULT now());'''
    )
    identity = scope()
    index = await service.call(
        'create_index', {**identity, 'name': 'scheduled', 'workspaceId': str(uuid4())}
    )
    source = await service.call(
        'save_source',
        {
            **identity,
            'indexId': index['id'],
            'expectedConfigurationRevision': 0,
            'name': 'Repository',
            'sourceType': 'repository',
            'datasetInputs': {'repositoryId': str(uuid4())},
            'syncMode': 'scheduled',
            'schedule': {'cron': '0 * * * *', 'timezone': 'America/Chicago'},
        },
    )
    assert source['source']['schedule']['status'] == 'inactive'
    active = await service.call(
        'set_source_schedule',
        {
            **identity,
            'indexId': index['id'],
            'sourceId': source['source']['id'],
            'expectedConfigurationRevision': 1,
            'action': 'activate',
        },
    )
    assert active['source']['schedule']['status'] == 'active'
    with pytest.raises(CatalogError):
        await service.call(
            'save_source',
            {
                **identity,
                'indexId': index['id'],
                'sourceId': source['source']['id'],
                'expectedConfigurationRevision': 2,
                'name': 'Changed',
                'sourceType': 'repository',
            },
        )
    assert (await service.call('get_index', {**identity, 'indexId': index['id']}))[
        'configurationRevision'
    ] == 2
    assert await service.call(
        'owns_schedule', {**identity, 'externalId': source['source']['triggerConfigId']}
    ) == {'owned': True}


async def test_source_registration_waits_for_upstream_admission_and_binds_once(catalog):
    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    await pool.execute(
        f'''CREATE TABLE {schema}.dag (
        id uuid PRIMARY KEY,ref_id text,ref_type text,project_id text,planner text,
        created_on timestamptz DEFAULT now(),completed_on timestamptz,state text);
        CREATE TABLE {schema}.job(id uuid PRIMARY KEY,dag_id uuid,data jsonb);'''
    )
    uri = f"s3://fixture/{upload['s3Key']}"
    upstream_id = uuid4()
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
                upstreamPlanner='extract',
                upstreamFabrics=['fabric-a'],
            )
        ],
    }
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(jsonb_build_array('kb_upstream'::text,$1::text,$2::text,$3::text)::text,0))",
                identity['tenant_id'],
                'extract',
                uri,
            )
            registration = asyncio.create_task(
                service.call('admit_documents', admission)
            )
            await asyncio.sleep(0.05)
            assert not registration.done()
            await connection.execute(
                f'INSERT INTO {schema}.dag(id,ref_id,ref_type,project_id,planner) VALUES($1,$2,$3,$4,$5)',
                upstream_id,
                'document-ref',
                'submission',
                identity['tenant_id'],
                'extract',
            )
            await connection.execute(
                f'INSERT INTO {schema}.job(id,dag_id,data) VALUES($1,$2,$3::jsonb)',
                uuid4(),
                upstream_id,
                json.dumps({'metadata': {'uri': uri, 'fabric_group_id': 'fabric-a'}}),
            )
    document = (await registration)[0]
    assert document['itemState']['input_ref']['dag_id'] == str(upstream_id)
    assert document['itemState']['input_ref']['fabric_group_id'] == 'fabric-a'
    assert (await service.call('admit_documents', admission))[0]['id'] == document['id']
    assert await pool.fetchval(f'SELECT count(*) FROM {schema}.kb_item_state') == 1


async def test_current_citation_and_shared_storage_reference_checks(catalog):
    from marie.kb.catalog import CatalogError

    service, pool, schema = catalog
    identity = scope()
    index = await service.call(
        'create_index',
        {**identity, 'name': 'text', 'retrievalBackend': 'postgresql_text'},
    )
    upload = await service.call(
        'prepare_upload', {**identity, 'indexId': index['id'], 'fileName': 'saved.md'}
    )
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
                        name='saved.md',
                        s3Key=upload['s3Key'],
                        size=12,
                    )
                ],
            },
        )
    )[0]
    item = document['metadata']['kbItem']
    await pool.execute(
        f"UPDATE {schema}.kb_item_state SET state='ready',indexed_dag_id=indexing_dag_id,record_count=1"
    )
    await pool.execute(
        f'''INSERT INTO {schema}.kb_text_section(tenant_id,index_id,source_id,item_id,section_id,
        indexed_dag_id,content,source_digest,input_ref,start_offset,end_offset)
        VALUES($1,$2,$3,$4,'section',$5,'saved passage',$6,'{{}}',0,13)''',
        UUID(identity['tenant_id']),
        UUID(index['id']),
        upload['source']['id'],
        item['itemId'],
        UUID(item['indexingDagId']),
        'a' * 64,
    )
    citation = dict(
        sourceId=upload['source']['id'],
        itemId=item['itemId'],
        sectionId='section',
        indexedDagId=item['indexingDagId'],
        sourceDigest='a' * 64,
    )
    query = {**identity, 'indexId': index['id'], 'citation': citation}
    assert (await service.call('check_text_citation', query))['document'][
        'id'
    ] == document['id']
    assert (
        await service.call(
            'document_evidence', {**identity, 'documentId': document['id']}
        )
        == citation
    )
    assert (
        await service.call('get_document', {**identity, 'documentId': document['id']})
    )['indexId'] == index['id']
    assert await service.call(
        'storage_references', {**identity, 'keys': [upload['s3Key']]}
    ) == {'keys': [upload['s3Key']]}
    assert await service.call(
        'storage_references',
        {**identity, 'keys': [upload['s3Key']], 'excludeDocumentId': document['id']},
    ) == {'keys': []}
    with pytest.raises(CatalogError):
        await service.call(
            'check_text_citation',
            {**query, 'citation': {**citation, 'sourceDigest': 'b' * 64}},
        )
    await service.call('withdraw_document', {**identity, 'documentId': document['id']})
    with pytest.raises(CatalogError):
        await service.call('check_text_citation', query)


async def test_real_gateway_dispatch_preserves_scope_and_safe_dtos(
    catalog, monkeypatch
):
    import httpx
    from fastapi import FastAPI

    import marie.kb.catalog_routes as routes
    from marie.auth.api_key_manager import APIKeyManager
    from marie.kb.catalog import KbCatalog

    service, pool, schema = catalog
    monkeypatch.setattr(
        APIKeyManager, 'is_valid', staticmethod(lambda token: token == 'test')
    )
    monkeypatch.setattr(
        routes, 'KbCatalog', lambda received: KbCatalog(received, schema=schema)
    )
    app = FastAPI()
    routes.register_kb_catalog_routes(app, lambda: pool)
    identity = scope()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url='http://fixture',
        headers={'Authorization': 'Bearer test'},
    ) as api:
        created = await api.post(
            '/api/v1/kb/catalog/create_index', json={**identity, 'name': 'gateway'}
        )
        assert created.status_code == 200
        index = created.json()['result']
        assert UUID(index['id'])
        conflict = await api.post(
            '/api/v1/kb/catalog/create_index', json={**identity, 'name': 'gateway'}
        )
        assert conflict.status_code == 409
        other = await api.post(
            '/api/v1/kb/catalog/get_index',
            json={**identity, 'organization_id': str(uuid4()), 'indexId': index['id']},
        )
        assert other.status_code == 404
        bad = await api.post(
            '/api/v1/kb/catalog/update_index_configuration',
            json={
                **identity,
                'indexId': index['id'],
                'expectedConfigurationRevision': 99,
                'data': {'chunkSize': 4096},
            },
        )
        assert bad.status_code == 409
        assert await pool.fetchval(f'SELECT chunk_size FROM {schema}.kb_index') == 1024


async def test_audit_remains_index_scoped_after_document_removal(catalog):
    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    await service.call(
        'record_audit',
        {
            **identity,
            'entityType': 'kb_index',
            'entityId': index['id'],
            'action': 'create',
        },
    )
    audit = await service.call(
        'record_audit',
        {
            **identity,
            'entityType': 'kb_source',
            'entityId': upload['source']['id'],
            'action': 'upload',
        },
    )
    assert audit['indexId'] == index['id']
    await service.call(
        'withdraw_source',
        {**identity, 'indexId': index['id'], 'sourceId': upload['source']['id']},
    )
    await service.call(
        'finalize_source_removal',
        {**identity, 'indexId': index['id'], 'sourceId': upload['source']['id']},
    )
    await service.call(
        'record_audit',
        {
            **identity,
            'entityType': 'kb_source',
            'entityId': upload['source']['id'],
            'action': 'delete',
        },
    )
    result = await service.call('list_audit', {**identity, 'indexId': index['id']})
    assert result['total'] == 3
    assert {entry['entityId'] for entry in result['entries']} == {
        index['id'],
        upload['source']['id'],
    }


async def test_reindex_keeps_committed_output_and_stale_mapping_cannot_remove_it(
    catalog,
):
    from marie.kb.catalog import CatalogError

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
    old = document['metadata']['kbItem']['indexingDagId']
    await pool.execute(
        f"UPDATE {schema}.kb_item_state SET state='ready',indexed_dag_id=indexing_dag_id,record_count=7"
    )
    query = {**identity, 'documentId': document['id'], 'expectedIndexingDagId': old}
    reindexed = await service.call('reindex_document', query)
    assert reindexed['itemState']['indexed_dag_id'] == old
    assert reindexed['itemState']['record_count'] == 7
    assert reindexed['itemState']['indexing_dag_id'] != old
    for operation in ('reindex_document', 'withdraw_document'):
        with pytest.raises(CatalogError):
            await service.call(operation, query)
    current = await service.call(
        'get_document', {**identity, 'documentId': document['id']}
    )
    assert current['itemState']['state'] == 'pending'
    assert not current['metadata']['removalPending']


async def test_configuration_refresh_preserves_failed_upstream_without_artifact(
    catalog,
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
    old = document['metadata']['kbItem']['indexingDagId']
    await pool.execute(
        f'''UPDATE {schema}.kb_item_state SET state='error',error='upstream failed',
        input_ref='{{"kind":"upstream","planner":"extract"}}'::jsonb'''
    )
    await service.call(
        'update_index_configuration',
        {
            **identity,
            'indexId': index['id'],
            'expectedConfigurationRevision': 0,
            'data': {'chunkSize': 2048},
        },
    )
    current = await service.call(
        'get_document', {**identity, 'documentId': document['id']}
    )
    assert current['itemState']['indexing_dag_id'] == old
    assert current['itemState']['state'] == 'error'
    assert current['itemState']['run_params']['chunk_size'] == 2048
    await pool.execute(f"UPDATE {schema}.kb_item_state SET input_ref=NULL")
    await service.call(
        'update_index_configuration',
        {
            **identity,
            'indexId': index['id'],
            'expectedConfigurationRevision': 1,
            'data': {'chunkSize': 4096},
        },
    )
    current = await service.call(
        'get_document', {**identity, 'documentId': document['id']}
    )
    assert current['itemState']['indexing_dag_id'] != old
    assert current['itemState']['state'] == 'pending'


async def test_registration_before_upstream_admission_is_bound_by_existing_scheduler_helper(
    catalog,
):
    from marie.kb.item_state import bind_upstream_input

    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    await pool.execute(
        f'''CREATE TABLE {schema}.dag (
        id uuid PRIMARY KEY,ref_id text,ref_type text,project_id text,planner text,
        created_on timestamptz DEFAULT now(),completed_on timestamptz,state text);
        CREATE TABLE {schema}.job(id uuid PRIMARY KEY,dag_id uuid,data jsonb);'''
    )
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
                        upstreamPlanner='extract',
                        upstreamFabrics=['fabric-a'],
                    )
                ],
            },
        )
    )[0]
    assert 'dag_id' not in document['itemState']['input_ref']
    upstream = uuid4()

    class ScopedConnection:
        def __init__(self, connection):
            self.connection = connection

        async def execute(self, statement, *args):
            return await self.connection.execute(
                statement.replace('marie_scheduler.', f'{schema}.'), *args
            )

        async def fetchval(self, statement, *args):
            return await self.connection.fetchval(
                statement.replace('marie_scheduler.', f'{schema}.'), *args
            )

    async with pool.acquire() as connection:
        async with connection.transaction():
            await bind_upstream_input(
                ScopedConnection(connection),
                str(upstream),
                {
                    'project_id': identity['tenant_id'],
                    'uri': f"s3://fixture/{upload['s3Key']}",
                    'ref_id': 'document-ref',
                    'ref_type': 'submission',
                    'fabric_group_id': 'fabric-a',
                },
                'extract',
            )
    current = await service.call(
        'get_document', {**identity, 'documentId': document['id']}
    )
    assert current['itemState']['input_ref']['dag_id'] == str(upstream)
    assert current['dagId'] == current['itemState']['indexing_dag_id']


async def test_concurrent_configuration_saves_admit_one_revision(catalog):
    from marie.kb.catalog import CatalogError

    service, pool, schema = catalog
    identity = scope()
    index, upload = await index_and_source(service, identity)
    results = await asyncio.gather(
        *[
            service.call(
                'update_index_configuration',
                {
                    **identity,
                    'indexId': index['id'],
                    'expectedConfigurationRevision': 0,
                    'data': {'chunkSize': size},
                },
            )
            for size in (2048, 4096)
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(result, CatalogError) for result in results) == 1
    winner = next(result for result in results if isinstance(result, dict))
    current = await service.call('get_index', {**identity, 'indexId': index['id']})
    assert current['configurationRevision'] == 1
    assert current['chunkSize'] == winner['chunkSize']
    assert current['workflowBinding']['runParams']['chunk_size'] == winner['chunkSize']


async def test_catalog_uses_scheduler_tuple_pool_without_changing_its_row_factory(
    catalog,
):
    from psycopg.rows import tuple_row

    from marie.kb.catalog import KbCatalog

    service, pool, schema = catalog
    scheduler_pool = AsyncPostgresConnectionPool()
    await scheduler_pool.initialize(
        dict(
            hostname='127.0.0.1',
            port=int(os.environ['MARIE_SENSOR_TEST_PORT']),
            username='postgres',
            password='',
            database='f10c_qualification',
        ),
        row_factory=tuple_row,
        autocommit=True,
    )
    try:
        identity = scope()
        index = await KbCatalog(scheduler_pool, schema=schema).call(
            'create_index', {**identity, 'name': 'scheduler-pool'}
        )
        assert index['name'] == 'scheduler-pool'
        assert await scheduler_pool.fetchrow('SELECT 1') == (1,)
    finally:
        await scheduler_pool.close()


@pytest.mark.parametrize('legacy_state', ['unmapped', 'missing', 'removed'])
async def test_reindex_repairs_legacy_registration_without_reusing_removed_items(
    catalog, legacy_state
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
    original = document['metadata']['kbItem']['itemId']
    if legacy_state == 'removed':
        await pool.execute(
            f"UPDATE {schema}.kb_item_state SET state='removed',cleanup_state='complete'"
        )
    else:
        await pool.execute(f'DELETE FROM {schema}.kb_item_state')
        if legacy_state == 'unmapped':
            await pool.execute(
                f"UPDATE {schema}.kb_document SET metadata=metadata - 'kbItem'"
            )
    repaired = await service.call(
        'reindex_document',
        {**identity, 'documentId': document['id'], 'bucket': 'fixture'},
    )
    assert repaired['itemState']['state'] == 'pending'
    assert repaired['itemState']['owner_id'] == document['id']
    if legacy_state == 'removed':
        assert repaired['metadata']['kbItem']['itemId'] != original


async def test_first_repository_source_binds_workspace_atomically_and_cannot_rebind(
    catalog,
):
    from marie.kb.catalog import CatalogError

    service, pool, schema = catalog
    identity = scope()
    index = await service.call('create_index', {**identity, 'name': 'code-workspace'})
    workspace = str(uuid4())
    source = {
        'indexId': index['id'],
        'expectedConfigurationRevision': 0,
        'name': 'Code',
        'sourceType': 'repository',
        'workspaceId': workspace,
    }
    saved = await service.call('save_source', {**identity, **source})
    current = await service.call('get_index', {**identity, 'indexId': index['id']})
    assert current['workspaceId'] == workspace and current['configurationRevision'] == 1
    with pytest.raises(CatalogError) as mismatch:
        await service.call(
            'save_source',
            {
                **identity,
                **source,
                'sourceId': saved['source']['id'],
                'expectedConfigurationRevision': 1,
                'workspaceId': str(uuid4()),
            },
        )
    assert mismatch.value.status_code == 409
    second = await service.call('create_index', {**identity, 'name': 'failed-binding'})
    with pytest.raises(CatalogError):
        await service.call(
            'save_source',
            {**identity, **source, 'indexId': second['id'], 'sourceId': str(uuid4())},
        )
    second = await service.call('get_index', {**identity, 'indexId': second['id']})
    assert second['workspaceId'] is None and second['configurationRevision'] == 0


async def test_repository_attachment_creates_once_and_skips_existing_binding_on_retry(
    catalog,
):
    service, pool, schema = catalog
    identity = scope()
    index = await service.call(
        'create_index',
        {**identity, 'name': 'repository-attachment', 'workspaceId': str(uuid4())},
    )
    repository_id = str(uuid4())
    command = {
        **identity,
        'indexId': index['id'],
        'expectedConfigurationRevision': 0,
        'name': 'Engine',
        'sourceType': 'repository',
        'datasetInputs': {
            'repositoryId': repository_id,
            'branch': 'main',
            'include': ['src/*'],
        },
        'schedule': None,
        'syncMode': 'manual',
        'status': 'registered',
    }
    first = await service.call('attach_repository_source', command)
    retried = await service.call(
        'attach_repository_source',
        {
            **command,
            'datasetInputs': {**command['datasetInputs'], 'include': ['different/*']},
        },
    )
    assert first['source']['id'] == retried['source']['id']
    assert first['alreadyAdded'] is False and retried['alreadyAdded'] is True
    assert retried['source']['datasetInputs']['include'] == ['src/*']
    assert (
        await pool.fetchval(
            f'SELECT count(*) FROM {schema}.kb_source WHERE index_id=$1',
            UUID(index['id']),
        )
        == 1
    )
