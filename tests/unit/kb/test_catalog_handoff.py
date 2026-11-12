import os
from pathlib import Path
from uuid import uuid4

import pytest

from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Requires an owned disposable PostgreSQL database',
)
SQL_DIRECTORY = Path(__file__).parents[3] / 'config/psql/schema'
HANDOFF = Path(__file__).parents[3] / 'tools/kb/kb-backend-handoff.sql'


@pytest.fixture
async def legacy():
    source, target = f'kb_legacy_{uuid4().hex}', f'kb_owned_{uuid4().hex}'
    pool = AsyncPostgresConnectionPool()
    await pool.initialize(
        dict(
            hostname='127.0.0.1',
            port=int(os.environ['MARIE_SENSOR_TEST_PORT']),
            username='postgres',
            password='',
            database='f10c_qualification',
        ),
        autocommit=True,
    )
    await pool.execute(
        f'CREATE SCHEMA {source}; CREATE SCHEMA {target}; CREATE SCHEMA {source}_runtime'
    )
    try:
        await pool.execute(
            (SQL_DIRECTORY / '101_kb_catalog.sql')
            .read_text()
            .replace('{schema}', source)
        )
        await pool.execute(
            f'''ALTER TABLE {source}.kb_index DROP COLUMN organization_id CASCADE;
            ALTER TABLE {source}.kb_audit_log DROP COLUMN organization_id CASCADE;
            ALTER TABLE {source}.kb_audit_log DROP COLUMN index_id CASCADE;
            CREATE TABLE {source}_runtime.tenant(id uuid PRIMARY KEY,organization_id uuid NOT NULL);
            CREATE TABLE {source}."user"(id uuid PRIMARY KEY);
            ALTER TABLE {source}.kb_index ADD CONSTRAINT tenant_fk FOREIGN KEY(tenant_id) REFERENCES {source}_runtime.tenant(id);
            ALTER TABLE {source}.kb_document ADD CONSTRAINT creator_fk FOREIGN KEY(created_by) REFERENCES {source}."user"(id);
            ALTER TABLE {source}.kb_audit_log ADD CONSTRAINT actor_fk FOREIGN KEY(user_id) REFERENCES {source}."user"(id);
            CREATE TYPE {source}.segmentation_mode AS ENUM('page','paragraph','semantic');
            ALTER TABLE {source}.kb_index DROP CONSTRAINT kb_index_segmentation_mode_check;
            ALTER TABLE {source}.kb_index ALTER COLUMN segmentation_mode DROP DEFAULT;
            ALTER TABLE {source}.kb_index ALTER COLUMN segmentation_mode TYPE {source}.segmentation_mode USING segmentation_mode::{source}.segmentation_mode;
            ALTER TABLE {source}.kb_index ALTER COLUMN segmentation_mode SET DEFAULT 'semantic';
            ALTER TABLE {source}.kb_source DROP COLUMN extension_install_id,DROP COLUMN datasource_ref,
                DROP COLUMN dataset_inputs,DROP COLUMN sync_mode,DROP COLUMN trigger_config_id;
            CREATE TABLE {target}.submissions(id uuid PRIMARY KEY,kb_index_id uuid REFERENCES {source}.kb_index(id));
            CREATE TABLE {target}.submission_kb_indexes(submission_id uuid,kb_index_id uuid REFERENCES {source}.kb_index(id) ON DELETE CASCADE);'''
        )
        yield pool, source, target
    finally:
        await pool.execute(
            f'DROP SCHEMA {source} CASCADE; DROP SCHEMA {target} CASCADE; DROP SCHEMA {source}_runtime CASCADE'
        )
        await pool.close()


async def test_handoff_preserves_catalog_objects_incoming_fks_and_row_identity(legacy):
    pool, source, target = legacy
    tenant, organization, index, source_id, document, actor, dag, submission = [
        uuid4() for _ in range(8)
    ]
    await pool.execute(
        f'INSERT INTO {source}_runtime.tenant VALUES($1,$2);', tenant, organization
    )
    await pool.execute(f'INSERT INTO {source}."user" VALUES($1)', actor)
    await pool.execute(
        f'''INSERT INTO {source}.kb_index(id,tenant_id,name,metadata,created_at,updated_at)
        VALUES($1,$2,'preserved','{{"configurationRevision":7,"toolId":"external"}}','2020-01-01Z','2021-01-01Z')''',
        index,
        tenant,
    )
    await pool.execute(
        f'''INSERT INTO {source}.kb_source(id,index_id,name,file_count,node_count) VALUES($1,$2,'legacy',1,5)''',
        source_id,
        index,
    )
    await pool.execute(
        f'''INSERT INTO {source}.kb_document(id,source_id,name,s3_key,size,dag_id,metadata)
        VALUES($1,$2,'saved.pdf','kb-indexes/legacy/saved.pdf',42,$3,
        jsonb_build_object('kbItem',jsonb_build_object('itemId','stable-item','indexingDagId',$3::text),'custom','value'))''',
        document,
        source_id,
        dag,
    )
    await pool.execute(
        f'''INSERT INTO {source}.kb_audit_log(tenant_id,entity_type,entity_id,action,user_id)
        VALUES($1,'kb_document',$2,'upload',$3)''',
        tenant,
        document,
        actor,
    )
    await pool.execute(
        f'INSERT INTO {target}.submissions VALUES($1,$2)', submission, index
    )
    await pool.execute(
        f'INSERT INTO {target}.submission_kb_indexes VALUES($1,$2)', submission, index
    )
    oid = await pool.fetchval('SELECT to_regclass($1)::oid', f'{source}.kb_index')
    before = await pool.fetchrow(
        f'SELECT * FROM {source}.kb_document WHERE id=$1', document
    )
    statement = (
        HANDOFF.read_text()
        .replace('marie_studio', source)
        .replace('marie_scheduler', target)
        .replace('marie_runtime', f'{source}_runtime')
    )
    await pool.execute(statement)
    assert (
        await pool.fetchval('SELECT to_regclass($1)::oid', f'{target}.kb_index') == oid
    )
    assert (
        await pool.fetchrow(f'SELECT * FROM {target}.kb_document WHERE id=$1', document)
        == before
    )
    moved = await pool.fetchrow(f'SELECT * FROM {target}.kb_index WHERE id=$1', index)
    assert moved['organization_id'] == organization
    assert moved['metadata'] == {'configurationRevision': 7, 'toolId': 'external'}
    assert moved['created_at'].year == 2020 and moved['updated_at'].year == 2021
    assert (
        await pool.fetchval(
            f'SELECT kb_index_id FROM {target}.submissions WHERE id=$1', submission
        )
        == index
    )
    assert (
        await pool.fetchval(f'SELECT count(*) FROM {target}.submission_kb_indexes') == 1
    )
    assert (
        await pool.fetchval(
            f'''SELECT count(*) FROM pg_constraint c WHERE c.contype='f'
        AND c.confrelid=$1::oid AND c.conrelid IN ($2::regclass,$3::regclass)''',
            oid,
            f'{target}.submissions',
            f'{target}.submission_kb_indexes',
        )
        == 2
    )
    assert await pool.fetchval(f'SELECT index_id FROM {target}.kb_audit_log') == index
    assert (
        await pool.fetchval(f'SELECT organization_id FROM {target}.kb_audit_log')
        == organization
    )
    assert await pool.fetchval(f'SELECT file_count FROM {target}.kb_source') == 1
    assert await pool.fetchval(f'SELECT dataset_inputs FROM {target}.kb_source') == {}
    assert (
        await pool.fetchval(
            f"SELECT pg_typeof(segmentation_mode)::text FROM {target}.kb_index"
        )
        == 'text'
    )
    assert not await pool.fetchval(
        f'''SELECT EXISTS(SELECT 1 FROM pg_constraint c JOIN pg_class r ON r.oid=c.conrelid
        JOIN pg_namespace n ON n.oid=r.relnamespace JOIN pg_class f ON f.oid=c.confrelid
        JOIN pg_namespace fn ON fn.oid=f.relnamespace WHERE n.nspname=$1 AND r.relname IN ('kb_index','kb_source','kb_document','kb_audit_log')
        AND c.contype='f' AND fn.nspname=$2)''',
        target,
        source,
    )
    await pool.execute(
        (SQL_DIRECTORY / '101_kb_catalog.sql').read_text().replace('{schema}', target)
    )


async def test_handoff_refuses_dual_catalogs_without_moving_any_source(legacy):
    from psycopg.errors import RaiseException

    pool, source, target = legacy
    await pool.execute(f'CREATE TABLE {target}.kb_index(id uuid PRIMARY KEY)')
    statement = (
        HANDOFF.read_text()
        .replace('marie_studio', source)
        .replace('marie_scheduler', target)
        .replace('marie_runtime', f'{source}_runtime')
    )
    with pytest.raises(RaiseException, match='already exists'):
        await pool.execute(statement)
    assert (
        await pool.fetchval('SELECT to_regclass($1)', f'{source}.kb_index') is not None
    )
    assert (
        await pool.fetchval('SELECT to_regclass($1)', f'{source}.kb_document')
        is not None
    )
    assert await pool.fetchval('SELECT to_regclass($1)', f'{target}.kb_source') is None


async def test_handoff_missing_tenant_authority_rolls_back_all_changes(legacy):
    from psycopg.errors import RaiseException

    pool, source, target = legacy
    await pool.execute(f'ALTER TABLE {source}.kb_index DROP CONSTRAINT tenant_fk')
    await pool.execute(
        f"INSERT INTO {source}.kb_index(id,tenant_id,name) VALUES($1,$2,'orphan')",
        uuid4(),
        uuid4(),
    )
    with pytest.raises(RaiseException, match='organization authority'):
        await pool.execute(
            HANDOFF.read_text()
            .replace('marie_studio', source)
            .replace('marie_scheduler', target)
            .replace('marie_runtime', f'{source}_runtime')
        )
    assert (
        await pool.fetchval('SELECT to_regclass($1)', f'{source}.kb_index') is not None
    )
    assert await pool.fetchval('SELECT to_regclass($1)', f'{target}.kb_index') is None
    assert not await pool.fetchval(
        '''SELECT EXISTS(SELECT 1 FROM information_schema.columns
        WHERE table_schema=$1 AND table_name='kb_index' AND column_name='organization_id')''',
        source,
    )
