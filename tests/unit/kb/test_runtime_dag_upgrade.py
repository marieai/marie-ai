from pathlib import Path
from uuid import uuid4

import pytest
from psycopg.errors import DuplicateColumn

from tests.unit.kb.test_current_schema import database, install, pytestmark

UPGRADE = Path(__file__).parents[3] / 'tools/kb/kb-runtime-dag-upgrade.sql'


async def upgrade(connection, schema: str) -> None:
    await connection.execute(UPGRADE.read_text().replace('marie_scheduler', schema))


@pytest.mark.parametrize('layout', ['mixed_without_view', 'legacy_with_view', 'current'])
async def test_installed_runtime_upgrade_preserves_rows_and_view_identity(database, layout):
    connection, schema = database
    await install(connection, schema, '094_kb_item_state.sql')
    await install(connection, schema, '099_kb_text_index.sql')
    tenant, index, dag = uuid4(), uuid4(), uuid4()
    await connection.execute(
        f'''INSERT INTO {schema}.kb_item_state
        (tenant_id,index_id,source_id,item_id,owner_type,owner_id,uri,
         indexing_dag_id,indexed_dag_id,state,backend,store_name)
        VALUES (%s,%s,'source','item','kb_document','owner','s3://fixture/item',
                %s,%s,'ready','postgresql_text','kb_text_section');''',
        (tenant, index, dag, dag),
    )
    await connection.execute(
        f'''INSERT INTO {schema}.kb_text_section
        (tenant_id,index_id,source_id,item_id,section_id,indexed_dag_id,content,
         source_digest,input_ref,start_offset,end_offset)
        VALUES (%s,%s,'source','item','section',%s,'saved text',%s,'{{}}',0,10)''',
        (tenant, index, dag, 'a' * 64),
    )
    if layout == 'mixed_without_view':
        await connection.execute(f'DROP VIEW {schema}.kb_text_section_current')
    else:
        await connection.execute(
            f'CREATE VIEW {schema}.dependent_passage AS SELECT content FROM {schema}.kb_text_section_current'
        )
    if layout != 'current':
        await connection.execute(
            f'ALTER TABLE {schema}.kb_text_section RENAME COLUMN indexed_dag_id TO attempt_id'
        )
    if layout == 'legacy_with_view':
        await connection.execute(
            f'''ALTER TABLE {schema}.kb_item_state RENAME COLUMN indexing_dag_id TO attempt_id;
            ALTER TABLE {schema}.kb_item_state RENAME COLUMN indexed_dag_id TO completed_attempt_id;
            ALTER VIEW {schema}.kb_text_section_current RENAME COLUMN indexed_dag_id TO attempt_id;
            ALTER VIEW {schema}.kb_text_section_current RENAME COLUMN indexing_dag_id TO current_attempt_id;
            DROP INDEX {schema}.idx_kb_item_upstream_input;
            ALTER TABLE {schema}.kb_item_state DROP COLUMN input_ref'''
        )
    cursor = await connection.execute(
        "SELECT to_regclass(%s)::oid AS table_oid,to_regclass(%s)::oid AS view_oid",
        (f'{schema}.kb_text_section', f'{schema}.kb_text_section_current'),
    )
    before = await cursor.fetchone()
    for _ in range(2):
        await upgrade(connection, schema)
        await install(connection, schema, '094_kb_item_state.sql')
        await install(connection, schema, '099_kb_text_index.sql')
        cursor = await connection.execute(f'SELECT * FROM {schema}.kb_text_section_current')
        row = await cursor.fetchone()
        assert row['content'] == 'saved text'
        assert row['indexed_dag_id'] == dag
        assert row['indexing_dag_id'] == dag
    cursor = await connection.execute(
        "SELECT to_regclass(%s)::oid AS table_oid,to_regclass(%s)::oid AS view_oid",
        (f'{schema}.kb_text_section', f'{schema}.kb_text_section_current'),
    )
    after = await cursor.fetchone()
    assert after['table_oid'] == before['table_oid']
    if before['view_oid']:
        assert after['view_oid'] == before['view_oid']
        cursor = await connection.execute(f'SELECT content FROM {schema}.dependent_passage')
        assert await cursor.fetchall() == [{'content': 'saved text'}]


async def test_ambiguous_runtime_columns_fail_atomically(database):
    connection, schema = database
    await install(connection, schema, '094_kb_item_state.sql')
    await install(connection, schema, '099_kb_text_index.sql')
    await connection.execute(
        f'''ALTER TABLE {schema}.kb_item_state RENAME COLUMN indexing_dag_id TO attempt_id;
        ALTER TABLE {schema}.kb_text_section ADD COLUMN attempt_id uuid'''
    )
    with pytest.raises(DuplicateColumn):
        await upgrade(connection, schema)
    cursor = await connection.execute(
        '''SELECT column_name FROM information_schema.columns
        WHERE table_schema=%s AND table_name='kb_item_state'
          AND column_name IN ('attempt_id','indexing_dag_id')''', (schema,),
    )
    assert await cursor.fetchall() == [{'column_name': 'attempt_id'}]
