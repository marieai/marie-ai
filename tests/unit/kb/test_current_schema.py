import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg.rows import dict_row

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Requires an owned disposable PostgreSQL database',
)
SQL_DIRECTORY = Path(__file__).parents[3] / 'config/psql/schema'


@pytest.fixture
async def database():
    schema = f'kb_schema_{uuid4().hex}'
    async with await psycopg.AsyncConnection.connect(
        host='127.0.0.1',
        port=int(os.environ['MARIE_SENSOR_TEST_PORT']),
        user='postgres',
        dbname='f10c_qualification',
        autocommit=True,
        row_factory=dict_row,
    ) as connection:
        await connection.execute(f'CREATE SCHEMA {schema}')
        try:
            yield connection, schema
        finally:
            await connection.execute(f'DROP SCHEMA {schema} CASCADE')


async def install(connection, schema: str, filename: str) -> None:
    await connection.execute(
        (SQL_DIRECTORY / filename).read_text().replace('{schema}', schema)
    )


async def test_current_kb_schema_installs_and_replays_without_renames(database):
    connection, schema = database
    await install(connection, schema, '094_kb_item_state.sql')
    await install(connection, schema, '099_kb_text_index.sql')
    tenant_id, index_id, indexing_dag_id = uuid4(), uuid4(), uuid4()
    await connection.execute(
        f'''INSERT INTO {schema}.kb_item_state
            (tenant_id,index_id,source_id,item_id,owner_type,owner_id,uri,
             indexing_dag_id,indexed_dag_id,state,backend,store_name,input_ref)
            VALUES (%s,%s,'source','item','kb_document','owner','s3://fixture/item',
                    %s,%s,'ready','postgresql_text','kb_text_section','{{"kind":"text"}}')''',
        (tenant_id, index_id, indexing_dag_id, indexing_dag_id),
    )
    await connection.execute(
        f'''INSERT INTO {schema}.kb_text_section
            (tenant_id,index_id,source_id,item_id,section_id,indexed_dag_id,content,
             source_digest,input_ref,start_offset,end_offset)
            VALUES (%s,%s,'source','item','section',%s,'saved text',%s,'{{}}',0,10)''',
        (tenant_id, index_id, indexing_dag_id, 'a' * 64),
    )
    for _ in range(2):
        await install(connection, schema, '094_kb_item_state.sql')
        await install(connection, schema, '099_kb_text_index.sql')
        cursor = await connection.execute(f'SELECT * FROM {schema}.kb_text_section_current')
        row = await cursor.fetchone()
        assert row['content'] == 'saved text'
        assert row['indexed_dag_id'] == indexing_dag_id
        assert row['indexing_dag_id'] == indexing_dag_id


async def test_existing_sensor_cleanup_preserves_only_replayable_receipts(database):
    connection, schema = database
    await connection.execute(
        f'''CREATE TABLE {schema}.sensor_run_key
            (sensor_id uuid,run_key text,created_at timestamptz);
            CREATE TABLE {schema}.dag_history(history_id bigint PRIMARY KEY);
            INSERT INTO {schema}.dag_history VALUES (42)'''
    )
    await install(connection, schema, '055_sensor_retention.sql')
    sensor_id = uuid4()
    retained = f'run_status:{sensor_id}:42'
    expired = [f'run_status:{sensor_id}:41', 'schedule:old', 'run_status:malformed:42']
    for key in [retained, *expired]:
        await connection.execute(
            f"INSERT INTO {schema}.sensor_run_key VALUES (%s,%s,now()-interval '45 days')",
            (sensor_id, key),
        )
    cursor = await connection.execute(f'SELECT {schema}.cleanup_old_run_keys() AS deleted')
    assert (await cursor.fetchone())['deleted'] == len(expired)
    cursor = await connection.execute(f'SELECT run_key FROM {schema}.sensor_run_key')
    assert await cursor.fetchall() == [{'run_key': retained}]
    await connection.execute(f'DELETE FROM {schema}.dag_history')
    cursor = await connection.execute(f'SELECT {schema}.cleanup_old_run_keys() AS deleted')
    assert (await cursor.fetchone())['deleted'] == 1
