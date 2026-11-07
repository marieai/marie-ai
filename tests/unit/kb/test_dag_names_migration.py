import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable PostgreSQL'
)


async def test_dag_names_upgrade_preserves_indexed_output_and_runtime_replay():
    pool = AsyncPostgresConnectionPool()
    await pool.initialize_dsn(
        f"postgresql://postgres@127.0.0.1:{os.environ['MARIE_KB_TEST_PORT']}/kb_dag_names",
        min_size=1,
        max_size=3,
    )
    sql_dir = Path(__file__).parents[3] / 'config/psql/schema'
    tenant, index, doc, previous, requested = [uuid4() for _ in range(5)]
    try:
        await pool.execute('DROP SCHEMA IF EXISTS marie_scheduler CASCADE')
        await pool.execute('DROP SCHEMA IF EXISTS marie_studio CASCADE')
        await pool.execute('CREATE SCHEMA marie_scheduler; CREATE SCHEMA marie_studio')
        for name in [
            '094_kb_item_state.sql',
            '098_kb_workflow_inputs.sql',
            '099_kb_text_index.sql',
        ]:
            await pool.execute(
                (sql_dir / name).read_text().replace('{schema}', 'marie_scheduler')
            )
        await pool.execute(
            'CREATE TABLE marie_studio.kb_document(id uuid PRIMARY KEY,metadata jsonb,dag_id uuid)'
        )
        await pool.execute(
            'INSERT INTO marie_studio.kb_document VALUES($1,$2::jsonb,$3)',
            doc,
            json.dumps(
                {
                    'other': 'retained',
                    'kbItem': {'itemId': 'one', 'attemptId': str(requested)},
                }
            ),
            requested,
        )
        await pool.execute(
            """INSERT INTO marie_scheduler.kb_item_state
          (tenant_id,index_id,source_id,item_id,owner_type,owner_id,uri,attempt_id,completed_attempt_id,state,backend,store_name)
          VALUES($1,$2,'source','one','record','one','s3://fixture/one.md',$3,$4,'pending','postgresql_text','kb_text_section')""",
            tenant,
            index,
            requested,
            previous,
        )
        await pool.execute(
            """INSERT INTO marie_scheduler.kb_text_section
          (tenant_id,index_id,source_id,item_id,section_id,attempt_id,content,source_digest,input_ref,start_offset,end_offset)
          VALUES($1,$2,'source','one','section',$3,'covered',$4,'{}',0,7)""",
            tenant,
            index,
            previous,
            'a' * 64,
        )
        await pool.execute(
            """INSERT INTO marie_scheduler.kb_item_state
          (tenant_id,index_id,source_id,item_id,owner_type,owner_id,uri,attempt_id,completed_attempt_id,state,backend,store_name)
          VALUES($1,$2,'vector','vector','record','vector','s3://fixture/vector.md',$3,$4,'pending','pgvector','kb_vectors')""",
            tenant,
            index,
            requested,
            previous,
        )
        await pool.execute('DROP TABLE IF EXISTS kb_vectors CASCADE')
        await pool.execute(
            'CREATE TABLE kb_vectors(node_id text,index_name text,source_id text,ref_doc_id text,metadata jsonb)'
        )
        await pool.execute(
            "INSERT INTO kb_vectors VALUES('node',$1,'vector','vector',$2::jsonb)",
            str(index),
            json.dumps({'kb_attempt_id': str(previous)}),
        )
        assert (
            await pool.fetchval(
                'SELECT count(*) FROM marie_scheduler.kb_text_section_current'
            )
            == 1
        )
        upgrade = (
            (sql_dir / '100_kb_indexing_dag_names.sql')
            .read_text()
            .replace('{schema}', 'marie_scheduler')
        )
        await pool.execute(upgrade)
        item = await pool.fetchrow(
            "SELECT * FROM marie_scheduler.kb_item_state WHERE item_id='one'"
        )
        assert (
            item['indexing_dag_id'] == requested and item['indexed_dag_id'] == previous
        )
        assert (
            await pool.fetchrow('SELECT * FROM marie_scheduler.kb_text_section_current')
        )['indexed_dag_id'] == previous
        from marie.vector_stores.pgvector import PGVectorStore

        vectors = PGVectorStore(
            connection_string='unused', table_name='kb_vectors', embedding_dim=2
        )
        vectors._pool = pool
        await vectors.initialize_current_view()
        assert await pool.fetchval('SELECT count(*) FROM kb_vectors_current') == 1
        metadata = await pool.fetchval('SELECT metadata FROM marie_studio.kb_document')
        assert metadata == {
            'other': 'retained',
            'kbItem': {'itemId': 'one', 'indexingDagId': str(requested)},
        }
        for name in [
            '094_kb_item_state.sql',
            '098_kb_workflow_inputs.sql',
            '099_kb_text_index.sql',
            '100_kb_indexing_dag_names.sql',
        ]:
            await pool.execute(
                (sql_dir / name).read_text().replace('{schema}', 'marie_scheduler')
            )
        assert (
            await pool.fetchval(
                'SELECT count(*) FROM marie_scheduler.kb_text_section_current'
            )
            == 1
        )
    finally:
        await pool.close()
