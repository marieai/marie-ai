import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from marie.kb.item_state import KbItemScope, KbItemStore, StaleIndexingRun
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable KB PostgreSQL'
)


@pytest.fixture
async def state():
    pool = AsyncPostgresConnectionPool()
    await pool.initialize(
        dict(
            hostname='127.0.0.1',
            port=int(os.environ['MARIE_KB_TEST_PORT']),
            username='postgres',
            password='',
            database='postgres',
            max_connections=8,
        ),
        autocommit=True,
    )
    await pool.execute('CREATE SCHEMA IF NOT EXISTS marie_scheduler')
    await pool.execute('DROP TABLE IF EXISTS marie_scheduler.kb_text_section CASCADE')
    await pool.execute('DROP TABLE IF EXISTS marie_scheduler.kb_item_state CASCADE')
    sql = Path(__file__).parents[3] / 'config/psql/schema/094_kb_item_state.sql'
    await pool.execute(sql.read_text().replace('{schema}', 'marie_scheduler'))
    await pool.execute((sql.parent / '098_kb_workflow_inputs.sql').read_text().replace('{schema}', 'marie_scheduler'))
    for name in ['099_kb_text_index.sql','100_kb_indexing_dag_names.sql']:
        await pool.execute((sql.parent/name).read_text().replace('{schema}','marie_scheduler'))
    await pool.execute('TRUNCATE marie_scheduler.kb_item_state')
    await pool.execute(
        'CREATE TABLE IF NOT EXISTS kb_structured_output (scope text, id text, payload jsonb, PRIMARY KEY(scope,id))'
    )
    await pool.execute('TRUNCATE kb_structured_output')
    store = KbItemStore(pool)
    scope = KbItemScope(str(uuid4()), str(uuid4()), str(uuid4()), 'game:one')
    attempt = str(uuid4())
    await store.register(
        scope,
        indexing_dag_id=attempt,
        owner_type='chess_game',
        owner_id='one',
        uri='s3://fixture/games/one.json',
        run_params={},
    )
    yield store, pool, scope, attempt
    await pool.close()


async def replace(state, records, *, scope=None, attempt=None, fail=False):
    import json

    store, pool, initial_scope, initial_attempt = state
    scope = scope or initial_scope
    attempt = attempt or initial_attempt
    scope_key = json.dumps([str(value) for value in scope.values()])
    async with store.write(scope, attempt, record_count=len(records)) as conn:
        await conn.execute('DELETE FROM kb_structured_output WHERE scope=$1', scope_key)
        for record_id, payload in records:
            await conn.execute(
                'INSERT INTO kb_structured_output VALUES ($1,$2,$3::jsonb)',
                scope_key,
                record_id,
                json.dumps(payload),
            )
        if fail:
            raise RuntimeError('adapter failed')


async def test_structured_records_need_no_text_fields(state):
    store, pool, scope, attempt = state
    await replace(state, [('position', {'fen': 'board', 'moves': ['e4']})])
    assert (await pool.fetchrow('SELECT payload FROM kb_structured_output'))['payload'][
        'moves'
    ] == ['e4']
    item = await store.get(scope)
    assert str(item['indexed_dag_id']) == attempt
    assert item['record_count'] == 1


async def test_nontext_replacement_is_scoped_and_can_complete_an_empty_result(state):
    store, pool, scope, attempt = state
    other = KbItemScope(scope.tenant_id, str(uuid4()), scope.source_id, scope.item_id)
    other_attempt = str(uuid4())
    await store.register(
        other,
        indexing_dag_id=other_attempt,
        owner_type='chess_game',
        owner_id='one',
        uri='s3://fixture/games/one.json',
        run_params={},
    )
    await replace(state, [('same', {'moves': ['e4']})])
    await replace(
        state, [('same', {'moves': ['d4']})], scope=other, attempt=other_attempt
    )
    retry = str(uuid4())
    await store.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=retry)
    await replace(state, [], attempt=retry)
    assert (await store.get(scope))['record_count'] == 0
    rows = await pool.fetch('SELECT payload FROM kb_structured_output')
    assert [row['payload']['moves'] for row in rows] == [['d4']]


async def test_failed_attempt_and_duplicate_completion_cannot_replace_output(state):
    store, pool, scope, attempt = state
    await replace(state, [('first', {})])
    with pytest.raises(StaleIndexingRun):
        await replace(state, [('duplicate', {})])
    second = str(uuid4())
    await store.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=second)
    await store.fail(scope, second, 'admission failed')
    with pytest.raises(StaleIndexingRun):
        await replace(state, [('failed', {})], attempt=second)
    assert [
        row['id'] for row in await pool.fetch('SELECT id FROM kb_structured_output')
    ] == ['first']


@pytest.mark.parametrize('legacy_envelope', [False, True])
async def test_checked_job_must_be_active_and_match_current_item_and_lease(state, legacy_envelope):
    import json
    from datetime import datetime, timedelta, timezone

    store, pool, scope, attempt = state
    await pool.execute('DROP TABLE IF EXISTS marie_scheduler.job CASCADE')
    await pool.execute('''CREATE TABLE marie_scheduler.job (
        id uuid PRIMARY KEY,dag_id uuid,state text,data jsonb,run_attempt_id uuid,
        run_owner text,run_lease_expires_at timestamptz)''')
    job, run = str(uuid4()), str(uuid4())
    from marie.kb.item_state import item_run_parameters

    metadata = item_run_parameters(await store.get(scope))
    if legacy_envelope:
        metadata['kb_attempt_id'] = metadata.pop('kb_indexing_dag_id')
    await pool.execute(
        'INSERT INTO marie_scheduler.job VALUES($1,$2,$3,$4::jsonb,$5,$6,$7)',
        UUID(job),
        UUID(attempt),
        'active',
        json.dumps({'metadata': metadata}),
        UUID(run),
        'worker',
        datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    async with store.write(
        scope,
        attempt,
        record_count=0,
        job_id=job,
        run_attempt_id=run,
        run_owner='worker',
    ):
        pass
    next_attempt = str(uuid4())
    await store.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=next_attempt)
    with pytest.raises(StaleIndexingRun):
        async with store.write(
            scope,
            next_attempt,
            record_count=0,
            job_id=job,
            run_attempt_id=run,
            run_owner='worker',
        ):
            pytest.fail('old execution admitted')
    metadata['kb_indexing_dag_id'] = next_attempt
    await pool.execute(
        "UPDATE marie_scheduler.job SET dag_id=$1,data=$2::jsonb,run_lease_expires_at=NOW()-INTERVAL '1 second'",
        UUID(next_attempt),
        json.dumps({'metadata': metadata}),
    )
    with pytest.raises(StaleIndexingRun):
        async with store.write(
            scope,
            next_attempt,
            record_count=0,
            job_id=job,
            run_attempt_id=run,
            run_owner='worker',
        ):
            pytest.fail('expired execution admitted')


async def test_failed_replacement_preserves_previous_result_and_completion(state):
    store, pool, scope, first = state
    await replace(state, [('old', {'moves': ['e4']})])
    second = str(uuid4())
    await store.reindex(scope, expected_indexing_dag_id=first, indexing_dag_id=second)
    with pytest.raises(RuntimeError, match='adapter failed'):
        await replace(
            state, [('partial', {'moves': ['d4']})], attempt=second, fail=True
        )
    assert [
        row['id'] for row in await pool.fetch('SELECT id FROM kb_structured_output')
    ] == ['old']
    assert str((await store.get(scope))['indexed_dag_id']) == first


async def test_retry_rejects_old_writer_and_stale_retry(state):
    store, _, scope, first = state
    second = str(uuid4())
    await store.reindex(scope, expected_indexing_dag_id=first, indexing_dag_id=second)
    with pytest.raises(StaleIndexingRun):
        await replace(state, [('late', {})])
    with pytest.raises(StaleIndexingRun):
        await store.reindex(scope, expected_indexing_dag_id=first, indexing_dag_id=str(uuid4()))
    await replace(state, [('new', {})], attempt=second)


async def test_removal_blocks_late_writes_without_recreating_item(state):
    store, _, scope, first = state
    await store.remove(scope)
    with pytest.raises(StaleIndexingRun):
        await replace(state, [('late', {})])
    with pytest.raises(StaleIndexingRun):
        await store.register(
            scope,
            indexing_dag_id=str(uuid4()),
            owner_type='chess_game',
            owner_id='one',
            uri='s3://fixture/one',
            run_params={},
        )
    assert (await store.get(scope))['state'] == 'removed'


async def test_registration_is_idempotent_and_does_not_merge_other_kbs(state):
    store, _, scope, first = state
    await store.register(
        scope,
        indexing_dag_id=first,
        owner_type='chess_game',
        owner_id='one',
        uri='s3://fixture/games/one.json',
        run_params={},
    )
    other = KbItemScope(scope.tenant_id, str(uuid4()), scope.source_id, scope.item_id)
    second = str(uuid4())
    await store.register(
        other,
        indexing_dag_id=second,
        owner_type='chess_game',
        owner_id='one',
        uri='s3://fixture/games/one.json',
        run_params={},
    )
    await store.remove(scope)
    assert (await store.get(other))['state'] == 'pending'


async def test_unknown_scope_does_not_gain_authority_from_a_write(state):
    _, _, scope, _ = state
    unknown = KbItemScope(
        scope.tenant_id, scope.index_id, 'unregistered-source', scope.item_id
    )
    with pytest.raises(StaleIndexingRun):
        await replace(state, [('forged', {})], scope=unknown)


async def test_legacy_status_callback_cannot_clear_owner_removal_marker(
    state, monkeypatch
):
    import psycopg

    from marie.storage.submission.storage import SubmissionStorage

    _, pool, _, _ = state
    await pool.execute('DROP TABLE IF EXISTS marie_scheduler.submission_documents')
    await pool.execute('''CREATE TABLE marie_scheduler.submission_documents (
      id uuid PRIMARY KEY,indexing_status text,indexing_error text,indexed_at timestamptz,updated_at timestamptz)''')
    document = uuid4()
    await pool.execute(
        "INSERT INTO marie_scheduler.submission_documents(id,indexing_status) VALUES($1,'removing')",
        document,
    )
    connection = psycopg.connect(
        host='127.0.0.1',
        port=os.environ['MARIE_KB_TEST_PORT'],
        user='postgres',
        dbname='postgres',
        autocommit=True,
    )
    storage = object.__new__(SubmissionStorage)
    monkeypatch.setattr(storage, '_ensure_started', lambda: None)
    monkeypatch.setattr(storage, '_get_connection', lambda: connection)
    monkeypatch.setattr(
        storage, '_qualified_table', lambda _: 'marie_scheduler.submission_documents'
    )
    monkeypatch.setattr(
        storage,
        '_execute_sql_gracefully',
        lambda sql, params, **kw: connection.execute(sql, params),
    )
    monkeypatch.setattr(storage, '_close_connection', lambda conn: None)
    try:
        storage.update_document_indexing_status(str(document), 'indexed')
        assert (
            await pool.fetchval(
                'SELECT indexing_status FROM marie_scheduler.submission_documents WHERE id=$1',
                document,
            )
            == 'removing'
        )
    finally:
        connection.close()


async def test_removal_waits_for_checked_write_then_wins_delivery(state):
    import asyncio

    store, _, scope, attempt = state
    entered = asyncio.Event()
    release = asyncio.Event()

    async def writer():
        async with store.write(scope, attempt, record_count=1):
            entered.set()
            await release.wait()

    write_task = asyncio.create_task(writer())
    await entered.wait()
    remove_task = asyncio.create_task(store.remove(scope))
    await asyncio.sleep(0.05)
    assert not remove_task.done()
    release.set()
    await asyncio.gather(write_task, remove_task)
    assert (await store.get(scope))['state'] == 'removed'


async def test_retry_cannot_change_attempt_inside_replacement_transaction(state):
    import asyncio

    store, _, scope, first = state
    entered = asyncio.Event()
    release = asyncio.Event()

    async def writer():
        async with store.write(scope, first, record_count=2):
            entered.set()
            await release.wait()

    task = asyncio.create_task(writer())
    await entered.wait()
    second = str(uuid4())
    retry = asyncio.create_task(
        store.reindex(scope, expected_indexing_dag_id=first, indexing_dag_id=second)
    )
    await asyncio.sleep(0.05)
    assert not retry.done()
    release.set()
    await asyncio.gather(task, retry)
    row = await store.get(scope)
    assert str(row['indexing_dag_id']) == second
    assert str(row['indexed_dag_id']) == first
    assert row['state'] == 'pending'
