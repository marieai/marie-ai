import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from marie.kb.code.store import CodeIndexStore
from marie.kb.item_state import StaleIndexingRun
from marie.sensors.state.psql_storage import PostgreSQLSensorStorage
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool
from tests.unit.kb.code.test_capture import git, repo
from tests.unit.kb.code.test_zoekt import zoekt

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable PostgreSQL'
)


@pytest.fixture
async def code_store(zoekt, repo):
    pool = AsyncPostgresConnectionPool()
    await pool.initialize(
        dict(
            hostname='127.0.0.1',
            port=int(os.environ['MARIE_KB_TEST_PORT']),
            username='postgres',
            password='',
            database='kb_f06_code',
            max_connections=10,
        ),
        autocommit=True,
    )
    await pool.execute(
        'CREATE SCHEMA IF NOT EXISTS marie_studio; CREATE SCHEMA IF NOT EXISTS marie_scheduler'
    )
    await pool.execute(
        '''DROP TABLE IF EXISTS marie_studio.kb_source,marie_studio.kb_index,marie_studio.tenant,marie_studio.git_repositories CASCADE;
        CREATE TABLE marie_studio.tenant(id uuid primary key,organization_id uuid,status text);
        CREATE TABLE marie_studio.kb_index(id uuid primary key,tenant_id uuid,workspace_id uuid,metadata jsonb);
        CREATE TABLE marie_studio.git_repositories(id uuid primary key,path text,name text,is_enabled boolean,git_enabled boolean,organization_id uuid,visibility text);
        CREATE TABLE marie_studio.kb_source(id uuid primary key,index_id uuid,name text,source_type text,status text,metadata jsonb,dataset_inputs jsonb,updated_at timestamptz default now());'''
    )
    schema = Path(__file__).parents[4] / 'config/psql/schema'
    await pool.execute(
        'DROP TABLE IF EXISTS marie_scheduler.job CASCADE; DROP TABLE IF EXISTS marie_scheduler.dag CASCADE'
    )
    # Canonical scheduler definitions supply existing state/timestamps/lease fields.
    await pool.execute(
        (schema / '007_dag.sql').read_text().replace('{schema}', 'marie_scheduler')
    )
    for filename in ['002_enums.sql', '005_job.sql']:
        await pool.execute(
            (schema / filename).read_text().replace('{schema}', 'marie_scheduler')
        )
    await pool.execute(
        'CREATE TABLE IF NOT EXISTS marie_scheduler.sensor_run_key(sensor_id uuid,run_key text,job_id uuid)'
    )
    scope = {
        key: str(uuid4())
        for key in [
            'tenant_id',
            'index_id',
            'source_id',
            'organization_id',
            'workspace_id',
        ]
    }
    repository_id = str(uuid4())
    await pool.execute(
        "INSERT INTO marie_studio.tenant VALUES($1,$2,'active')",
        UUID(scope['tenant_id']),
        UUID(scope['organization_id']),
    )
    await pool.execute(
        "INSERT INTO marie_studio.kb_index VALUES($1,$2,$3,'{}')",
        UUID(scope['index_id']),
        UUID(scope['tenant_id']),
        UUID(scope['workspace_id']),
    )
    await pool.execute(
        "INSERT INTO marie_studio.git_repositories VALUES($1,$2,'fixture',true,true,$3,'organization')",
        UUID(repository_id),
        str(repo),
        UUID(scope['organization_id']),
    )
    await pool.execute(
        "INSERT INTO marie_studio.kb_source(id,index_id,name,source_type,status,metadata,dataset_inputs) VALUES($1,$2,'fixture','repository','ready','{}',$3::jsonb)",
        UUID(scope['source_id']),
        UUID(scope['index_id']),
        json.dumps({'repositoryId': repository_id, 'branch': 'main', 'fetch': False}),
    )
    store = CodeIndexStore(pool, zoekt, [str(repo.parent)])

    async def execution():
        params = {
            'job_id': str(uuid4()),
            'run_attempt_id': str(uuid4()),
            'run_owner': 'fixture',
        }
        dag = str(uuid4())
        await pool.execute(
            "INSERT INTO marie_scheduler.dag(id,state,name) VALUES($1,'active','kb_repository_refresh')",
            UUID(dag),
        )
        await pool.execute(
            "INSERT INTO marie_scheduler.job(id,dag_id,state,data,run_attempt_id,run_owner,run_lease_expires_at,name) VALUES($1,$2,'active',$3::jsonb,$4,'fixture',clock_timestamp()+interval '5 minutes','kb_repository_refresh')",
            UUID(params['job_id']),
            UUID(dag),
            json.dumps({'metadata': {**scope, 'planner': 'kb_repository_refresh'}}),
            UUID(params['run_attempt_id']),
        )
        return params

    yield store, pool, scope, execution, repo
    await pool.close()


async def test_real_capture_search_refresh_and_old_locator(code_store):
    store, pool, scope, execution, repo = code_store
    initial = await store.refresh(await execution())
    assert initial['files'] == 1
    results = await store.search(scope, 'alpha', 'literal')
    row = results['results'][0]
    assert row['path'] == 'same.py'
    opened = await store.open(scope, row['citation'])
    assert opened['outcome'] == 'ready' and 'alpha' in opened['passage']
    skipped = await store.refresh(await execution())
    assert skipped['skipped'] and skipped['handle'] == initial['handle']
    (repo / 'same.py').write_text('beta = 2\n')
    git(repo, 'add', 'same.py')
    git(repo, 'commit', '-m', 'changed')
    next_result = await store.refresh(await execution())
    assert next_result['handle'] != initial['handle']
    assert not (await store.search(scope, 'alpha', 'literal'))['results']
    assert (await store.open(scope, row['citation']))['outcome'] == 'changed'
    assert not store.zoekt.directory(initial['handle']).exists()


async def test_failed_build_revocation_and_stale_execution_do_not_replace(
    code_store, monkeypatch
):
    store, pool, scope, execution, repo = code_store
    await store.refresh(await execution())
    (repo / 'same.py').write_text('beta = 2\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'changed')
    old = await execution()

    async def fail(*args):
        raise RuntimeError('partial fixture')

    monkeypatch.setattr(store.zoekt, 'build', fail)
    with pytest.raises(RuntimeError, match='partial'):
        await store.refresh(old)
    assert (await store.search(scope, 'alpha', 'literal'))['results']
    await pool.execute(
        "UPDATE marie_scheduler.job SET run_owner='other' WHERE id=$1",
        UUID(old['job_id']),
    )
    with pytest.raises(StaleIndexingRun):
        await store.refresh(old)
    await pool.execute("UPDATE marie_studio.kb_source SET status='disabled'")
    with pytest.raises(PermissionError):
        await store.search(scope, 'alpha', 'literal')


async def test_two_repositories_same_paths_and_branch_are_separate(
    code_store, tmp_path
):
    store, pool, scope, execution, repo = code_store
    await store.refresh(await execution())
    other = tmp_path / 'second'
    other.mkdir()
    git(other, 'init', '-b', 'main')
    git(other, 'config', 'user.name', 'Fixture')
    git(other, 'config', 'user.email', 'fixture@example.invalid')
    (other / 'same.py').write_text('different = 3\n')
    git(other, 'add', '.')
    git(other, 'commit', '-m', 'second')
    original_id = scope['source_id']
    other_id = uuid4()
    repo_id = uuid4()
    await pool.execute(
        "INSERT INTO marie_studio.git_repositories VALUES($1,$2,'other',true,true,$3,'organization')",
        repo_id,
        str(other),
        UUID(scope['organization_id']),
    )
    await pool.execute(
        "INSERT INTO marie_studio.kb_source(id,index_id,name,source_type,status,metadata,dataset_inputs) VALUES($1,$2,'other','repository','ready','{}',$3::jsonb)",
        other_id,
        UUID(scope['index_id']),
        json.dumps({'repositoryId': str(repo_id), 'branch': 'main', 'fetch': False}),
    )
    scope['source_id'] = str(other_id)
    await store.refresh(await execution())
    assert not (await store.search(scope, 'alpha', 'literal'))['results']
    row = (await store.search(scope, 'different', 'literal'))['results'][0]
    assert row['path'] == 'same.py'
    scope['source_id'] = original_id
    with pytest.raises(PermissionError):
        await store.open(scope, row['citation'])
    assert (await store.search(scope, 'alpha', 'literal'))['results']
    await store.remove(scope)
    scope['source_id'] = str(other_id)
    assert (await store.search(scope, 'different', 'literal'))['results']


async def test_missing_index_and_changed_settings_rebuild_at_same_commit(code_store):
    store, pool, scope, execution, repo = code_store
    initial = await store.refresh(await execution())
    await store.zoekt.remove(initial['handle'])
    second = await store.refresh(await execution())
    assert not second['skipped'] and second['handle'] != initial['handle']
    await pool.execute(
        "UPDATE marie_studio.kb_source SET dataset_inputs=dataset_inputs || '{\"include\":[\"nothing/*\"]}'::jsonb"
    )
    third = await store.refresh(await execution())
    assert (
        third['commit'] == second['commit']
        and not third['skipped']
        and third['files'] == 0
    )
    assert not (await store.search(scope, 'alpha', 'literal'))['results']


async def test_moving_head_completes_actual_commit_and_exposes_later_input(
    code_store, monkeypatch
):
    store, pool, scope, execution, repo = code_store
    build = store.zoekt.build

    async def advance(*args):
        result = await build(*args)
        (repo / 'same.py').write_text('later = 7\n')
        git(repo, 'add', '.')
        git(repo, 'commit', '-m', 'later')
        return result

    monkeypatch.setattr(store.zoekt, 'build', advance)
    result = await store.refresh(await execution())
    assert result['pendingCommit'] != result['commit']
    assert (await store.search(scope, 'alpha', 'literal'))['results']
    monkeypatch.setattr(store.zoekt, 'build', build)
    next_result = await store.refresh(await execution())
    assert (
        next_result['commit'] == result['pendingCommit']
        and 'pendingCommit' not in next_result
    )
    assert (await store.search(scope, 'later', 'literal'))['results']


async def test_lease_expiring_during_real_build_keeps_previous_result(
    code_store, monkeypatch
):
    store, pool, scope, execution, repo = code_store
    await store.refresh(await execution())
    (repo / 'same.py').write_text('later = 7\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'later')
    params = await execution()
    build = store.zoekt.build

    async def expire(*args):
        result = await build(*args)
        await pool.execute(
            "UPDATE marie_scheduler.job SET run_lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
            UUID(params['job_id']),
        )
        return result

    monkeypatch.setattr(store.zoekt, 'build', expire)
    with pytest.raises(StaleIndexingRun):
        await store.refresh(params)
    assert (await store.search(scope, 'alpha', 'literal'))['results']


async def test_replacement_waits_for_reader_and_retires_only_its_own_build(
    code_store, monkeypatch
):
    import asyncio

    store, pool, scope, execution, repo = code_store
    first = await store.refresh(await execution())
    original = store.zoekt.search
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(store.zoekt, 'search', hold)
    read = asyncio.create_task(store.search(scope, 'alpha', 'literal'))
    await entered.wait()
    (repo / 'same.py').write_text('later = 7\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'later')
    write = asyncio.create_task(store.refresh(await execution()))
    await asyncio.sleep(0.15)
    assert store.zoekt.directory(first['handle']).exists()
    release.set()
    assert (await read)['results']
    await write
    assert not store.zoekt.directory(first['handle']).exists()


async def test_source_scope_failure_and_stale_dag_never_select_or_retire(code_store):
    store, pool, scope, execution, repo = code_store
    older = await execution()
    current = await store.refresh(await execution())
    with pytest.raises(StaleIndexingRun):
        await store.refresh(older)
    assert store.zoekt.directory(current['handle']).exists()
    with pytest.raises(PermissionError):
        await store.search(
            {**scope, 'organization_id': str(uuid4())}, 'alpha', 'literal'
        )
    await pool.execute("UPDATE marie_studio.git_repositories SET is_enabled=false")
    with pytest.raises(PermissionError):
        await store.search(scope, 'alpha', 'literal')


@pytest.mark.skipif(
    not os.environ.get('MARIE_REAL_CODE_REPOS'),
    reason='Explicit real repository qualification',
)
async def test_two_marie_repositories_qualification(code_store):
    store, pool, scope, execution, _ = code_store
    results = []
    paths = os.environ['MARIE_REAL_CODE_REPOS'].split(':')
    for i, value in enumerate(paths):
        path = Path(value).resolve()
        store.roots.append(str(path.parent))
        repo_id = uuid4()
        source_id = uuid4()
        branch = git(path, 'symbolic-ref', '--short', 'HEAD')
        includes = (
            ['marie/*.py', 'packages/*/src/*.py']
            if i == 0
            else [
                'packages/api/src/*.ts',
                'packages/core/src/*.ts',
                'packages/frontend/ui/src/*.tsx',
                'packages/frontend/ui/src/*.ts',
            ]
        )
        await pool.execute(
            "INSERT INTO marie_studio.git_repositories VALUES($1,$2,$3,true,true,$4,'organization')",
            repo_id,
            str(path),
            path.name,
            UUID(scope['organization_id']),
        )
        await pool.execute(
            "INSERT INTO marie_studio.kb_source(id,index_id,name,source_type,status,metadata,dataset_inputs) VALUES($1,$2,$3,'repository','pending','{}',$4::jsonb)",
            source_id,
            UUID(scope['index_id']),
            path.name,
            json.dumps(
                {
                    'repositoryId': str(repo_id),
                    'branch': branch,
                    'include': includes,
                    'exclude': ['*/generated/*'],
                    'fetch': False,
                }
            ),
        )
        scope['source_id'] = str(source_id)
        first = await store.refresh(await execution())
        second = await store.refresh(await execution())
        assert (
            first['files'] > 100
            and second['skipped']
            and first['handle'] == second['handle']
        )
        hits = await store.search(scope, 'def ' if i == 0 else 'export ', 'literal', 10)
        assert hits['results']
        assert (await store.open(scope, hits['results'][0]['citation']))[
            'outcome'
        ] == 'ready'
        results.append(
            {
                'repository': path.name,
                'sourceId': str(source_id),
                'scope': dict(scope),
                'initial': first,
                'unchanged': second,
                'searchResults': len(hits['results']),
                'truncated': hits['truncated'],
            }
        )
    report = Path(os.environ['MARIE_REAL_CODE_REPORT'])
    report.write_text(json.dumps(results, indent=2))


async def test_native_schedule_requires_current_source_link_and_current_repository(
    code_store,
):
    from marie.sensors.state.psql_storage import PostgreSQLSensorStorage

    store, pool, scope, execution, repo = code_store
    trigger = uuid4()
    await pool.execute(
        'ALTER TABLE marie_studio.kb_source ADD COLUMN trigger_config_id uuid; ALTER TABLE marie_studio.kb_source ADD COLUMN sync_mode text'
    )
    await pool.execute(
        "UPDATE marie_studio.kb_source SET trigger_config_id=$1,sync_mode='scheduled'",
        trigger,
    )
    sensor = {
        'external_id': str(trigger),
        'config': {
            'kbSourceId': scope['source_id'],
            'kbIndexId': scope['index_id'],
            'run_config': scope,
        },
    }
    storage = PostgreSQLSensorStorage(pool)
    assert await storage.repository_schedule_active(sensor)
    await pool.execute("UPDATE marie_studio.kb_source SET trigger_config_id=NULL")
    assert not await storage.repository_schedule_active(sensor)


async def test_pending_input_reuses_schedule_admission_without_consuming_future_cron(
    code_store,
):
    from marie.sensors.context import SensorEvaluationContext
    from marie.sensors.definitions.schedule_sensor import ScheduleSensor
    from marie.sensors.types import SensorType

    store, pool, scope, execution, repo = code_store
    await store.refresh(await execution())
    await pool.execute(
        "UPDATE marie_studio.kb_source SET metadata=jsonb_set(metadata,'{codeIndex,pendingCommit}',to_jsonb($1::text))",
        'd' * 40,
    )
    sensor_id = str(uuid4())
    sensor = ScheduleSensor(
        {
            'id': sensor_id,
            'name': 'repository daily',
            'target_job_name': 'kb_repository_refresh',
            'config': {'cron': '0 2 * * *', 'timezone': 'UTC', 'run_config': scope},
        }
    )
    cursor = '2027-01-01T00:00:00+00:00'
    context = SensorEvaluationContext(
        sensor_id,
        'repository daily',
        SensorType.SCHEDULE,
        cursor=cursor,
        resources={'postgres_pool': PostgreSQLSensorStorage(pool)},
    )
    result = await sensor.evaluate(context)
    assert len(result.run_requests) == 1 and result.cursor == cursor
    assert result.run_requests[0].job_name == 'kb_repository_refresh'
    assert (
        result.run_requests[0].run_key
        == (await sensor.evaluate(context)).run_requests[0].run_key
    )
    await pool.execute(
        'INSERT INTO marie_scheduler.sensor_run_key VALUES($1,$2,$3)',
        UUID(sensor_id),
        result.run_requests[0].run_key,
        uuid4(),
    )
    assert not (await sensor.evaluate(context)).run_requests
    context.cursor = '2020-01-01T00:00:00+00:00'
    due = await sensor.evaluate(context)
    assert (
        due.run_requests
        and due.run_requests[0].run_key != result.run_requests[0].run_key
    )
    context.cursor = cursor
    await pool.execute(
        "UPDATE marie_studio.kb_source SET metadata=metadata #- '{codeIndex,pendingCommit}'"
    )
    assert not (await sensor.evaluate(context)).run_requests


async def test_removal_does_not_race_an_inflight_capture_or_orphan_its_shards(
    code_store, monkeypatch
):
    import asyncio

    store, pool, scope, execution, repo = code_store
    await store.refresh(await execution())
    (repo / 'same.py').write_text('later = 9\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'later')
    entered, release = asyncio.Event(), asyncio.Event()
    build = store.zoekt.build

    async def blocked(*args):
        entered.set()
        await release.wait()
        return await build(*args)

    monkeypatch.setattr(store.zoekt, 'build', blocked)
    running = asyncio.create_task(store.refresh(await execution()))
    await entered.wait()
    await pool.execute("UPDATE marie_studio.kb_source SET status='disabled'")
    try:
        with pytest.raises(RuntimeError, match='refresh'):
            await store.remove(scope)
    finally:
        release.set()
    with pytest.raises(PermissionError):
        await running
    await store.remove(scope)
    assert not list(store.zoekt.root.glob('*/manifest.json'))


async def test_retirement_failure_reports_cleanup_without_failing_a_complete_index(
    code_store, monkeypatch
):
    store, pool, scope, execution, repo = code_store
    first = await store.refresh(await execution())
    (repo / 'same.py').write_text('later = 9\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'later')
    remove = store.zoekt.remove

    async def fail(handle):
        if handle == first['handle']:
            raise OSError('fixture retirement failure')
        await remove(handle)

    monkeypatch.setattr(store.zoekt, 'remove', fail)
    result = await store.refresh(await execution())
    assert result['cleanupError']
    assert (await store.search(scope, 'later', 'literal'))['results']
    monkeypatch.setattr(store.zoekt, 'remove', remove)
    assert 'cleanupError' not in await store.refresh(await execution())
    assert not store.zoekt.directory(first['handle']).exists()


async def test_corrupt_capture_reports_unavailable_and_rebuilds_without_using_old_checkout(
    code_store,
):
    store, pool, scope, execution, repo = code_store
    current = await store.refresh(await execution())
    row = (await store.search(scope, 'alpha', 'literal'))['results'][0]
    (store.zoekt.directory(current['handle']) / 'manifest.json').write_text('{broken')
    assert (await store.open(scope, row['citation']))['outcome'] == 'unavailable'
    next_result = await store.refresh(await execution())
    assert not next_result['skipped'] and next_result['handle'] != current['handle']
    assert not store.zoekt.directory(current['handle']).exists()


async def test_force_pushed_branch_uses_complete_tree_without_assuming_ancestry(
    code_store,
):
    store, pool, scope, execution, repo = code_store
    first = await store.refresh(await execution())
    git(repo, 'checkout', '--orphan', 'rewritten')
    (repo / 'same.py').unlink()
    (repo / 'replacement.py').write_text('replacement = 10\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-m', 'new root')
    git(repo, 'branch', '-f', 'main', 'HEAD')
    next_result = await store.refresh(await execution())
    assert next_result['commit'] != first['commit'] and next_result['deleted'] == 1
    assert not (await store.search(scope, 'alpha', 'literal'))['results']
    assert (await store.search(scope, 'replacement', 'literal'))['results'][0][
        'path'
    ] == 'replacement.py'


async def test_admitted_dirty_input_rebuilds_at_the_same_commit(code_store):
    store, pool, scope, execution, repo = code_store
    await pool.execute(
        "UPDATE marie_studio.kb_source SET dataset_inputs=dataset_inputs || '{\"dirty\":true,\"fetch\":false}'::jsonb"
    )
    first = await store.refresh(await execution())
    (repo / 'same.py').write_text('dirty = 11\n')
    second = await store.refresh(await execution())
    assert (
        first['commit'] == second['commit']
        and first['handle'] != second['handle']
        and not second['skipped']
    )
    assert (await store.search(scope, 'dirty', 'literal'))['results']


async def test_source_scope_edit_during_search_withholds_old_binding_results(
    code_store, monkeypatch
):
    store, pool, scope, execution, repo = code_store
    await store.refresh(await execution())
    search = store.zoekt.search

    async def changed(*args):
        result = await search(*args)
        await pool.execute(
            "UPDATE marie_studio.kb_source SET dataset_inputs=dataset_inputs || '{\"exclude\":[\"*.py\"]}'::jsonb"
        )
        return result

    monkeypatch.setattr(store.zoekt, 'search', changed)
    with pytest.raises(PermissionError, match='scope changed'):
        await store.search(scope, 'alpha', 'literal')


async def test_later_head_fetch_failure_cannot_discard_a_complete_build(
    code_store, monkeypatch
):
    import subprocess

    from marie.kb.code import store as module

    store, pool, scope, execution, repo = code_store

    def offline(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ['git', 'fetch'])

    monkeypatch.setattr(module, 'branch_commit', offline)
    result = await store.refresh(await execution())
    assert (
        result['files'] == 1
        and (await store.search(scope, 'alpha', 'literal'))['results']
    )


async def test_real_indexer_sees_only_the_captured_eligible_files(code_store):
    store, pool, scope, execution, repo = code_store
    for name in ['.sourcegraph/ignore', 'vendor/.svn/entries', 'vendor/.hg/hgrc']:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('*\n')
    git(repo, 'add', '-f', '.')
    git(repo, 'commit', '-m', 'indexer controls')
    result = await store.refresh(await execution())
    assert result['files'] == 1 and result['omitted']['indexer_scope'] == 3
    assert (await store.search(scope, 'alpha', 'literal'))['results']
