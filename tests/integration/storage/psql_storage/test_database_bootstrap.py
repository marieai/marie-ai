from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from marie.excepts import BadConfigSource, RuntimeFailToStart
from marie.logging_core.logger import MarieLogger
from marie.scheduler.repository.async_job_repository import AsyncJobRepository
from marie.storage.database.postgres import PostgresqlMixin
from marie.storage.kv.psql import PostgreSQLKV


@pytest.fixture(scope='module', params=['baked', 'empty'])
def native_server(request: pytest.FixtureRequest):
    if os.environ.get('MARIE_TEST_NATIVE_DOCUMENTDB') != '1':
        pytest.skip('Set MARIE_TEST_NATIVE_DOCUMENTDB=1 to test the owned native stack')
    suffix = uuid4().hex[:16]
    sql_user = f'marie_smoke_sql_{suffix}'
    document_user = f'marie_smoke_doc_{suffix}'
    sql_password = uuid4().hex
    document_password = uuid4().hex
    image = os.environ.get(
        'MARIE_TEST_NATIVE_IMAGE',
        'marieai/postgres-documentdb:17-1.2.0',
    )
    volume = subprocess.check_output(
        [
            'docker',
            'volume',
            'create',
            '--label',
            'com.marieai.test=native-documentdb',
            f'marie_smoke_data_{suffix}',
        ],
        text=True,
    ).strip()
    try:
        container = subprocess.check_output(
            [
                'docker',
                'run',
                '--detach',
                '--pull',
                'never',
                '--name',
                f'marie_smoke_{suffix}',
                '--label',
                'com.marieai.test=native-documentdb',
                '--publish',
                '127.0.0.1::5432',
                '--publish',
                '127.0.0.1::27017',
                '--mount',
                f'type=volume,source={volume},target=/data'
                + (',volume-nocopy' if request.param == 'empty' else ''),
                '--env',
                f'POSTGRES_USER={sql_user}',
                '--env',
                f'POSTGRES_PASSWORD={sql_password}',
                '--env',
                'POSTGRES_DB=postgres',
                '--env',
                f'DOCUMENTDB_USERNAME={document_user}',
                '--env',
                f'DOCUMENTDB_PASSWORD={document_password}',
                '--env',
                'POSTGRESQL_PORT=5432',
                '--env',
                'DOCUMENTDB_PORT=27017',
                '--env',
                'ALLOW_EXTERNAL_CONNECTIONS=true',
                '--env',
                'DOCUMENTDB_USAGE_TELEMETRY=false',
                '--mount',
                f'type=bind,source={Path(__file__).resolve().parents[4] / "config/psql/init.sql"},target=/docker-entrypoint-initdb.d/init.sql,readonly',
                image,
            ],
            text=True,
        ).strip()
    except BaseException:
        subprocess.run(
            ['docker', 'volume', 'rm', volume], check=True, stdout=subprocess.DEVNULL
        )
        raise

    def wait_ready() -> None:
        deadline = time.monotonic() + 90
        while subprocess.run(
            [
                'docker',
                'exec',
                container,
                'bash',
                '-c',
                'if command -v marie-documentdb-healthcheck >/dev/null; then '
                'exec marie-documentdb-healthcheck; else exec documentdb-healthcheck; fi',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    'Owned native DocumentDB container did not become ready'
                )
            running = subprocess.check_output(
                ['docker', 'inspect', '--format', '{{.State.Running}}', container],
                text=True,
            ).strip()
            if running != 'true':
                raise RuntimeError(
                    'Owned native DocumentDB container exited during startup'
                )
            time.sleep(0.2)

    def config() -> dict:
        address = subprocess.check_output(
            ['docker', 'port', container, '5432/tcp'], text=True
        ).strip()
        host, port = address.rsplit(':', 1)
        return {
            'hostname': host,
            'port': int(port),
            'username': sql_user,
            'password': sql_password,
            'database': 'postgres',
            'max_connections': 2,
        }

    def mongo(script: str) -> None:
        setup = (
            'const connection = new Mongo("mongodb://" + '
            'encodeURIComponent(process.env.DOCUMENTDB_USERNAME) + ":" + '
            'encodeURIComponent(process.env.DOCUMENTDB_PASSWORD) + '
            '"@127.0.0.1:27017/?authMechanism=SCRAM-SHA-256");'
        )
        subprocess.run(
            [
                'docker',
                'exec',
                container,
                'mongosh',
                '--nodb',
                '--quiet',
                '--norc',
                '--eval',
                setup + script,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=30,
        )

    try:
        wait_ready()
        yield container, config, mongo, wait_ready
    finally:
        subprocess.run(
            ['docker', 'rm', '--force', '--volumes', container],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ['docker', 'volume', 'rm', volume],
            check=True,
            stdout=subprocess.DEVNULL,
        )


async def test_native_fresh_install_sql_wire_indexes_and_restart(
    native_server, monkeypatch
):
    container, config, mongo, wait_ready = native_server
    root = Path(__file__).resolve().parents[4]
    monkeypatch.setenv('MARIE_PSQL_DIR', str(root / 'config/psql'))
    monkeypatch.setenv('MARIE_SCHEMA_DIR', str(root / 'config/psql/schema'))
    params = config()
    dsn = {
        'host': params['hostname'],
        'port': params['port'],
        'user': params['username'],
        'password': params['password'],
        'dbname': 'postgres',
    }
    with psycopg.connect(**dsn, autocommit=True) as connection:
        assert connection.execute('SELECT current_database()').fetchone() == (
            'postgres',
        )
        assert connection.execute('SHOW cron.database_name').fetchone() == ('postgres',)
        extensions = {
            row[0] for row in connection.execute('SELECT extname FROM pg_extension')
        }
        assert {
            'documentdb',
            'documentdb_core',
            'documentdb_extended_rum',
            'pg_cron',
            'pg_stat_statements',
            'vector',
            'pgstattuple',
            'pg_repack',
        } <= extensions
        assert (
            connection.execute(
                "SELECT 1 FROM pg_database WHERE datname='marie'"
            ).fetchone()
            is None
        )
        assert connection.execute(
            'SHOW documentdb.alternate_index_handler_name'
        ).fetchone() == ('extended_rum',)
        assert connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname='pg_repack'"
        ).fetchone() == ('1.5.3',)
        assert (
            connection.execute('SELECT count(*) FROM pg_stat_statements').fetchone()[0]
            >= 0
        )
        assert connection.execute(
            "SELECT count(*) FROM cron.job WHERE jobname ~ '^documentdb_' "
            "AND nodeport <> current_setting('port')::integer"
        ).fetchone() == (0,)
        logical_database = f'marie_smoke_{uuid4().hex[:16]}'
        mongo(
            f'const d = connection.getDB("{logical_database}");'
            'd.documents.insertMany([{_id:1, amount:7}, {_id:2, amount:9}]);'
            'if(d.documents.createIndex({amount:1}) !== "amount_1") throw Error("index creation failed");'
            'if(d.documents.getIndexes().length !== 2) throw Error("index missing");'
            'if(d.documents.aggregate([{$group:{_id:null,total:{$sum:"$amount"}}}]).toArray()[0].total !== 16) throw Error("aggregation failed");'
        )
        assert connection.execute(
            sql.SQL('SELECT count(*) FROM documentdb_api.collection({}, {})').format(
                sql.Literal(logical_database), sql.Literal('documents')
            )
        ).fetchone() == (2,)
        mongo(
            'let rejected = false; try {'
            'new Mongo("mongodb://" + encodeURIComponent(process.env.DOCUMENTDB_USERNAME) + '
            '":wrong-password@127.0.0.1:27017/?authMechanism=SCRAM-SHA-256&serverSelectionTimeoutMS=2000")'
            '.getDB("admin").runCommand({ping:1});'
            '} catch (error) { rejected = true; }'
            'if (!rejected) throw Error("invalid password was accepted");'
        )
    repository = AsyncJobRepository(params)
    try:
        await repository.initialize()
        await repository.initialize_cron()
        await repository.create_tables('marie_scheduler')
        await repository.validate_durable_scheduler_schema('marie_scheduler')
    finally:
        await repository.close()
    subprocess.run(
        ['docker', 'restart', container], check=True, stdout=subprocess.DEVNULL
    )
    wait_ready()
    params = config()
    with psycopg.connect(
        host=params['hostname'],
        port=params['port'],
        user=params['username'],
        password=params['password'],
        dbname='postgres',
    ) as connection:
        assert connection.execute(
            "SELECT to_regclass('marie_scheduler.job') IS NOT NULL"
        ).fetchone() == (True,)
    mongo(
        f'const d = connection.getDB("{logical_database}");'
        'if(d.documents.countDocuments({}) !== 2) throw Error("documents missing after restart");'
        'if(d.documents.createIndex({other:1}) !== "other_1") throw Error("index build failed after restart");'
        'if(d.documents.getIndexes().length !== 3) throw Error("index missing after restart");'
    )


@pytest.fixture(scope='module')
def smoke_server(request: pytest.FixtureRequest):
    if os.environ.get('MARIE_TEST_DATABASE_BOOTSTRAP') != '1':
        pytest.skip(
            'Set MARIE_TEST_DATABASE_BOOTSTRAP=1 to create an isolated smoke container'
        )
    name = f'marie_smoke_{uuid4().hex}'
    container = subprocess.check_output(
        [
            'docker',
            'run',
            '--detach',
            '--pull',
            'never',
            '--name',
            name,
            '--label',
            'com.marieai.test=database-bootstrap',
            '--publish',
            '127.0.0.1::5432',
            '--tmpfs',
            '/var/lib/postgresql/data',
            '--env',
            'POSTGRES_HOST_AUTH_METHOD=trust',
            getattr(request, 'param', 'pgvector/pgvector:pg17'),
        ],
        text=True,
    ).strip()
    try:
        subprocess.run(
            [
                'docker',
                'exec',
                container,
                'sh',
                '-c',
                'for attempt in $(seq 1 200); do psql -h 127.0.0.1 -U postgres -d postgres -Atqc "SELECT 1" >/dev/null 2>&1 && exit 0; sleep 0.1; done; exit 1',
            ],
            check=True,
        )
        address = subprocess.check_output(
            ['docker', 'port', container, '5432/tcp'], text=True
        ).strip()
        yield f'postgresql://postgres@{address}/postgres'
    finally:
        subprocess.run(
            ['docker', 'rm', '--force', '--volumes', container],
            check=True,
            stdout=subprocess.DEVNULL,
        )


@pytest.fixture
def database(smoke_server):
    dsn = smoke_server
    suffix = uuid4().hex
    name = f'marie_smoke_{suffix} "quoted"'
    role = f'marie_smoke_role_{suffix}'
    password = uuid4().hex
    params = psycopg.conninfo.conninfo_to_dict(dsn)
    config = {
        'hostname': params['host'],
        'port': int(params['port']),
        'username': role,
        'password': password,
        'database': name,
        'schema': 'marie_scheduler',
        'default_table': 'kv_store_worker',
        'max_connections': 1,
        'pool_open_timeout_seconds': 0.5,
    }
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(
            sql.SQL('CREATE ROLE {} LOGIN CREATEDB PASSWORD {}').format(
                sql.Identifier(role), sql.Literal(password)
            )
        )
        try:
            yield config, admin
        finally:
            admin.execute(
                sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(
                    sql.Identifier(name)
                )
            )
            admin.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))


async def test_gateway_kv_creates_missing_database_and_restart_preserves_data(database):
    config, admin = database
    assert (
        admin.execute(
            'SELECT 1 FROM pg_database WHERE datname = %s', (config['database'],)
        ).fetchone()
        is None
    )

    storage = PostgreSQLKV(config)
    try:
        await storage.internal_kv_put(
            b'bootstrap', b'{"retained": true}', True, b'test'
        )
    finally:
        await storage.close()

    storage = PostgreSQLKV(config)
    try:
        assert await storage.internal_kv_get(b'bootstrap', b'test') == {
            'retained': True
        }
        assert admin.execute(
            'SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname = %s',
            (config['database'],),
        ).fetchone() == (config['username'],)
    finally:
        await storage.close()


async def test_existing_database_works_without_createdb_or_maintenance_access(
    database, monkeypatch
):
    config, admin = database
    admin.execute(
        sql.SQL('CREATE DATABASE {} OWNER {}').format(
            sql.Identifier(config['database']), sql.Identifier(config['username'])
        )
    )
    admin.execute(
        sql.SQL('ALTER ROLE {} NOCREATEDB').format(sql.Identifier(config['username']))
    )
    connect = psycopg.connect

    def target_only_connect(*args, **kwargs):
        if kwargs.get('dbname') == 'postgres':
            raise AssertionError(
                'An existing database must not require maintenance access'
            )
        return connect(*args, **kwargs)

    monkeypatch.setattr(psycopg, 'connect', target_only_connect)
    storage = PostgreSQLKV(config)
    try:
        await storage.internal_kv_put(b'existing', b'1', True, b'test')
        assert await storage.internal_kv_get(b'existing', b'test') == 1
    finally:
        await storage.close()


async def test_bootstrap_inherits_template1_customizations(database, smoke_server):
    config, admin = database
    marker = f'bootstrap_marker_{uuid4().hex}'
    template_dsn = psycopg.conninfo.make_conninfo(smoke_server, dbname='template1')
    with psycopg.connect(template_dsn) as template:
        template.execute(
            sql.SQL(
                "CREATE FUNCTION public.{}() RETURNS integer LANGUAGE SQL AS 'SELECT 7'"
            ).format(sql.Identifier(marker))
        )
    try:
        storage = PostgreSQLKV(config)
        try:
            target = psycopg.conninfo.make_conninfo(
                smoke_server, dbname=config['database']
            )
            with psycopg.connect(target) as connection:
                assert (
                    connection.execute(
                        'SELECT to_regprocedure(%s)', (f'public.{marker}()',)
                    ).fetchone()[0]
                    is not None
                )
                assert connection.execute(
                    sql.SQL('SELECT public.{}()').format(sql.Identifier(marker))
                ).fetchone() == (7,)
        finally:
            await storage.close()
    finally:
        with psycopg.connect(template_dsn) as template:
            template.execute(
                sql.SQL('DROP FUNCTION public.{}()').format(sql.Identifier(marker))
            )


def test_missing_database_reports_createdb_permission(database):
    config, admin = database
    admin.execute(
        sql.SQL('ALTER ROLE {} NOCREATEDB').format(sql.Identifier(config['username']))
    )

    with pytest.raises(BadConfigSource, match='CREATEDB'):
        PostgreSQLKV(config)

    assert (
        admin.execute(
            'SELECT 1 FROM pg_database WHERE datname = %s', (config['database'],)
        ).fetchone()
        is None
    )


def test_existing_database_connection_failure_does_not_recreate_it(database):
    config, admin = database
    admin.execute(
        sql.SQL('CREATE DATABASE {} OWNER {}').format(
            sql.Identifier(config['database']), sql.Identifier(config['username'])
        )
    )
    identity = admin.execute(
        'SELECT oid FROM pg_database WHERE datname = %s', (config['database'],)
    ).fetchone()
    admin.execute(
        sql.SQL('ALTER DATABASE {} ALLOW_CONNECTIONS false').format(
            sql.Identifier(config['database'])
        )
    )

    with pytest.raises(BadConfigSource, match='not currently accepting connections'):
        PostgreSQLKV(config)

    assert (
        admin.execute(
            'SELECT oid FROM pg_database WHERE datname = %s', (config['database'],)
        ).fetchone()
        == identity
    )


def test_concurrent_storage_startups_create_one_database(database):
    config, admin = database
    barrier = threading.Barrier(4)

    def start_storage(_: int) -> str:
        storage = PostgresqlMixin()
        storage.logger = MarieLogger('database_bootstrap_test')
        barrier.wait(timeout=10)
        try:
            storage._setup_storage(config, connection_only=True)
            with storage.postgreSQL_pool.connection() as connection:
                return connection.execute('SELECT current_database()').fetchone()[0]
        finally:
            pool = getattr(storage, 'postgreSQL_pool', None)
            if pool is not None:
                pool.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(start_storage, range(4))) == [config['database']] * 4
    assert admin.execute(
        'SELECT count(*) FROM pg_database WHERE datname = %s', (config['database'],)
    ).fetchone() == (1,)


async def test_scheduler_bootstrap_rejects_missing_cron_coordinator(
    database, smoke_server, monkeypatch
):
    config, _ = database
    config = {**config, 'username': 'postgres', 'password': ''}
    psql_dir = Path(
        os.environ.get(
            'MARIE_TEST_PSQL_DIR', Path(__file__).resolve().parents[4] / 'config/psql'
        )
    )
    monkeypatch.setenv('MARIE_PSQL_DIR', str(psql_dir))
    monkeypatch.setenv('MARIE_SCHEMA_DIR', str(psql_dir / 'schema'))
    storage = PostgreSQLKV(config)
    await storage.close()
    repository = AsyncJobRepository(config)
    try:
        await repository.initialize()
        with pytest.raises(
            RuntimeFailToStart,
            match='pg_cron is not preloaded',
        ):
            await repository.initialize_cron()
    finally:
        await repository.close()
    target = psycopg.conninfo.make_conninfo(smoke_server, dbname=config['database'])
    with psycopg.connect(target) as connection:
        assert connection.execute("SELECT to_regclass('cron.job')").fetchone() == (
            None,
        )
        assert connection.execute(
            "SELECT to_regclass('marie_scheduler.job')"
        ).fetchone() == (None,)
