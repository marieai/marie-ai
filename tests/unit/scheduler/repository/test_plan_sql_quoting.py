import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest

from marie.scheduler.repository import plans

JOB_ID = '00000000-0000-0000-0000-000000000001'
SIBLING_ID = '00000000-0000-0000-0000-000000000002'
ATTEMPT_ID = '00000000-0000-0000-0000-000000000003'
TRANSITIONS = [
    ('cancel_jobs', 'active', 'cancelled'),
    ('resume_jobs', 'cancelled', 'created'),
    ('mark_as_active_jobs', 'created', 'active'),
    ('complete_jobs', 'active', 'completed'),
    ('complete_jobs_by_id', 'created', 'completed'),
    ('complete_jobs_by_attempt', 'active', 'completed'),
    ('fail_jobs_by_id', 'active', 'failed'),
    ('fail_jobs_by_attempt', 'active', 'failed'),
]


@pytest.fixture
def database() -> Iterator[tuple[psycopg.Connection, str]]:
    port = os.environ.get('MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT')
    if not port:
        pytest.skip(
            'Set MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT to an owned test database'
        )
    schema = 'sql_quoting_' + uuid4().hex
    with psycopg.connect(
        host='127.0.0.1',
        port=int(port),
        user='postgres',
        dbname='postgres',
        autocommit=True,
    ) as connection:
        with connection.transaction(force_rollback=True):
            connection.execute(
                f'''
                CREATE SCHEMA {schema};
                CREATE TYPE {schema}.job_state AS ENUM
                    ('created', 'retry', 'active', 'completed', 'cancelled', 'failed');
                CREATE TABLE {schema}.job (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), dag_id uuid,
                    name text, state {schema}.job_state, started_on timestamptz,
                    completed_on timestamptz, lease_owner text, lease_expires_at timestamptz,
                    run_owner text, run_attempt_id uuid, run_lease_expires_at timestamptz,
                    retry_count int DEFAULT 0, retry_limit int DEFAULT 0,
                    retry_delay int DEFAULT 1, retry_backoff boolean DEFAULT false,
                    start_after timestamptz DEFAULT now(), keep_until timestamptz,
                    expire_in interval, priority int, data jsonb, output jsonb, dead_letter text
                );
                CREATE TABLE {schema}.dag (
                    id uuid PRIMARY KEY, name text, state {schema}.job_state,
                    started_on timestamptz, serialized_dag jsonb
                );
                CREATE TABLE {schema}.version (version text PRIMARY KEY);
                CREATE FUNCTION {schema}.exponential_backoff(int, int)
                    RETURNS timestamptz LANGUAGE sql AS 'SELECT now()';
                CREATE FUNCTION {schema}.create_queue(text, json)
                    RETURNS jsonb LANGUAGE sql AS
                    'SELECT jsonb_build_object(''name'', $1, ''options'', $2)';
                CREATE FUNCTION {schema}.delete_queue(text)
                    RETURNS text LANGUAGE sql AS 'SELECT $1';
            '''
            )
            yield connection, schema


def seed_jobs(
    connection: psycopg.Connection, schema: str, name: str, state: str
) -> None:
    connection.execute(
        f'''INSERT INTO {schema}.job (id, dag_id, name, state, run_owner, run_attempt_id)
            VALUES (%s::uuid, %s::uuid, %s, %s::{schema}.job_state, %s, %s::uuid),
                   (%s::uuid, %s::uuid, %s, %s::{schema}.job_state, %s, %s::uuid)''',
        (
            JOB_ID,
            JOB_ID,
            name,
            state,
            'owner',
            ATTEMPT_ID,
            SIBLING_ID,
            SIBLING_ID,
            'unrelated',
            state,
            'owner',
            ATTEMPT_ID,
        ),
    )


def transition(operation: str, schema: str, name: str, ids: list[str]) -> str:
    kwargs = {}
    if operation.startswith(('complete_', 'fail_')):
        kwargs['output'] = {'message': "quoted ' output"}
    if operation.endswith('_by_attempt'):
        kwargs.update(run_owner='owner', run_attempt_id=ATTEMPT_ID)
    return getattr(plans, operation)(schema, name, ids, **kwargs)


@pytest.mark.parametrize('name', ["O'Reilly", r"q\'; SELECT 1; --", '文档'])
def test_create_queue_keeps_name_as_data(
    database: tuple[psycopg.Connection, str], name: str
) -> None:
    connection, schema = database
    result = connection.execute(plans.create_queue(schema, name, {})).fetchone()[0]
    assert result == {'name': name, 'options': {'retry_limit': 2}}


@pytest.mark.parametrize('operation,initial,expected', TRANSITIONS)
@pytest.mark.parametrize('name', ["O'Reilly", "queue' OR TRUE --"])
def test_transition_keeps_queue_name_as_data(
    database: tuple[psycopg.Connection, str],
    operation: str,
    initial: str,
    expected: str,
    name: str,
) -> None:
    connection, schema = database
    seed_jobs(connection, schema, name, initial)
    connection.execute(transition(operation, schema, name, [JOB_ID])).fetchall()
    states = connection.execute(
        f'SELECT id::text, state::text FROM {schema}.job ORDER BY id'
    ).fetchall()
    assert states == [(JOB_ID, expected), (SIBLING_ID, initial)]


@pytest.mark.parametrize('operation,initial,expected', TRANSITIONS)
def test_transition_rejects_sql_in_an_id_as_invalid_uuid(
    database: tuple[psycopg.Connection, str],
    operation: str,
    initial: str,
    expected: str,
) -> None:
    connection, schema = database
    seed_jobs(connection, schema, 'queue', initial)
    malicious_id = JOB_ID + "']::uuid[])) OR TRUE --"
    with connection.transaction(force_rollback=True):
        with pytest.raises(psycopg.errors.InvalidTextRepresentation):
            connection.execute(transition(operation, schema, 'queue', [malicious_id]))
    states = connection.execute(
        f'SELECT state::text FROM {schema}.job ORDER BY id'
    ).fetchall()
    assert states == [(initial,), (initial,)]


@pytest.mark.parametrize('operation,initial,expected', TRANSITIONS)
def test_empty_ids_do_not_mutate_jobs(
    database: tuple[psycopg.Connection, str],
    operation: str,
    initial: str,
    expected: str,
) -> None:
    connection, schema = database
    seed_jobs(connection, schema, 'queue', initial)
    connection.execute(transition(operation, schema, 'queue', [])).fetchall()
    states = connection.execute(
        f'SELECT state::text FROM {schema}.job ORDER BY id'
    ).fetchall()
    assert states == [(initial,), (initial,)]


def test_dag_activation_rejects_sql_in_ids(
    database: tuple[psycopg.Connection, str]
) -> None:
    connection, schema = database
    connection.execute(
        f"INSERT INTO {schema}.dag VALUES (%s::uuid, 'dag', 'created', NULL, '{{}}')",
        (JOB_ID,),
    )
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        connection.execute(
            plans.mark_as_active_dags(schema, [JOB_ID + "']::uuid[])) OR TRUE --"])
        )


@pytest.mark.parametrize('operation', ['load_dag', 'cancel_pending_jobs_for_dag'])
def test_dag_id_remains_a_uuid_value(
    database: tuple[psycopg.Connection, str],
    operation: str,
) -> None:
    connection, schema = database
    dag_id = JOB_ID + "' OR TRUE --"
    kwargs = {'output': {}} if operation == 'cancel_pending_jobs_for_dag' else {}
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        connection.execute(getattr(plans, operation)(schema, dag_id, **kwargs))


def test_delete_queue_keeps_name_as_data(
    database: tuple[psycopg.Connection, str]
) -> None:
    connection, schema = database
    assert connection.execute(plans.delete_queue(schema, "O'Reilly")).fetchone() == (
        "O'Reilly",
    )


def test_insert_version_keeps_value_as_data(
    database: tuple[psycopg.Connection, str]
) -> None:
    connection, schema = database
    connection.execute(plans.insert_version(schema, "95'); SELECT 1; --"))
    assert connection.execute(f'SELECT version FROM {schema}.version').fetchall() == [
        ("95'); SELECT 1; --",)
    ]


@pytest.mark.parametrize(
    'operation', ['fail_jobs_by_id', 'fail_jobs_by_attempt', 'fail_jobs_by_timeout']
)
@pytest.mark.parametrize('retry_delay', [0, 2, 30])
@pytest.mark.parametrize('retries_remaining', [True, False])
def test_fixed_retry_delay_is_in_seconds(
    database: tuple[psycopg.Connection, str],
    operation: str,
    retry_delay: int,
    retries_remaining: bool,
) -> None:
    connection, schema = database
    seed_jobs(connection, schema, 'queue', 'active')
    connection.execute(
        f'''UPDATE {schema}.job
            SET retry_count = %s, retry_limit = 2, retry_delay = %s,
                retry_backoff = false, start_after = now() - interval '1 minute',
                started_on = now() - interval '2 minutes', expire_in = interval '1 minute'
            WHERE id = %s::uuid''',
        (1 if retries_remaining else 2, retry_delay, JOB_ID),
    )
    query = (
        plans.fail_jobs_by_timeout(schema)
        if operation == 'fail_jobs_by_timeout'
        else transition(operation, schema, 'queue', [JOB_ID])
    )

    result = connection.execute(query).fetchone()
    delay = connection.execute(
        f'SELECT extract(epoch FROM start_after - now()) FROM {schema}.job WHERE id = %s::uuid',
        (JOB_ID,),
    ).fetchone()[0]

    assert result == (1, 'retry' if retries_remaining else 'failed')
    assert delay == (retry_delay if retries_remaining else -60)
    assert connection.execute(
        f'SELECT state::text FROM {schema}.job WHERE id = %s::uuid',
        (SIBLING_ID,),
    ).fetchone() == ('active',)
