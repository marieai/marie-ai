from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

MIGRATION = Path(__file__).resolve().parents[3] / 'config/psql/schema/093_runtime_registry.sql'


def test_registry_ddl_contains_no_application_migration_logic():
    ddl = MIGRATION.read_text()
    assert 'studio' not in ddl.lower()
    assert 'SET SCHEMA' not in ddl
    bundled = MIGRATION.parents[3] / 'deploy/helm/charts/marie/charts/server/files/psql/schema/093_runtime_registry.sql'
    assert bundled.read_text() == ddl


@pytest.fixture
def database():
    dsn = os.environ.get('TEST_RUNTIME_DSN')
    if not dsn:
        pytest.skip('TEST_RUNTIME_DSN must point to a disposable PostgreSQL server')
    name = 'runtime_registry_test_' + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
        info = psycopg.conninfo.conninfo_to_dict(dsn)
        info['dbname'] = name
        try:
            with psycopg.connect(**info, autocommit=True) as connection:
                yield connection
        finally:
            admin.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))


def install(connection):
    assert MIGRATION.is_file(), 'The standalone runtime registry migration is missing'
    connection.execute(MIGRATION.read_text().replace('{schema}', 'marie_scheduler'))


def test_runtime_registry_installs_independently_and_is_repeatable(database):
    install(database)
    install(database)
    namespaces = database.execute(
        "SELECT nspname FROM pg_namespace WHERE nspname !~ '^pg_' "
        "AND nspname NOT IN ('public', 'information_schema')"
    ).fetchall()
    assert namespaces == [('marie_runtime',)]
    tables = database.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='marie_runtime' ORDER BY tablename"
    ).fetchall()
    assert tables == [('fabric_groups',), ('gateway_definitions',), ('storage_connections',)]


def add_group(connection, group_id='test', environment='production', default=True):
    return connection.execute(
        'INSERT INTO marie_runtime.fabric_groups (group_id, name, environment, is_default) '
        'VALUES (%s, %s, %s, %s) RETURNING id',
        (group_id, group_id, environment, default),
    ).fetchone()[0]


def add_gateway(connection, group, gateway_id='writer', role='READ_WRITE'):
    return connection.execute(
        'INSERT INTO marie_runtime.gateway_definitions '
        '(gateway_id, name, endpoint, fabric_group_id, service_type) '
        'VALUES (%s, %s, %s, %s, %s) RETURNING id',
        (gateway_id, gateway_id, 'http://runtime.invalid', group, role),
    ).fetchone()[0]


def test_default_writer_must_be_enabled_write_capable_and_in_same_fabric(database):
    install(database)
    group = add_group(database)
    other = add_group(database, 'other', default=False)
    reader = add_gateway(database, group, 'reader', 'READ_ONLY')
    foreign_writer = add_gateway(database, other, 'foreign')
    for gateway in [reader, foreign_writer]:
        with pytest.raises(psycopg.errors.CheckViolation):
            database.execute(
                'UPDATE marie_runtime.fabric_groups SET default_write_gateway_id = %s WHERE id = %s',
                (gateway, group),
            )
    writer = add_gateway(database, group)
    database.execute(
        'UPDATE marie_runtime.fabric_groups SET default_write_gateway_id = %s WHERE id = %s',
        (writer, group),
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        database.execute('UPDATE marie_runtime.gateway_definitions SET is_enabled = false WHERE id = %s', (writer,))


def test_only_one_default_fabric_per_environment(database):
    install(database)
    add_group(database)
    with pytest.raises(psycopg.errors.UniqueViolation):
        add_group(database, 'duplicate')
    add_group(database, 'staging', environment='staging')


def test_storage_target_revision_changes_only_with_target(database):
    install(database)
    profile = database.execute(
        "INSERT INTO marie_runtime.storage_connections (name, provider, default_bucket, visibility) "
        "VALUES ('test', 's3', 'original', 'system') RETURNING id"
    ).fetchone()[0]
    database.execute('UPDATE marie_runtime.storage_connections SET name = %s WHERE id = %s', ('renamed', profile))
    assert database.execute('SELECT target_revision FROM marie_runtime.storage_connections WHERE id = %s', (profile,)).fetchone()[0] == 1
    database.execute('UPDATE marie_runtime.storage_connections SET default_bucket = %s WHERE id = %s', ('other', profile))
    assert database.execute('SELECT target_revision FROM marie_runtime.storage_connections WHERE id = %s', (profile,)).fetchone()[0] == 2


def test_reinstall_preserves_selected_and_cleared_writer(database):
    install(database)
    group = add_group(database)
    writer = add_gateway(database, group)
    database.execute(
        'UPDATE marie_runtime.fabric_groups SET default_write_gateway_id=%s WHERE id=%s',
        (writer, group),
    )
    install(database)
    assert database.execute(
        'SELECT default_write_gateway_id FROM marie_runtime.fabric_groups WHERE id=%s', (group,)
    ).fetchone()[0] == writer
    database.execute(
        'UPDATE marie_runtime.fabric_groups SET default_write_gateway_id=NULL WHERE id=%s', (group,)
    )
    install(database)
    assert database.execute(
        'SELECT default_write_gateway_id FROM marie_runtime.fabric_groups WHERE id=%s', (group,)
    ).fetchone()[0] is None
