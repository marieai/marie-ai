from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg_pool import ConnectionPool

from marie.storage.runtime import resolve_storage_reference, system_storage_target


def test_system_bucket_uses_existing_gateway_environment(monkeypatch):
    monkeypatch.setenv('S3_STORAGE_BUCKET_NAME', 'system-bucket')
    monkeypatch.setenv('S3_BUCKET_NAME', 'other-bucket')
    assert system_storage_target().bucket == 'system-bucket'


def test_unconfigured_system_bucket_fails_before_s3(monkeypatch):
    for key in ['S3_STORAGE_BUCKET_NAME', 'S3_BUCKET_NAME', 'S3_BUCKET']:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match='bucket'):
        system_storage_target()


@pytest.fixture
def registry_pool():
    from pathlib import Path

    dsn = os.environ.get('TEST_RUNTIME_DSN')
    if not dsn:
        pytest.skip('TEST_RUNTIME_DSN must point to a disposable PostgreSQL server')
    name = 'runtime_storage_test_' + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
        params = psycopg.conninfo.conninfo_to_dict(dsn)
        params['dbname'] = name
        try:
            with ConnectionPool(psycopg.conninfo.make_conninfo(**params), min_size=1, max_size=2) as pool:
                with pool.connection() as connection:
                    migration = Path(__file__).resolve().parents[3] / 'config/psql/schema/093_runtime_registry.sql'
                    connection.execute(migration.read_text())
                yield pool
        finally:
            admin.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))


def profile(pool, *, visibility='system', organization=None, owner=None):
    with pool.connection() as connection:
        return str(connection.execute(
            'INSERT INTO marie_runtime.storage_connections '
            '(name, provider, default_bucket, visibility, organization_id, owner_id, access_key_secret_ref, secret_key_secret_ref) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id',
            ('test', 's3', 'source-bucket', visibility, organization, owner, 'env:RUNTIME_TEST_ACCESS', 'secrets:RUNTIME_TEST_SECRET'),
        ).fetchone()[0])


def test_registered_storage_resolves_without_studio_and_pins_revision(registry_pool, monkeypatch):
    monkeypatch.setenv('RUNTIME_TEST_ACCESS', 'synthetic-access')
    monkeypatch.setenv('RUNTIME_TEST_SECRET', 'synthetic-secret')
    monkeypatch.setenv('SECRET_PROVIDER', 'env')
    connection_id = profile(registry_pool)
    reference = {'source': 'connection', 'connection_id': connection_id, 'fabric_group_id': 'default', 'target_revision': 1}
    target = resolve_storage_reference(reference, registry_pool)
    assert target.bucket == 'source-bucket'
    assert target.reference['connection_id'] == connection_id
    assert 'synthetic-secret' not in repr(target)
    with registry_pool.connection() as connection:
        connection.execute('UPDATE marie_runtime.storage_connections SET default_bucket=%s WHERE id=%s', ('changed', connection_id))
    with pytest.raises(ValueError, match='revision'):
        resolve_storage_reference(reference, registry_pool)


def test_registered_lookup_needs_only_its_connection_profile(registry_pool, monkeypatch):
    monkeypatch.setenv('RUNTIME_TEST_ACCESS', 'synthetic-access')
    monkeypatch.setenv('RUNTIME_TEST_SECRET', 'synthetic-secret')
    monkeypatch.setenv('SECRET_PROVIDER', 'env')
    connection_id = profile(registry_pool)
    with registry_pool.connection() as connection:
        connection.execute('DROP TABLE IF EXISTS marie_runtime.schema_version')
    target = resolve_storage_reference({
        'source': 'connection', 'connection_id': connection_id, 'fabric_group_id': 'default',
    }, registry_pool)
    assert target.bucket == 'source-bucket'


def test_registered_storage_never_falls_back_to_system_credentials(registry_pool, monkeypatch):
    monkeypatch.setenv('S3_ACCESS_KEY_ID', 'system-access')
    monkeypatch.setenv('S3_SECRET_ACCESS_KEY', 'system-secret')
    monkeypatch.delenv('RUNTIME_TEST_ACCESS', raising=False)
    monkeypatch.delenv('RUNTIME_TEST_SECRET', raising=False)
    connection_id = profile(registry_pool)
    with pytest.raises(ValueError, match='credential'):
        resolve_storage_reference({'source': 'connection', 'connection_id': connection_id, 'fabric_group_id': 'default'}, registry_pool)


def test_connection_visibility_is_independent_of_fabric_membership(registry_pool):
    organization, owner = uuid4(), uuid4()
    connection_id = profile(registry_pool, visibility='personal', organization=organization, owner=owner)
    reference = {
        'source': 'connection', 'connection_id': connection_id, 'fabric_group_id': 'default',
        'organization_id': str(organization), 'owner_id': str(uuid4()),
    }
    with pytest.raises(ValueError, match='access'):
        resolve_storage_reference(reference, registry_pool)


async def test_sensor_resolves_system_source_and_emits_pinned_reference(monkeypatch):
    from marie.sensors.context import SensorEvaluationContext
    from marie.sensors.definitions.data_sink.s3_sensor import S3DataSinkSensor
    from marie.sensors.types import SensorType
    from marie.storage.runtime import StorageTarget

    class Client:
        def get_paginator(self, operation):
            return self

        def paginate(self, **kwargs):
            assert kwargs['Bucket'] == 'system-bucket'
            return [{'Contents': [{'Key': 'incoming/file.txt', 'Size': 5, 'LastModified': datetime.now(timezone.utc)}]}]

        def close(self):
            pass

    monkeypatch.setenv('S3_STORAGE_BUCKET_NAME', 'system-bucket')
    monkeypatch.setattr(StorageTarget, 'create_client', lambda self: Client())
    config = {'provider': 's3', 'storage_ref': {'source': 'system', 'fabric_group_id': 'default'}}
    sensor = S3DataSinkSensor({'id': str(uuid4()), 'name': 'source', 'config': config})
    context = SensorEvaluationContext(sensor_id=sensor.sensor_id, sensor_name='source', sensor_type=SensorType.DATA_SINK, config=config)
    result = await sensor.evaluate(context)
    assert result.run_requests[0].run_config['storage_ref']['target_fingerprint']
    assert result.run_requests[0].run_config['uri'] == 's3://system-bucket/incoming/file.txt'


def test_worker_keeps_fabric_targeting_separate_from_connection_ownership():
    from marie.sensors.daemon.worker import sensor_targets_fabric

    sensor = {'config': {'fabric_group_id': 'production', 'storage_ref': {'source': 'connection', 'organization_id': str(uuid4())}}}
    assert sensor_targets_fabric(sensor, 'production')
    assert not sensor_targets_fabric(sensor, 'staging')
    assert not sensor_targets_fabric(sensor, None)


async def test_sensors_poll_selected_endpoints_with_their_own_credentials(registry_pool, monkeypatch):
    import boto3
    from botocore.config import Config
    from psycopg.rows import dict_row

    from marie.sensors.context import SensorEvaluationContext
    from marie.sensors.daemon.worker import build_job_data
    from marie.sensors.definitions.data_sink.s3_sensor import S3DataSinkSensor
    from marie.sensors.types import SensorType

    first, second = os.getenv('TEST_S3_FIRST_ENDPOINT'), os.getenv('TEST_S3_SECOND_ENDPOINT')
    if not first or not second:
        pytest.skip('Two isolated S3 endpoints are required')
    bucket, key = 'shared-source-bucket', 'incoming/sample.md'
    for endpoint, user, secret, body in [
        (first, 'runtime-first', 'runtime-first-test-secret', b'First endpoint document'),
        (second, 'runtime-second', 'runtime-second-test-secret', b'Second endpoint document'),
    ]:
        client = boto3.client('s3', endpoint_url=endpoint, region_name='us-east-1', aws_access_key_id=user,
                              aws_secret_access_key=secret, config=Config(s3={'addressing_style': 'path'}))
        if bucket not in [entry['Name'] for entry in client.list_buckets()['Buckets']]:
            client.create_bucket(Bucket=bucket)
        client.put_object(Bucket=bucket, Key=key, Body=body)
        client.close()
    for name, value in {
        'S3_STORAGE_BUCKET_NAME': bucket, 'S3_ENDPOINT_URL': first,
        'S3_ACCESS_KEY_ID': 'runtime-first', 'S3_SECRET_ACCESS_KEY': 'runtime-first-test-secret',
        'RUNTIME_SECOND_ACCESS': 'runtime-second', 'RUNTIME_SECOND_SECRET': 'runtime-second-test-secret',
        'SECRET_PROVIDER': 'env',
    }.items():
        monkeypatch.setenv(name, value)
    with registry_pool.connection() as connection:
        profile_id = str(connection.execute(
            'INSERT INTO marie_runtime.storage_connections '
            '(name, provider, endpoint, default_bucket, visibility, access_key_secret_ref, secret_key_secret_ref) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id',
            ('second', 's3', second, bucket, 'system', 'env:RUNTIME_SECOND_ACCESS', 'secrets:RUNTIME_SECOND_SECRET'),
        ).fetchone()[0])

    class SensorPool:
        async def fetchrow(self, query, *args):
            def read():
                with registry_pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
                    return cursor.execute(query.replace('$1', '%s'), args).fetchone()
            return await asyncio.to_thread(read)

    for reference, expected in [
        ({'source': 'system', 'fabric_group_id': 'default'}, b'First endpoint document'),
        ({'source': 'connection', 'connection_id': profile_id, 'fabric_group_id': 'default', 'target_revision': 1}, b'Second endpoint document'),
    ]:
        config = {'provider': 's3', 'prefix': 'incoming/', 'file_patterns': ['*.md'], 'storage_ref': reference}
        sensor = S3DataSinkSensor({'id': str(uuid4()), 'name': 'source', 'config': config})
        context = SensorEvaluationContext(sensor_id=sensor.sensor_id, sensor_name='source', sensor_type=SensorType.DATA_SINK,
                                          config=config, resources={'postgres_pool': SensorPool(), 'fabric_group_id': 'default'})
        result = await sensor.evaluate(context)
        payload = build_job_data(result.run_requests[0], sensor.sensor_id, sensor.name)['metadata']
        if reference['source'] == 'connection':
            assert 'uri' not in payload
            assert payload['remote_source']['storage_ref']['connection_id'] == profile_id
            assert payload['remote_source']['uri'] == f's3://{bucket}/{key}'
        else:
            assert payload['storage_ref']['source'] == 'system'
            assert payload['uri'] == f's3://{bucket}/{key}'
        assert result.run_requests[0].run_config['size'] == len(expected)
