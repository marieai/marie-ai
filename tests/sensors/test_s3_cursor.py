from datetime import datetime, timezone

import pytest

from marie.sensors.definitions.data_sink.s3_sensor import S3DataSinkSensor


def _mk_sensor(**config_overrides) -> S3DataSinkSensor:
    return S3DataSinkSensor(
        {
            "id": "t",
            "name": "t",
            "config": {
                "provider": "s3",
                "bucket": "b",
                "subtype": "s3",
                **config_overrides,
            },
        }
    )


@pytest.mark.asyncio
async def test_equal_timestamp_objects_not_skipped(monkeypatch) -> None:
    ts = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)
    sensor = _mk_sensor()
    # simulate the paginator returning one object whose LastModified equals the cursor
    page = {"Contents": [{"Key": "a.pdf", "Size": 1, "LastModified": ts, "ETag": '"x"'}]}

    class _FakePaginator:
        def paginate(self, **kwargs):
            yield page

    class _FakeClient:
        def get_paginator(self, name):
            return _FakePaginator()

    monkeypatch.setattr(sensor, "_get_client", lambda: _FakeClient())

    objects = await sensor.list_objects(after_timestamp=ts)

    assert [o.key for o in objects] == ["a.pdf"]


def test_get_client_passes_config_credentials(monkeypatch) -> None:
    sensor = _mk_sensor(
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="miniosecret",
    )

    recorded_kwargs = {}

    def _fake_client(service_name, **kwargs):
        recorded_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("boto3.client", _fake_client)

    sensor._get_client()

    assert recorded_kwargs["aws_access_key_id"] == "minioadmin"
    assert recorded_kwargs["aws_secret_access_key"] == "miniosecret"
    assert recorded_kwargs["endpoint_url"] == "http://minio:9000"


def test_get_client_falls_back_to_credential_chain_without_config_creds(
    monkeypatch,
) -> None:
    sensor = _mk_sensor(endpoint_url="http://minio:9000")

    recorded_kwargs = {}

    def _fake_client(service_name, **kwargs):
        recorded_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("boto3.client", _fake_client)

    sensor._get_client()

    assert "aws_access_key_id" not in recorded_kwargs
    assert "aws_secret_access_key" not in recorded_kwargs
