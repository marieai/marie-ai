import asyncio

import pytest

import marie.scheduler.services.notification_service as notification_service_module
from marie.scheduler.services.notification_service import NotificationService


class FakeCursor:
    def __init__(self):
        self.executed = []
        self.closed = False

    def execute(self, sql):
        self.executed.append(sql)

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.closed = False
        self.notifies = []
        self.cursor_instance = FakeCursor()
        self.autocommit = False

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True

    def notifies(self, **_kwargs):
        return iter(())


@pytest.fixture
def config():
    return {
        "username": "user",
        "password": "pass",
        "database": "db",
        "hostname": "localhost",
        "port": 5432,
        "application_name": "scheduler-test",
    }


def test_setup_connection_uses_keepalive_and_registers_channels(monkeypatch, config):
    service = NotificationService(config)
    service.register_handler("dag_state_changed", lambda _payload: None)

    captured = {}
    connection = FakeConnection()

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return connection

    monkeypatch.setattr(notification_service_module.psycopg, "connect", fake_connect)

    service._setup_connection()

    assert captured["keepalives"] == 1
    assert captured["keepalives_idle"] == 60
    assert captured["keepalives_interval"] == 10
    assert captured["keepalives_count"] == 5
    assert captured["application_name"] == "scheduler-test_listener"
    assert connection.autocommit is True
    assert connection.cursor_instance.executed == ["LISTEN dag_state_changed;"]


@pytest.mark.asyncio
async def test_notification_listener_reconnects_after_runtime_failure(
    monkeypatch, config
):
    service = NotificationService(config)
    service.register_handler("dag_state_changed", lambda _payload: None)
    service.running = True

    setup_calls = []
    close_calls = []
    sleep_calls = []
    connections = [FakeConnection(), FakeConnection()]

    def fake_setup():
        connection = connections[len(setup_calls)]
        service._listen_connection = connection
        setup_calls.append(connection)

    def fake_close():
        close_calls.append(True)
        if service._listen_connection is not None:
            service._listen_connection.closed = True
        service._listen_connection = None

    notification_calls = {"count": 0}

    def fake_next_notification():
        notification_calls["count"] += 1
        if notification_calls["count"] == 1:
            raise RuntimeError("socket gone")
        service.running = False
        return None

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(service, "_setup_connection", fake_setup)
    monkeypatch.setattr(service, "_close_connection", fake_close)
    monkeypatch.setattr(service, "_next_notification", fake_next_notification)
    monkeypatch.setattr(notification_service_module.asyncio, "sleep", fake_sleep)

    await service._listen_for_notifications()

    assert len(setup_calls) == 2
    assert len(close_calls) >= 2
    assert sleep_calls == [service._reconnect_base_delay]
    assert service.connected is False
    assert service._ever_connected is True
